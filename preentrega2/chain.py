import logging
import os
import time
from functools import cache

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from schemas import ExtraccionTecnica, LLMConfig

logger = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# 1. El modelo
# ---------------------------------------------------------------------------

# Registro de proveedores: nombre -> (clase de LangChain, modelo por defecto).
# LangChain ya trae un cliente por proveedor (ChatOpenAI, ChatAnthropic) con la misma interfaz,
# así que el resto del pipeline no sabe qué proveedor hay atrás.
PROVIDERS: dict[str, tuple[type[BaseChatModel], str]] = {
    "openai": (ChatOpenAI, "gpt-4o-mini"),
    "anthropic": (ChatAnthropic, "claude-opus-5-5"),
}


def load_config() -> LLMConfig:
    """Lee LLM_PROVIDER y, según el proveedor, <PROVEEDOR>_API_KEY y <PROVEEDOR>_MODEL."""
    provider = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if provider not in PROVIDERS:
        raise ValueError(f"Proveedor desconocido: {provider!r}. Opciones: {', '.join(PROVIDERS)}")

    prefix = provider.upper()
    api_key = os.getenv(f"{prefix}_API_KEY")
    if not api_key:
        raise ValueError(f"Falta {prefix}_API_KEY para usar el proveedor {provider}")

    # Una variable vacía del .env cuenta como no definida
    datos = {"provider": provider, "api_key": api_key, "model": os.getenv(f"{prefix}_MODEL") or None}

    # Opcional: bajar el tope de tokens sirve para probar la detección de respuestas cortadas.
    # Llega como string: Pydantic lo convierte a int y valida que sea mayor que 0.
    if max_tokens := os.getenv("LLM_MAX_TOKENS"):
        datos["max_tokens"] = max_tokens
    return LLMConfig(**datos)


def build_model(config: LLMConfig) -> BaseChatModel:
    model_class, default_model = PROVIDERS[config.provider]
    args = {
        "model": config.model or default_model,
        "api_key": config.api_key,          # LangChain también recibe SecretStr: la key no se ve en los logs
        "max_tokens": config.max_tokens,
        # Reintentos del SDK: errores de red, timeouts y rate limit (la primera capa de resiliencia).
        # Los errores de formato de la respuesta los maneja .with_retry() más adelante (la segunda capa).
        "max_retries": 2,
        "timeout": 60,
    }
    # temperature=0 hace la salida más estable, que es lo que conviene al extraer datos.
    # Solo para OpenAI: los modelos recientes de Anthropic no la aceptan.
    if config.provider == "openai":
        args["temperature"] = 0
    return model_class(**args)


# ---------------------------------------------------------------------------
# 2. El prompt
# ---------------------------------------------------------------------------

# Las instrucciones fijas van aparte del template: se pueden cambiar o testear sin tocar el prompt.
INSTRUCCIONES_FORMATO = """\
- tecnologias: solo las que aparecen en el texto, con su nombre propio (por ejemplo "PostgreSQL", no "base de datos"). No inventes ninguna.
  Si el texto no menciona ninguna tecnología, no uses valores de relleno como "<UNKNOWN>" o "N/A".
- nivel_de_criticidad:
  - alta: caída del servicio, pérdida o corrupción de datos, problema de seguridad o impacto directo en usuarios.
  - media: degradación de rendimiento, errores intermitentes o un riesgo que todavía no causa impacto.
  - baja: descripción de arquitectura, mejoras o tareas sin un problema activo.
- resumen_tecnico: una o dos oraciones en español, sin repetir el texto original."""

# ChatPromptTemplate en lugar de f-strings: {texto} y {instrucciones_formato} no se reemplazan acá.
# LangChain los completa al ejecutar la cadena, valida que no falte ninguno
# y el mismo template sirve para cualquier texto.
PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Sos un ingeniero de software senior que analiza textos técnicos "
        "(descripciones de arquitectura, logs de error, reportes de incidentes) "
        "y extrae su información en forma estructurada.\n\n"
        "Reglas para cada campo:\n{instrucciones_formato}",
    ),
    ("human", "Texto a analizar:\n<texto>\n{texto}\n</texto>"),
]).partial(instrucciones_formato=INSTRUCCIONES_FORMATO)
# .partial() completa una variable por adelantado. El prompt que queda solo pide {texto},
# que es lo único que cambia en cada llamada.


# ---------------------------------------------------------------------------
# 3. Validación de la salida y cadena LCEL con reintento
# ---------------------------------------------------------------------------

class SalidaIncompletaError(Exception):
    """El modelo cortó la respuesta al llegar a max_tokens: el JSON quedó a medias."""


class SalidaInvalidaError(Exception):
    """El modelo respondió completo, pero la respuesta no cumple el schema."""


# Errores que vale la pena reintentar: son de la RESPUESTA, y otra llamada puede salir bien.
# Los errores de red no van acá: ya los reintenta el SDK (max_retries). Si también los reintentara
# .with_retry(), un corte de red podría generar 3 x 3 = 9 llamadas.
ERRORES_REINTENTABLES = (SalidaIncompletaError, SalidaInvalidaError)

# Motivos de fin que indican que la respuesta se cortó por tokens:
# "length" es el finish_reason de OpenAI y "max_tokens" el stop_reason de Anthropic.
MOTIVOS_DE_CORTE = {"length", "max_tokens"}


def resumir_error(exc: BaseException) -> str:
    """Pasa un ValidationError de varias líneas a una sola: 'campo: mensaje; campo: mensaje'."""
    causa = exc if isinstance(exc, ValidationError) else exc.__cause__
    if isinstance(causa, ValidationError):
        return "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in causa.errors())
    return str(exc).splitlines()[0]


def validar_salida(resultado: dict, config: RunnableConfig) -> ExtraccionTecnica:
    """
    Recibe lo que devuelve with_structured_output(include_raw=True):
      - raw: el mensaje crudo del modelo (trae el finish_reason en response_metadata)
      - parsed: el objeto ExtraccionTecnica, o None si no se pudo armar
      - parsing_error: la excepción del parseo o de la validación de Pydantic, o None
    Devuelve el objeto validado o lanza una excepción que .with_retry() sabe reintentar.
    """
    # La etiqueta viaja en la metadata de la config y sirve para distinguir los logs de cada texto
    etiqueta = config.get("metadata", {}).get("etiqueta", "-")
    metadata = resultado["raw"].response_metadata
    motivo = metadata.get("finish_reason") or metadata.get("stop_reason")

    # 1. Primero el corte: si la respuesta quedó a medias, cualquier error de parseo es una
    #    consecuencia, y un objeto que igual se haya podido armar no es confiable.
    if motivo in MOTIVOS_DE_CORTE:
        logger.warning("[%s] Respuesta cortada por falta de tokens (motivo=%s)", etiqueta, motivo)
        raise SalidaIncompletaError(f"El modelo cortó la respuesta por falta de tokens (motivo={motivo})")

    # 2. Respuesta completa pero inválida: JSON mal formado o datos que no cumplen el schema
    if resultado["parsing_error"] is not None:
        detalle = resumir_error(resultado["parsing_error"])
        logger.warning("[%s] La respuesta no cumple el schema: %s", etiqueta, detalle)
        raise SalidaInvalidaError(detalle) from resultado["parsing_error"]

    # 3. Sin error y sin objeto: el modelo contestó con texto en vez de usar el schema
    if resultado["parsed"] is None:
        logger.warning("[%s] El modelo no devolvió datos estructurados (motivo=%s)", etiqueta, motivo)
        raise SalidaInvalidaError("El modelo no devolvió datos estructurados")

    logger.info("[%s] Validación OK (motivo=%s)", etiqueta, motivo)
    return resultado["parsed"]


def build_chain(config: LLMConfig) -> Runnable:
    model = build_model(config)

    # with_structured_output le pasa el schema de Pydantic al modelo y parsea la respuesta.
    # - method="function_calling": el schema viaja como una herramienta que el modelo está obligado
    #   a llamar. Se fija para que los dos proveedores usen el mismo mecanismo y el finish_reason
    #   se controle igual en ambos.
    # - include_raw=True: además del objeto devuelve el mensaje crudo, que trae el finish_reason.
    #   Sin esto, un error de parseo lanzaría una excepción y no podríamos ver por qué falló.
    structured_model = model.with_structured_output(
        ExtraccionTecnica, method="function_calling", include_raw=True
    )

    # LCEL: cada | conecta la salida de un paso con la entrada del siguiente.
    #   {"texto"} -> PROMPT -> mensajes -> modelo -> {raw, parsed, parsing_error} -> validar_salida -> objeto
    chain = PROMPT | structured_model | RunnableLambda(validar_salida)

    # .with_retry() envuelve la cadena ENTERA: si validar_salida lanza un error reintentable,
    # se vuelve a ejecutar desde el prompt, con una nueva llamada al modelo.
    return chain.with_retry(
        retry_if_exception_type=ERRORES_REINTENTABLES,
        stop_after_attempt=3,               # 1 intento + 2 reintentos
        wait_exponential_jitter=True,       # espera creciente con algo de azar entre intentos
    )


# ---------------------------------------------------------------------------
# 4. Función pública del pipeline
# ---------------------------------------------------------------------------

class ExtraccionError(Exception):
    """No se obtuvo una extracción válida después de todos los intentos."""


@cache
def get_chain() -> Runnable:
    """
    Arma la cadena la primera vez que se usa y después devuelve siempre la misma.
    No se arma al importar el módulo porque en ese momento el .env todavía puede no estar cargado.
    """
    return build_chain(load_config())


async def process_text(text: str, etiqueta: str = "-") -> ExtraccionTecnica:
    """
    Extrae las entidades técnicas de un texto y devuelve el objeto validado.
    etiqueta identifica el texto en los logs, útil cuando se procesan varios en paralelo.

    Lanza:
      - ValueError si el texto está vacío (no tiene sentido gastar una llamada al LLM).
      - ExtraccionError si después de todos los intentos la respuesta sigue incompleta o inválida.
      - Los errores del proveedor (key inválida, sin saldo, red caída tras los reintentos del SDK)
        se dejan pasar tal cual: no son un problema de la respuesta y reintentar acá no los arregla.
    """
    if not text.strip():
        raise ValueError("El texto a procesar está vacío")

    logger.info("[%s] Procesando texto de %d caracteres", etiqueta, len(text))
    inicio = time.perf_counter()
    try:
        # .ainvoke() es la versión asíncrona de .invoke(): mientras espera la respuesta del LLM,
        # el event loop puede avanzar con otros textos (por ejemplo, con asyncio.gather).
        resultado: ExtraccionTecnica = await get_chain().ainvoke(
            {"texto": text},
            config={"metadata": {"etiqueta": etiqueta}},
        )
    except ERRORES_REINTENTABLES as exc:
        logger.error("[%s] Se agotaron los intentos. Último error: %s", etiqueta, exc)
        raise ExtraccionError(f"No se pudo extraer información válida: {exc}") from exc

    logger.info(
        "[%s] Extracción lista en %.1fs: %d tecnologías, criticidad %s",
        etiqueta, time.perf_counter() - inicio, len(resultado.tecnologias), resultado.nivel_de_criticidad,
    )
    return resultado
