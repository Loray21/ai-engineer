import asyncio
import os
import sys
from functools import cache

from chromadb.api.models.Collection import Collection
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

from ingest import ingest
from schemas import RespuestaRAG

# Cantidad de chunks que se le pasan al LLM. Pocos y relevantes: con demasiados se pasa el límite
# de tokens y el modelo presta menos atención a lo que queda en el medio del contexto.
TOP_K = 4


# ---------------------------------------------------------------------------
# 1. Recuperación
# ---------------------------------------------------------------------------

@cache
def get_collection() -> Collection:
    """
    Abre la colección una sola vez y la reutiliza en todas las consultas.
    ingest() indexa solo si la colección está vacía, así que la consulta funciona
    aunque no se haya corrido antes ingest.py.
    """
    return ingest()


def retrieve(query: str) -> list[dict]:
    """Devuelve los TOP_K chunks más parecidos a la pregunta, del más al menos parecido."""
    # query_texts: Chroma convierte la pregunta en un embedding con la misma función con la que
    # se indexaron los documentos y busca los vectores más cercanos
    results = get_collection().query(query_texts=[query], n_results=TOP_K)

    # query acepta varias preguntas a la vez y devuelve una lista por pregunta: [0] es la única que mandamos
    return [
        {"id": chunk_id, "text": text, "distance": distance}
        for chunk_id, text, distance in zip(
            results["ids"][0], results["documents"][0], results["distances"][0]
        )
    ]


# RunnableLambda convierte la función en un Runnable: se puede encadenar con | y
# tiene .ainvoke() (corre la función en un hilo aparte para no bloquear el event loop)
retriever = RunnableLambda(retrieve)


def formatear_contexto(datos: dict) -> str:
    """Une los chunks en un solo texto, cada uno precedido por su id para que el modelo pueda citarlo."""
    return "\n\n".join(f"[{chunk['id']}]\n{chunk['text']}" for chunk in datos["chunks"])


# ---------------------------------------------------------------------------
# 2. El modelo
# ---------------------------------------------------------------------------

# Proveedor -> (clase de LangChain, modelo por defecto). Las dos clases tienen la misma interfaz,
# así que el resto de la cadena no depende del proveedor elegido.
PROVIDERS: dict[str, tuple[type[BaseChatModel], str]] = {
    "openai": (ChatOpenAI, "gpt-4o-mini"),
    "anthropic": (ChatAnthropic, "claude-opus-5-5"),
}


def build_model() -> BaseChatModel:
    """Arma el modelo a partir de LLM_PROVIDER, <PROVEEDOR>_API_KEY y <PROVEEDOR>_MODEL del entorno."""
    provider = (os.getenv("LLM_PROVIDER") or "openai").strip().lower()
    if provider not in PROVIDERS:
        raise ValueError(f"Proveedor desconocido: {provider!r}. Opciones: {', '.join(PROVIDERS)}")

    prefix = provider.upper()
    api_key = os.getenv(f"{prefix}_API_KEY")
    if not api_key:
        raise ValueError(f"Falta {prefix}_API_KEY en el .env para usar el proveedor {provider}")

    model_class, default_model = PROVIDERS[provider]
    args = {
        "model": os.getenv(f"{prefix}_MODEL") or default_model,
        "api_key": api_key,
        "max_tokens": 1000,
        "max_retries": 2,       # reintentos del SDK ante errores de red o rate limit
        "timeout": 60,
    }
    # temperature=0: respuestas más estables, conviene cuando solo se tiene que repetir lo que dice el contexto.
    # Solo para OpenAI: los modelos recientes de Anthropic no la aceptan.
    if provider == "openai":
        args["temperature"] = 0
    return model_class(**args)


# ---------------------------------------------------------------------------
# 3. El prompt
# ---------------------------------------------------------------------------

# El parser genera las instrucciones de formato (el JSON Schema de RespuestaRAG) y después
# convierte el texto que devuelve el modelo en un objeto RespuestaRAG validado
parser = PydanticOutputParser(pydantic_object=RespuestaRAG)

# El system prompt funciona como filtro de veracidad: el modelo solo puede usar el CONTEXTO
PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Sos un asistente que responde preguntas sobre el Refugio Frey, en Bariloche.\n\n"
        "Reglas:\n"
        "1. Respondé únicamente con la información del CONTEXTO. No uses tu conocimiento propio, "
        "aunque sepas la respuesta.\n"
        "2. Si el CONTEXTO no contiene la respuesta, respondé exactamente \"No lo sé\", "
        "con encontrada=false y sin referencias. No completes con datos que no estén escritos "
        "en el CONTEXTO (cifras, fechas, nombres).\n"
        "3. En referencias poné los ids, tal como aparecen entre corchetes, "
        "de los fragmentos que usaste para responder.\n\n"
        "{format_instructions}",
    ),
    ("human", "CONTEXTO:\n<contexto>\n{contexto}\n</contexto>\n\nPREGUNTA: {pregunta}"),
]).partial(format_instructions=parser.get_format_instructions())
# .partial() completa format_instructions una sola vez: lo que cambia en cada llamada es contexto y pregunta


# ---------------------------------------------------------------------------
# 4. La cadena LCEL
# ---------------------------------------------------------------------------

class ReferenciaInvalidaError(Exception):
    """El modelo citó un fragmento que no estaba entre los recuperados."""


def validar_referencias(datos: dict) -> RespuestaRAG:
    """Controla que cada referencia citada sea uno de los chunks que se le pasaron al modelo."""
    respuesta: RespuestaRAG = datos["respuesta"]
    ids_recuperados = {chunk["id"] for chunk in datos["chunks"]}
    invalidas = [ref for ref in respuesta.referencias if ref not in ids_recuperados]
    if invalidas:
        raise ReferenciaInvalidaError(f"Referencias que no están en el contexto: {invalidas}")
    return respuesta


def build_chain() -> Runnable:
    model = build_model()

    # Cada paso recibe el diccionario del anterior. RunnablePassthrough.assign le AGREGA una clave
    # sin borrar las que ya tenía: así los chunks llegan hasta el final para validar las referencias.
    #   "pregunta"
    #   -> {chunks, pregunta}
    #   -> {chunks, pregunta, contexto}
    #   -> {chunks, pregunta, contexto, respuesta}
    #   -> RespuestaRAG
    chain = (
        {"chunks": retriever, "pregunta": RunnablePassthrough()}
        | RunnablePassthrough.assign(contexto=formatear_contexto)
        | RunnablePassthrough.assign(respuesta=PROMPT | model | parser)
        | RunnableLambda(validar_referencias)
    )

    # Si el modelo devuelve un JSON inválido, incoherente o con referencias inventadas,
    # se reintenta la cadena una vez más con una nueva llamada al modelo
    return chain.with_retry(
        retry_if_exception_type=(OutputParserException, ReferenciaInvalidaError),
        stop_after_attempt=2,
    )


@cache
def get_chain() -> Runnable:
    """Arma la cadena la primera vez que se usa y después devuelve siempre la misma."""
    load_dotenv()   # las keys se leen del .env, nunca se escriben en el código
    return build_chain()


async def get_rag_response(query: str) -> RespuestaRAG:
    """Responde la pregunta usando solo los documentos indexados en ChromaDB."""
    if not query.strip():
        raise ValueError("La pregunta está vacía")
    # ainvoke: mientras espera al LLM, el event loop puede atender otras tareas
    return await get_chain().ainvoke(query)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")   # acentos en la consola de Windows
    query = " ".join(sys.argv[1:]) or "¿Cuánto se tarda en llegar al refugio?"
    print(f"\nPregunta: {query}\n")
    resultado = asyncio.run(get_rag_response(query))
    print(resultado.model_dump_json(indent=2))
