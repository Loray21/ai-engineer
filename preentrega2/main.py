import asyncio
import logging
import sys

from dotenv import load_dotenv

from chain import ExtraccionError, get_chain, process_text

# Cada caso declara qué se espera: "ok" (objeto válido) o "error" (ExtraccionError después de los reintentos).
# Así el script verifica el comportamiento del pipeline y no solo que corra.
CASOS = [
    {
        "etiqueta": "arquitectura",
        "espera": "ok",
        "texto": (
            "Arquitectura de microservicios: el frontend en React consume una API en FastAPI "
            "que persiste en PostgreSQL, usa Redis como caché y publica eventos en Kafka. "
            "Todo se despliega con Docker sobre Kubernetes."
        ),
    },
    {
        "etiqueta": "log-error",
        "espera": "ok",
        "texto": (
            "2026-11-10 03:14:07 ERROR [payment-service] psycopg2.OperationalError: FATAL: "
            "remaining connection slots are reserved for non-replication superuser connections\n"
            "2026-11-10 03:14:07 ERROR [payment-service] Celery task process_payment failed after 3 retries\n"
            "2026-11-10 03:14:08 WARN  [nginx] upstream timed out (110: Connection timed out) "
            "while reading response header from upstream"
        ),
    },
    {
        # Ambiguo: casi no nombra tecnologías y no queda claro qué tan grave es
        "etiqueta": "ambiguo",
        "espera": "ok",
        "texto": (
            "Algo anda raro con el login desde ayer: a algunos usuarios no les llega el mail "
            "de verificación y a otros la página les queda cargando. Puede ser el Nginx o el "
            "servicio de mails, todavía no sabemos."
        ),
    },
    {
        # Sin ninguna tecnología: el schema exige al menos una, así que debe terminar en error controlado
        "etiqueta": "sin-tecnologias",
        "espera": "error",
        "texto": "Ayer la reunión de planificación se extendió y quedaron temas pendientes para el viernes.",
    },
]


async def procesar_caso(caso: dict) -> bool:
    """Procesa un caso, muestra el resultado y devuelve True si salió como se esperaba."""
    etiqueta = caso["etiqueta"]
    try:
        resultado = await process_text(caso["texto"], etiqueta=etiqueta)
        obtenido = "ok"
        detalle = resultado.model_dump_json(indent=2)
    except ExtraccionError as exc:
        obtenido = "error"
        detalle = str(exc)

    cumple = obtenido == caso["espera"]
    marca = "OK " if cumple else "FALLA"
    print(f"\n[{marca}] {etiqueta}: se esperaba '{caso['espera']}' y se obtuvo '{obtenido}'\n{detalle}")
    return cumple


async def main() -> int:
    load_dotenv()

    # La cadena se arma acá, antes de lanzar los casos: si falta la key o el proveedor es inválido,
    # se ve un solo mensaje claro en lugar de cuatro errores iguales en paralelo.
    try:
        get_chain()
    except ValueError as exc:                    # los errores de validación de Pydantic también son ValueError
        print(f"Error de configuración: {exc}")
        return 1

    # asyncio.gather lanza los cuatro casos a la vez: mientras uno espera al LLM, avanzan los otros.
    # El tiempo total se acerca al del caso más lento y no a la suma de todos.
    resultados = await asyncio.gather(*(procesar_caso(caso) for caso in CASOS))

    print(f"\n{sum(resultados)}/{len(resultados)} casos se comportaron como se esperaba")
    return 0 if all(resultados) else 1


if __name__ == "__main__":
    # Evita errores con acentos y símbolos en la consola de Windows.
    # stderr también, porque logging escribe ahí por defecto.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Cuando la respuesta se corta, el parser de LangChain loguea un ERROR con un traceback largo.
    # validar_salida ya registra ese caso en una línea con la causa real, así que se silencia el duplicado.
    logging.getLogger("langchain_core.output_parsers.openai_tools").setLevel(logging.CRITICAL)
    sys.exit(asyncio.run(main()))
