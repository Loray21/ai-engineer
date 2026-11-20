import asyncio
import sys

from rag import get_chain, get_collection, get_rag_response

# Cada caso declara si la respuesta está en los documentos. El script verifica que el modelo
# responda cuando corresponde y que diga "No lo sé" cuando no, en lugar de inventar.
CASOS = [
    {
        "etiqueta": "en-documentos",
        "pregunta": "¿Cuánto se tarda en llegar al refugio caminando desde Villa Catedral?",
        "encontrada": True,
    },
    {
        # Pregunta trampa: los documentos hablan del pernocte y de las reservas, pero nunca de precios.
        # El retriever va a traer chunks parecidos, y el modelo no tiene que completar el dato que falta.
        "etiqueta": "trampa",
        "pregunta": "¿Cuánto cuesta pasar la noche en el refugio?",
        "encontrada": False,
    },
]


async def probar(caso: dict) -> bool:
    """Hace la pregunta, muestra la respuesta y devuelve True si se comportó como se esperaba."""
    resultado = await get_rag_response(caso["pregunta"])
    cumple = resultado.encontrada == caso["encontrada"]
    marca = "OK " if cumple else "FALLA"
    print(
        f"\n[{marca}] {caso['etiqueta']}: {caso['pregunta']}\n"
        f"Se esperaba encontrada={caso['encontrada']} y se obtuvo encontrada={resultado.encontrada}\n"
        f"{resultado.model_dump_json(indent=2)}"
    )
    return cumple


async def main() -> int:
    # Se preparan la cadena y la colección antes de lanzar las preguntas en paralelo:
    # - si falta la key, se ve un solo error de configuración y no uno por pregunta
    # - la colección se abre (y, si hace falta, se indexa) una sola vez, en vez de que
    #   dos preguntas intenten indexar al mismo tiempo
    try:
        get_chain()
    except ValueError as exc:
        print(f"Error de configuración: {exc}")
        return 1
    get_collection()

    # asyncio.gather lanza las preguntas a la vez: mientras una espera al LLM, avanza la otra
    resultados = await asyncio.gather(*(probar(caso) for caso in CASOS))

    print(f"\n{sum(resultados)}/{len(resultados)} casos se comportaron como se esperaba")
    return 0 if all(resultados) else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")   # acentos en la consola de Windows
    sys.exit(asyncio.run(main()))
