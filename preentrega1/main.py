import asyncio
import sys

from dotenv import load_dotenv

from manager import AsyncLLMManager
from schemas import ChatMessage

QUESTION = "¿Qué es la entropía? Respondé en dos oraciones."


async def main() -> int:
    load_dotenv()

    try:
        manager = AsyncLLMManager.from_env()
    except ValueError as exc:                    # los errores de validación de Pydantic también son ValueError
        print(f"Error de configuración: {exc}")
        return 1

    messages = [ChatMessage(role="user", content=QUESTION)]
    exit_code = 0

    print("--- Modo normal ---")
    response = await manager.generate(messages)
    if response.ok:
        print(f"[{response.provider.value} / {response.model}] {response.text}")
    else:
        print(f"Error controlado ({response.error.code}): {response.error.message}")
        exit_code = 1

    print("\n--- Modo streaming ---")
    async for chunk in manager.stream(messages):
        if chunk.error:
            print(f"\nError controlado ({chunk.error.code}): {chunk.error.message}")
            exit_code = 1
        else:
            print(chunk.text, end="", flush=True)    # muestra cada fragmento apenas llega
    print()

    return exit_code


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")     # evita errores con símbolos de la respuesta en Windows
    sys.exit(asyncio.run(main()))
