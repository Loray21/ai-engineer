import asyncio
import logging
import random

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d | %(message)s",  # con milisegundos para ver el solapamiento
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


async def gpt_4_call(sem: asyncio.Semaphore, timeout: float):
    async with sem:                              # espera turno si ya hay N corriendo
        try:
            async with asyncio.timeout(timeout):     # cuenta solo la llamada, no la espera del semáforo
                latency = random.uniform(0.5, 2.5)       # latencia de red simulada
                logger.info("gpt-4       -> inicio (latencia %.2fs)", latency)
                await asyncio.sleep(latency)
                logger.info("gpt-4       <- fin")
                return {"model": "gpt-4", "ok": True, "content": "the day is cloud"}
        except TimeoutError:
            logger.warning("gpt-4       xx TimeoutError")
            raise                                # se re-lanza para que gather la devuelva como resultado

         # await, NO time.sleep (bloquearía el loop)
async def claude_3_call(sem: asyncio.Semaphore, timeout: float):
    async with sem:
        try:
            async with asyncio.timeout(timeout):
                latency = random.uniform(0.7, 3)       # latencia de red simulada
                logger.info("claude-3    -> inicio (latencia %.2fs)", latency)
                await asyncio.sleep(latency)
                logger.info("claude-3    <- fin")
                return {"model": "claude-3", "ok": True, "content": "the day is cloud"}
        except TimeoutError:
            logger.warning("claude-3    xx TimeoutError")
            raise


async def local_llama_call(sem: asyncio.Semaphore, timeout: float):
    async with sem:
        try:
            async with asyncio.timeout(timeout):
                latency = random.uniform(0.9, 3.)       # latencia de red simulada
                logger.info("local-llama -> inicio (latencia %.2fs)", latency)
                await asyncio.sleep(latency)
                logger.info("local-llama <- fin")
                return {"model": "local-llama", "ok": True, "content": "the day is cloud"}
        except TimeoutError:
            logger.warning("local-llama xx TimeoutError")
            raise


async def orcchestor(timeout: float = 2.0):
    sem = asyncio.Semaphore(3)                   # máximo 3 llamadas en paralelo
    tasks = [
        gpt_4_call(sem, timeout),
        claude_3_call(sem, timeout),
        local_llama_call(sem, timeout)
    ]
    return await asyncio.gather(*tasks, return_exceptions=True,)


if __name__ == "__main__":
    asyncio.run(orcchestor())
