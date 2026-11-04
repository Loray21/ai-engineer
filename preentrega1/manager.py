import os
from collections.abc import AsyncGenerator
from contextlib import aclosing

import anthropic
import openai

from clients import AnthropicClient, BaseLLMClient, OpenAIClient, RefusalError
from schemas import ChatMessage, GenerationParams, LLMConfig, LLMError, ModelResponse, Provider, StreamChunk

# Errores que se devuelven como LLMError. Cualquier otro es un bug nuestro y conviene que se vea.
HANDLED_ERRORS = (openai.APIError, anthropic.APIError, RefusalError, ValueError)


def to_llm_error(exc: Exception) -> LLMError:
    # El orden importa: APITimeoutError hereda de APIConnectionError
    if isinstance(exc, (openai.RateLimitError, anthropic.RateLimitError)):
        code = "rate_limit"
    elif isinstance(exc, (openai.APITimeoutError, anthropic.APITimeoutError)):
        code = "timeout"
    elif isinstance(exc, (openai.APIConnectionError, anthropic.APIConnectionError)):
        code = "connection"
    elif isinstance(exc, (openai.AuthenticationError, anthropic.AuthenticationError)):
        code = "auth"
    elif isinstance(exc, (openai.BadRequestError, anthropic.BadRequestError, ValueError)):
        code = "invalid_request"
    elif isinstance(exc, RefusalError):
        code = "refusal"
    else:
        code = "api"
    return LLMError(code=code, message=str(exc))


def create_client(config: LLMConfig) -> BaseLLMClient:
    if config.provider == Provider.OPENAI:
        if config.openai_api_key is None:
            raise ValueError("Falta OPENAI_API_KEY para usar el proveedor openai")
        return OpenAIClient(config.openai_api_key, config.openai_model)

    if config.anthropic_api_key is None:
        raise ValueError("Falta ANTHROPIC_API_KEY para usar el proveedor anthropic")
    return AnthropicClient(config.anthropic_api_key, config.anthropic_model)


class AsyncLLMManager:
    def __init__(self, config: LLMConfig):
        self._client = create_client(config)

    @classmethod
    def from_env(cls) -> "AsyncLLMManager":
        values = {
            "provider": os.getenv("LLM_PROVIDER"),
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
            "openai_model": os.getenv("OPENAI_MODEL"),
            "anthropic_model": os.getenv("ANTHROPIC_MODEL"),
        }
        # Las variables vacías del .env cuentan como no definidas, así aplican los valores por defecto
        return cls(LLMConfig(**{name: value for name, value in values.items() if value}))

    async def generate(self, messages: list[ChatMessage], params: GenerationParams | None = None) -> ModelResponse:
        try:
            return await self._client.generate(messages, params or GenerationParams())
        except HANDLED_ERRORS as exc:
            return ModelResponse(
                ok=False,
                provider=self._client.provider,
                model=self._client.model,
                error=to_llm_error(exc),
            )

    async def stream(
        self, messages: list[ChatMessage], params: GenerationParams | None = None
    ) -> AsyncGenerator[StreamChunk, None]:
        # Generador que envuelve a otro: pide tokens al cliente y los reentrega como StreamChunk.
        # Un generador async no puede hacer "return valor", por eso el error sale como un último yield.
        # No se reintenta a mitad de camino: el consumidor ya recibió texto y se duplicaría.
        try:
            # aclosing cierra el generador del cliente aunque el consumidor corte con break
            async with aclosing(self._client.stream(messages, params or GenerationParams())) as tokens:
                async for text in tokens:
                    yield StreamChunk(text=text)
        except HANDLED_ERRORS as exc:
            yield StreamChunk(error=to_llm_error(exc))
