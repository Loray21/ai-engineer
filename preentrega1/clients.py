from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from pydantic import SecretStr

from schemas import ChatMessage, GenerationParams, ModelResponse, Provider


class RefusalError(Exception):
    """
    Error propio para cuando el modelo se niega a contestar.
    La API responde bien a nivel HTTP, así que el SDK no lanza ningún error por su cuenta.
    Sin esta clase, el rechazo parecería una respuesta correcta pero vacía.
    """


class BaseLLMClient(ABC):
    """
    Contrato común: define qué tiene que saber hacer cualquier cliente de LLM.
    - generate(): espera la respuesta completa y la devuelve de una sola vez.
    - stream(): entrega la respuesta de a pedazos, a medida que llega.
    El resto del programa usa solo esta interfaz, sin saber qué proveedor hay atrás.
    Es abstracta: no se puede instanciar, solo heredar.
    """
    provider: Provider
    model: str

    @abstractmethod
    async def generate(self, messages: list[ChatMessage], params: GenerationParams) -> ModelResponse:
        pass

    # Con def y no async def: acá no hay yield, así que async def la convertiría
    # en una corrutina y no coincidiría con los generadores de las clases hijas.
    @abstractmethod
    def stream(self, messages: list[ChatMessage], params: GenerationParams) -> AsyncGenerator[str, None]:
        pass


class OpenAIClient(BaseLLMClient):
    """
    Implementa el contrato usando el SDK asíncrono de OpenAI.
    Traduce nuestros ChatMessage y GenerationParams al formato de OpenAI,
    y de su respuesta saca solo el texto (choices[0].message.content).
    """
    provider = Provider.OPENAI

    def __init__(self, api_key: SecretStr, model: str):
        # max_retries reintenta el pedido inicial ante errores de red y de rate limit
        self._client = AsyncOpenAI(api_key=api_key.get_secret_value(), max_retries=2)
        self.model = model

    def _request_args(self, messages: list[ChatMessage], params: GenerationParams) -> dict:
        args = {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
            "max_completion_tokens": params.max_tokens,  # max_tokens está deprecado en OpenAI
        }
        if params.temperature is not None:
            args["temperature"] = params.temperature
        return args

    async def generate(self, messages: list[ChatMessage], params: GenerationParams) -> ModelResponse:
        response = await self._client.chat.completions.create(**self._request_args(messages, params))

        # OpenAI: el texto viene en choices[0].message.content
        # Ejemplo de respuesta (objeto ChatCompletion del SDK, con valores ilustrativos):
        # ChatCompletion(
        #     id="chatcmpl-B9MBs8CjcvOU2jLn4n570S5qMJKcT",
        #     object="chat.completion",
        #     created=1741569952,
        #     model="gpt-4o-mini-2024-07-18",
        #     choices=[
        #         Choice(
        #             index=0,
        #             message=ChatCompletionMessage(
        #                 role="assistant",
        #                 content="La entropía mide el desorden de un sistema.",
        #                 refusal=None,
        #                 tool_calls=None,
        #             ),
        #             finish_reason="stop",
        #         ),
        #     ],
        #     usage=CompletionUsage(prompt_tokens=10, completion_tokens=12, total_tokens=22),
        # )
        # -> "La entropía mide el desorden de un sistema."
        message = response.choices[0].message
        if message.refusal:
            raise RefusalError(message.refusal)
        return ModelResponse(ok=True, text=message.content or "", provider=self.provider, model=self.model)

    async def stream(self, messages: list[ChatMessage], params: GenerationParams) -> AsyncGenerator[str, None]:
        response = await self._client.chat.completions.create(
            **self._request_args(messages, params), stream=True
        )
        # async with cierra la conexión aunque el consumidor corte el stream antes de tiempo
        async with response:
            async for chunk in response:
                if not chunk.choices:                # hay chunks sin choices, por ejemplo con métricas de uso
                    continue
                delta = chunk.choices[0].delta
                if delta.refusal:
                    raise RefusalError(delta.refusal)
                if delta.content:                    # el primer y el último chunk pueden venir sin texto
                    yield delta.content


class AnthropicClient(BaseLLMClient):
    """
    Implementa el contrato usando el SDK asíncrono de Anthropic.
    Resuelve lo que cambia respecto de OpenAI: los mensajes system van en un parámetro aparte,
    max_tokens es obligatorio, temperature no se acepta y el texto viene en una lista de bloques.
    """
    provider = Provider.ANTHROPIC

    def __init__(self, api_key: SecretStr, model: str):
        self._client = AsyncAnthropic(api_key=api_key.get_secret_value(), max_retries=2)
        self.model = model

    def _request_args(self, messages: list[ChatMessage], params: GenerationParams) -> dict:
        # No se ajusta ni se descarta en silencio: quien la pidió tiene que enterarse
        if params.temperature is not None:
            raise ValueError("La API de Anthropic no acepta temperature: dejala sin definir para este proveedor")

        args = {
            "model": self.model,
            "max_tokens": params.max_tokens,         # en Anthropic es obligatorio
            "messages": [m.model_dump() for m in messages if m.role != "system"],
        }
        # Anthropic recibe las instrucciones de sistema en un parámetro aparte
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        if system:
            args["system"] = system
        return args

    async def generate(self, messages: list[ChatMessage], params: GenerationParams) -> ModelResponse:
        response = await self._client.messages.create(**self._request_args(messages, params))
        print(f"[claude normal] {response!r}")       # respuesta cruda, solo para ver su forma
        if response.stop_reason == "refusal":
            raise RefusalError("El modelo se negó a responder")
        # Anthropic: content es una lista de bloques; nos quedamos con los de texto
        text = "".join(block.text for block in response.content if block.type == "text")
        return ModelResponse(ok=True, text=text, provider=self.provider, model=self.model)

    async def stream(self, messages: list[ChatMessage], params: GenerationParams) -> AsyncGenerator[str, None]:
        async with self._client.messages.stream(**self._request_args(messages, params)) as response:
            async for event in response:
                # Evento crudo, solo para ver su forma. snapshot repite todo lo acumulado, por eso se saca.
                print(f"[claude stream] {event.model_dump(exclude={'snapshot', 'message'}, exclude_none=True)}")
                if event.type == "text":             # los eventos de tipo text traen el pedazo nuevo
                    yield event.text
            final = await response.get_final_message()
            print(f"[claude stream final] {final!r}")
            if final.stop_reason == "refusal":
                raise RefusalError("El modelo se negó a responder")
