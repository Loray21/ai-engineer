from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, SecretStr


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class GenerationParams(BaseModel):
    # None significa "no enviarla": la API de Anthropic ya no acepta temperature
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int = Field(default=16000, gt=0)


class LLMConfig(BaseModel):
    """
    Las keys son opcionales: solo se exige la del proveedor elegido.
    SecretStr las enmascara al imprimir o loguear la config ('**********').
    """
    provider: Provider
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-opus-5-5"


class LLMError(BaseModel):
    code: Literal["rate_limit", "connection", "timeout", "auth", "invalid_request", "refusal", "api"]
    message: str


class ModelResponse(BaseModel):
    ok: bool
    text: str = ""
    provider: Provider
    model: str
    error: LLMError | None = None


class StreamChunk(BaseModel):
    """Un fragmento del stream: trae texto o, si algo falló, el error."""
    text: str = ""
    error: LLMError | None = None
