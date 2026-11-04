# Unified Async LLM Client

Cliente asíncrono para usar OpenAI o Anthropic con la misma interfaz, en modo normal y en streaming, con validación de Pydantic y errores controlados.

## Requisitos

- Python 3.12
- Una API key de OpenAI o de Anthropic

## Instalación

Desde esta carpeta:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux o macOS
pip install -r requirements.txt
```

## Variables de entorno

Copiá `.env.example` como `.env` y completalo. El archivo `.env` no se sube al repositorio.

| Variable | Obligatoria | Descripción |
|---|---|---|
| `LLM_PROVIDER` | Sí | `openai` o `anthropic` |
| `OPENAI_API_KEY` | Si usás `openai` | Key de OpenAI |
| `ANTHROPIC_API_KEY` | Si usás `anthropic` | Key de Anthropic |
| `OPENAI_MODEL` | No | Por defecto `gpt-4o-mini` |
| `ANTHROPIC_MODEL` | No | Por defecto `claude-opus-5-5` |

## Cómo ejecutarlo

```bash
python main.py
```

El script pregunta "¿Qué es la entropía?" al proveedor configurado: primero en modo normal y después en streaming, donde el texto aparece a medida que llega. Termina con código 0 si todo salió bien y con 1 si hubo algún error.

## Uso

```python
from manager import AsyncLLMManager
from schemas import ChatMessage, GenerationParams

manager = AsyncLLMManager.from_env()
messages = [ChatMessage(role="user", content="Hola")]

# Modo normal
response = await manager.generate(messages, GenerationParams(max_tokens=500))
print(response.text if response.ok else response.error.message)

# Streaming
async for chunk in manager.stream(messages):
    print(chunk.error.message if chunk.error else chunk.text, end="", flush=True)
```

## Estructura

| Archivo | Contenido |
|---|---|
| `schemas.py` | Modelos Pydantic: mensajes, parámetros, configuración, respuestas y errores |
| `clients.py` | `BaseLLMClient`, `OpenAIClient` y `AnthropicClient` |
| `manager.py` | `AsyncLLMManager`: elige el proveedor y convierte las excepciones en errores controlados |
| `main.py` | Script de prueba |

`generator.py`, si aparece en una copia local, es práctica de `yield` y no forma parte de la entrega.

## Manejo de errores

`generate()` y `stream()` no lanzan excepciones por errores de la API, de red o de parámetros:

- `generate()` devuelve un `ModelResponse` con `ok=False` y el `error`.
- `stream()` entrega lo que haya recibido y termina con un `StreamChunk` que trae el `error`.

| `error.code` | Cuándo ocurre |
|---|---|
| `rate_limit` | Se superó el límite de pedidos o la cuota |
| `connection` / `timeout` | Falla de red o tiempo de espera agotado |
| `auth` | API key inválida |
| `invalid_request` | Parámetros que el proveedor no acepta |
| `refusal` | El modelo se negó a responder |
| `api` | Cualquier otro error de la API |

Los SDK reintentan hasta 2 veces el pedido inicial ante errores de red y de rate limit. Una vez que el stream entregó texto no se reintenta, porque se duplicaría.

Una configuración incompleta, como una key faltante, sí lanza `ValueError` al crear el manager, y `main.py` la muestra como mensaje.

## Diferencias entre proveedores

- **`temperature`:** `GenerationParams` valida un rango de 0 a 2 y es opcional. OpenAI la usa si la definís. La API de Anthropic ya no acepta este parámetro, así que definirla con ese proveedor devuelve `invalid_request`.
- **Mensajes `system`:** en Anthropic se envían en un parámetro aparte; `AnthropicClient` hace esa conversión.
- **Rechazos del modelo:** se devuelven como error `refusal`, sin reintentar con otro modelo.

## Estado de las pruebas

- **Con las APIs reales y una key inválida:** se comprobó en OpenAI y Anthropic que devuelven `auth` en ambos modos, sin crash.
- **Sin red, con respuestas simuladas:** se comprobó el texto de cada proveedor, el streaming, el corte del stream, los errores de rate limit y de red, y la validación de parámetros.
- **Pendiente:** todavía no se probó una respuesta real con una key válida.
