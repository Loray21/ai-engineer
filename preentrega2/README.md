# Pipeline de Extracción de Entidades Técnicas

Recibe un texto técnico sin procesar (una descripción de arquitectura, un log de error o un reporte de incidente) y devuelve un objeto Pydantic validado con las tecnologías mencionadas, un nivel de criticidad y un resumen técnico.

Está construido con LangChain (LCEL): `prompt | model.with_structured_output(Schema)`, más un paso de validación propio y reintentos automáticos con `.with_retry()`. Funciona con OpenAI o con Anthropic.

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
| `LLM_MAX_TOKENS` | No | Tope de tokens de la respuesta. Por defecto `1000`. Sirve para la prueba de estrés |

## Cómo ejecutarlo

```bash
python main.py
```

`main.py` procesa cuatro textos en paralelo con `asyncio.gather`:

| Caso | Texto | Se espera |
|---|---|---|
| `arquitectura` | Descripción de microservicios con React, FastAPI, PostgreSQL, Redis, Kafka, Docker y Kubernetes | Objeto válido |
| `log-error` | Log real de un servicio de pagos con errores de PostgreSQL, Celery y nginx | Objeto válido |
| `ambiguo` | Problema de login descrito sin precisión, con una sola tecnología nombrada | Objeto válido |
| `sin-tecnologias` | Texto sobre una reunión, sin ninguna tecnología | Error controlado después de 3 intentos |

Cada caso declara qué espera y el script verifica que se cumpla. Termina con código 0 si los cuatro se comportaron como se esperaba y con 1 si alguno no.

### Prueba de estrés: respuesta cortada por tokens

Con un tope de tokens muy bajo, el modelo no llega a completar el JSON. Sirve para ver que el pipeline detecta el corte y no deja pasar una respuesta incompleta.

```powershell
# PowerShell
$env:LLM_MAX_TOKENS = "50"
python main.py
Remove-Item Env:LLM_MAX_TOKENS
```

```bash
# Bash
LLM_MAX_TOKENS=50 python main.py
```

En esta prueba es correcto que los casos fallen: los logs muestran `Respuesta cortada por falta de tokens (motivo=max_tokens)` en cada intento y, al final, un `ExtraccionError` con esa causa.

## Uso desde código

```python
from dotenv import load_dotenv
from chain import ExtraccionError, process_text

load_dotenv()

try:
    resultado = await process_text("La API en FastAPI tarda 8 s por timeouts contra Redis.")
    print(resultado.tecnologias)            # por ejemplo ['FastAPI', 'Redis']
    print(resultado.nivel_de_criticidad)    # por ejemplo media
    print(resultado.model_dump_json(indent=2))
except ExtraccionError as exc:
    print(f"No se pudo extraer: {exc}")
```

`process_text(text, etiqueta="-")` recibe un parámetro opcional, `etiqueta`, que identifica el texto en los logs cuando se procesan varios en paralelo.

## Ejemplo de salida

Entrada (caso `log-error`):

```
2026-11-10 03:14:07 ERROR [payment-service] psycopg2.OperationalError: FATAL: remaining connection slots are reserved for non-replication superuser connections
2026-11-10 03:14:07 ERROR [payment-service] Celery task process_payment failed after 3 retries
2026-11-10 03:14:08 WARN  [nginx] upstream timed out (110: Connection timed out) while reading response header from upstream
```

Salida obtenida con `claude-haiku-5-5`:

```json
{
  "tecnologias": ["psycopg2", "PostgreSQL", "Celery", "nginx"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "El servicio de pagos (payment-service) no puede abrir conexiones a la base de datos porque se agotaron los slots de conexión, lo que provoca que la tarea Celery process_payment falle tras 3 reintentos. nginx registra además timeouts de upstream al leer la cabecera de respuesta."
}
```

Logs de la misma corrida, donde se ve un caso que se resuelve en un intento y otro que agota los reintentos:

```
11:35:32 | INFO    | pipeline | [log-error] Procesando texto de 379 caracteres
11:35:32 | INFO    | pipeline | [sin-tecnologias] Procesando texto de 89 caracteres
11:35:34 | WARNING | pipeline | [sin-tecnologias] La respuesta no cumple el schema: tecnologias: Value error, La lista de tecnologías no puede quedar vacía ni tener solo valores de relleno
11:35:34 | INFO    | pipeline | [log-error] Validación OK (motivo=tool_use)
11:35:34 | INFO    | pipeline | [log-error] Extracción lista en 2.6s: 4 tecnologías, criticidad alta
11:35:36 | WARNING | pipeline | [sin-tecnologias] La respuesta no cumple el schema: ...
11:35:40 | WARNING | pipeline | [sin-tecnologias] La respuesta no cumple el schema: ...
11:35:40 | ERROR   | pipeline | [sin-tecnologias] Se agotaron los intentos. Último error: ...
```

## Estructura

| Archivo | Contenido |
|---|---|
| `schemas.py` | `ExtraccionTecnica` (el contrato de salida), `NivelCriticidad` (enum) y `LLMConfig` |
| `chain.py` | Modelo, prompt, cadena LCEL, validación de la salida, reintentos y `process_text()` |
| `main.py` | Mini-script de prueba asíncrono |

## Cómo funciona

```
{"texto": "..."}
   │  PROMPT                 ChatPromptTemplate: mensaje system con las reglas + mensaje human con el texto
   ▼
mensajes
   │  model.with_structured_output(ExtraccionTecnica, method="function_calling", include_raw=True)
   ▼
{"raw": mensaje crudo, "parsed": objeto o None, "parsing_error": error o None}
   │  RunnableLambda(validar_salida)
   ▼
ExtraccionTecnica  ── o una excepción que .with_retry() reintenta (hasta 3 intentos)
```

### El contrato (`schemas.py`)

| Campo | Regla |
|---|---|
| `tecnologias` | Lista de strings con al menos un elemento. Un validador quita espacios, elimina duplicados sin distinguir mayúsculas y descarta valores de relleno como `<UNKNOWN>`, `N/A` o `ninguna` |
| `nivel_de_criticidad` | Enum: `baja`, `media` o `alta` |
| `resumen_tecnico` | Entre 20 y 400 caracteres |

Las `description` de los campos y el docstring de la clase forman parte del JSON Schema que recibe el modelo, así que funcionan como instrucciones. Las notas para desarrolladores van en comentarios `#`, que no se envían.

### El prompt (`chain.py`)

`ChatPromptTemplate` con dos variables, `{texto}` y `{instrucciones_formato}`, sin f-strings. Las instrucciones (qué cuenta como tecnología y los criterios de cada nivel de criticidad) se cargan con `.partial()`, así que en cada llamada solo se pasa el texto.

### La salida estructurada

`with_structured_output` usa `method="function_calling"`: el schema viaja como una herramienta que el modelo está obligado a llamar, y los argumentos de esa llamada son el JSON. Se fija este método para que OpenAI y Anthropic usen el mismo mecanismo. `ChatOpenAI` usa por defecto otro (`json_schema`), con el que el SDK de OpenAI maneja los cortes por su cuenta y el control del `finish_reason` quedaría distinto para cada proveedor.

`include_raw=True` devuelve, además del objeto, el mensaje crudo del modelo, que es donde viene el `finish_reason`.

## Resiliencia

### Dos capas de reintento

| Capa | Qué reintenta | Configuración |
|---|---|---|
| SDK del proveedor | Errores de red, timeouts y rate limit | `max_retries=2`, `timeout=60` |
| `.with_retry()` | Respuestas cortadas o que no cumplen el schema | `stop_after_attempt=3`, espera exponencial con jitter |

`.with_retry()` solo reintenta `SalidaIncompletaError` y `SalidaInvalidaError`. Los errores de red no se reintentan en esta capa porque ya lo hace el SDK: si se reintentaran en las dos, un corte de red podría generar 3 × 3 = 9 llamadas.

### Control del `finish_reason`

El LLM genera hasta llegar a `max_tokens`. Si se queda sin tokens, la respuesta llega cortada y la API lo indica en el motivo de fin: `finish_reason="length"` en OpenAI y `stop_reason="max_tokens"` en Anthropic.

`validar_salida` revisa ese motivo **antes** de usar el objeto parseado:

1. **Respuesta cortada:** lanza `SalidaIncompletaError`, aunque se haya podido armar un objeto, porque un objeto que viene de una respuesta incompleta no es confiable.
2. **Respuesta completa pero inválida** (JSON mal formado o datos fuera del schema): lanza `SalidaInvalidaError` con el detalle de cada campo.
3. **Sin objeto y sin error** (el modelo respondió texto en lugar de usar la herramienta): lanza `SalidaInvalidaError`.

Sin este control, una respuesta cortada se reporta como `Field required` en todos los campos, y parece que el modelo respondió mal cuando en realidad le faltaron tokens.

Reintentar un corte con el mismo `max_tokens` suele volver a cortarse. Si pasa en uso real, la solución es subir `LLM_MAX_TOKENS`.

### Errores de `process_text()`

| Error | Cuándo |
|---|---|
| `ValueError` | El texto está vacío. No se llama al modelo |
| `ExtraccionError` | Después de 3 intentos la respuesta sigue cortada o inválida. Trae el último error como causa |
| Errores del SDK del proveedor | Key inválida, sin saldo o red caída después de los reintentos del SDK. Se dejan pasar tal cual, porque reintentar no los resuelve |

### Logs

Cada línea lleva la etiqueta del texto, así se pueden seguir varios procesos en paralelo. Se registra el inicio, cada fallo de validación con su causa, el resultado y el agotamiento de los intentos. Las líneas `httpx2 | HTTP Request` son de la librería HTTP del SDK y marcan cada llamada real a la API.

## Estado de las pruebas

Todas se hicieron con `claude-haiku-5-5`.

- **`python main.py`:** los 4 casos se comportaron como se esperaba, en unos 8 segundos en total, gracias a la ejecución en paralelo.
- **Texto ambiguo:** el modelo se recuperó. Devolvió solo `Nginx`, sin inventar el "servicio de mails" como tecnología, y criticidad `media` porque la falla es intermitente.
- **Texto sin tecnologías:** antes de agregar el filtro de valores de relleno, el modelo devolvía `["<UNKNOWN>"]` para cumplir el mínimo de un elemento, y Pydantic lo aceptaba. Con el filtro, la respuesta se rechaza, se reintenta 3 veces y termina en `ExtraccionError`.
- **Prueba de estrés con `LLM_MAX_TOKENS=50`:** el corte se detectó en todos los intentos de los 4 casos, y cada uno terminó en `ExtraccionError` con la causa correcta.
- **Validación y reintento sin red:** se probó `validar_salida` con respuestas simuladas (cortada en OpenAI y en Anthropic, fuera del schema, sin objeto) y `.with_retry()` con un modelo falso que falla dos veces y responde bien a la tercera.
- **Pendiente:** no se probó con OpenAI porque no había una key disponible. El código es el mismo para los dos proveedores, salvo `temperature=0`, que solo se envía a OpenAI.
