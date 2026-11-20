# RAG local sobre el Refugio Frey

Sistema de recuperación semántica (RAG) que responde preguntas sobre el Refugio Frey, en Bariloche, usando únicamente cuatro documentos de ejemplo. Si la respuesta no está en los documentos, responde "No lo sé" en lugar de inventarla.

- **Ingesta:** los documentos de `data/` se dividen en fragmentos de 500 tokens y se guardan en una base vectorial ChromaDB persistida en disco.
- **Recuperación:** la pregunta se convierte en un embedding y se buscan los 4 fragmentos más parecidos.
- **Generación:** una cadena de LangChain (LCEL) le pasa esos fragmentos y la pregunta al LLM, de forma asíncrona, y convierte la respuesta en un objeto Pydantic con el texto y las referencias usadas.

## Requisitos

- Python 3.12
- Una API key de OpenAI o de Anthropic, para el LLM. Los embeddings corren en local y no necesitan key.

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
| `LLM_PROVIDER` | No | `openai` (por defecto) o `anthropic` |
| `OPENAI_API_KEY` | Si usás `openai` | Key de OpenAI |
| `ANTHROPIC_API_KEY` | Si usás `anthropic` | Key de Anthropic |
| `OPENAI_MODEL` | No | Por defecto `gpt-4o-mini` |
| `ANTHROPIC_MODEL` | No | Por defecto `claude-opus-5-5` |

## Cómo ejecutarlo

### 1. Indexar los documentos

```bash
python ingest.py           # indexa data/ en ./vectorstore
python ingest.py --reset   # borra la colección y vuelve a indexar (por ejemplo, después de cambiar data/)
```

Si la colección ya tiene datos, `ingest.py` no reindexa: muestra cuántos fragmentos hay y termina. La primera ejecución descarga el modelo de embeddings (unos 80 MB).

Este paso es opcional: si la base no existe, `main.py` la crea solo.

### 2. Correr las dos pruebas

```bash
python main.py
```

| Caso | Pregunta | Se espera |
|---|---|---|
| `en-documentos` | ¿Cuánto se tarda en llegar al refugio caminando desde Villa Catedral? | `encontrada: true`, con la respuesta y sus referencias |
| `trampa` | ¿Cuánto cuesta pasar la noche en el refugio? | `encontrada: false` y `"No lo sé"`: los documentos hablan del pernocte y de las reservas, pero nunca de precios |

Las dos preguntas se procesan en paralelo con `asyncio.gather`. El script termina con código 0 si las dos se comportaron como se esperaba y con 1 si alguna no.

Para probar otras preguntas, agregalas a la lista `CASOS` de `main.py` o usá `get_rag_response()` desde código (ver abajo).

## Uso desde código

```python
from rag import get_rag_response

resultado = await get_rag_response("¿Qué hay que llevar para dormir?")
print(resultado.respuesta)     # el texto de la respuesta
print(resultado.referencias)   # por ejemplo ['servicios_y_reglas.md#0']
print(resultado.encontrada)    # False si la respuesta no está en los documentos
```

## Estructura

| Archivo | Contenido |
|---|---|
| `data/` | Dataset de ejemplo: 4 archivos `.md` sobre el Refugio Frey (historia, cómo llegar, servicios y reglas, escalada y seguridad) |
| `ingest.py` | Lectura de `data/`, chunking y carga en ChromaDB. `get_collection()` es el único lugar donde se abre la colección |
| `schemas.py` | `RespuestaRAG`, el modelo Pydantic de la respuesta |
| `rag.py` | Retriever, modelo, prompt, cadena LCEL y `get_rag_response()` |
| `main.py` | Las dos pruebas: pregunta con respuesta en los documentos y pregunta trampa |

Los datos del dataset son material de ejemplo para la práctica: algunas cifras (distancias, tiempos, desnivel) son aproximadas.

## Cómo funciona

### Ingesta (`ingest.py`)

- **Chunking:** `RecursiveCharacterTextSplitter.from_tiktoken_encoder` con `chunk_size=500` y `chunk_overlap=50`, medidos en tokens (tokenizer `cl100k_base`). Intenta cortar primero entre párrafos (`\n\n`), después entre líneas, oraciones y palabras, así los fragmentos tienden a respetar las secciones de los documentos. Cada documento queda en unos 2 fragmentos.
- **Ids:** cada fragmento se guarda con el id `archivo#n` (por ejemplo `como_llegar.md#0`) y la metadata `source` y `chunk`. El id es estable entre ejecuciones y se usa como referencia en las respuestas.
- **Persistencia:** `chromadb.PersistentClient` guarda la base en `./vectorstore`. Antes de indexar se comprueba `collection.count()`: si ya hay datos, no se recalculan los embeddings.

### Embeddings

Se usa la función por defecto de ChromaDB (`all-MiniLM-L6-v2`, corre en local con ONNX). La función queda asociada a la colección y tanto la ingesta como la consulta abren la colección con `get_collection()`, así que es imposible indexar con un modelo y consultar con otro.

Es un modelo entrenado principalmente en inglés: con estos cuatro documentos de temas bien diferenciados alcanza, pero para un corpus grande en español convendría un modelo multilingüe.

### Recuperación

`collection.query(query_texts=[pregunta], n_results=4)`: Chroma convierte la pregunta en un embedding y devuelve los 4 fragmentos más cercanos (distancia L2: cuanto más baja, más parecido). Se usan 4 fragmentos para no saturar el contexto del modelo.

### La cadena LCEL (`rag.py`)

```
"pregunta"
   │  {"chunks": retriever, "pregunta": RunnablePassthrough()}
   ▼
{chunks, pregunta}
   │  RunnablePassthrough.assign(contexto=formatear_contexto)
   ▼
{chunks, pregunta, contexto}
   │  RunnablePassthrough.assign(respuesta=PROMPT | model | PydanticOutputParser)
   ▼
{chunks, pregunta, contexto, respuesta}
   │  validar_referencias
   ▼
RespuestaRAG   ── o una excepción que .with_retry() reintenta una vez
```

- `retriever` es la búsqueda en Chroma envuelta en un `RunnableLambda`.
- `formatear_contexto` une los fragmentos en un solo texto, cada uno precedido por su id entre corchetes, para que el modelo pueda citarlo.
- `RunnablePassthrough.assign` agrega una clave al diccionario sin borrar las anteriores. Así los fragmentos recuperados llegan hasta el final y se pueden usar para validar las referencias.
- `get_rag_response(query)` es asíncrona: llama a la cadena con `await chain.ainvoke(query)`.

### El prompt: filtro de veracidad

El mensaje de sistema indica que:

1. Se responde únicamente con la información del CONTEXTO, aunque el modelo conozca la respuesta por su cuenta.
2. Si el CONTEXTO no la contiene, se responde exactamente "No lo sé", con `encontrada=false` y sin referencias, sin completar cifras, fechas ni nombres que no estén escritos.
3. Las referencias son los ids de los fragmentos usados.

El formato de salida se agrega al prompt con `PydanticOutputParser.get_format_instructions()`, que genera el JSON Schema de `RespuestaRAG`.

### La respuesta (`schemas.py`)

| Campo | Contenido |
|---|---|
| `respuesta` | El texto de la respuesta, o "No lo sé" |
| `referencias` | Ids de los fragmentos usados, por ejemplo `["como_llegar.md#0"]` |
| `encontrada` | `true` si la respuesta está en los documentos |

### Controles contra las alucinaciones

| Control | Dónde | Qué rechaza |
|---|---|---|
| Prompt de sistema | `rag.py` | Respuestas basadas en conocimiento propio del modelo |
| `model_validator` | `schemas.py` | `encontrada=true` sin ninguna referencia. Con `encontrada=false`, normaliza la salida a "No lo sé" y sin referencias |
| `validar_referencias` | `rag.py` | Referencias a fragmentos que no estaban entre los recuperados |
| `.with_retry()` | `rag.py` | Si falla el parseo o alguno de los controles anteriores, la cadena se reintenta una vez |

Los errores de red y de rate limit los reintenta el SDK del proveedor (`max_retries=2`).
