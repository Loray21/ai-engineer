import sys
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Rutas relativas a este archivo: el script funciona igual desde cualquier carpeta en la que se corra
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
COLLECTION_NAME = "refugio_frey"

# Tamaños medidos en tokens, no en caracteres
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def get_collection(reset: bool = False) -> Collection:
    """
    Abre (o crea) la colección persistida en disco.
    La usan tanto la ingesta como la consulta: así las dos usan siempre el mismo modelo de embeddings.
    Si se indexa con un modelo y se consulta con otro, los vectores no son comparables
    y la búsqueda devuelve resultados al azar.
    """
    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
    # DefaultEmbeddingFunction: modelo chico (all-MiniLM-L6-v2) que corre local, sin API key ni costo
    embedding_function = DefaultEmbeddingFunction()
    collection = client.get_or_create_collection(COLLECTION_NAME, embedding_function=embedding_function)

    if reset:
        client.delete_collection(COLLECTION_NAME)
        collection = client.create_collection(COLLECTION_NAME, embedding_function=embedding_function)
    return collection


def load_and_split() -> tuple[list[str], list[str], list[dict]]:
    """Lee los .md y .txt de data/ y los corta en chunks. Devuelve ids, textos y metadatas."""
    # from_tiktoken_encoder mide el largo de cada chunk en tokens con el tokenizer cl100k_base
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Primero intenta cortar entre párrafos, después entre líneas, oraciones y palabras
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    ids, documents, metadatas = [], [], []
    files = sorted(p for p in DATA_DIR.iterdir() if p.suffix in {".md", ".txt"})
    for path in files:
        chunks = splitter.split_text(path.read_text(encoding="utf-8"))
        for i, chunk in enumerate(chunks):
            # El id combina archivo y número de chunk: es estable entre ejecuciones
            # y sirve como referencia legible en las respuestas ("como_llegar.md#1")
            ids.append(f"{path.name}#{i}")
            documents.append(chunk)
            metadatas.append({"source": path.name, "chunk": i})
        print(f"{path.name}: {len(chunks)} chunks")
    return ids, documents, metadatas


def ingest(reset: bool = False) -> Collection:
    """Indexa los documentos de data/ en ChromaDB, salvo que la colección ya tenga datos."""
    collection = get_collection(reset=reset)

    # Evita reindexar en cada ejecución: calcular los embeddings es la parte lenta
    # (y, con un modelo pago, la parte que cuesta plata)
    if collection.count() > 0:
        print(f"La colección '{COLLECTION_NAME}' ya tiene {collection.count()} chunks. "
              "No se reindexa (usar --reset para rehacerla).")
        return collection

    ids, documents, metadatas = load_and_split()
    # upsert en lugar de add: si un id ya existe lo reemplaza en vez de fallar
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Indexados {collection.count()} chunks en {VECTORSTORE_DIR}")
    return collection


if __name__ == "__main__":
    ingest(reset="--reset" in sys.argv)
