"""Build the Chroma knowledge base used by the agentic RAG application."""

import hashlib
from pathlib import Path

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DB_PATH,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    KNOWLEDGE_BASE_PATH,
)


def read_documents() -> list[dict]:
    """Load Markdown documents and retain useful source metadata."""
    documents = []
    for path in sorted(KNOWLEDGE_BASE_PATH.rglob("*.md")):
        relative_path = path.relative_to(KNOWLEDGE_BASE_PATH).as_posix()
        documents.append(
            {
                "source": relative_path,
                "type": relative_path.split("/", maxsplit=1)[0],
                "text": path.read_text(encoding="utf-8"),
            }
        )
    return documents


def split_text(text: str) -> list[str]:
    """Create overlapping chunks, preferring paragraph boundaries."""
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip()
        if current and len(candidate) > CHUNK_SIZE:
            chunks.append(current)
            overlap = current[-CHUNK_OVERLAP:]
            current = f"{overlap}\n\n{paragraph}".strip()
        else:
            current = candidate

    if current:
        chunks.append(current)
    return chunks


def make_records(documents: list[dict]) -> list[dict]:
    """Turn source documents into deterministic Chroma records."""
    records = []
    for document in documents:
        for index, text in enumerate(split_text(document["text"])):
            record_id = hashlib.sha256(
                f"{document['source']}:{index}:{text}".encode("utf-8")
            ).hexdigest()
            records.append(
                {
                    "id": record_id,
                    "document": f"Source: {document['source']}\n\n{text}",
                    "metadata": {
                        "source": document["source"],
                        "type": document["type"],
                        "chunk": index,
                    },
                }
            )
    return records


def batched(items: list[dict], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def create_vector_store(records: list[dict]) -> None:
    """Embed records in batches and replace the existing collection."""
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    chroma = PersistentClient(path=str(DB_PATH))

    existing = {collection.name for collection in chroma.list_collections()}
    if COLLECTION_NAME in existing:
        chroma.delete_collection(COLLECTION_NAME)

    collection = chroma.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    batches = list(batched(records, EMBEDDING_BATCH_SIZE))
    for batch in tqdm(batches, desc="Embedding knowledge base"):
        texts = [record["document"] for record in batch]
        embeddings = embedding_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        collection.add(
            ids=[record["id"] for record in batch],
            documents=texts,
            metadatas=[record["metadata"] for record in batch],
            embeddings=embeddings,
        )

    print(f"Created {collection.count()} chunks in {DB_PATH}")


def main() -> None:
    if not KNOWLEDGE_BASE_PATH.exists():
        raise FileNotFoundError(f"Knowledge base not found: {KNOWLEDGE_BASE_PATH}")

    documents = read_documents()
    if not documents:
        raise RuntimeError(f"No Markdown documents found in {KNOWLEDGE_BASE_PATH}")

    print(f"Loaded {len(documents)} documents")
    create_vector_store(make_records(documents))


if __name__ == "__main__":
    main()
