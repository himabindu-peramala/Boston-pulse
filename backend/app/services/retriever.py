"""
Boston Pulse — Retriever
Stores and retrieves text chunks from ChromaDB vector store.
"""
import logging
from typing import List
import chromadb

from app.core.config import settings
from app.models.schemas import RetrievedChunk
from app.services.embedder import embed_query, embed_texts

logger = logging.getLogger(__name__)

_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
        )
        _collection = _client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB collection '{settings.chroma_collection_name}' "
            f"loaded with {_collection.count()} chunks"
        )
    return _collection


def add_chunks(chunks: List[dict]) -> int:
    """
    Add text chunks to ChromaDB.

    Args:
        chunks: List of dicts with keys: text, dataset, metadata

    Returns:
        Number of chunks added
    """
    if not chunks:
        return 0

    collection = _get_collection()

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    # Build IDs and metadata
    existing_count = collection.count()
    ids = [f"chunk_{existing_count + i}" for i in range(len(chunks))]
    metadatas = [
        {"dataset": c["dataset"], "text": c["text"][:500]}
        for c in chunks
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    logger.info(f"Added {len(chunks)} chunks to ChromaDB")
    return len(chunks)


def retrieve(query: str, top_k: int = None) -> List[RetrievedChunk]:
    """
    Retrieve the most relevant chunks for a query.

    Args:
        query: User's question
        top_k: Number of results to return (defaults to settings.top_k_results)

    Returns:
        List of RetrievedChunk objects
    """
    collection = _get_collection()

    if collection.count() == 0:
        logger.warning("ChromaDB is empty — run ingest.py first!")
        return []

    k = top_k or settings.top_k_results
    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        # Convert cosine distance to similarity score (0-1)
        score = round(1 - distance, 4)

        chunks.append(RetrievedChunk(
            dataset=metadata.get("dataset", "unknown"),
            content=doc,
            score=score,
        ))

    logger.info(f"Retrieved {len(chunks)} chunks for query: '{query[:50]}...'")
    return chunks


def get_collection_stats() -> dict:
    """Return stats about the vector store."""
    collection = _get_collection()
    return {
        "total_chunks": collection.count(),
        "collection_name": settings.chroma_collection_name,
        "persist_dir": settings.chroma_persist_dir,
    }
