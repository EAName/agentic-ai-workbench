"""
ChromaDB Vector Store Wrapper.

Based on Ch 8: Understanding agent memory and knowledge.

Thin wrapper around ChromaDB that provides a clean interface
for storing and retrieving document embeddings.
"""

from __future__ import annotations

from typing import Any

import structlog

from config.settings import settings

logger = structlog.get_logger()


class VectorStore:
    """
    ChromaDB vector store for semantic search.

    Usage:
        store = VectorStore(collection_name="my_docs")
        store.add(ids=["doc1"], documents=["Hello world"], metadatas=[{"source": "test"}])
        results = store.query("greeting", top_k=3)
    """

    def __init__(
        self,
        collection_name: str = "default",
        persist_dir: str | None = None,
    ):
        import chromadb

        self.persist_dir = persist_dir or settings.vector_store.chroma_persist_dir
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.collection_name = collection_name

    def add(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add documents to the store."""
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("vectors_added", collection=self.collection_name, count=len(ids))

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Query the store for similar documents."""
        kwargs: dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": top_k,
        }
        if where:
            kwargs["where"] = where

        return self.collection.query(**kwargs)

    def delete(self, ids: list[str]) -> None:
        """Delete documents by ID."""
        self.collection.delete(ids=ids)

    @property
    def count(self) -> int:
        """Number of documents in the collection."""
        return self.collection.count()
