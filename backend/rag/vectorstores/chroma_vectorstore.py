from uuid import uuid4
import chromadb

from .base import BaseVectorStore


class ChromaVectorStore(BaseVectorStore):
    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str | None = None,
    ):
        self.persist_directory = persist_directory

        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add_documents(
        self,
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ):
        if not documents:
            return

        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings length mismatch")

        if metadatas and len(metadatas) != len(documents):
            raise ValueError("metadatas and documents length mismatch")

        ids = ids or [str(uuid4()) for _ in documents]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        self.persist()

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filter: dict | None = None,
        include_scores: bool = False,
    ):
        """
        retrieval layer
        """

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=metadata_filter,
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not documents:
            return []

        if not include_scores:
            return documents

        return [
            {
                "text": doc,
                "metadata": meta,
                "score": 1 - dist,
            }
            for doc, meta, dist in zip(documents, metadatas, distances)
        ]

    def persist(self):
        if self.persist_directory:
            pass