from uuid import uuid4
from langchain_chroma import Chroma

from .base import BaseVectorStore


class ChromaVectorStore(BaseVectorStore):
    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str | None = "./chroma_db",
    ):
        self.persist_directory = persist_directory
        self.vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
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

        self.vectorstore._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        self.persist()

    def vector_query(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        results = self.vectorstore._collection.query(
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

        return [
            {
                "document": doc,
                "metadata": meta,
                "distance": dist,
            }
            for doc, meta, dist in zip(documents, metadatas, distances)
        ]

    def persist(self):
        if self.persist_directory:
            pass