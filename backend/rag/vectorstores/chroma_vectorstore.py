from uuid import uuid4
from langchain_chroma import Chroma
from langchain_core.documents import Document

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

    def get_all_documents(
        self,
        metadata_filter: dict | None = None,
    ) -> list[Document]:
        results = self.vectorstore._collection.get(
            where=metadata_filter,
            include=["documents", "metadatas"],
        )

        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        if not documents:
            return []

        return [
            Document(
                page_content=doc,
                metadata={
                    **(meta or {}),
                },
            )
            for doc, meta in zip(documents, metadatas)
        ]

    def persist(self):
        if self.persist_directory:
            pass
        
if __name__ == "__main__":
    vectorstore = ChromaVectorStore(collection_name="test_collection")
    vectorstore.add_documents(
        documents=["Hello world", "Hi there"],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        metadatas=[{"source": "greeting"}, {"source": "salutation"}],
    )

    query_embedding = [0.1, 0.2]
    results = vectorstore.vector_query(query_embedding=query_embedding, top_k=2)
    print(results)

    all_docs = vectorstore.get_all_documents()
    print(all_docs)