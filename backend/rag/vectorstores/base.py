from abc import ABC, abstractmethod
from langchain_core.documents import Document

class BaseVectorStore(ABC):
    @abstractmethod
    def add_documents(
        self,
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ):
        """Add documents, embeddings, metadata, and optional ids to the vector store."""
        raise NotImplementedError
    
    @abstractmethod
    def vector_query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        """
        return:
        [
            {
                "document": str,
                "metadata": dict,
                "distance": float
            }
        ]
        """
        raise NotImplementedError

    @abstractmethod
    def get_all_documents(
        self,
        metadata_filter: dict | None = None,
    ) -> list[Document]:
        """Return all stored documents, optionally filtered by metadata."""
        raise NotImplementedError

    @abstractmethod
    def delete_documents(
        self,
        metadata_filter: dict | None = None,
    ):
        """Delete stored documents, optionally filtered by metadata."""
        raise NotImplementedError

    @abstractmethod
    def persist(self):
        """Persist vector store data if the backend supports explicit flush."""
        raise NotImplementedError