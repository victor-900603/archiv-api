from abc import ABC, abstractmethod

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
                "text": str,
                "metadata": dict,
                "distance": float
            }
        ]
        """
        raise NotImplementedError

    @abstractmethod
    def persist(self):
        """Persist vector store data if the backend supports explicit flush."""
        raise NotImplementedError