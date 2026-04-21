from abc import ABC, abstractmethod


class BaseEmbedding(ABC):
    @abstractmethod
    def embed_documents(self, documents) -> list[list[float]]:
        """Convert a list of documents into their corresponding vector embeddings."""
        raise NotImplementedError
    
    @abstractmethod
    def embed_query(self, query) -> list[float]:
        """Convert a query into its corresponding vector embedding."""
        raise NotImplementedError