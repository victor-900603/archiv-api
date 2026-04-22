from abc import ABC, abstractmethod
from langchain_core.documents import Document
class BaseSplitter(ABC):
    @abstractmethod
    def split(self, documents: list[Document]) -> list[Document]:
        """Split chunk dictionaries while preserving and enriching metadata."""
        raise NotImplementedError