from abc import ABC, abstractmethod

class BaseSplitter(ABC):
    @abstractmethod
    def split(self, documents) -> list[str]:
        """Split the input documents into smaller chunks."""
        raise NotImplementedError