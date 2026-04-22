from abc import ABC, abstractmethod

class BaseSplitter(ABC):
    @abstractmethod
    def split(self, documents: list[dict]) -> list[dict]:
        """Split chunk dictionaries while preserving and enriching metadata."""
        raise NotImplementedError