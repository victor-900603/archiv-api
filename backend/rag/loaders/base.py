from abc import ABC, abstractmethod

class BaseLoader(ABC):
    @abstractmethod
    def load(self) -> list[dict]:
        """Load source files into chunk dictionaries with text and metadata."""
        raise NotImplementedError