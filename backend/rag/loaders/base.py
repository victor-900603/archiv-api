from abc import ABC, abstractmethod
from langchain_core.documents import Document

class BaseLoader(ABC):
    @abstractmethod
    def load(self) -> list[Document]:
        """Load source files into chunk dictionaries with text and metadata."""
        raise NotImplementedError
    
    def _clean_text(self, text: str) -> str:
        import re

        text = re.sub(r"\n+", "\n", text)
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"\x0c", "", text)

        return text.strip()