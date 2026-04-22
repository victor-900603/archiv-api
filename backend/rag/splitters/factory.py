from .recursive_splitter import RecursiveSplitter
from .sentence_splitter import SentenceSplitter
from .base import BaseSplitter
from pathlib import Path

class SplitterFactory:
    _registry = {
        ".pdf": RecursiveSplitter,
        ".txt": RecursiveSplitter,
    }
    @staticmethod
    def get(file_path: str, document_type: str = None, **kwargs) -> BaseSplitter:
        if document_type:
            extension = f".{document_type}"
        else:
            extension = Path(file_path).suffix.lower()
        splitter_class = SplitterFactory._registry.get(extension)
        if not splitter_class:
            raise ValueError(f"No splitter found for extension: {extension}")
        return splitter_class(**kwargs)
