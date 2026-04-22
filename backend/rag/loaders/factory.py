from .text_loader import TextLoader
from .pdf_loader import PDFLoader
from .base import BaseLoader
from pathlib import Path

class LoaderFactory:
    _registry = {
        ".txt": TextLoader,
        ".pdf": PDFLoader,
    }
    @staticmethod
    def get(file_path: str, document_type: str = None) -> BaseLoader:
        if document_type:
            extension = f".{document_type}"
        else:
            extension = Path(file_path).suffix.lower()
        loader_class = LoaderFactory._registry.get(extension)
        if not loader_class:
            raise ValueError(f"No loader found for extension: {extension}")
        return loader_class(file_path)