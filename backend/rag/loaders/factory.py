from .text_loader import TextLoader
from .pdf_loader import PDFLoader

class LoaderFactory:
    _registry = {
        ".txt": TextLoader,
        ".pdf": PDFLoader,
    }
    @staticmethod
    def get_loader(file_path: str):
        extension = file_path.split(".")[-1].lower()
        loader_class = LoaderFactory._registry.get(f".{extension}")
        if not loader_class:
            raise ValueError(f"No loader found for extension: .{extension}")
        return loader_class(file_path)