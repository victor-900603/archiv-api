from .text_loader import TextLoader
from .pdf_loader import PDFLoader

class LoaderFactory:
    @staticmethod
    def get_loader(file_path: str):
        if file_path.endswith(".txt"):
            return TextLoader(file_path)
        elif file_path.endswith(".pdf"):
            return PDFLoader(file_path)
        else:
            raise ValueError("Unsupported file type")