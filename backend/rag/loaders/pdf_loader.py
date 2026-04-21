from .base import BaseLoader
from langchain_community.document_loaders import PyPDFLoader

class PDFLoader(BaseLoader):
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        return PyPDFLoader(self.file_path).load()