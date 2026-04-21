from .base import BaseLoader
from langchain_community.document_loaders import TextLoader

class TextLoader(BaseLoader):
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        return TextLoader(self.file_path).load()