from langchain_text_splitters import RecursiveCharacterTextSplitter
from .base import BaseSplitter

class RecursiveSplitter(BaseSplitter):
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split(self, documents):
        return self.splitter.split_documents(documents)