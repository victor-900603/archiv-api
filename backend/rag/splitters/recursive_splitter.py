from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .base import BaseSplitter

class RecursiveSplitter(BaseSplitter):
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def split(self, documents: list[Document]) -> list[Document]:
        split_docs = self.splitter.split_documents(documents)
        
        for idx, doc in enumerate(split_docs):
            doc.metadata = {
                **doc.metadata,
                "splitter": "recursive",
                "chunk_index": idx,
            }
        
        return split_docs
           
if __name__ == "__main__":
    from backend.rag.loaders.text_loader import TextLoader
    loader = TextLoader(r"backend\docs\example\example.txt")
    docs = loader.load()
    
    splitter = RecursiveSplitter(chunk_size=100, chunk_overlap=20)
    chunks = splitter.split(docs)
    
    for chunk in chunks:
        print(chunk.metadata)