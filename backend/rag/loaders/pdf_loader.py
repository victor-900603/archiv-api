from .base import BaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

class PDFLoader(BaseLoader):
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self) -> list[Document]:
        documents = PyPDFLoader(self.file_path).load()
        
        for doc in documents:
            doc.metadata = {
                "type": "pdf",
                "file_path": self.file_path,
                "source": doc.metadata.get("source", self.file_path),
                
                "page": doc.metadata.get("page"),
                "author": doc.metadata.get("author"),
                "title": doc.metadata.get("title"),
            }
        
        return documents
    
if __name__ == "__main__":
    loader = PDFLoader(r"backend\docs\example\example.pdf")
    docs = loader.load()
    for doc in docs:
        print(doc.metadata)