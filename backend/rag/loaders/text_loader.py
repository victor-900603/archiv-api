from .base import BaseLoader
from langchain_community.document_loaders import TextLoader as LCTextLoader

class TextLoader(BaseLoader):
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        documents = LCTextLoader(self.file_path).load()
        
        for doc in documents:
            doc.metadata = {
                "type": "text",
                "file_path": self.file_path,
                "source": doc.metadata.get("source", self.file_path),
            }
            
        return documents
        
if __name__ == "__main__":
    loader = TextLoader(r"backend\docs\example\example.txt")
    docs = loader.load()
    for doc in docs:
        print(doc.metadata)