from langchain_huggingface import HuggingFaceEmbeddings
from .base import BaseEmbedding

class HFEmbedding(BaseEmbedding):
    def __init__(self):
        self.model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
        )

    def embed_documents(self, documents):
        return self.model.embed_documents(documents)

    def embed_query(self, text):
        return self.model.embed_query(text)