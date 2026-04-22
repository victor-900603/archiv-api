from langchain_community.retrievers import BM25Retriever as LCBM25Retriever
from .base import BaseRetriever


class BM25Retriever(BaseRetriever):
    def __init__(self, documents: list):
        self.retriever = LCBM25Retriever.from_documents(documents)
        self.default_k = 10

    def retrieve(self, query: str, k: int = None, **kwargs):
        k = k or self.default_k
        
        docs = self.retriever.invoke(query)

        return [
            {
                "text": d.page_content,
                "metadata": d.metadata,
                "score": 1.0,
                "source": "bm25"
            }
            for d in docs[:k]
        ]