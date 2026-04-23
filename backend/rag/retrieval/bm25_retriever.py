from langchain_community.retrievers import BM25Retriever as LCBM25Retriever
from ..vectorstores.base import BaseVectorStore
from .base import BaseRetriever


class BM25Retriever(BaseRetriever):
    def __init__(
        self,
        documents: list | None = None,
        vectorstore: BaseVectorStore | None = None,
        default_k: int = 10,
    ):
        self.vectorstore = vectorstore
        self.default_k = default_k
        self.retriever = None

        if documents is not None:
            self.refresh(documents=documents)
        elif vectorstore is not None:
            self.refresh()

    def refresh(self, documents: list | None = None):
        if documents is None:
            if self.vectorstore is None:
                raise ValueError("documents or vectorstore is required to build BM25 index")

            documents = self.vectorstore.get_all_documents()

        self.retriever = LCBM25Retriever.from_documents(documents)
        return self

    def retrieve(self, query: str, k: int = None, **kwargs):
        k = k or self.default_k

        if self.retriever is None:
            self.refresh()
        
        docs = self.retriever.invoke(query)

        return [
            {
                "document": d.page_content,
                "metadata": d.metadata,
                "score": 1.0,
                "source": "bm25"
            }
            for d in docs[:k]
        ]