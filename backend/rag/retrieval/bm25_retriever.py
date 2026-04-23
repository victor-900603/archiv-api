from langchain_community.retrievers import BM25Retriever as LCBM25Retriever
from ..vectorstores.base import BaseVectorStore
from .base import BaseRetriever
from time import perf_counter
import logging

logger = logging.getLogger('retrieval')


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

        if not documents:
            logger.info("[BM25] No documents available; skipping BM25 index build.")
            self.retriever = None
            return self

        self.retriever = LCBM25Retriever.from_documents(documents)
        return self

    def retrieve(self, query: str, k: int = None, **kwargs):
        t0 = perf_counter()
        logger.debug(f"[BM25] Retrieving with query: {query}, k: {k}")
        
        k = k or self.default_k

        if self.retriever is None:
            if self.vectorstore is None:
                logger.debug("[BM25] No vectorstore or retriever available; returning empty result set.")
                return []

            self.refresh()

        if self.retriever is None:
            logger.debug("[BM25] No retriever available; returning empty result set.")
            return []
        
        docs = self.retriever.invoke(query)
        
        t1 = perf_counter()
        logger.debug(f"[BM25] Retrieved {len(docs)} documents in {t1 - t0:.2f} seconds.")
        
        return [
            {
                "document": d.page_content,
                "metadata": d.metadata,
                "score": 1.0,
                "source": "bm25"
            }
            for d in docs[:k]
        ]