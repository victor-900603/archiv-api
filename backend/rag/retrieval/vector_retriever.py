from .base import BaseRetriever
from ..vectorstores.base import BaseVectorStore
from ..embeddings.base import BaseEmbedding
import logging
from time import perf_counter

logger = logging.getLogger('retrieval')

class VectorRetriever(BaseRetriever):
    def __init__(self, vectorstore: BaseVectorStore, embedder: BaseEmbedding, k=10):
        self.vectorstore = vectorstore
        self.embedder = embedder
        self.default_k = k

    def retrieve(self, query: str, k: int = None, **kwargs):
        t0 = perf_counter()
        logger.debug(f"[Vector] Retrieving with query: {query}, k: {k}")
        k = k or self.default_k

        query_vec = self.embedder.embed_query(query)

        docs = self.vectorstore.vector_query(
            query_embedding=query_vec,
            top_k=k
        )
        
        t1 = perf_counter()
        logger.debug(
            f"[Vector] Retrieved {len(docs)} documents in {t1 - t0:.2f} seconds."
        )

        return [
            {
                "document": d["document"],
                "metadata": d["metadata"],
                "score": 1 - d["distance"],
                "source": "vector"
            }
            for d in docs
        ]