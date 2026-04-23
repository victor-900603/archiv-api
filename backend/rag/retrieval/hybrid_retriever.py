from .base import BaseRetriever
from time import perf_counter
import logging

logger = logging.getLogger('retrieval')

class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        vector_retriever: BaseRetriever,
        bm25_retriever: BaseRetriever,
        weight_vector: float = 0.5,
        weight_bm25: float = 0.5,
        k: int = 10
    ):
        self.vector = vector_retriever
        self.bm25 = bm25_retriever
        self.weight_vector = weight_vector
        self.weight_bm25 = weight_bm25
        self.default_k = k

    def retrieve(self, query: str, k: int = None, **kwargs):
        t0 = perf_counter()
        logger.debug(f"[Hybrid] Retrieving with query: {query}, k: {k}")
        
        k = k or self.default_k

        v_docs = self.vector.retrieve(query, k=k)
        b_docs = self.bm25.retrieve(query, k=k)

        v_docs = self._normalize(v_docs)
        b_docs = self._normalize(b_docs)

        fused = self._fusion(v_docs, b_docs)

        fused.sort(key=lambda x: x["score"], reverse=True)
        
        t1 = perf_counter()
        logger.debug(f"[Hybrid] Retrieved {len(fused)} fused documents in {t1 - t0:.2f} seconds.")
        
        return fused[:k]
    
    def _normalize(self, docs):
        scores = [d["score"] for d in docs if d["score"] is not None]

        if not scores:
            return docs

        min_s = min(scores)
        max_s = max(scores)

        for d in docs:
            if d["score"] is None:
                d["score"] = 0.0
            elif max_s == min_s:
                d["score"] = 1.0
            else:
                d["score"] = (d["score"] - min_s) / (max_s - min_s)

        return docs
    
    def _fusion(self, v_docs, b_docs):
        combined = {}

        # vector
        for d in v_docs:
            key = d["document"]

            combined[key] = {
                "document": d["document"],
                "metadata": d["metadata"],
                "score": d["score"] * self.weight_vector,
                "source": {"vector"}
            }

        # bm25
        for d in b_docs:
            key = d["document"]

            if key not in combined:
                combined[key] = {
                    "document": d["document"],
                    "metadata": d["metadata"],
                    "score": 0.0,
                    "source": set()
                }

            combined[key]["score"] += d["score"] * self.weight_bm25
            combined[key]["source"].add("bm25")

        # convert source set → list
        return [
            {
                **v,
                "source": list(v["source"])
            }
            for v in combined.values()
        ]