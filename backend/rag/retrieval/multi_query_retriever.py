from .query_rewriter import QueryRewriter
from .base import BaseRetriever

class MultiQueryRetriever(BaseRetriever):
    def __init__(self, rewriter: QueryRewriter, retriever: BaseRetriever):
        self.rewriter = rewriter
        self.retriever = retriever

    def retrieve(self, query: str, k: int = 10):
        queries = [query] + self.rewriter.rewrite(query)

        all_docs = []

        for q in queries:
            docs = self.retriever.retrieve(q, k=k)
            all_docs.extend(docs)

        return self._dedup(all_docs)[:k]
    
    def _dedup(self, docs):
        seen = set()
        unique_docs = []

        for d in docs:
            key = d["document"]

            if key not in seen:
                seen.add(key)
                unique_docs.append(d)

        return unique_docs