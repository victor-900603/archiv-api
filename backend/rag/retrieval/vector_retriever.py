from .base import BaseRetriever
from ..vectorstores.base import BaseVectorStore
from ..embeddings.base import BaseEmbedding

class VectorRetriever(BaseRetriever):
    def __init__(self, vectorstore: BaseVectorStore, embedder: BaseEmbedding, k=10):
        self.vectorstore = vectorstore
        self.embedder = embedder
        self.default_k = k

    def retrieve(self, query: str, k: int = None, **kwargs):
        k = k or self.default_k

        query_vec = self.embedder.embed_query(query)

        docs = self.vectorstore.query(
            query_embedding=query_vec,
            top_k=k
        )

        return [
            {
                "text": d["text"],
                "metadata": d["metadata"],
                "score": 1 - d["distance"],
                "source": "vector"
            }
            for d in docs
        ]