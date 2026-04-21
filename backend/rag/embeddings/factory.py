from .hf_embedding import HFEmbedding
from .bge_embedding import BGEEmbedding

class EmbeddingFactory:
    _embeddings = {
        "hf": HFEmbedding,
        "bge": BGEEmbedding,
    }
    @staticmethod
    def get(name: str, **kwargs):
        embedding_cls = EmbeddingFactory._embeddings.get(name)
        if not embedding_cls:
            raise ValueError(f"Embedding '{name}' not found. Available embeddings: {list(EmbeddingFactory._embeddings.keys())}")
        return embedding_cls(**kwargs)