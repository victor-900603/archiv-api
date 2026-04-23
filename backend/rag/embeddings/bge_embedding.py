from sentence_transformers import SentenceTransformer
from configs.settings import settings
from .base import BaseEmbedding

class BGEEmbedding(BaseEmbedding):
    def __init__(self, model_name="BAAI/bge-base-zh", batch_size=32):
        self.model = SentenceTransformer(
            model_name,
            token=settings.hf_token,
        )
        self.batch_size = batch_size

    def embed_query(self, text):
        return self.model.encode(
            [f"query: {text}"],
            normalize_embeddings=True
        )[0]

    def embed_documents(self, documents):
        return self.model.encode(
            [f"passage: {t}" for t in documents],
            batch_size=self.batch_size,
            normalize_embeddings=True
        )