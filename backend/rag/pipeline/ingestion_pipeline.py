from pathlib import Path
from uuid import uuid4

from langchain_core.documents import Document

from ..embeddings.factory import EmbeddingFactory
from ..loaders.factory import LoaderFactory
from ..splitters.factory import SplitterFactory
from ..vectorstores.factory import VectorStoreFactory


class IngestionPipeline:
    def __init__(
        self,
        embedding_name: str = "hf",
        vectorstore_name: str = "chroma",
        embedding_kwargs: dict | None = None,
        vectorstore_kwargs: dict | None = None,
    ):
        self.embedder = EmbeddingFactory.get(embedding_name, **(embedding_kwargs or {}))
        self.vectorstore = VectorStoreFactory.get(
            vectorstore_name, **(vectorstore_kwargs or {})
        )

    def ingest(self, file_path: str) -> dict:
        loader = LoaderFactory.get(file_path)
        documents = loader.load()

        if not documents:
            return {
                "file_path": file_path,
                "doc_id": None,
                "document_count": 0,
                "chunk_count": 0,
                "indexed_count": 0,
            }

        splitter = SplitterFactory.get(file_path)
        chunks = splitter.split(documents)

        if not chunks:
            return {
                "file_path": file_path,
                "doc_id": documents[0].metadata.get("doc_id"),
                "document_count": len(documents),
                "chunk_count": len(chunks),
                "indexed_count": 0,
            }

        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [
            chunk.metadata.get("chunk_id") or str(uuid4()) for chunk in chunks
        ]

        embeddings = self.embedder.embed_documents(texts)
        self.vectorstore.add_documents(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        return {
            "file_path": file_path,
            "doc_id": documents[0].metadata.get("doc_id"),
            "document_count": len(documents),
            "chunk_count": len(chunks),
        }


if __name__ == "__main__":
    pipeline = IngestionPipeline()
    result = pipeline.ingest(r"backend\docs\example\example.pdf")
    print(result)