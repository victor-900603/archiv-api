from typing import Any
from uuid import uuid4
from ..loaders.factory import LoaderFactory
from ..splitters.factory import SplitterFactory

class IngestionPipeline:
    def __init__(
        self,
        embedder: Any,
        vectorstore: Any,
    ):
        self.embedder = embedder
        self.vectorstore = vectorstore

    def ingest(self, file_path: str) -> dict:
        loader = LoaderFactory.get(file_path)
        documents = loader.load()

        if not documents:
            return {
                "file_path": file_path,
                "doc_id": None,
                "document_count": 0,
                "chunk_count": 0,
            }

        splitter = SplitterFactory.get(file_path)
        chunks = splitter.split(documents)

        if not chunks:
            return {
                "file_path": file_path,
                "doc_id": documents[0].metadata.get("doc_id"),
                "document_count": len(documents),
                "chunk_count": len(chunks),
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
    from ..embeddings.factory import EmbeddingFactory
    from ..vectorstores.factory import VectorStoreFactory

    pipeline = IngestionPipeline(
        embedder=EmbeddingFactory.get("hf"),
        vectorstore=VectorStoreFactory.get("chroma"),
    )
    result = pipeline.ingest(r"backend\docs\example\example.pdf")
    print(result)