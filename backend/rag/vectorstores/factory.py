from .chroma_vectorstore import ChromaVectorStore


class VectorStoreFactory:
    _vectorstores = {
        "chroma": ChromaVectorStore,
    }

    @staticmethod
    def get(name: str, **kwargs):
        vectorstore_cls = VectorStoreFactory._vectorstores.get(name)
        if not vectorstore_cls:
            raise ValueError(
                f"Vector store '{name}' not found. Available vector stores: {list(VectorStoreFactory._vectorstores.keys())}"
            )
        return vectorstore_cls(**kwargs)