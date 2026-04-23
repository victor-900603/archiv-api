from ..embeddings import EmbeddingFactory
from ..vectorstores import VectorStoreFactory
from ..retrieval import (
    VectorRetriever,
    BM25Retriever,
    HybridRetriever,
    MultiQueryRetriever,
    QueryRewriter,
    CrossEncoderReranker,
    ContextBuilder,
)
from ..llm import GroqLLM
from .rag_pipeline import RAGPipeline
from .ingestion_pipeline import IngestionPipeline

def build_rag_pipeline(config: dict = {}) -> RAGPipeline:
    llm = GroqLLM()

    vectorstore = VectorStoreFactory.get(config.get("vectorstore", "chroma"))
    vector_retriever = VectorRetriever(
        vectorstore=vectorstore,
        embedder=EmbeddingFactory.get(config.get("embedding", "bge")),
        default_k=config.get("retriever_k", 10),
    )
    bm25_retriever = BM25Retriever(
        vectorstore=vectorstore,
        default_k=config.get("retriever_k", 10),
    )
    
    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        weight_bm25=config.get("weight_bm25", 0.5),
        weight_vector=config.get("weight_vector", 0.5),
        default_k=config.get("retriever_k", 10),
    )

    rewriter = QueryRewriter(
        llm=llm,
        n_queries=config.get("n_queries", 3),
    )
    
    multi_query_retriever = MultiQueryRetriever(
        retriever=hybrid_retriever,
        rewriter=rewriter,
    )
        
    rag_pipeline = RAGPipeline(
        retriever=multi_query_retriever,
        reranker=CrossEncoderReranker(),
        context_builder=ContextBuilder(),
        llm=llm,
    )

    return rag_pipeline

def build_ingestion_pipeline(config: dict = {}) -> IngestionPipeline:
    ingestion_pipeline = IngestionPipeline(
        embedder=EmbeddingFactory.get(config.get("embedding", "bge")),
        vectorstore=VectorStoreFactory.get(config.get("vectorstore", "chroma")),
    )
    
    return ingestion_pipeline