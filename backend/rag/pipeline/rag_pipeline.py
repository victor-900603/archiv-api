from ..retrieval import BaseReranker, BaseRetriever, ContextBuilder
from ..llm.base import BaseLLM
from ..prompts import RAGPrompt

class RAGPipeline:
    def __init__(
        self,
        retriever: BaseRetriever,
        reranker: BaseReranker,
        context_builder: ContextBuilder,
        llm: BaseLLM,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.context_builder = context_builder
        self.prompt = RAGPrompt()
        self.llm = llm

    def ask(self, query: str, history: list[dict] | None = None) -> str:
        docs = self.retriever.retrieve(query)

        docs = self.reranker.rerank(query, docs)

        context = self.context_builder.build(docs)

        prompt = self.prompt.format(query=query, context=context, history=history)

        return self.llm.generate(prompt)


if __name__ == "__main__":
    from configs.logging import configure_logging
    from ..retrieval import *
    from ..vectorstores.factory import VectorStoreFactory
    from ..embeddings.factory import EmbeddingFactory
    from..llm.groq_llm import GroqLLM

    configure_logging()
    
    llm = GroqLLM()

    vectorstore = VectorStoreFactory.get("chroma")
    vector_retriever = VectorRetriever(
        vectorstore=vectorstore,
        embedder=EmbeddingFactory.get("bge"),
    )
    bm25_retriever = BM25Retriever(
        vectorstore=vectorstore,
    )
    
    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
    )

    rewriter = QueryRewriter(
        llm=llm,
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
    
    query = "銀行業建立公司治理制度有什麼？"
    response = rag_pipeline.ask(query)
    
    print("Response:", response)