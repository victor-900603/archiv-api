from ..retrieval import BaseReranker, BaseRetriever, ContextBuilder
from ..llm.base import BaseLLM

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
        self.llm = llm

    def ask(self, query: str):
        docs = self.retriever.retrieve(query)

        docs = self.reranker.rerank(query, docs)

        context = self.context_builder.build(docs)

        prompt = self._build_prompt(query, context)

        return self.llm.generate(prompt)

    def _build_prompt(self, query: str, context: str) -> str:
        return f"""
### 角色與目標
你是一位專業的知識庫助手。你的任務是閱讀 [檢索內容]，並針對 [問題] 提供準確、客觀且易於理解的回答。

### 執行規則
1. **嚴格限制**：回答必須「完全」基於 [檢索內容]。禁止帶入任何檢索內容之外的外部知識或假設。
2. **誠實原則**：若檢索內容中找不到答案，請回答：「根據目前的知識庫，我無法回答這個問題。」
3. **引文規範**：如果可能，請標註你引用了哪一部分的內容。
4. **語言風格**：使用繁體中文，保持專業且親切的語氣。

### [檢索內容]
{context}

---
### [問題]
{query}

### [回答]
""".strip()


if __name__ == "__main__":
    from ..retrieval import *
    from ..vectorstores.factory import VectorStoreFactory
    from ..embeddings.factory import EmbeddingFactory
    from..llm.groq_llm import GroqLLM
    
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
    
    query = "這篇論文主要在說什麼？"
    response = rag_pipeline.ask(query)
    
    print("Response:", response)