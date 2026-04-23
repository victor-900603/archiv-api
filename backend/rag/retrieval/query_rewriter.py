from ..llm.base import BaseLLM

class QueryRewriter:
    def __init__(self, llm: BaseLLM, n_queries: int = 4):
        self.llm = llm
        self.n_queries = n_queries

    def rewrite(self, query: str) -> list[str]:
        prompt = f"""
請將以下問題改寫成 {self.n_queries} 個語意相同但表達不同的查詢，用於文件檢索。
請用簡短句子，每行一個查詢，不要解釋、不要重複、不要回答其他內容。

問題：
{query}
"""

        response = self.llm.generate(prompt)

        queries = [
            q.strip("- ").strip()
            for q in response.split("\n")
            if q.strip()
        ]

        return queries[:self.n_queries]