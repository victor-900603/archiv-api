from .base import BasePrompt

USER_PROMPT = """
請將以下問題改寫成 {n_queries} 個語意相同但表達不同的查詢。

要求：
- 每行一個
- 簡短清楚
- 不要解釋

問題：
{query}
"""


class RewritePrompt(BasePrompt):
    def __init__(self, n_queries: int = 4):
        self.n_queries = n_queries

    def format(self, query: str) -> list[dict]:
        return [
            {
                "role": "system",
                "content": "你是一個搜尋專家，負責產生多個查詢語句。"
            },
            {
                "role": "user",
                "content": USER_PROMPT.format(
                    n_queries=self.n_queries,
                    query=query
                )
            }
        ]