from ..llm.base import BaseLLM
from ..prompts.rewrite_prompt import RewritePrompt

class QueryRewriter:
    def __init__(self, llm: BaseLLM, n_queries: int = 4):
        self.llm = llm
        self.n_queries = n_queries
        self.prompt = RewritePrompt(n_queries=n_queries)

    def rewrite(self, query: str) -> list[str]:
        prompt = self.prompt.format(query=query)

        response = self.llm.generate(prompt)

        queries = [
            q.strip("- ").strip()
            for q in response.split("\n")
            if q.strip()
        ]

        return queries[:self.n_queries]