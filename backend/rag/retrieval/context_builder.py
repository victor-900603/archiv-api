class ContextBuilder:
    def __init__(self, max_tokens: int = 3000):
        self.max_tokens = max_tokens

    def build(self, docs: list[dict]) -> str:
        docs = sorted(
            docs,
            key=lambda x: x.get("rerank_score", x.get("score", 0)),
            reverse=True
        )

        seen = set()
        unique_docs = []

        for d in docs:
            key = d["document"]

            if key not in seen:
                seen.add(key)
                unique_docs.append(d)

        context_parts = []
        total_tokens = 0

        for i, d in enumerate(unique_docs):
            chunk = self._format_chunk(i, d)

            chunk_tokens = len(chunk) // 4

            if total_tokens + chunk_tokens > self.max_tokens:
                break

            context_parts.append(chunk)
            total_tokens += chunk_tokens

        return "\n\n".join(context_parts)
    
    def _format_chunk(self, i: int, doc: dict) -> str:
        return f"""
[Chunk {i}]
Source: {doc.get("source", "unknown")}
Score: {doc.get("rerank_score", doc.get("score", 0))}

{doc["document"]}
""".strip()