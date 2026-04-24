from .base import BasePrompt

SYSTEM_PROMPT = """你是一個專業助理，請根據提供的資料回答問題。

請嚴格遵守以下規則：
1. 只能根據提供的資料回答
2. 如果資料不足，請回答「不知道」
3. 回答要清楚、有條理
4. 優先使用條列式整理
"""

USER_PROMPT = """
【參考資料】
{context}

【問題】
{query}
"""

class RAGPrompt(BasePrompt):
    def __init__(self, max_history: int = 5):
        self.max_history = max_history

    def format(
        self,
        query: str,
        context: str,
        history: list[dict] | None = None
    ) -> list[dict]:

        messages = []

        messages.append({
            "role": "system",
            "content": SYSTEM_PROMPT
        })

        if history:
            history = history[-self.max_history:]

            for h in history:
                if h["role"] not in ["user", "assistant"]:
                    continue
                messages.append({
                    "role": h["role"],
                    "content": h["content"]
                })

        messages.append({
            "role": "user",
            "content": USER_PROMPT.format(
                context=context,
                query=query
            )
        })

        return messages