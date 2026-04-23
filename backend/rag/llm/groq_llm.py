from .base import BaseLLM
from configs.settings import settings
from langchain_groq import ChatGroq

class GroqLLM(BaseLLM):
    def __init__(self, model: str = "llama-3.1-8b-instant", api_key: str | None = None):
        self.api_key = api_key or settings.groq_api_key
        self.client = ChatGroq(api_key=self.api_key, model=model)

    def generate(self, prompt: str) -> str:
        response = self.client.invoke(prompt)
        return response.text.strip()