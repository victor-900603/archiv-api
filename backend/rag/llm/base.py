from abc import ABC, abstractmethod

class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: list[dict]) -> str:
        raise NotImplementedError("LLM must implement generate method")