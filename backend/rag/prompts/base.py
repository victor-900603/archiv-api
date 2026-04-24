from abc import ABC, abstractmethod


class BasePrompt(ABC):
    @abstractmethod
    def format(self, **kwargs) -> list[dict]:
        """
        return:
        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
        """
        pass