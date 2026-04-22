from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseRetriever(ABC):

    @abstractmethod
    def retrieve(
        self,
        query: str,
        k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        return:
        [
            {
                "text": str,
                "metadata": dict,
                "score": float (optional),
                "source": str (optional)
            }
        ]
        """
        pass