from abc import ABC, abstractmethod
from collections.abc import Iterator


class LLMProvider(ABC):
    """
    Abstract base class for all LLM providers.
    Every provider must implement generate().
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

    @abstractmethod
    def stream(
        self,
        prompt: str,
    ) -> Iterator[str]:
        pass
