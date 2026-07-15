from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """
    Abstract base class for all LLM providers.
    Every provider must implement generate().
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response from the LLM."""
        pass
