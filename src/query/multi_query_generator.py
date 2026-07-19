from abc import ABC, abstractmethod


class MultiQueryGenerator(ABC):

    @abstractmethod
    def generate(
        self,
        question: str,
    ) -> list[str]:
        """
        Generate multiple search queries from the user's question.
        """
        pass
