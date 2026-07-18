from abc import ABC, abstractmethod

from src.chat.message import Message


class QueryRewriter(ABC):

    @abstractmethod
    def rewrite(
        self,
        question: str,
        history: list[Message],
    ) -> str:
        pass
