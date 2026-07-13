from abc import ABC, abstractmethod

from src.chat.message import Message


class Memory(ABC):

    @abstractmethod
    def add_message(self, message: Message) -> None:
        pass

    @abstractmethod
    def get_messages(self) -> list[Message]:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass