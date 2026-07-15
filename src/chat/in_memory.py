from src.chat.memory import Memory
from src.chat.message import Message


class InMemoryMemory(Memory):

    def __init__(self):
        self._messages: list[Message] = []

    def add_message(self, message: Message) -> None:
        self._messages.append(message)

    def get_messages(self) -> list[Message]:
        return self._messages.copy()

    def clear(self) -> None:
        self._messages.clear()
