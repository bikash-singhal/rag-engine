from src.chat.conversation import Conversation
from src.chat.memory import Memory
from src.chat.message import Message


class InMemoryMemory(Memory):

    def __init__(
        self,
        history_limit: int = 10,
    ):
        self._conversation = Conversation()
        self.history_limit = history_limit

    def add_message(self, message: Message) -> None:
        self._conversation.add_message(message)

        if len(self._conversation.messages) > self.history_limit:
            del self._conversation.messages[0]

    def get_messages(self) -> list[Message]:
        return self._conversation.messages.copy()

    def clear(self) -> None:
        self._conversation.messages.clear()
