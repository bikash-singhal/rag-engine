from src.chat.memory import Memory
from src.chat.message import Message
from src.llm.base import LLMProvider


class ChatEngine:

    def __init__(
        self,
        memory: Memory,
        llm: LLMProvider,
    ):
        self.memory = memory
        self.llm = llm

    def ask(self, question: str) -> str:

        user_message = Message(
            role="user",
            content=question,
        )

        self.memory.add_message(user_message)

        response = self.llm.generate(question)

        assistant_message = Message(
            role="assistant",
            content=response,
        )

        self.memory.add_message(assistant_message)

        return response