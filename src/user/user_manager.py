from typing import Dict
from src.user.chat_history import ChatHistory
from src.types.conversation import Thread, ThreadCollection

class UserManager:
    def __init__(self, gptConversationsInputFile: str):
        self.chat_history = ChatHistory()
        self.gptConversationsInputFile = gptConversationsInputFile

    def load_chat_history(self) -> None:
        """
        Initiate loading of chat history.

        Args:
        gptConversationsInputFile (str): Path to the input file containing GPT conversations
        """
        self.chat_history.load_history(self.gptConversationsInputFile)

    def get_chat_history(self) -> ChatHistory:
        return self.chat_history

    def get_user_id(self) -> str:
        return self.chat_history.get_user_id()

    def get_all_threads(self) -> ThreadCollection:
        return self.chat_history.get_all_threads()