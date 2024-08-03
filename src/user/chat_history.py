from typing import Dict
from src.data.gpt_parser import GPTConversationsParser
from src.types.conversation import Thread, ThreadCollection
from src.utils.logger import log


class ChatHistory:
    def __init__(self):
        self.threads: ThreadCollection = {}
        self.parser: GPTConversationsParser | None = None

    def load_history(self, gptConversationsInputFile: str) -> None:
        """
        Load and parse the chat history from the input file.

        Args:
        gptConversationsInputFile (str): Path to the input file containing GPT conversations
        """
        self.parser = GPTConversationsParser(gptConversationsInputFile)
        self.threads = self.parser.parse_conversations_json()

    def get_all_threads(self) -> ThreadCollection:
        return self.threads

    def get_user_id(self) -> str:
        if self.parser:
            return "single_user"
        else:
            log.error("Chat history has not been loaded yet.")
            raise ValueError("Chat history has not been loaded yet.")