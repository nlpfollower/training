import json
from typing import Dict, Any, List
from src.types.conversation import Chat, Thread, ThreadCollection

class GPTConversationsParser:
    def __init__(self, input_file: str):
        self.input_file = input_file

    def parse_conversations_json(self) -> ThreadCollection:
        """
        Parse the input JSON file and return a structured collection of Thread objects.

        Returns:
        ThreadCollection: A collection containing parsed chat history as Thread objects
        """
        with open(self.input_file, 'r') as infile:
            data = json.load(infile)

        threads = []
        for item in data:
            title = item.get("title", "")
            created_time = item.get("create_time", "")
            updated_time = item.get("update_time", "")
            mapping = item.get("mapping", {})

            chats = self._extract_ordered_chats(mapping)

            thread = Thread(
                title=title,
                created_time=created_time,
                updated_time=updated_time,
                chats=chats
            )
            threads.append(thread)

        # Reverse the order of threads
        threads.reverse()

        # Create the final ThreadCollection
        return {"threads": {i: thread for i, thread in enumerate(threads)}}

    def _extract_ordered_chats(self, mapping: Dict[str, Any]) -> List[Chat]:
        chats = []
        root_id = next((key for key, value in mapping.items() if value["parent"] is None), None)

        if root_id:
            self._add_messages(root_id, mapping, chats)

        return chats

    def _add_messages(self, node_id: str, mapping: Dict[str, Any], chats: List[Chat]):
        if node_id not in mapping:
            return

        node = mapping[node_id]
        message = node.get("message")

        if message:
            role = message["author"]["role"]
            content = message["content"]
            text_content = self.extract_text_content(content)

            if text_content.strip():
                chat = Chat(role=role, message=text_content)
                chats.append(chat)

        for child_id in node.get("children", []):
            self._add_messages(child_id, mapping, chats)

    def extract_text_content(self, content: Dict[str, Any]) -> str:
        if "parts" in content:
            parts = content["parts"]
            return " ".join(part for part in parts if isinstance(part, str))
        elif "text" in content:
            return content["text"]
        else:
            return ""