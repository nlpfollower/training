from dataclasses import dataclass
from typing import List, Dict
from enum import Enum
from strong_typing.schema import json_schema_type

@json_schema_type
class Role(Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    ipython = "ipython"

@dataclass
class Chat:
    role: str
    message: str

    def is_turn(self) -> bool:
        return self.role in [Role.user, Role.assistant]

    def is_assistant(self) -> bool:
        return self.role == Role.assistant

@dataclass
class Thread:
    title: str
    created_time: str
    updated_time: str
    chats: List[Chat]

ThreadCollection = Dict[str, Dict[int, Thread]]