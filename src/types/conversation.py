from dataclasses import dataclass
from typing import List, Dict
from enum import Enum

class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    ipython = "ipython"

@dataclass
class Chat:
    role: Role
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