from abc import ABC, abstractmethod
from typing import Dict, Any
from AgentMemory.Memory import Memory

class AssistantMemory(ABC):

    @abstractmethod
    def save_memory(self, memory: Memory) -> str:
        pass

    @abstractmethod
    def retrieve_memory(self, memory_id: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def update_memory(self, memory: Memory) -> str:
        pass

    @abstractmethod
    def delete_memory(self, memory_id: str) -> str:
        pass