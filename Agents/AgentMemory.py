from typing import Any, Dict
from pydantic import BaseModel, Field
from AgentMemory.WeaviateImplementation import WeaviateImplementation
from AgentMemory.Memory import Memory
from datetime import datetime
import uuid

class AgentMemory(BaseModel):
    """
    AgentMemory serves as a class for managing agent's memory using Weaviate.
    It uses the agent's identity (UUID) when creating memories of AgentMemory type.
    """
    agent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    weaviate_client: WeaviateImplementation = Field(default_factory=WeaviateImplementation)
    thought_chain: Dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **data):
        super().__init__(**data)
        if not self.thought_chain:
            self.thought_chain = {
                'base_instruction': data.get('base_instruction', ''),
                'thoughts': [f"{datetime.now()}: I was created"]
            }

    def set_id(self, agent_id: str) -> None:
        """
        Method for the agent to set its UUID in the class.
        """
        self.agent_id = agent_id

    def save_thoughts(self) -> str:
        """
        Method for the agent to save its thoughts in the memory.
        The memory is labelled as 'thoughts' and agent_id is required as an additional property.
        """
        memory = Memory(type='thoughts', content=self.thought_chain, agent_id=self.agent_id)
        memory_id = self.create_memory('thoughts', self.thought_chain)
        return memory_id

    def restore_thoughts(self) -> None:
        """
        Method for the agent to restore its thoughts from the memory.
        The memory is labelled as 'thoughts' and agent_id is used to retrieve the memory.
        """
        # Retrieve the memory labelled as 'thoughts' for the agent
        memories = self.search_memories('thoughts')
        # If the memory exists, restore the thought chain
        if memories:
            self.thought_chain = memories[0]['content']
        else:
            raise ValueError("No thoughts found for the agent.")

    def create_memory(self, memory_type: str, content: Dict[str, Any]) -> str:
        """
        Method for the agent to create a memory of AgentMemory type.
        """
        memory = Memory(type=memory_type, content=content, agent_id=self.agent_id)
        memory_id = self.weaviate_client.save_memory(memory)
        return memory_id

    def list_memories(self) -> Dict[str, Any]:
        """
        Method for the agent to list all its memories.
        """
        memories = self.weaviate_client.get_memories_by_property({'agent_id': self.agent_id})
        return memories

    def search_memories(self, query_concept: str, distance: float = 0.55) -> Dict[str, Any]:
        """
        Method for the agent to search its memories based on a concept and a distance.
        """
        search_results = self.weaviate_client.search_memories(query_concept, distance)
        # Filter the search results to only include memories related to the agent
        agent_memories = [memory for memory in search_results if memory['agent_id'] == self.agent_id]
        return agent_memories

    def update_memory(self, memory_id: str, content: Dict[str, Any]) -> None:
        """
        Method for the agent to update a memory of AgentMemory type.
        """
        memory = Memory(id=memory_id, type=content['type'], content=content['content'], agent_id=self.agent_id)
        self.weaviate_client.update_memory(memory)

