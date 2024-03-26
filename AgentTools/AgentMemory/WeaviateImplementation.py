import weaviate
import uuid
from config import config
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel
from AgentMemory import Memory

class WeaviateImplementation:
    def __init__(self):
        self.client = weaviate.Client(url=config.weaviate_url)

    def save_memory(self, memory: Memory) -> str:
        memory.agent_id = str(uuid.uuid4())
        memory_dict = memory.model_dump(exclude_none=True)
        memory_id: Optional[str] = memory_dict.get('uuid')
        if memory_id:
            self.client.data_object.update(uuid=memory_id, class_name='Memory', data_object=memory_dict)
        else:
            memory_id = self.client.data_object.create(memory_dict, 'Memory')
        return memory_id

    def retrieve_memory(self, memory_id: str) -> Dict[str, Any]:
        return self.client.data_object.get_by_id(memory_id, class_name='Memory')

    def update_memory(self, memory: Memory, memory_id: str = None) -> str:
        memory_dict = memory.model_dump(exclude_none=True)
        if not memory_id:
            memory_id = memory_dict.get('uuid')
        self.client.data_object.update(uuid=memory_id, class_name='Memory', data_object=memory_dict)
        return memory_id

    def delete_memory(self, memory_id: str) -> str:
        self.client.data_object.delete(memory_id, class_name='Memory')
        return memory_id

    def get_memories_by_property(self, properties: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """
        Method to get all weaviate entries matching all of properties.
        If no matching entries are found, return None.
        """
        # Prepare the where filter
        where_filter = {"operator": "And", "operands": []}
        for key, value in properties.items():
            where_filter["operands"].append({
                "path": [key],
                "operator": "Equal",
                "valueString": value
            })
        
        # Execute the query
        result = (
            self.client.query
            .get('Memory', ['content', 'type'])
            .with_additional(['id'])
            .with_where(where_filter)
            .do()
        )
        
        # If no results found, return None
        if not result['data']:
            return None
        
        return result['data']

    def search_memories(self, query_concept: str, distance: float = 0.55) -> Dict[str, Any]:
        """
        Method for the agent to search its memories based on a concept and a distance.
        """
        # Define the near_text_filter
        near_text_filter = {
            "concepts": [query_concept],
            "distance": distance
        }

        # Execute the search query using with_near_text method
        result = (
            self.client.query
            .get('Memory', ['content', 'type'])
            .with_near_text(near_text_filter)
            .with_additional(['id'])
            .with_limit(30)
            .do()
        )
        return result
