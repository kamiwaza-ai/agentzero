import pytest
from typing import Optional
from pydantic import ValidationError
from AgentMemory.Memory import Memory
from AgentMemory.WeaviateImplementation import WeaviateImplementation

def test_memory_model():
    """
    Test to check the Memory model
    """
    try:
        memory = Memory(content="Test content", type="Test type")
        assert memory.content == "Test content"
        assert memory.type == "Test type"
        assert memory.agent_id is None
        assert memory.additional_properties is None
    except ValidationError as e:
        pytest.fail(f"Validation error while creating Memory: {e}")

def test_weaviate_implementation():
    """
    Test to check the WeaviateImplementation
    """
    weaviate_impl = WeaviateImplementation()
    memory_id: Optional[str] = None
    try:
        memory = Memory(content="Test content", type="Test type")
        memory_id = weaviate_impl.save_memory(memory)
        assert memory_id is not None

        retrieved_memory = weaviate_impl.retrieve_memory(memory_id)
        assert retrieved_memory['properties']['content'] == "Test content"
        assert retrieved_memory['properties']['type'] == "Test type"

        memory.content = "Updated content"
        updated_memory_id: Optional[str] = weaviate_impl.update_memory(memory, memory_id)
        assert updated_memory_id == memory_id

        updated_memory = weaviate_impl.retrieve_memory(updated_memory_id)
        assert updated_memory['properties']['content'] == "Updated content"

        # Test get_memories_by_property method
        properties = {"content": "Test content", "type": "Test type"}
        memories = weaviate_impl.get_memories_by_property(properties)
        assert memories is not None

        # Test search_memories method
        search_result = weaviate_impl.search_memories("Test content")
        assert search_result is not None
    except ValidationError as e:
        pytest.fail(f"Validation error while testing WeaviateImplementation: {e}")
    finally:
        # Ensure the memory is deleted after the test
        if memory_id:
            try:
                weaviate_impl.delete_memory(memory_id)
            except Exception:
                pass  # Ignore errors during deletion
