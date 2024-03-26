import os
import sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import weaviate
from config import config


def create_memory_schema():
    """ not meant to be generally callable, scaffolding """
    client = weaviate.Client(url=config.weaviate_url)

    class_name = "Memory"
    schema = client.schema.get()
    class_names = [cls['class'] for cls in schema['classes']]
    if class_name in class_names:
        print(f"Before deletion: {schema}")
        client.schema.delete_class(class_name)
        time.sleep(5)  # wait for 5 seconds
        schema = client.schema.get()
        print(f"After deletion: {schema}")

    memory_schema = {
        "class": class_name,
        "properties": [
            {
                "name": "type",
                "dataType": ["string"],
                "description": "The type of the memory."
            },
            {
                "name": "content",
                "dataType": ["string"],
                "description": "The content of the memory."
            },
            {
                "name": "agent_id",
                "dataType": ["string"],
                "description": "The identifier of the assistant that the memory belongs to."
            }
        ]
    }

    client.schema.create_class(memory_schema)

if __name__ == "__main__":
    create_memory_schema()