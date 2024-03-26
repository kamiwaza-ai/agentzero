from pydantic import BaseModel
from typing import Optional, Dict, Any

class Memory(BaseModel):
    content: str
    type: str
    agent_id: Optional[str] = None
    additional_properties: Optional[Dict[str, Any]] = None