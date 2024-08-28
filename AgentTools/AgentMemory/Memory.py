from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class Memory(BaseModel):
    content: str
    type: str
    agent_id: Optional[str] = Field(default=None)
    additional_properties: Optional[Dict[str, Any]] = Field(default=None)