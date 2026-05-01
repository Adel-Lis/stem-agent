from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime, timezone
import uuid


class NodeType(str, Enum):
    prompt = "prompt"
    tool = "tool"


class NodeRole(str, Enum):
    builder = "builder"                 # Represents the builder node role. This is the initial node of the stem agent
    evaluator = "evaluator"             # Represents the evaluator node that has the task to check the graph. This is also the initial node of the stem agent
    domain_kn = "domain_knowledge"      # Represents what the agent learned about the domain
    strategy = "strategy"               # Represents how the agent decided to approach the domain
    tool = "tool"                       # Represents a capacity/tool the agent has aquired


class Node(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    node_type: NodeType
    role: NodeRole
    content: str
    version: int = 1
    status: str = "active"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
