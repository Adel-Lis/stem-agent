from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime, timezone
import uuid


class EdgeRelation(str, Enum):
    feeds_into = "feeds_into"       # The output of source becomes input of target
    requires = "requires"           # Defines that target cannot run without source completing first
    validates = "validates"         # This is the evaluator edge 
    contradicts = "contradicts"     # this signals a conflict that the builder must resolve


class Edge(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    source_id: str
    target_id: str
    relation: EdgeRelation
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
