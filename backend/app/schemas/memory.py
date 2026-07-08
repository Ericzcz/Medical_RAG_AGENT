from typing import Literal

from pydantic import BaseModel, Field


class ExtractedMemory(BaseModel):
    memory_type: Literal[
        "communication_preference",
        "behavior_correction",
        "project_context",
        "user_context",
    ]
    content: str
    importance: int = Field(ge=1, le=5)


class MemoryUpdateDecision(BaseModel):
    action: Literal["skip", "merge", "create"]
    target_index: int | None = None
    merged_content: str | None = None
    reason: str | None = None
