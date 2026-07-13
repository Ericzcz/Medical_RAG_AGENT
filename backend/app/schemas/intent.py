from typing import Literal
from pydantic import BaseModel, Field


class IntentDecision(BaseModel):
    intent: Literal[
        "local_medical_qa",
        "web_search",
        "medical_record_insert",
        "medical_record_query",
        "general_chat",
    ]
    confidence: float = Field(ge=0, le=1)
    reason: str