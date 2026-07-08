from typing import Any, Optional

from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str
    model: str = "gpt-4.0"


class AskResponse(BaseModel):
    source: str
    answer: str


class IndexRequest(BaseModel):
    filename: str


class TaskResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Any] = None
    progress: Optional[Any] = None
