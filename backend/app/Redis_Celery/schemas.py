from pydantic import BaseModel, Field
from typing import Optional, Any


class QueryRequest(BaseModel):
    query: str = Field(..., description="The user question to ask.")
    session_id: str | None = Field(default=None, description="Conversation session id.")

class QueryResponse(BaseModel):
    answer: str
    mode: str

class BatchQueryRequest(BaseModel):
    queries: list[str]

class BatchQueryItem(BaseModel):
    query: str
    answer: str | None = None
    error: str | None = None
    mode: str

class BatchQueryResponse(BaseModel):
    items: list[BatchQueryItem]
    mode: str

class AskRequest(BaseModel):
    question: str
    model: str = 'gpt-4.0'

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