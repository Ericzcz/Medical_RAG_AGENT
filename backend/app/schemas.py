from typing import Literal
from typing import Optional, Any
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    query: str = Field(..., description="The user question to ask.")
    session_id: str | None = Field(default=None, description="Conversation session id.")
    user_id: str | None = Field(default=None, description="User id.")

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


class ExtractedMemory(BaseModel):
    memory_type: Literal["preference", "project", "fact", "correction"]
    content: str
    importance: int = Field(ge=1, le=5)