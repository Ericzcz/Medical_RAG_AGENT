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
