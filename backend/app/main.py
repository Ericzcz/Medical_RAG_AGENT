from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
import redis.asyncio as redis

from app.api.routes.cache import router as cache_router
from app.api.routes.health import router as health_router
from app.api.routes.indexing import router as indexing_router
from app.api.routes.query import router as query_router
from app.logging_config import setup_logging
from app.schemas import BatchQueryRequest, QueryRequest, QueryResponse, TaskResponse

# Backward-compatible exports for scripts/tests that import endpoint functions
# from app.main directly.
from app.api.routes.health import health_check, root
from app.api.routes.indexing import get_task, index
from app.api.routes.query import (
    agent_query,
    batch_agent_query,
    batch_local_query,
    local_query,
)


setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis = redis.Redis(
        host="redis",
        port=6383,
        decode_responses=True,
        db=2
    )

    yield

    await app.state.redis.aclose()


app = FastAPI(
    title="Medical_RAG_Agent_API",
    description="Medical RAG Agent API with Redis, Celery, Milvus, and observability.",
    version="1.0.0",
    lifespan=lifespan,
)

Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    excluded_handlers=["/metrics"],
).instrument(app).expose(
    app,
    endpoint="/metrics",
    include_in_schema=False,
)

app.include_router(health_router)
app.include_router(query_router)
app.include_router(cache_router)
app.include_router(indexing_router)
