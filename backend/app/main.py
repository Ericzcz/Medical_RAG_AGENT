from fastapi import FastAPI, Request
from fastapi import HTTPException
import redis.asyncio as redis
from contextlib import asynccontextmanager

from app.agent import run_agent, run_agent_batch

from app.rag_chain import search_local_knowledge, search_local_knowledge_batch

from app.Redis_Celery.schemas import QueryRequest, QueryResponse, TaskResponse, BatchQueryItem, BatchQueryRequest, BatchQueryResponse
from app.Redis_Celery.cache import get_cache, set_cache, make_cache_key, delete_cache
from app.Redis_Celery.tasks import index_document
from app.Redis_Celery.celery_app import celery_app
from app.Redis_Celery.memory import get_memory_context, save_memory_turn

from prometheus_fastapi_instrumentator import Instrumentator

import logging
from app.logging_config import setup_logging

LOCAL_SCOPE = "local"
LOCAL_MODEL = "gpt-5.5"

AGENT_SCOPE = "agent"
AGENT_MODEL = "gpt-5.5"

setup_logging()
logger = logging.getLogger(__name__)

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

@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "Medical_RAG_Agent_API",
        "docs": "/docs",
        "health": "/health",
        "endpoints": [
            "/health",
            "/local_query",
            "/batch_local_query",
            "/agent_query",
            "/batch_agent_query",
            "/cache",
            "/index",
            "/tasks/{task_id}",
        ],
    }


@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/local_query", response_model=QueryResponse)
async def local_query(req: QueryRequest, request: Request):
    redis_client = request.app.state.redis
    
    if req.session_id:
        chat_history = await get_memory_context(redis_client, req.session_id)

        answer = await search_local_knowledge(
            req.query,
            LOCAL_MODEL,
            chat_history=chat_history,
        )

        await save_memory_turn(
            redis_client,
            req.session_id,
            req.query,
            answer,
            LOCAL_MODEL,
        )

        return QueryResponse(answer=answer, mode="local_memory")

    cached = await get_cache(redis_client, req.query, LOCAL_SCOPE, LOCAL_MODEL)

    if cached is not None:
        return QueryResponse(answer=cached["answer"], mode="cached")

    answer = await search_local_knowledge(req.query, LOCAL_MODEL)

    await set_cache(
        redis_client,
        req.query,
        LOCAL_SCOPE,
        LOCAL_MODEL,
        {"answer": answer},
    )

    return QueryResponse(answer=answer, mode="local")

# @app.post("/local_query", response_model=QueryResponse)
# async def local_query(req: QueryRequest, request: Request):
#     redis_client = request.app.state.redis
#     cached = await get_cache(redis_client, req.query, LOCAL_SCOPE, LOCAL_MODEL)

#     if cached is not None:
#         logger.info(
#             "local query cache hit",
#             extra={
#                 "scope": LOCAL_SCOPE,
#                 "model": LOCAL_MODEL,
#                 "query_length": len(req.query),
#             },
#         )
#         return QueryResponse(
#             answer=cached["answer"],
#             mode="cached"
#         )

#     logger.info(
#         "local query cache miss",
#         extra={
#             "scope": LOCAL_SCOPE,
#             "model": LOCAL_MODEL,
#             "query_length": len(req.query),
#         },
#     )

#     try:
#         answer = await search_local_knowledge(req.query, LOCAL_MODEL)
#     except RuntimeError as e:
#         logger.exception(
#             "local query failed",
#             extra={
#                 "scope": LOCAL_SCOPE,
#                 "model": LOCAL_MODEL,
#                 "query_length": len(req.query),
#             },
#         )
#         raise HTTPException(status_code=400, detail=str(e))
    
#     # answer = await search_local_knowledge(req.query, LOCAL_MODEL)

#     data = {
#         "answer": answer
#         }
    
#     await set_cache(redis_client, req.query, LOCAL_SCOPE, LOCAL_MODEL, data)

#     return QueryResponse(answer=answer, mode="local")


@app.post("/batch_local_query", response_model=BatchQueryResponse)
async def batch_local_query(req: BatchQueryRequest, request: Request):
    redis_client = request.app.state.redis

    results = [None] * len(req.queries)
    misses = []

    logger.info(
        "batch local query received",
        extra={
            "scope": LOCAL_SCOPE,
            "model": LOCAL_MODEL,
            "total": len(req.queries),
        },
    )

    for idx, query in enumerate(req.queries):
        cached = await get_cache(redis_client, query, LOCAL_SCOPE, LOCAL_MODEL)

        if cached is not None:
            results[idx] = BatchQueryItem(
                query=query,
                answer=cached["answer"],
                error=None,
                mode="cached",
            )
        else:
            misses.append((idx, query))
    
    miss_queries = [query for _, query in misses]

    logger.info(
        "batch local cache summary",
        extra={
            "scope": LOCAL_SCOPE,
            "model": LOCAL_MODEL,
            "total": len(req.queries),
            "cache_hits": len(req.queries) - len(misses),
            "cache_misses": len(misses),
        },
    )

    if not miss_queries:
        return BatchQueryResponse(
            items=results,
            mode="batch_local",
        )

    try:
        fresh_results = await search_local_knowledge_batch(
            miss_queries,
            LOCAL_MODEL,
        )
    except RuntimeError as e:
        logger.exception(
            "batch local query failed",
            extra={
                "scope": LOCAL_SCOPE,
                "model": LOCAL_MODEL,
                "total": len(req.queries),
                "cache_misses": len(misses),
            },
        )
        raise HTTPException(status_code=400, detail=str(e))

    for (idx, query), item in zip(misses, fresh_results):
        if item["error"] is None:
            await set_cache(
                redis_client,
                query,
                LOCAL_SCOPE,
                LOCAL_MODEL,
                {"answer": item["answer"]},
            )

            results[idx] = BatchQueryItem(
                query=query,
                answer=item["answer"],
                error=None,
                mode="local",
            )
        else:
            results[idx] = BatchQueryItem(
                query=query,
                answer=None,
                error=item["error"],
                mode="error",
            )

    return BatchQueryResponse(
        items=results,
        mode="batch_local",
    )

@app.post("/agent_query", response_model=QueryResponse)
async def agent_query(req: QueryRequest, request: Request):
    redis_client = request.app.state.redis

    if req.session_id:
        chat_history = await get_memory_context(redis_client, req.session_id)

        try:
            answer = await run_agent(
                req.query,
                model=AGENT_MODEL,
                chat_history=chat_history,
            )
        except RuntimeError as e:
            logger.exception(
                "agent query failed",
                extra={
                    "scope": AGENT_SCOPE,
                    "model": AGENT_MODEL,
                    "query_length": len(req.query),
                },
            )
            raise HTTPException(status_code=400, detail=str(e))

        await save_memory_turn(
            redis_client,
            req.session_id,
            req.query,
            answer,
            AGENT_MODEL,
        )

        return QueryResponse(answer=answer, mode="agent_memory")



    cached = await get_cache(redis_client, req.query, AGENT_SCOPE, AGENT_MODEL)

   
    if cached is not None:
        logger.info(
            "agent query cache hit",
            extra={
                "scope": AGENT_SCOPE,
                "model": AGENT_MODEL,
                "query_length": len(req.query),
            },
        )
        return QueryResponse(
            answer=cached["answer"],
            mode="cached"
        )
    
    logger.info(
        "agent query cache miss",
        extra={
            "scope": AGENT_SCOPE,
            "model": AGENT_MODEL,
            "query_length": len(req.query),
        },
    )

    try:
        answer = await run_agent(req.query, model=AGENT_MODEL)
    except RuntimeError as e:
        logger.exception(
            "agent query failed",
            extra={
                "scope": AGENT_SCOPE,
                "model": AGENT_MODEL,
                "query_length": len(req.query),
            },
        )
        raise HTTPException(status_code=400, detail=str(e))
    
    #answer = await run_agent(req.query, model=AGENT_MODEL)

    data = {
        'answer': answer
        }
    
    await set_cache(redis_client, req.query, AGENT_SCOPE, AGENT_MODEL, data)

    return QueryResponse(answer=answer, mode="agent")


@app.post("/batch_agent_query", response_model=BatchQueryResponse)
async def batch_agent_query(req: BatchQueryRequest, request: Request):
    redis_client = request.app.state.redis
    results = [None] * len(req.queries)
    misses = []

    logger.info(
        "batch agent query received",
        extra={
            "scope": AGENT_SCOPE,
            "model": AGENT_MODEL,
            "total": len(req.queries),
        },
    )

    for idx, query in enumerate(req.queries):
        cached = await get_cache(redis_client, query, AGENT_SCOPE, AGENT_MODEL)

        if cached is not None:
            results[idx] = BatchQueryItem(
                query=query,
                answer=cached["answer"],
                error=None,
                mode="cached",
            )
        else:
            misses.append((idx, query))

    miss_queries = [query for _, query in misses]

    logger.info(
        "batch agent cache summary",
        extra={
            "scope": AGENT_SCOPE,
            "model": AGENT_MODEL,
            "total": len(req.queries),
            "cache_hits": len(req.queries) - len(misses),
            "cache_misses": len(misses),
        },
    )

    if not miss_queries:
        return BatchQueryResponse(
            items=results,
            mode="batch_agent",
        )

    try:
        fresh_results = await run_agent_batch(
            queries=miss_queries,
            model=AGENT_MODEL,
        )
    except RuntimeError as e:
        logger.exception(
            "batch agent query failed",
            extra={
                "scope": AGENT_SCOPE,
                "model": AGENT_MODEL,
                "total": len(req.queries),
                "cache_misses": len(misses),
            },
        )
        raise HTTPException(status_code=400, detail=str(e))

    for (idx, query), item in zip(misses, fresh_results):
        if item["error"] is None:
            await set_cache(
                redis_client,
                query,
                AGENT_SCOPE,
                AGENT_MODEL,
                {"answer": item["answer"]},
            )

            results[idx] = BatchQueryItem(
                query=query,
                answer=item["answer"],
                error=None,
                mode="agent",
            )
        else:
            results[idx] = BatchQueryItem(
                query=query,
                answer=None,
                error=item["error"],
                mode="error",
            )

    return BatchQueryResponse(
        items=results,
        mode="batch_agent",
    )


@app.delete("/cache")
async def delete(question: str, scope: str, model: str, request: Request):
    r = request.app.state.redis
    key = make_cache_key(question, scope, model)
    deleted = await delete_cache(r, question, scope, model)

    logger.info(
        "cache delete requested",
        extra={
            "scope": scope,
            "model": model,
            "query_length": len(question),
            "deleted": bool(deleted),
        },
    )

    return {
        "key": key,
        "deleted": bool(deleted)
    }

@app.post("/index")
def index():
    try:
        task = index_document.delay()
    except Exception:
        logger.exception("index task submit failed")
        raise
    
    logger.info("index task submitted", extra={"task_id": task.id})
    
    return {
        "task_id": task.id,
        "status": "indexing_submitted",
    }

@app.get("/tasks/{task_id}", response_model=TaskResponse)
def get_task(task_id : str):
    task = celery_app.AsyncResult(task_id)
    
    response = {
        "task_id": task_id,
        "status": task.status,
        "result": None,
        "progress": None,
    }

    if task.status == "PROGRESS":
        response["progress"] = task.info
    elif task.ready():
        response["result"] = task.result
    return response
