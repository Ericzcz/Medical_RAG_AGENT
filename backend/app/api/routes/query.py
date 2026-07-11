import logging

from fastapi import APIRouter, HTTPException, Request

from app.Redis_Celery.cache import get_cache, set_cache
from app.agent import run_agent, run_agent_batch
from app.long_term_memory import (
    extract_long_term_memory,
    get_long_term_context,
    save_long_term_memory,
)
from app.services.local_rag_service import (
    search_local_knowledge,
    search_local_knowledge_batch,
)
from app.schemas import (
    BatchQueryItem,
    BatchQueryRequest,
    BatchQueryResponse,
    QueryRequest,
    QueryResponse,
)
from app.short_term_memory import get_memory_context, save_short_memory


router = APIRouter()
logger = logging.getLogger(__name__)

LOCAL_SCOPE = "local"
LOCAL_MODEL = "gpt-5.5"

AGENT_SCOPE = "agent"
AGENT_MODEL = "gpt-5.5"

MEMORY_MODEL = "gpt-5.5"


@router.post("/local_query", response_model=QueryResponse)
async def local_query(req: QueryRequest, request: Request):
    redis_client = request.app.state.redis

    if req.session_id:
        chat_history = await get_memory_context(redis_client, req.session_id)

        answer = await search_local_knowledge(
            req.query,
            LOCAL_MODEL,
            chat_history=chat_history,
        )

        await save_short_memory(
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


@router.post("/batch_local_query", response_model=BatchQueryResponse)
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


@router.post("/agent_query", response_model=QueryResponse)
async def agent_query(req: QueryRequest, request: Request):
    redis_client = request.app.state.redis

    if req.session_id:
        chat_history = await get_memory_context(redis_client, req.session_id)

        long_term_context = ""
        if req.user_id:
            long_term_context = await get_long_term_context(
                redis_client,
                req.user_id,
                req.query,
            )

        instructions = None

        if long_term_context:
            instructions = f"""
                The following is long-term memory about the user. Use it to understand
                user preferences, project background, and durable context.

                - You must follow global long-term memories with memory_type=communication_preference
                  or memory_type=behavior_correction unless the current user question explicitly overrides them.
                - If you read legacy memory_type=preference or memory_type=correction, treat it as global long-term memory too.
                - If global long-term memory contains an answer-language preference, such as answering in Chinese,
                  follow that preferred language even when the current question is written in another language.

                Memory source notes:
                - Global long-term memory comes from Redis and is always injected. It usually contains user preferences and behavior corrections.
                - Relevant long-term memory comes from Milvus semantic retrieval. It usually contains project background or user context related to the current question.

                Usage rules:
                - Use long-term memory to understand who the user is, what project they are working on, and how they prefer answers.
                - Do not treat long-term memory as a source of medical facts. Medical facts must come primarily from the local knowledge base, reliable search results, or known medical knowledge.
                - When explaining this project's architecture, distinguish accurately: Redis stores short-term memory and global long-term memory, while Milvus provides semantic recall for retrievable long-term memory.

                {long_term_context}
                """

        try:
            answer = await run_agent(
                req.query,
                model=AGENT_MODEL,
                chat_history=chat_history,
                instructions = instructions,
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

        await save_short_memory(
            redis_client,
            req.session_id,
            req.query,
            answer,
            AGENT_MODEL,
        )

        if req.user_id:
            try:
                memories = await extract_long_term_memory(
                    user_query=req.query,
                    assistant_answer=answer,
                    model=MEMORY_MODEL,
                )

                memory_stats = await save_long_term_memory(
                    redis_client=redis_client,
                    user_id=req.user_id,
                    session_id=req.session_id,
                    memories=memories,
                    model=MEMORY_MODEL,
                )

                logger.info(
                    "long-term memory processed",
                    extra={
                       "user_id": req.user_id,
                        "session_id": req.session_id,
                        "extracted": memory_stats["extracted"],
                        "saved": memory_stats["saved"],
                        "merged": memory_stats["merged"],
                        "skipped_exact_duplicates": memory_stats["skipped_exact_duplicates"],
                        "skipped_semantic_duplicates": memory_stats["skipped_semantic_duplicates"],
                    },
                )

            except Exception:
                logger.exception("Failed to extract or save long-term memory")

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


@router.post("/batch_agent_query", response_model=BatchQueryResponse)
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
