import logging

from fastapi import APIRouter, HTTPException, Request

from app.Redis_Celery.cache import get_cache, set_cache
from app.agent import run_agent, run_agent_batch
from app.long_term_memory import (
    extract_long_term_memory,
    get_long_term_context,
    save_long_term_memory,
)
from app.rag_chain import search_local_knowledge, search_local_knowledge_batch
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
                以下是关于用户的长期记忆。它们用于理解用户偏好、项目背景和长期上下文。

                - 对 memory_type=communication_preference 和 memory_type=behavior_correction 的全局长期记忆必须遵守，除非用户在当前问题中明确提出相反要求。
                - 如果读取到旧版本 memory_type=preference 或 memory_type=correction，也按全局长期记忆处理。
                - 如果全局长期记忆中包含回答语言偏好，例如“用中文回答”，即使用户当前问题是英文，也应该使用该偏好语言回答。

                记忆来源说明：
                - 全局长期记忆来自 Redis，会始终注入，通常包含用户偏好和行为纠正。
                - 相关长期记忆来自 Milvus 语义检索，通常包含和当前问题相关的项目背景或用户上下文。

                使用规则：
                - 可以用长期记忆理解用户是谁、正在做什么项目、偏好什么回答方式。
                - 不要把长期记忆当作医学事实来源；医学事实必须优先基于本地知识库、可靠搜索结果或模型已知医学常识。
                - 当解释本项目架构时，请准确区分：Redis 负责短期记忆和全局长期记忆，Milvus 负责可检索长期记忆的语义召回。

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
