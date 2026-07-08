import logging

from fastapi import APIRouter, Request

from app.Redis_Celery.cache import delete_cache, make_cache_key


router = APIRouter()
logger = logging.getLogger(__name__)


@router.delete("/cache")
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
