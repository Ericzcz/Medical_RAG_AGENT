import hashlib
import json
from typing import Optional

def make_cache_key(
        question: str, 
        scope: str,
        model: str = '',
    ) -> str:
    
    normalized = question.strip().lower()
    hashed = hashlib.sha256(normalized.encode()).hexdigest()
    
    return f"cache:{scope}:{hashed}:{model}"


async def get_cache(
        redis_client, 
        question: str,
        scope: str,
        model: str = "",
    ) -> Optional[dict]:
    
    key = make_cache_key(question, scope, model)
    cached = await redis_client.get(key)

    if cached is None:
        return None

    try: 
        return json.loads(cached)
    except json.JSONDecodeError as e:
        return None

async def set_cache(
        redis_client, 
        question: str, 
        scope: str,
        model: str, 
        data: dict, 
        ttl: int = 3600
    ) -> None:
    
    key = make_cache_key(question, scope, model)
    await redis_client.set(
        key,
        json.dumps(data, ensure_ascii=False),
        ex=ttl
    )

async def delete_cache(
        redis_client, 
        question: str, 
        scope: str,
        model: str, 
    ) -> bool:

    key = make_cache_key(question, scope, model)
    deleted = await redis_client.delete(key)

    return bool(deleted)