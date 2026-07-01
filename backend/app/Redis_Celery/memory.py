import json

def make_messages_key(session_id: str) -> str:
    return f"user:chat:{session_id}:messages"

async def get_recent_messages(
    redis_client,
    session_id: str,
    limit: int = 10,
) -> list[dict]:
    key = make_messages_key(session_id)
    raw_items = await redis_client.lrange(key, -limit, -1)

    messages = []
    for item in raw_items:
        try:
            messages.append(json.loads(item))
        except json.JSONDecodeError:
            continue

    return messages


async def append_turn(
    redis_client,
    session_id: str,
    user_query: str,
    assistant_answer: str,
    max_messages: int = 10,
    ttl: int = 86400,
) -> None:
    key = make_messages_key(session_id)

    messages = [
        {"role": "human", "content": user_query},
        {"role": "ai", "content": assistant_answer},
    ]

    await redis_client.rpush(
        key,
        *[json.dumps(message, ensure_ascii=False) for message in messages],
    )
    await redis_client.ltrim(key, -max_messages, -1)
    await redis_client.expire(key, ttl)