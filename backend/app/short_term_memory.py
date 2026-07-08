import json
from langchain_openai import ChatOpenAI

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


def make_summary_key(session_id: str) -> str:
    return f"user:chat:{session_id}:summary"

async def get_summary(redis_client, session_id: str) -> str:
    summary = await redis_client.get(make_summary_key(session_id))
    return summary or ""

async def set_summary(redis_client, session_id: str, summary: str, ttl: int = 86400) -> None:
    key = make_summary_key(session_id)
    await redis_client.set(key, summary, ex=ttl)

async def summarize_messages(
    current_summary: str,
    messages: list[dict],
    model: str,
) -> str:
    transcript = "\n".join(
        f"{message.get('role')}: {message.get('content')}"
        for message in messages
    )

    prompt = f"""
        You maintain a concise short-term conversation memory summary.

        Existing summary:
        {current_summary or "None"}

        New conversation turns to compress into the summary:
        {transcript}

        Output the updated summary with these requirements:
        1. Preserve user questions, key facts, and contextual references
        2. Remove greetings and repeated content
        3. Do not invent information
        4. Keep it within 200 characters
        """

    llm = ChatOpenAI(model=model, temperature=0)
    response = await llm.ainvoke(prompt)
    return response.content


async def compress_memory_if_needed(
    redis_client,
    session_id: str,
    model: str,
    max_messages: int = 10,
    ttl: int = 86400,
) -> None:
    key = make_messages_key(session_id)
    count = await redis_client.llen(key)

    overflow = count - max_messages
    if overflow <= 0:
        return

    raw_messages = await redis_client.lrange(key, 0, overflow - 1)

    messages = []
    for item in raw_messages:
        try:
            messages.append(json.loads(item))
        except json.JSONDecodeError:
            continue

    if not messages:
        return

    current_summary = await get_summary(redis_client, session_id)
    new_summary = await summarize_messages(current_summary, messages, model)

    await set_summary(redis_client, session_id, new_summary, ttl=ttl)
    await redis_client.ltrim(key, overflow, -1)
    await redis_client.expire(key, ttl)

async def get_memory_context(redis_client, session_id: str) -> list[dict]:
    summary = await get_summary(redis_client, session_id)
    recent_messages = await get_recent_messages(redis_client, session_id)

    chat_history = []

    if summary:
        chat_history.append({
            "role": "system",
            "content": f"Summary of earlier conversation: {summary}",
        })

    chat_history.extend(recent_messages)
    return chat_history

# Save the current conversation turn.
async def append_turn(
    redis_client,
    session_id: str,
    user_query: str,
    assistant_answer: str,
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
    await redis_client.expire(key, ttl)

async def save_short_memory(
    redis_client,
    session_id: str,
    user_query: str,
    assistant_answer: str,
    model: str,
    max_messages: int = 10,
    ttl: int = 86400,
) -> None:
    await append_turn(
        redis_client,
        session_id,
        user_query,
        assistant_answer,
        ttl=ttl,
    )

    await compress_memory_if_needed(
        redis_client,
        session_id,
        model,
        max_messages=max_messages,
        ttl=ttl,
    )
