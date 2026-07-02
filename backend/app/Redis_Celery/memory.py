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
        你负责维护一段短期会话记忆摘要。

        已有摘要：
        {current_summary or "无"}

        需要压缩进摘要的新对话：
        {transcript}

        请输出更新后的摘要，要求：
        1. 保留用户问题、关键事实、上下文指代关系
        2. 删除寒暄和重复内容
        3. 不要编造信息
        4. 控制在 200 字以内
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

    raw_old_messages = await redis_client.lrange(key, 0, overflow - 1)

    old_messages = []
    for item in raw_old_messages:
        try:
            old_messages.append(json.loads(item))
        except json.JSONDecodeError:
            continue

    if not old_messages:
        return

    current_summary = await get_summary(redis_client, session_id)
    new_summary = await summarize_messages(current_summary, old_messages, model)

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
            "content": f"以下是之前对话的摘要：{summary}",
        })

    chat_history.extend(recent_messages)
    return chat_history

async def save_memory_turn(
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