import json
from langchain_openai import ChatOpenAI
from app.schemas import ExtractedMemory
from datetime import datetime, timezone

MEMORY_EXTRACT_PROMPT = """
    你是一个长期记忆提取器。

    你的任务是判断当前这一轮用户问题和助手回答中，是否包含未来对话仍然有价值的信息。

    只保存以下类型的信息：
    1. preference: 用户长期偏好，例如语言、解释风格、代码风格
    2. project: 用户正在做的项目背景、技术栈、目标
    3. fact: 稳定事实，例如用户使用的工具、项目名称
    4. correction: 用户对助手行为的纠正，例如不要直接改代码、要一步一步教

    不要保存：
    1. 临时问题
    2. 一次性的测试数据
    3. session_id、临时命令结果
    4. 普通技术解释
    5. 敏感医疗个人信息，除非用户明确要求长期保存

    请只返回 JSON 数组，不要返回 markdown，不要解释。

    如果没有值得保存的信息，返回空数组 []。

    格式：
    [
    {
        "memory_type": "preference",
        "content": "用户希望用中文解释技术问题。",
        "importance": 4
    }
    ]
    """


async def extract_long_term_memory(
    user_query: str,
    assistant_answer: str,
    model: str,
) -> list[ExtractedMemory]:
    llm = ChatOpenAI(model=model, temperature=0)

    message = f"""
        用户问题：
        {user_query}

        助手回答：
        {assistant_answer}
        """

    response = await llm.ainvoke(
        [
            ("system", MEMORY_EXTRACT_PROMPT),
            ("human", message),
        ]
    )

    try:
        raw_memories = json.loads(response.content)
    except json.JSONDecodeError:
        return []

    memories: list[ExtractedMemory] = []

    for item in raw_memories:
        try:
            memories.append(ExtractedMemory(**item))
        except Exception:
            continue

    return memories

def make_long_term_memory_key(user_id: str) -> str:
    return f"user:{user_id}:long_term_memory"
    
async def get_long_term_memory(
    redis_client,
    user_id: str,
    limit: int = 20,
) -> list[dict]:
    key = make_long_term_memory_key(user_id)
    raw_items = await redis_client.lrange(key, -limit, -1)

    memories = []
    for item in raw_items:
        try:
            memories.append(json.loads(item))
        except json.JSONDecodeError:
            continue
    
    return memories

async def get_long_term_context(
    redis_client,
    user_id: str,
    limit: int = 20,
) -> str:
    memories = await get_long_term_memory(
        redis_client=redis_client,
        user_id=user_id,
        limit=limit,
        )
    
    if not memories:
        return ""
    
    context = ["长期记忆:"]

    for memory in memories:
        memory_type = memory.get("memory_type", "fact")
        content = memory.get("content", "")
        importance = memory.get("importance", 1)

        if not content:
            continue

        context.append(f"- [{memory_type}, importance={importance}] {content}")

    return "\n".join(context)

async def is_duplicate_memory(
    new_memory: ExtractedMemory,
    existing_memories: list[dict],
    model: str,
) -> bool:
    if not existing_memories:
        return False
    
    existing_text = "\n".join(
        f"{index + 1}. [{memory.get('memory_type', 'fact')}] {memory.get('content', '')}"
        for index, memory in enumerate(existing_memories)
        if memory.get("content")
    )

    if not existing_text:
        return False
    
    prompt = f"""
        你是长期记忆去重器。

        判断“新记忆”是否已经被“已有记忆”表达过。
        只要语义基本相同，就认为重复。
        不要因为措辞不同就认为不是重复。

        判断标准：
        1. 如果新记忆只是已有记忆的改写、同义表达或更笼统表达，返回 true。
        2. 如果新记忆包含已有记忆没有的新偏好、新事实或新项目背景，返回 false。
        3. 不要因为 memory_type 不同就直接认为不重复，重点看 content 语义。

        已有记忆：
        {existing_text}

        新记忆：
        [{new_memory.memory_type}] {new_memory.content}

        请只返回 JSON，不要解释。
        如果重复，返回：
        {{"is_duplicate": true}}

        如果不重复，返回：
        {{"is_duplicate": false}}
        """

    llm = ChatOpenAI(model=model, temperature=0)
    result = await llm.ainvoke(prompt)

    try:
        data = json.loads(result.content)
    except json.JSONDecodeError:
        return False

    return bool(data.get("is_duplicate", False))

async def save_long_term_memory(
    redis_client,
    user_id: str,
    memories:list[ExtractedMemory],
    model: str,
    session_id: str | None = None,
    max_memories: int = 100,
) -> dict:
    
    stats = {
        "extracted": len(memories),
        "saved": 0,
        "skipped_duplicates": 0,
        "skipped_semantic_duplicates": 0,
    }
        
    if not memories:
        return stats
    
    key = make_long_term_memory_key(user_id)

    existing_memories = await get_long_term_memory(
        redis_client=redis_client,
        user_id=user_id,
        limit=max_memories,
    )
    existing_contents = {
        memory.get("content")
        for memory in existing_memories if memory.get("content")
    }

    for memory in memories:
        if memory.content in existing_contents:
            stats["skipped_duplicates"] += 1
            continue
        
        is_duplicate = await is_duplicate_memory(
            memory, 
            existing_memories, 
            model,
        )
            
        if is_duplicate:
            stats["skipped_semantic_duplicates"] += 1
            continue

        record = memory.model_dump()
        record["user_id"] = user_id
        record["source_session_id"] = session_id
        record["created_at"] = datetime.now(timezone.utc).isoformat()

        
        await redis_client.rpush(key, json.dumps(record, ensure_ascii=False))
        stats["saved"] += 1
        existing_contents.add(memory.content)
        existing_memories.append(record)
    
    if stats["saved"] > 0:
        await redis_client.ltrim(key, -max_memories, -1)

    return stats




