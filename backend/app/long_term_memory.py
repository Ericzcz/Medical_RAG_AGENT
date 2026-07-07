import json
from langchain_openai import ChatOpenAI
from app.schemas import ExtractedMemory
from datetime import datetime, timezone
from app.schemas import MemoryUpdateDecision
from uuid import uuid4
from pymilvus import DataType
from app.rag_chain import get_embedding_model, get_milvus_client

LONG_TERM_MEMORY_COLLECTION = "long_term_memory_collection"

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

def make_global_memory_key(user_id: str) -> str:
    return f"user:{user_id}:global_memory"

def make_retrievable_memory_key(user_id: str) -> str:
    return f"user:{user_id}:retrievable_memory"

def get_memory_storage_key(user_id: str, memory: dict | ExtractedMemory) -> str:
    if is_global_memory(memory):
        return make_global_memory_key(user_id)

    return make_retrievable_memory_key(user_id)

def is_global_memory(memory: dict | ExtractedMemory) -> bool:
    memory_type = (
        memory.memory_type
        if isinstance(memory, ExtractedMemory)
        else memory.get("memory_type")
    )
    return memory_type in {"preference", "correction"}


def is_retrievable_memory(memory: dict | ExtractedMemory) -> bool:
    memory_type = (
        memory.memory_type
        if isinstance(memory, ExtractedMemory)
        else memory.get("memory_type")
    )
    return memory_type in {"project", "fact"}

async def get_memories_from_key(
    redis_client,
    key: str,
    limit: int = 20,
) -> list[dict]:
    total = await redis_client.llen(key)
    start_index = max(total - limit, 0)

    raw_items = await redis_client.lrange(key, -limit, -1)

    memories = []
    for offset, item in enumerate(raw_items):
        try:
            memory = json.loads(item)
            memory["_index"] = start_index + offset
            memories.append(memory)
        except json.JSONDecodeError:
            continue

    return memories

async def get_global_memories(redis_client, user_id: str, limit: int = 20) -> list[dict]:
    return await get_memories_from_key(
        redis_client,
        make_global_memory_key(user_id),
        limit=limit,
    )

async def get_retrievable_memories_from_milvus(
    user_id: str,
    query: str,
    top_k: int = 5,
) -> list[dict]:
    client = get_milvus_client()

    if not client.has_collection(collection_name=LONG_TERM_MEMORY_COLLECTION):
        return []

    embedding_model = get_embedding_model()
    query_vector = await embedding_model.aembed_query(query)

    results = client.search(
        collection_name=LONG_TERM_MEMORY_COLLECTION,
        data=[query_vector],
        limit=top_k,
        filter=f'user_id == "{user_id}"',
        output_fields=[
            "memory_id",
            "user_id",
            "memory_type",
            "content",
            "importance",
        ],
    )

    memories = []

    for item in results[0]:
        entity = item.get("entity", {})
        memory = {
            "memory_id": entity.get("memory_id"),
            "user_id": entity.get("user_id"),
            "memory_type": entity.get("memory_type"),
            "content": entity.get("content"),
            "importance": entity.get("importance", 1),
            "score": item.get("distance"),
        }

        if memory["content"]:
            memories.append(memory)

    return memories

    
async def get_long_term_context(
    redis_client,
    user_id: str,
    query: str,
    limit: int = 20,
    top_k: int = 5,
) -> str:
    global_memories = await get_global_memories(
        redis_client,
        user_id,
        limit=limit,
    )

    retrievable_memories = await get_retrievable_memories_from_milvus(
        user_id=user_id,
        query=query,
        top_k=top_k,
    )
    
    context = []

    if global_memories:
        context.append("全局长期记忆：")
        for memory in global_memories:
            content = memory.get("content")
            if content:
                context.append(
                    f"- [{memory.get('memory_type')}, importance={memory.get('importance', 1)}] {content}"
                )

    if retrievable_memories:
        context.append("")
        context.append("相关长期记忆：")
        for memory in retrievable_memories:
            content = memory.get("content")
            if content:
                context.append(
                    f"- [{memory.get('memory_type')}, importance={memory.get('importance', 1)}] {content}"
                )

    return "\n".join(context)


async def decide_memory_update(
    new_memory: ExtractedMemory,
    existing_memories: list[dict],
    model: str,
) -> MemoryUpdateDecision:
    if not existing_memories:
        return MemoryUpdateDecision(action="create")
    
    existing_text = "\n".join(
        f"{memory.get('_index')}. [{memory.get('memory_type', 'fact')}] {memory.get('content', '')}"
        for memory in existing_memories if memory.get("content")
    )

    if not existing_text:
        return MemoryUpdateDecision(action="create")
    
    prompt = f"""
        你是长期记忆更新决策器。

        你的任务是判断一条新长期记忆应该如何处理。

        你只能选择三种动作之一：

        1. skip
        含义：新记忆和已有记忆表达的是同一件事，没有任何值得补充的新信息。

        2. merge
        含义：新记忆和某条已有记忆属于同一主题，并且新记忆补充了有价值的新细节。
        你需要返回 target_index，并给出合并后的 merged_content。

        3. create
        含义：新记忆表达的是新的长期偏好、项目背景、稳定事实或纠正信息，应该新增保存。

        判断原则：
        - 不要因为措辞不同就 create。
        - 不要因为新记忆更具体就 create；如果它是在补充已有记忆，应该 merge。
        - 如果新记忆包含独立的新主题，才 create。
        - 如果不确定，优先 create，避免丢失信息。
        - 不要保存敏感医疗个人信息，除非用户明确要求长期保存。

        已有记忆：
        {existing_text}

        新记忆：
        [{new_memory.memory_type}] {new_memory.content}

        请只返回 JSON，不要解释。

        如果完全重复：
        {{"action": "skip"}}

        如果应该合并：
        {{"action": "merge", "target_index": 0, "merged_content": "合并后的记忆内容"}}

        如果应该新增：
        {{"action": "create"}}
        """
    
    llm = ChatOpenAI(model=model, temperature=0)
    result = await llm.ainvoke(prompt)

    try:
        data = json.loads(result.content)
    except json.JSONDecodeError:
        return MemoryUpdateDecision(action="create")

    try:
        decision = MemoryUpdateDecision(**data)
    except Exception:
        return MemoryUpdateDecision(action="create")
    
    if decision.action == "merge":
        if decision.target_index is None or not decision.merged_content:
            return MemoryUpdateDecision(action="create")
        
    return decision


def ensure_long_term_memory_collection() -> None:
    client = get_milvus_client()

    if client.has_collection(collection_name=LONG_TERM_MEMORY_COLLECTION):
        return

    embedding_model = get_embedding_model()
    dimension = len(embedding_model.embed_query("test"))

    schema = client.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
    )

    schema.add_field(
        field_name="memory_id",
        datatype=DataType.VARCHAR,
        is_primary=True,
        max_length=64,
    )
    schema.add_field(
        field_name="vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=dimension,
    )
    schema.add_field(
        field_name="user_id",
        datatype=DataType.VARCHAR,
        max_length=128,
    )
    schema.add_field(
        field_name="memory_type",
        datatype=DataType.VARCHAR,
        max_length=32,
    )
    schema.add_field(
        field_name="content",
        datatype=DataType.VARCHAR,
        max_length=4096,
    )
    schema.add_field(
        field_name="importance",
        datatype=DataType.INT64,
    )

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )
    index_params.add_index(
        field_name="user_id",
        index_type="AUTOINDEX",
        index_name="user_id_index",
    )

    client.create_collection(
        collection_name=LONG_TERM_MEMORY_COLLECTION,
        schema=schema,
        index_params=index_params,
    )

def format_memory_for_embedding(record: dict) -> str:
    return f"[{record.get('memory_type')}] {record.get('content')}"


async def upsert_retrievable_memory_to_milvus(record: dict) -> None:
    if not is_retrievable_memory(record):
        return

    if not record.get("memory_id"):
        return

    if not record.get("user_id"):
        return

    if not record.get("content"):
        return

    ensure_long_term_memory_collection()

    embedding_model = get_embedding_model()
    client = get_milvus_client()

    text = format_memory_for_embedding(record)
    vector = await embedding_model.aembed_query(text)

    client.upsert(
        collection_name=LONG_TERM_MEMORY_COLLECTION,
        data=[
            {
                "memory_id": record["memory_id"],
                "vector": vector,
                "user_id": record["user_id"],
                "memory_type": record["memory_type"],
                "content": record["content"],
                "importance": record.get("importance", 1),
            }
        ],
    )


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
        "merged": 0,
        "skipped_exact_duplicates": 0,
        "skipped_semantic_duplicates": 0,
    }
        
    if not memories:
        return stats

    for memory in memories:
        key = get_memory_storage_key(user_id, memory)

        existing_memories = await get_memories_from_key(
            redis_client=redis_client,
            key=key,
            limit=max_memories,
        )
        existing_contents = {
            memory.get("content")
            for memory in existing_memories if memory.get("content")
        }

        if memory.content in existing_contents:
            stats["skipped_exact_duplicates"] += 1
            continue
        
        decision = await decide_memory_update(
            new_memory=memory,
            existing_memories=existing_memories,
            model=model,
        )

        if decision.action == "skip":
            stats["skipped_semantic_duplicates"] += 1
            continue
        
        if decision.action == "merge":
            target_index = decision.target_index

            target_memory = next(
                (
                    item for item in existing_memories
                    if item.get("_index") == target_index
                ),
                None,
            )

            if target_memory is None or not decision.merged_content:
                decision.action = "create"
            else:
                updated_record = {
                    **target_memory,
                    "content": decision.merged_content,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }

                updated_record.setdefault("memory_id", str(uuid4()))
                updated_record.pop("_index", None)

                await redis_client.lset(
                    key,
                    target_index,
                    json.dumps(updated_record, ensure_ascii=False),
                )

                await upsert_retrievable_memory_to_milvus(updated_record)

                stats["merged"] += 1
                continue
        
            
        record = memory.model_dump()
        record["user_id"] = user_id
        record["source_session_id"] = session_id
        record["created_at"] = datetime.now(timezone.utc).isoformat()
        record["memory_id"] = str(uuid4())

        await redis_client.rpush(key, json.dumps(record, ensure_ascii=False))

        await redis_client.ltrim(key, -max_memories, -1)

        await upsert_retrievable_memory_to_milvus(record)

        stats["saved"] += 1
    

    return stats




