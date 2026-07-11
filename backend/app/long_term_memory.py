import json
from langchain_openai import ChatOpenAI
from app.schemas import ExtractedMemory
from datetime import datetime, timezone
from app.schemas import MemoryUpdateDecision
from uuid import uuid4
from pymilvus import DataType
from app.rag_chain import get_embedding_model, get_milvus_client

LONG_TERM_MEMORY_COLLECTION = "long_term_memory_collection"

GLOBAL_MEMORY_TYPES = {"communication_preference", "behavior_correction"}
RETRIEVABLE_MEMORY_TYPES = {"project_context", "user_context"}

# Keep old records readable so existing Redis and Milvus memory is not orphaned.
LEGACY_GLOBAL_MEMORY_TYPES = {"preference", "correction"}
LEGACY_RETRIEVABLE_MEMORY_TYPES = {"project", "fact"}

MEMORY_EXTRACT_PROMPT = """
    You are a long-term memory extractor.

    Your task is to decide whether the current user question and assistant answer
    contain durable information that will remain useful in future conversations.

    Only save the following types of information:
    1. communication_preference: long-term communication preferences, such as language, explanation style, or code style
    2. behavior_correction: corrections to assistant behavior, such as not editing code directly or teaching step by step first
    3. project_context: stable project background, tech stack, goals, or architecture
    4. user_context: non-sensitive stable user context, such as resume preparation or a desire to learn through guided explanations

    Do not save:
    1. Temporary questions
    2. One-off test data
    3. session_id values or temporary command outputs
    4. Ordinary technical explanations
    5. Medical knowledge-base facts, such as disease definitions, inheritance patterns, or treatments; these should come from the RAG knowledge base, not user memory
    6. Personal medical record facts, including allergies, symptoms, medications, diagnoses, procedures, vitals, lab results, visit history, family history, or past medical history. These belong to the medical record system, not long-term memory. Return only a JSON array. Do not return markdown. Do not explain.
    7. If the user asks to record, save, store, or remember medical record facts, do not extract them as long-term memory. They should be handled by the medical record skill instead.

    If there is nothing worth saving, return an empty array: [].

    Format:
    [
    {
        "memory_type": "communication_preference",
        "content": "The user prefers Chinese explanations for technical questions.",
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
        User question:
        {user_query}

        Assistant answer:
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
    return memory_type in GLOBAL_MEMORY_TYPES | LEGACY_GLOBAL_MEMORY_TYPES


def is_retrievable_memory(memory: dict | ExtractedMemory) -> bool:
    memory_type = (
        memory.memory_type
        if isinstance(memory, ExtractedMemory)
        else memory.get("memory_type")
    )
    return memory_type in RETRIEVABLE_MEMORY_TYPES | LEGACY_RETRIEVABLE_MEMORY_TYPES

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
        context.append("Global long-term memory:")
        for memory in global_memories:
            content = memory.get("content")
            if content:
                context.append(
                    f"- [{memory.get('memory_type')}, importance={memory.get('importance', 1)}] {content}"
                )

    if retrievable_memories:
        context.append("")
        context.append("Relevant long-term memory:")
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
        f"{memory.get('_index')}. [{memory.get('memory_type', 'user_context')}] {memory.get('content', '')}"
        for memory in existing_memories if memory.get("content")
    )

    if not existing_text:
        return MemoryUpdateDecision(action="create")
    
    prompt = f"""
        You are a long-term memory update decision maker.

        Your task is to decide how a new long-term memory candidate should be handled.

        You can choose only one of three actions:

        1. skip
        Meaning: the new memory expresses the same thing as an existing memory and adds no useful new information.

        2. merge
        Meaning: the new memory belongs to the same topic as an existing memory and adds useful new details.
        Return target_index and the merged_content.

        3. create
        Meaning: the new memory expresses a new long-term communication preference, behavior correction, project context, or non-sensitive user context.

        Decision rules:
        - Do not create a new memory only because the wording is different.
        - Do not create a new memory only because the new memory is more specific; if it extends an existing memory, merge it.
        - Create only when the new memory contains an independent new topic.
        - If uncertain, prefer create to avoid losing information.
        - Do not save medical knowledge-base facts such as disease definitions, inheritance patterns, or treatments as user long-term memory.
        - Do not save personal medical record facts such as allergies, symptoms, medications, diagnoses, procedures, vitals, lab results, or visit history as long-term memory. These belong to the medical record system.
        
        Existing memories:
        {existing_text}

        New memory:
        [{new_memory.memory_type}] {new_memory.content}

        Return only JSON. Do not explain.

        If it is fully duplicated:
        {{"action": "skip"}}

        If it should be merged:
        {{"action": "merge", "target_index": 0, "merged_content": "Merged memory content"}}

        If it should be created:
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


