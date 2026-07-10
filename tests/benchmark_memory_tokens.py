import argparse
import asyncio
import re
import sys
from pathlib import Path

try:
    import tiktoken
except ImportError:
    tiktoken = None


BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.short_term_memory import (  # noqa: E402
    get_memory_context,
    get_recent_messages,
    get_summary,
    make_messages_key,
    make_summary_key,
    save_short_memory,
)


SAMPLE_TURNS = [
    (
        "What is Aarskog-Scott syndrome?",
        "Aarskog-Scott syndrome is a rare genetic disorder affecting development, facial features, stature, and sometimes genital development. It is commonly associated with changes in the FGD1 gene.",
    ),
    (
        "How is it inherited?",
        "Aarskog-Scott syndrome is usually inherited in an X-linked recessive pattern. Males are more often affected because they have one X chromosome.",
    ),
    (
        "Explain why fathers do not pass it to sons.",
        "Fathers pass their Y chromosome to sons and their X chromosome to daughters, so X-linked traits are not transmitted from father to son.",
    ),
    (
        "What are the common symptoms?",
        "Common features can include widely spaced eyes, short stature, shawl scrotum, mild learning differences, and sometimes hernias or heart defects.",
    ),
    (
        "What is Noonan syndrome?",
        "Noonan syndrome is a genetic condition that can affect facial appearance, growth, heart structure, bleeding tendency, and development.",
    ),
    (
        "How is Noonan syndrome treated?",
        "Treatment is based on specific symptoms, such as cardiac monitoring, developmental support, growth management, and treatment for bleeding issues when present.",
    ),
    (
        "What causes celiac disease?",
        "Celiac disease is an autoimmune response to gluten in genetically susceptible people, leading to inflammation and damage in the small intestine.",
    ),
    (
        "What are its symptoms?",
        "Celiac disease symptoms may include diarrhea, abdominal pain, bloating, fatigue, anemia, weight loss, and nutrient deficiencies.",
    ),
    (
        "What is Addison disease?",
        "Addison disease is adrenal insufficiency caused by inadequate production of adrenal hormones, often due to autoimmune adrenal gland damage.",
    ),
    (
        "How is Addison disease diagnosed?",
        "Diagnosis often involves blood tests for cortisol and ACTH, electrolyte evaluation, and stimulation testing to assess adrenal response.",
    ),
    (
        "What is autoimmune hepatitis?",
        "Autoimmune hepatitis is a chronic liver disease in which the immune system attacks liver cells, causing inflammation and possible liver damage.",
    ),
    (
        "How is autoimmune hepatitis treated?",
        "Treatment often uses immunosuppressive medicines such as corticosteroids and azathioprine, with monitoring of liver function over time.",
    ),
    (
        "What is retrieval augmented generation?",
        "Retrieval augmented generation combines external knowledge retrieval with language generation so answers can be grounded in relevant documents.",
    ),
    (
        "Why does this project use Milvus?",
        "Milvus stores vector embeddings and enables semantic search over medical knowledge chunks and retrievable long-term memories.",
    ),
    (
        "Why does this project use Redis?",
        "Redis stores short-term conversation messages, rolling summaries, query cache entries, and complete long-term memory records.",
    ),
    (
        "Explain the short-term memory design.",
        "The short-term memory keeps the latest messages in a Redis List and compresses older turns into a rolling summary to bound prompt size.",
    ),
    (
        "Explain the long-term memory design.",
        "The long-term memory extracts durable user preferences, project context, and user context, stores full records in Redis, and indexes retrievable memories in Milvus.",
    ),
    (
        "Why use query rewriting?",
        "Query rewriting converts context-dependent follow-up questions into standalone queries, improving retrieval quality for RAG.",
    ),
    (
        "Why use reranking?",
        "Reranking reorders candidate documents so the most relevant chunks are more likely to appear in the final context passed to the LLM.",
    ),
    (
        "Summarize the project architecture.",
        "The project combines FastAPI, a ReAct-style agent, local medical RAG retrieval, Redis-backed memory and cache, Milvus vector search, Celery indexing, and observability tools.",
    ),
]


def get_encoding(model: str):
    if tiktoken is None:
        return None

    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def count_tokens_locally(text: str) -> int:
    pieces = re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\w\s]", text)
    total = 0
    for piece in pieces:
        if re.fullmatch(r"[A-Za-z0-9_]+", piece):
            total += max(1, (len(piece) + 3) // 4)
        else:
            total += 1
    return total


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def build_messages(turns: int) -> list[dict]:
    messages = []
    selected_turns = SAMPLE_TURNS[:turns]
    for user_query, assistant_answer in selected_turns:
        messages.append({"role": "human", "content": user_query})
        messages.append({"role": "ai", "content": assistant_answer})
    return messages


def count_message_tokens(messages: list[dict], model: str) -> int:
    text = "\n".join(
        f"{message.get('role')}: {message.get('content')}"
        for message in messages
    )
    encoding = get_encoding(model)
    if encoding is None:
        return count_tokens_locally(text)
    return len(encoding.encode(text))


def build_offline_summary(messages: list[dict], max_chars: int) -> str:
    transcript = " ".join(
        f"{message.get('role')}: {message.get('content')}"
        for message in messages
    )
    return normalize_whitespace(transcript)[:max_chars]


def print_metrics(
    *,
    mode: str,
    turns: int,
    max_messages: int,
    baseline_messages: list[dict],
    optimized_messages: list[dict],
    summary: str,
    model: str,
) -> None:
    baseline_tokens = count_message_tokens(baseline_messages, model)
    optimized_tokens = count_message_tokens(optimized_messages, model)
    reduction = (
        ((baseline_tokens - optimized_tokens) / baseline_tokens) * 100
        if baseline_tokens > 0
        else 0.0
    )

    print("SHORT-TERM MEMORY TOKEN BENCHMARK")
    print(f"mode={mode}")
    print(f"turns={turns}")
    print(f"max_messages={max_messages}")
    print(f"baseline_messages={len(baseline_messages)}")
    print(f"optimized_messages={len(optimized_messages)}")
    print(f"summary_chars={len(summary)}")
    print(f"baseline_tokens={baseline_tokens}")
    print(f"optimized_tokens={optimized_tokens}")
    print(f"token_reduction={reduction:.1f}%")
    print(
        "resume_metric="
        f"Reduced prompt memory tokens by {reduction:.1f}% "
        f"({baseline_tokens} -> {optimized_tokens}) using a "
        f"{max_messages}-message sliding window plus rolling summary."
    )


def run_offline(args) -> None:
    baseline_messages = build_messages(args.turns)
    old_messages = baseline_messages[:-args.max_messages]
    recent_messages = baseline_messages[-args.max_messages:]
    summary = build_offline_summary(old_messages, args.summary_chars)

    optimized_messages = [
        {"role": "system", "content": f"Summary of earlier conversation: {summary}"},
        *recent_messages,
    ]

    print_metrics(
        mode="offline",
        turns=args.turns,
        max_messages=args.max_messages,
        baseline_messages=baseline_messages,
        optimized_messages=optimized_messages,
        summary=summary,
        model=args.model,
    )


async def run_redis(args) -> None:
    import redis.asyncio as redis

    redis_client = redis.Redis(
        host=args.redis_host,
        port=args.redis_port,
        db=args.redis_db,
        decode_responses=True,
    )

    await redis_client.delete(
        make_messages_key(args.session_id),
        make_summary_key(args.session_id),
    )

    baseline_messages = []
    for user_query, assistant_answer in SAMPLE_TURNS[:args.turns]:
        baseline_messages.extend(
            [
                {"role": "human", "content": user_query},
                {"role": "ai", "content": assistant_answer},
            ]
        )
        await save_short_memory(
            redis_client,
            args.session_id,
            user_query,
            assistant_answer,
            args.model,
            max_messages=args.max_messages,
            ttl=args.ttl,
        )

    optimized_messages = await get_memory_context(redis_client, args.session_id)
    summary = await get_summary(redis_client, args.session_id)
    recent_messages = await get_recent_messages(
        redis_client,
        args.session_id,
        limit=args.max_messages,
    )

    print_metrics(
        mode="redis",
        turns=args.turns,
        max_messages=args.max_messages,
        baseline_messages=baseline_messages,
        optimized_messages=optimized_messages,
        summary=summary,
        model=args.model,
    )
    print(f"redis_recent_messages={len(recent_messages)}")
    print(f"redis_session_id={args.session_id}")

    await redis_client.aclose()


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark short-term memory prompt token reduction."
    )
    parser.add_argument(
        "--mode",
        choices=["offline", "redis"],
        default="offline",
        help="offline estimates with a fixed summary; redis uses real save_short_memory and Redis.",
    )
    parser.add_argument("--turns", type=int, default=20)
    parser.add_argument("--max-messages", type=int, default=10)
    parser.add_argument("--summary-chars", type=int, default=200)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--session-id", default="resume_token_benchmark")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6383)
    parser.add_argument("--redis-db", type=int, default=2)
    parser.add_argument("--ttl", type=int, default=86400)
    args = parser.parse_args()

    if args.turns > len(SAMPLE_TURNS):
        raise ValueError(f"Maximum supported turns is {len(SAMPLE_TURNS)}.")

    if args.max_messages <= 0:
        raise ValueError("--max-messages must be positive.")

    if args.mode == "offline":
        run_offline(args)
    else:
        await run_redis(args)


if __name__ == "__main__":
    asyncio.run(main())
