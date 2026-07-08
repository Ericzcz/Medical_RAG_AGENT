# Medical RAG Agent

Medical RAG Agent is a FastAPI-based medical retrieval-augmented generation system with a ReAct-style agent, local knowledge-base retrieval, web search, Redis-backed memory, Milvus vector search, Celery background indexing, and production-style observability.

The main endpoint is `/agent_query`. It can combine:

- Short-term session memory from Redis
- Long-term user memory from Redis and Milvus
- Local medical RAG retrieval
- Web search through the agent tool layer

## Highlights

- Built a ReAct-style agent that can route between local medical knowledge retrieval and web search.
- Implemented hybrid local RAG retrieval with Milvus vector search, BM25 retrieval, and Cohere reranking.
- Added Redis query cache for context-free local and batch queries.
- Built short-term memory with Redis sliding window storage and active LLM summarization.
- Built long-term memory with LLM extraction, `skip / merge / create` update decisions, Redis record storage, and Milvus semantic recall.
- Split long-term memory into global memory and retrievable memory to balance personalization and relevance.
- Added Celery background jobs for knowledge-base indexing.
- Added Prometheus metrics, Grafana dashboards, Loki logs, and Promtail log collection.

## Architecture

```text
Client
  |
  v
FastAPI
  |
  |-- /local_query
  |     |-- Redis query cache, for context-free queries
  |     |-- Local RAG retrieval
  |
  |-- /agent_query
        |-- Load short-term memory by session_id
        |-- Load long-term memory by user_id
        |     |-- Redis global memory
        |     |-- Milvus retrievable memory search
        |-- Run ReAct agent
        |     |-- Local medical RAG tool
        |     |-- Web search tool
        |-- Save short-term memory
        |-- Extract and update long-term memory
```

## Services

| Service | Purpose | Local Address |
| --- | --- | --- |
| `api` | FastAPI API service | `http://127.0.0.1:8000` |
| `worker` | Celery worker for background indexing | Docker internal |
| `redis` | Query cache, Celery broker/backend, short-term memory, long-term memory records | `127.0.0.1:6383` |
| `milvus` | Vector database for local RAG and retrievable long-term memory | `127.0.0.1:19530` |
| `etcd` | Milvus metadata dependency | Docker internal |
| `minio` | Milvus object storage | `http://127.0.0.1:9001` |
| `prometheus` | Metrics collection | `http://127.0.0.1:9090` |
| `grafana` | Metrics and log dashboards | `http://127.0.0.1:3000` |
| `loki` | Log storage API | `http://127.0.0.1:3100` |
| `promtail` | Log collector | Docker internal |

Grafana default login:

```text
admin / admin
```

## Tech Stack

- API: FastAPI, Uvicorn, Pydantic
- Agent and LLM: LangChain, OpenAI-compatible chat models
- Retrieval: Milvus, BM25, Cohere Rerank
- Memory and cache: Redis
- Background jobs: Celery
- Observability: Prometheus, Grafana, Loki, Promtail
- Container runtime: Docker Compose

## Setup

Create `.env` in the project root:

```text
OPENAI_API_KEY=...
TAVILY_API_KEY=...
COHERE_API_KEY=...
LANGSMITH_API_KEY=...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=medical-rag-agent
```

Start all services:

```bash
cd /Users/eric_zcz/Desktop/Eric_Project/agent/medical_rag_agent/backend
docker compose up --build -d
```

Check service status:

```bash
docker compose ps
```

Check API logs:

```bash
docker logs medical_rag_api
```

Open API docs:

```text
http://127.0.0.1:8000/docs
```

Stop services:

```bash
docker compose down
```

## API Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Service metadata |
| `GET` | `/health` | Health check |
| `POST` | `/local_query` | Direct local RAG query |
| `POST` | `/batch_local_query` | Batch local RAG queries |
| `POST` | `/agent_query` | Main agent endpoint with memory |
| `POST` | `/batch_agent_query` | Batch agent queries |
| `DELETE` | `/cache` | Delete cached answer |
| `POST` | `/index` | Submit background indexing task |
| `GET` | `/tasks/{task_id}` | Check Celery task status |
| `GET` | `/metrics` | Prometheus metrics |

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Context-free local RAG query:

```bash
curl -X POST http://127.0.0.1:8000/local_query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is retrieval-augmented generation?"
  }'
```

Agent query with short-term and long-term memory:

```bash
curl -X POST http://127.0.0.1:8000/agent_query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Remember that my project is medical_rag_agent and I want Chinese step-by-step explanations.",
    "session_id": "demo_session_1",
    "user_id": "demo_user"
  }'
```

Follow-up in the same session:

```bash
curl -X POST http://127.0.0.1:8000/agent_query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does its memory system work?",
    "session_id": "demo_session_1",
    "user_id": "demo_user"
  }'
```

Cross-session long-term memory test:

```bash
curl -X POST http://127.0.0.1:8000/agent_query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What project am I working on and what explanation style do I prefer?",
    "session_id": "demo_session_2",
    "user_id": "demo_user"
  }'
```

## Memory System

The project has two memory layers:

```text
Short-term memory:
session_id -> Redis messages + rolling summary

Long-term memory:
user_id -> Redis complete memory records
user_id + query -> Milvus semantic recall for retrievable memory
```

### Short-Term Memory

Short-term memory is implemented in `backend/app/short_term_memory.py`.

It is scoped by `session_id` and is designed for current-session continuity, follow-up questions, pronouns, and recent context.

Redis keys:

```text
user:chat:{session_id}:messages
user:chat:{session_id}:summary
```

The `messages` key is a Redis List containing recent user and assistant messages:

```json
{"role": "human", "content": "What is RAG?"}
{"role": "ai", "content": "RAG means retrieval-augmented generation..."}
```

The `summary` key is a Redis String containing a compressed summary of older turns.

Flow:

```text
Request starts
  |
  |-- get_memory_context()
        |-- get_summary()
        |-- get_recent_messages()
        |-- summary + recent messages -> chat_history

Request finishes
  |
  |-- save_short_memory()
        |-- append_turn()
        |-- compress_memory_if_needed()
              |-- old summary + overflow messages -> new summary
              |-- ltrim messages to keep the latest window
```

When the Redis List grows beyond `max_messages`, older messages are summarized by an LLM and removed from the List. The latest messages stay in raw form, while older context is preserved as a rolling summary.

### Long-Term Memory

Long-term memory is implemented in `backend/app/long_term_memory.py`.

It is scoped by `user_id` and is designed for durable personalization across sessions.

The extractor only keeps durable information:

- `communication_preference`: user preferences such as language, explanation style, or code style
- `behavior_correction`: user corrections to assistant behavior
- `project_context`: project background, goals, architecture, or stack
- `user_context`: non-sensitive stable user context, such as learning goals or collaboration preferences

It avoids temporary questions, one-off test data, session IDs, ordinary technical explanations, medical knowledge-base facts, and sensitive medical personal information unless explicitly requested.

### Redis as Source of Truth

Redis stores the complete long-term memory records.

Global memories:

```text
user:{user_id}:global_memory
```

Stores:

```text
communication_preference
behavior_correction
```

Retrievable memories:

```text
user:{user_id}:retrievable_memory
```

Stores:

```text
project_context
user_context
```

Example Redis memory record:

```json
{
  "memory_id": "550e8400-e29b-41d4-a716-446655440000",
  "memory_type": "project_context",
  "content": "The user is building medical_rag_agent with Redis short-term memory and Milvus long-term semantic recall.",
  "importance": 5,
  "user_id": "demo_user",
  "source_session_id": "demo_session_1",
  "created_at": "2026-07-06T12:00:00+00:00"
}
```

### Milvus as Semantic Index

Milvus stores vector indexes only for retrievable long-term memory:

```text
project_context
user_context
```

Collection:

```text
long_term_memory_collection
```

Milvus records use `memory_id` as the primary key. This allows a merged memory to update the same vector record through `upsert`.

At query time:

```text
current query
  -> embedding
  -> Milvus search top_k
  -> filter by user_id
  -> return related project/user context memories
```

Global memories are not searched in Milvus. They are loaded directly from Redis and always injected.

### Long-Term Memory Write Flow

After each `/agent_query` response:

```text
user query + assistant answer
  -> extract_long_term_memory()
  -> list[ExtractedMemory]
  -> save_long_term_memory()
```

Each candidate memory goes through:

```text
exact duplicate check
  -> LLM update decision
       |-- skip
       |-- merge
       |-- create
```

Update policy:

```text
skip:
  ignore fully duplicated memory

merge:
  update an existing Redis List item with lset
  upsert the updated project/user context record into Milvus

create:
  append a new Redis record with rpush
  upsert project/user context records into Milvus
```

Tracked stats:

```text
extracted
saved
merged
skipped_exact_duplicates
skipped_semantic_duplicates
```

### Long-Term Memory Read Flow

When `/agent_query` receives `user_id`:

```text
get_long_term_context(redis_client, user_id, query)
  |
  |-- Redis: get_global_memories()
  |     |-- communication_preference/behavior_correction
  |
  |-- Milvus: get_retrievable_memories_from_milvus()
        |-- project_context/user_context related to current query
        |-- user_id filter prevents cross-user recall
```

The result is injected into the agent instructions:

```text
Global long-term memory:
- communication preferences and behavior corrections

Relevant long-term memory:
- project/user context memories retrieved from Milvus
```

## Query Cache

The Redis query cache is implemented in `backend/app/Redis_Celery/cache.py`.

Cache key format:

```text
cache:{scope}:{sha256(normalized_query)}:{model}
```

The cache is used for context-free queries:

- `/local_query` without `session_id`
- `/batch_local_query`
- `/batch_agent_query`

For `/local_query` with `session_id`, the response depends on chat history, so the local query cache is bypassed.

For `/agent_query` with memory, the response depends on short-term and long-term context, so answer-level caching should be used carefully.

## Knowledge Indexing

Submit a background indexing task:

```bash
curl -X POST http://127.0.0.1:8000/index
```

Check task status:

```bash
curl http://127.0.0.1:8000/tasks/{task_id}
```

The worker builds the Milvus local RAG collection from documents under:

```text
data_base/knowledge_db
```

The compose environment maps this path inside the container:

```text
KNOWLEDGE_BASE_DIR=/workspace/medical_rag_agent/data_base/knowledge_db
```

## Redis Inspection

Connect to Redis DB 2:

```bash
docker exec medical_rag_redis redis-cli -p 6383 -n 2
```

List memory keys:

```bash
docker exec medical_rag_redis redis-cli -p 6383 -n 2 KEYS "user:*"
```

Inspect short-term messages:

```bash
docker exec medical_rag_redis redis-cli -p 6383 -n 2 \
  LRANGE user:chat:demo_session_1:messages 0 -1
```

Inspect short-term summary:

```bash
docker exec medical_rag_redis redis-cli -p 6383 -n 2 \
  GET user:chat:demo_session_1:summary
```

Inspect global long-term memory:

```bash
docker exec medical_rag_redis redis-cli -p 6383 -n 2 \
  LRANGE user:demo_user:global_memory 0 -1
```

Inspect retrievable long-term memory:

```bash
docker exec medical_rag_redis redis-cli -p 6383 -n 2 \
  LRANGE user:demo_user:retrievable_memory 0 -1
```

## Observability

### Prometheus

FastAPI exposes metrics at:

```text
http://127.0.0.1:8000/metrics
```

Prometheus targets are configured in `backend/prometheus.yml`.

Open targets:

```text
http://127.0.0.1:9090/targets
```

Useful PromQL:

```promql
up
```

```promql
http_requests_total
```

```promql
sum by (handler, method, status) (
  rate(http_requests_total[5m])
)
```

```promql
histogram_quantile(
  0.95,
  sum by (le, handler) (
    rate(http_request_duration_seconds_bucket[5m])
  )
)
```

### Grafana

Open Grafana:

```text
http://127.0.0.1:3000
```

Prometheus data source:

```text
http://prometheus:9090
```

Loki data source:

```text
http://loki:3100
```

Recommended panels:

- API request rate
- API p95 latency
- API status-code distribution
- Service health with `up`
- Long-term memory processing logs

### Loki Logs

Application logs are written to:

```text
backend/logs/app.log
```

Promtail reads:

```text
backend/logs/*.log
```

LogQL examples:

```logql
{job="medical_rag"}
```

```logql
{job="medical_rag"} | json
```

```logql
{job="medical_rag"} | json | levelname="ERROR"
```

```logql
{job="medical_rag"} |= "long-term memory processed"
```

`http://127.0.0.1:3100` is the Loki API. Use Grafana at `http://127.0.0.1:3000` for log exploration.

## Benchmarking

Run benchmarks from the project root:

```bash
cd /Users/eric_zcz/Desktop/Eric_Project/agent/medical_rag_agent
```

Single endpoint benchmark:

```bash
python3 tests/benchmark_api.py \
  --base-url http://127.0.0.1:8000 \
  --endpoint local_query \
  --counts 1 5 10 \
  --mode serial
```

Concurrent agent benchmark:

```bash
python3 tests/benchmark_api.py \
  --base-url http://127.0.0.1:8000 \
  --endpoint agent_query \
  --counts 1 5 10 \
  --mode concurrent
```

Batch endpoint benchmark:

```bash
python3 tests/benchmark_api.py \
  --base-url http://127.0.0.1:8000 \
  --endpoint batch_local_query \
  --counts 1 5 10
```

Mixed workload:

```bash
python3 tests/benchmark_api.py \
  --base-url http://127.0.0.1:8000 \
  --workload mixed \
  --counts 10 \
  --mode concurrent
```

## Project Structure

```text
medical_rag_agent/
  backend/
    app/
      main.py                    # FastAPI endpoints and request flow
      agent.py                   # ReAct-style agent tools and orchestration
      rag_chain.py               # Local RAG retrieval, Milvus, BM25, reranking
      short_term_memory.py       # Redis sliding-window session memory
      long_term_memory.py        # Redis/Milvus long-term memory
      Redis_Celery/
        cache.py                 # Redis query cache
        celery_app.py            # Celery app configuration
        tasks.py                 # Background indexing tasks
    compose.yaml                 # Docker Compose stack
    prometheus.yml               # Prometheus targets
    promtail-config.yml          # Promtail log collection
  tests/
    benchmark_api.py             # API benchmark utility
    test_long_term_memory.py     # Long-term memory test script
  requirements.txt
```

## Troubleshooting

API docs do not open:

1. Confirm the API container is running.
2. Open `http://127.0.0.1:8000/docs`.
3. Remember `/agent_query` is a POST endpoint and cannot be opened directly in a browser.

```bash
docker ps --format '{{.Names}} {{.Ports}}'
docker logs medical_rag_api
```

Redis memory keys do not appear:

1. Make sure the request includes `session_id` for short-term memory.
2. Make sure the request includes `user_id` for long-term memory.
3. Use Redis DB 2 and port `6383`.

```bash
docker exec medical_rag_redis redis-cli -p 6383 -n 2 KEYS "user:*"
```

Long-term memory does not appear in Milvus:

1. Confirm the extracted memory type is `project_context` or `user_context`.
2. `communication_preference` and `behavior_correction` are stored only in Redis global memory.
3. Confirm `long_term_memory_collection` exists.
4. Check API logs for memory extraction or Milvus schema errors.

The long-term memory answer seems to come from short-term memory:

1. Test with the same `user_id`.
2. Use a new `session_id`.
3. Ask a semantically related but differently phrased question.

Prometheus cannot find API metrics:

1. Open `http://127.0.0.1:9090/targets`.
2. Confirm `medical-rag-api` is `UP`.
3. In Docker Compose, Prometheus should scrape `api:80`, not `localhost:8000`.

Grafana does not show PromQL results:

1. Use `Code` mode for full PromQL.
2. Set the time range to `Last 5 minutes` or `Last 15 minutes`.
3. Query `up` first, then `http_requests_total`.

Loki does not show logs:

1. Confirm `backend/logs/app.log` exists and has recent content.
2. Check Promtail logs.
3. Use Loki data source URL `http://loki:3100`.
4. Query `{job="medical_rag"}` before adding filters.

Containers still use old dependencies:

```bash
cd /Users/eric_zcz/Desktop/Eric_Project/agent/medical_rag_agent/backend
docker compose up --build -d
```
