# Medical RAG Agent

Medical RAG Agent is a FastAPI-based medical AI assistant that combines local medical RAG, ReAct-style tool use, short-term and long-term memory, PostgreSQL-backed medical record workflows, and Docker-based observability.

The main endpoint is `/agent_query`. It can:

- Answer medical questions from a local knowledge base
- Search the web for recent or time-sensitive information
- Store and retrieve user-specific medical records
- Maintain short-term conversation context by `session_id`
- Maintain long-term user memory by `user_id`
- Route requests through an intent router and confidence-driven policy

## Quantified Results

Benchmarks were run locally on the current Docker Compose setup.

| Area | Result |
| --- | --- |
| Knowledge base scale | Indexed `11,274` XML files into `16,407` QA documents and `72,877` chunks |
| Short-term memory compression | Reduced prompt memory tokens by `61.9%`, from `737` to `281`, using a 10-message sliding window plus rolling summary |
| Retrieval quality | Improved Recall@5 by `+9.4pp` and Precision@5 by `+2.0pp` with contextual rewriting + hybrid retrieval + reranking over vector-only retrieval |
| Retrieval hit rate | Improved Hit@5 from `84.0%` to `100.0%` in the contextual retrieval benchmark |
| Query cache | Reduced repeated context-free local query latency from `45.875s` to `0.008s` in a 10-query benchmark |
| Agent concurrency | Handled 10 concurrent memory-aware agent requests with `~4.5x` throughput over single-request execution in local testing |
| Skill layer | Orchestrated `4` specialized skills: local RAG, web search, medical record insertion, and medical record querying |
| Medical records | Designed `2` PostgreSQL tables with `3` custom indexes for user and field-level medical record retrieval |

## Core Features

- ReAct-style agent loop with explicit Skill abstraction
- Intent router with confidence policy: `strong`, `weak`, and `uncertain`
- Local medical RAG with Milvus vector search, BM25 retrieval, hybrid search, and reranking
- Redis query cache for context-free local and batch queries
- Redis short-term memory with sliding window and active summarization
- Redis + Milvus long-term memory with `skip / merge / create` update decisions
- PostgreSQL medical record insertion and field-level querying
- Celery background indexing for knowledge-base ingestion
- Prometheus, Grafana, Loki, and Promtail for metrics and logs

## Architecture

```text
Client
  |
  v
FastAPI
  |
  |-- /local_query
  |     |-- Redis query cache
  |     |-- Local medical RAG retrieval
  |
  |-- /agent_query
        |-- Intent Router
        |-- Confidence Policy
        |-- Load short-term memory by session_id
        |-- Load long-term memory by user_id
        |     |-- Redis global memory
        |     |-- Milvus semantic memory recall
        |-- Run ReAct Agent
        |     |-- search_local_knowledge
        |     |-- search_web
        |     |-- insert_medical_record
        |     |-- query_medical_records
        |-- Save short-term memory
        |-- Extract and update long-term memory
```

## Skill Layer

The agent uses an explicit Skill abstraction instead of hard-coded tool functions.

Current skills:

| Skill | Purpose |
| --- | --- |
| `search_local_knowledge` | Query the local medical RAG knowledge base |
| `search_web` | Search the web for recent or external information |
| `insert_medical_record` | Extract structured medical facts from user text and save them |
| `query_medical_records` | Retrieve stored medical record items by field |

The intent router predicts:

```text
local_medical_qa
web_search
medical_record_insert
medical_record_query
general_chat
```

The confidence policy converts router confidence into:

```text
strong    -> strongly prefer the mapped skill
weak      -> use intent as a hint
uncertain -> reason normally or ask a clarifying question
```

This keeps routing flexible: the router guides the ReAct agent, while the agent can still reason about whether a tool call is needed.

## Memory System

The project has two memory layers.

```text
Short-term memory:
session_id -> Redis messages + rolling summary

Long-term memory:
user_id -> Redis complete memory records
user_id + query -> Milvus semantic recall for retrievable memories
```

### Short-Term Memory

Short-term memory is implemented in `backend/app/short_term_memory.py`.

It is scoped by `session_id` and is designed for current-session continuity, follow-up questions, pronouns, and recent context.

Redis keys:

```text
user:chat:{session_id}:messages
user:chat:{session_id}:summary
```

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

Memory types:

| Type | Storage | Purpose |
| --- | --- | --- |
| `communication_preference` | Redis global memory | Language, explanation style, formatting preference |
| `behavior_correction` | Redis global memory | User corrections to assistant behavior |
| `project_context` | Redis + Milvus | Durable project background and architecture |
| `user_context` | Redis + Milvus | Stable non-sensitive user context |

Redis stores complete long-term memory records:

```text
user:{user_id}:global_memory
user:{user_id}:retrievable_memory
```

Milvus stores vector indexes for retrievable memory:

```text
project_context
user_context
```

Collection:

```text
long_term_memory_collection
```

Write flow:

```text
user query + assistant answer
  -> extract_long_term_memory()
  -> save_long_term_memory()
       |-- exact duplicate check
       |-- LLM decision: skip / merge / create
       |-- Redis write
       |-- Milvus upsert for retrievable memories
```

Read flow:

```text
get_long_term_context(redis_client, user_id, query)
  |
  |-- Redis global memory
  |     |-- communication_preference
  |     |-- behavior_correction
  |
  |-- Milvus semantic search
        |-- project_context
        |-- user_context
        |-- user_id filter prevents cross-user recall
```

Medical facts such as allergies, medications, symptoms, diagnoses, and procedures are stored in the medical record system, not in long-term memory.

## Medical Records

Medical records are stored in PostgreSQL for structured and indexed retrieval.

Tables:

```text
medical_records
medical_record_items
```

Custom indexes:

```text
idx_medical_records_user_created
idx_medical_record_items_user_field_created
idx_medical_record_items_record_id
```

The insert workflow extracts structured medical facts from free text:

```text
"Please record that I am allergic to aspirin and currently taking metformin."
  |
  v
insert_medical_record
  |
  |-- medical_records
  |-- medical_record_items
        |-- allergy: aspirin
        |-- medication: metformin
```

The query workflow retrieves field-level records:

```text
"What allergies do I have in my medical record?"
  |
  v
query_medical_records(field="allergies")
  |
  v
"Your medical record lists an allergy to aspirin."
```

Supported field groups:

```text
allergies
symptoms
medications
diagnoses
procedures
vitals
notes
all
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

For memory-aware `/agent_query` requests, the answer depends on short-term and long-term context, so answer-level caching is used carefully.

## Services

| Service | Purpose | Local Address |
| --- | --- | --- |
| `api` | FastAPI API service | `http://127.0.0.1:8000` |
| `worker` | Celery worker for background indexing | Docker internal |
| `redis` | Query cache, Celery broker/backend, short-term memory, long-term memory records | `127.0.0.1:6383` |
| `postgres` | Structured medical record storage | `127.0.0.1:5433` |
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

PostgreSQL local connection:

```text
Host: 127.0.0.1
Port: 5433
Database: medical_rag
User: medical_rag
Password: medical_rag_password
```

## Tech Stack

- API: FastAPI, Uvicorn, Pydantic
- Agent and LLM: OpenAI-compatible models, LangChain utilities
- Retrieval: Milvus, BM25, Cohere Rerank
- Memory and cache: Redis
- Medical records: PostgreSQL, asyncpg
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
| `POST` | `/agent_query` | Main agent endpoint with memory and skills |
| `POST` | `/batch_agent_query` | Batch agent queries |
| `DELETE` | `/cache` | Delete cached answer |
| `POST` | `/index` | Submit background indexing task |
| `GET` | `/tasks/{task_id}` | Check Celery task status |
| `GET` | `/metrics` | Prometheus metrics |

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Local medical RAG query:

```bash
curl -X POST http://127.0.0.1:8000/local_query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Aarskog-Scott syndrome?"
  }'
```

Agent query with memory:

```bash
curl -X POST http://127.0.0.1:8000/agent_query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Remember that my project is medical_rag_agent and I prefer Chinese step-by-step explanations.",
    "session_id": "demo_session_1",
    "user_id": "demo_user"
  }'
```

Insert medical record:

```bash
curl -X POST http://127.0.0.1:8000/agent_query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Please record that I am allergic to aspirin and currently taking metformin.",
    "session_id": "record_insert_demo_1",
    "user_id": "demo_user"
  }'
```

Query medical record:

```bash
curl -X POST http://127.0.0.1:8000/agent_query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What allergies do I have in my medical record?",
    "session_id": "record_query_demo_1",
    "user_id": "demo_user"
  }'
```

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

## Inspection Commands

Connect to Redis DB 2:

```bash
docker exec medical_rag_redis redis-cli -p 6383 -n 2
```

List Redis memory keys:

```bash
docker exec medical_rag_redis redis-cli -p 6383 -n 2 KEYS "user:*"
```

Inspect short-term messages:

```bash
docker exec medical_rag_redis redis-cli -p 6383 -n 2 \
  LRANGE user:chat:demo_session_1:messages 0 -1
```

Inspect long-term global memory:

```bash
docker exec medical_rag_redis redis-cli -p 6383 -n 2 \
  LRANGE user:demo_user:global_memory 0 -1
```

Inspect long-term retrievable memory:

```bash
docker exec medical_rag_redis redis-cli -p 6383 -n 2 \
  LRANGE user:demo_user:retrievable_memory 0 -1
```

Count medical records:

```bash
docker exec medical_rag_postgres psql \
  -U medical_rag \
  -d medical_rag \
  -c "SELECT COUNT(*) AS medical_records_count FROM medical_records;"
```

Count medical record items:

```bash
docker exec medical_rag_postgres psql \
  -U medical_rag \
  -d medical_rag \
  -c "SELECT COUNT(*) AS medical_record_items_count FROM medical_record_items;"
```

List PostgreSQL indexes:

```bash
docker exec medical_rag_postgres psql \
  -U medical_rag \
  -d medical_rag \
  -c "\di"
```

## Benchmarking

Run benchmarks from the project root:

```bash
cd /Users/eric_zcz/Desktop/Eric_Project/agent/medical_rag_agent
```

Knowledge-base scale:

```bash
python3 tests/benchmark_knowledge_base_stats.py
```

Short-term memory token reduction:

```bash
python3 tests/benchmark_memory_tokens.py \
  --mode redis \
  --turns 20 \
  --max-messages 10 \
  --session-id resume_token_benchmark_20
```

Retrieval quality:

```bash
MILVUS_URI=http://localhost:19530 python3 tests/benchmark_retrieval_quality.py \
  --strategies vector hybrid hybrid_rerank \
  --limit 50 \
  --top-k 5 \
  --candidate-k 20 \
  --query-mode contextual \
  --rewrite-contextual \
  --compare-to vector
```

Concurrent agent benchmark:

```bash
python3 tests/benchmark_api.py \
  --base-url http://127.0.0.1:8000 \
  --endpoint agent_query \
  --counts 1 5 10 \
  --mode concurrent \
  --question-set medical \
  --cache-mode uncached
```

Query-cache benchmark:

```bash
python3 tests/benchmark_api.py \
  --base-url http://127.0.0.1:8000 \
  --endpoint local_query \
  --counts 10 \
  --mode concurrent \
  --question-set medical \
  --cache-mode both
```

Intent router smoke test:

```bash
PYTHONPATH=backend python3 tests/test_intent_router.py
```

Long-term memory smoke test:

```bash
PYTHONPATH=backend python3 -m tests.test_long_term_memory
```

## Observability

FastAPI exposes metrics at:

```text
http://127.0.0.1:8000/metrics
```

Prometheus:

```text
http://127.0.0.1:9090/targets
```

Grafana:

```text
http://127.0.0.1:3000
```

Loki data source in Grafana:

```text
http://loki:3100
```

Useful PromQL:

```promql
up
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

Useful LogQL:

```logql
{job="medical_rag"}
```

```logql
{job="medical_rag"} |= "long-term memory processed"
```

## Project Structure

```text
medical_rag_agent/
  backend/
    app/
      main.py                         # FastAPI app setup
      agent.py                        # ReAct agent loop
      rag_chain.py                    # Local RAG retrieval pipeline
      short_term_memory.py            # Redis sliding-window memory
      long_term_memory.py             # Redis/Milvus long-term memory
      logging_config.py               # Application logging setup
      api/
        routes/
          health.py                   # Root and health endpoints
          query.py                    # Local, batch, and agent query endpoints
          cache.py                    # Cache endpoint
          indexing.py                 # Indexing task endpoints
      db/
        schema.sql                    # PostgreSQL medical record schema
        session.py                    # asyncpg connection helper
      schemas/
        query.py                      # Query DTOs
        memory.py                     # Long-term memory DTOs
        medical_record.py             # Medical record DTOs
        intent.py                     # Intent router DTOs
        task.py                       # Task/indexing DTOs
      services/
        intent_router.py              # Intent classification and confidence policy
        local_rag_service.py          # Local RAG service wrapper
        web_search_service.py         # Web search service wrapper
        medical_record_service.py     # PostgreSQL medical record service
      skills/
        base.py                       # BaseSkill, SkillContext, SkillResult
        registry.py                   # Skill registry and tool schemas
        local_rag_skill.py            # Local RAG skill
        web_search_skill.py           # Web search skill
        medical_record_insert_skill.py
        medical_record_query_skill.py
      Redis_Celery/
        cache.py                      # Redis query cache
        celery_app.py                 # Celery app configuration
        tasks.py                      # Background indexing tasks
    compose.yaml                      # Docker Compose stack
    prometheus.yml                    # Prometheus targets
    promtail-config.yml               # Promtail log collection
  data_base/
    knowledge_db/                     # Medical XML/QA knowledge base
  tests/
    benchmark_api.py
    benchmark_knowledge_base_stats.py
    benchmark_memory_tokens.py
    benchmark_retrieval_quality.py
    test_intent_router.py
    test_long_term_memory.py
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

Medical record queries return empty results:

1. Confirm the same `user_id` was used for insert and query.
2. Check `medical_record_items` in PostgreSQL.
3. Confirm the requested field maps to the stored field type, such as `allergies -> allergy`.

```bash
docker exec medical_rag_postgres psql \
  -U medical_rag \
  -d medical_rag \
  -c "SELECT user_id, field_type, field_value, created_at FROM medical_record_items ORDER BY created_at DESC LIMIT 20;"
```

Long-term memory does not appear in Milvus:

1. Confirm the extracted memory type is `project_context` or `user_context`.
2. `communication_preference` and `behavior_correction` are stored only in Redis global memory.
3. Confirm `long_term_memory_collection` exists.
4. Check API logs for memory extraction or Milvus schema errors.

Prometheus cannot find API metrics:

1. Open `http://127.0.0.1:9090/targets`.
2. Confirm `medical-rag-api` is `UP`.
3. In Docker Compose, Prometheus should scrape the internal API service, not the host port.

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
