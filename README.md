# Medical RAG Agent

Medical RAG Agent is a FastAPI-based retrieval-augmented generation system with a ReAct-style agent, local medical knowledge retrieval, web search, Redis-backed memory, Celery background tasks, Milvus vector search, and observability through Prometheus, Grafana, Loki, and Promtail.

The main user-facing endpoint is `/agent_query`. It can combine short-term session memory, long-term user memory, local RAG retrieval, and web search. The `/local_query` endpoint is kept as a direct local RAG interface for debugging, benchmarking, and service-level reuse.

## Features

- FastAPI API service with Swagger docs and Prometheus metrics
- ReAct-style Agent that can call local knowledge search and web search tools
- Local medical RAG retrieval backed by Milvus
- Redis cache for stateless query results
- Redis short-term memory scoped by `session_id`
- Redis long-term memory scoped by `user_id`
- LLM-based long-term memory extraction and `skip / merge / create` updates
- Celery worker for background indexing tasks
- JSON logging to stdout and `backend/logs/app.log`
- Prometheus + Grafana metrics dashboards
- Loki + Promtail log aggregation

## Services

| Service | Purpose | Local Address |
| --- | --- | --- |
| `api` | FastAPI API service | http://127.0.0.1:8000 |
| `worker` | Celery background worker | compose internal |
| `redis` | Cache, Celery broker/backend, memory storage | `127.0.0.1:6383` |
| `milvus` | Vector database | `127.0.0.1:19530` |
| `minio` | Milvus object storage | http://127.0.0.1:9001 |
| `prometheus` | Metrics collection | http://127.0.0.1:9090 |
| `grafana` | Metrics and log visualization | http://127.0.0.1:3000 |
| `loki` | Log storage API | http://127.0.0.1:3100 |
| `promtail` | Log collector | compose internal |

Grafana default login:

```text
admin / admin
```

## Quick Start

Create `.env` in the project root with the required OpenAI, Tavily, and LangSmith variables.

Start all services:

```bash
cd /Users/eric_zcz/Desktop/Eric_Project/agent/medical_rag_agent/backend
docker compose up --build -d
```

Check containers:

```bash
docker compose ps
```

Check API logs:

```bash
docker logs medical_rag_api
```

Open Swagger docs:

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
| `GET` | `/health` | Health check |
| `POST` | `/local_query` | Direct local medical RAG query |
| `POST` | `/batch_local_query` | Batch local RAG queries |
| `POST` | `/agent_query` | Main Agent query endpoint |
| `POST` | `/batch_agent_query` | Batch Agent queries |
| `DELETE` | `/cache` | Delete cached response |
| `POST` | `/index` | Submit Milvus indexing task |
| `GET` | `/tasks/{task_id}` | Query Celery task status |
| `GET` | `/metrics` | Prometheus metrics |

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Stateless local RAG query:

```bash
curl -X POST http://127.0.0.1:8000/local_query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "什么是 RAG？"
  }'
```

Agent query with short-term and long-term memory:

```bash
curl -X POST http://127.0.0.1:8000/agent_query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "以后回答我项目相关问题时，请默认用中文，并一步一步教我。",
    "session_id": "demo_session_1",
    "user_id": "demo_user"
  }'
```

Submit indexing task:

```bash
curl -X POST http://127.0.0.1:8000/index
```

## Memory Architecture

This project implements both short-term and long-term memory.

Short-term memory is scoped by `session_id`. It helps the Agent understand follow-up questions, pronouns, and context within the same conversation.

Long-term memory is scoped by `user_id`. It stores durable user preferences, project facts, stable context, and corrections across sessions.

### Short-Term Memory

Short-term memory is stored in Redis DB 2.

Each session stores recent user/assistant turns in a Redis List:

```text
user:chat:{session_id}:messages
```

When the message count exceeds the configured sliding window, older messages are summarized and moved into:

```text
user:chat:{session_id}:summary
```

At query time, the system builds memory context with:

```text
summary + recent messages
```

This keeps prompts efficient while preserving the important context from earlier turns.

### Long-Term Memory

Long-term memory is also stored in Redis DB 2 and keyed by user:

```text
user:{user_id}:long_term_memory
```

After each `/agent_query` response, the system runs a memory extractor over the current user query and assistant answer. The extractor only keeps durable information such as:

- User preferences
- Project background
- Stable facts
- Corrections to assistant behavior

It avoids one-off questions, temporary test data, session IDs, ordinary technical explanations, and sensitive medical personal information unless the user explicitly asks to save it.

### Long-Term Memory Update Policy

Each candidate memory is passed into an LLM-based update decision step:

```text
skip   -> the memory is fully duplicated, so it is ignored
merge  -> the memory extends an existing memory, so the old Redis List item is updated with lset
create -> the memory is new, so it is appended with rpush
```

The save layer tracks:

```text
extracted
saved
merged
skipped_exact_duplicates
skipped_semantic_duplicates
```

This prevents unbounded duplicate growth while allowing useful new details to refine existing memories.

### Memory Flow

```text
User query
  ↓
Load short-term memory by session_id
  ↓
Load long-term memory by user_id
  ↓
Agent answers using memory context, local RAG, and web search
  ↓
Save current turn to short-term memory
  ↓
Extract long-term memory candidates
  ↓
Decide skip / merge / create
  ↓
Update Redis long-term memory
```

## Memory Testing

Use a fresh `user_id` when testing long-term memory so previous memories do not affect results.

Create a long-term preference:

```bash
curl -X POST http://127.0.0.1:8000/agent_query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "以后回答我项目相关问题时，请默认用中文，并一步一步教我。",
    "session_id": "memory_create_1",
    "user_id": "memory_demo_user"
  }'
```

Merge additional detail into that preference:

```bash
curl -X POST http://127.0.0.1:8000/agent_query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "之后讲项目代码的时候，请先讲背景和整体流程，再进入具体函数。",
    "session_id": "memory_merge_1",
    "user_id": "memory_demo_user"
  }'
```

Verify Redis long-term memory:

```bash
docker exec medical_rag_redis redis-cli -p 6383 -n 2 \
  LRANGE user:memory_demo_user:long_term_memory 0 -1
```

Verify short-term messages:

```bash
docker exec medical_rag_redis redis-cli -p 6383 -n 2 \
  LRANGE user:chat:memory_create_1:messages 0 -1
```

Verify short-term summary:

```bash
docker exec medical_rag_redis redis-cli -p 6383 -n 2 \
  GET user:chat:memory_create_1:summary
```

## Observability

### Prometheus

FastAPI exposes metrics at:

```text
http://127.0.0.1:8000/metrics
```

Prometheus targets are configured in `backend/prometheus.yml`:

- `medical-rag-api`: `api:80/metrics`
- `milvus`: `milvus:9091`

Open Prometheus targets:

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
- API status code distribution
- API p95 latency
- Service health with `up`
- Long-term memory logs filtered by `long-term memory processed`

### Logs With Loki + Promtail

Application logs are written to:

```text
backend/logs/app.log
```

Promtail reads:

```text
backend/logs/*.log
```

Inside the Promtail container, this is mounted as:

```text
/var/log/medical_rag/*.log
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

Note: `http://127.0.0.1:3100` is the Loki API, not a browser UI. Use Grafana at `http://127.0.0.1:3000` to explore logs.

## Benchmark

Run benchmark from the project root:

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

Concurrent Agent benchmark:

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

## Troubleshooting

API docs do not open:

1. Check that `medical_rag_api` is running.
2. Open `http://127.0.0.1:8000/docs`.
3. `/agent_query` is a POST endpoint and cannot be opened directly in the browser.

```bash
docker ps --format '{{.Names}} {{.Ports}}'
docker logs medical_rag_api
```

Prometheus cannot find API metrics:

1. Open `http://127.0.0.1:9090/targets`.
2. Confirm `medical-rag-api` is `UP`.
3. The Docker Compose target should be `api:80`, not `localhost:8000`.
4. Call a few API endpoints before querying `http_requests_total`.

Grafana does not show PromQL results:

1. Use `Code` mode for full PromQL.
2. Set the time range to `Last 5 minutes` or `Last 15 minutes`.
3. Query `up` first, then `http_requests_total`.
4. If traffic is low, prefer `rate(...[5m])` over `rate(...[1m])`.

Loki does not show logs:

1. Confirm `backend/logs/app.log` exists and has recent content.
2. Check Promtail logs with `docker compose logs -f promtail`.
3. Use Loki data source URL `http://loki:3100`.
4. Query `{job="medical_rag"}` before adding filters.

Redis memory keys do not appear:

1. Make sure you are connected to Redis DB 2.
2. Use port `6383`.
3. Check keys with:

```bash
docker exec medical_rag_redis redis-cli -p 6383 -n 2 KEYS "user:*"
```

Dependencies changed but containers still use old packages:

```bash
cd /Users/eric_zcz/Desktop/Eric_Project/agent/medical_rag_agent/backend
docker compose up --build -d
```
