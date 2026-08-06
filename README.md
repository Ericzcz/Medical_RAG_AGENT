# Medical RAG Agent

Medical RAG Agent is a full-stack medical AI assistant with an English chat interface, intent-aware tool routing, local retrieval-augmented generation (RAG), web search, conversation memory, structured medical records, background indexing, and observability.

The browser application sends every question to the main `/agent_query` endpoint. The backend then decides whether to use the local medical knowledge base, search the web, save a medical record, retrieve a medical record, or answer without a tool.

> This project provides general health information and is not a substitute for professional diagnosis or treatment. It is a development project, not a production clinical system.

## At a Glance

| Layer | Implementation | Default address |
| --- | --- | --- |
| Web application | React 19, TypeScript, vinext, Vite | `http://localhost:3001` |
| API | FastAPI, Uvicorn | `http://localhost:8000` |
| API documentation | OpenAPI / Swagger UI | `http://localhost:8000/docs` |
| Short-term memory and cache | Redis | `localhost:6383` |
| Vector retrieval and semantic memory | Milvus | `localhost:19530` |
| Medical records | PostgreSQL | `localhost:5433` |
| Background indexing | Celery | Internal worker |
| Metrics | Prometheus | `http://localhost:9090` |
| Dashboards | Grafana | `http://localhost:3000` |
| Logs | Loki and Promtail | `http://localhost:3100` |

## What Changed From the Backend-Only Version

| Area | Previous version | Current version |
| --- | --- | --- |
| User experience | API, Swagger UI, and command-line requests | Browser chat application |
| Main entry point | Call `/local_query` or `/agent_query` manually | UI always calls `/agent_query`; the agent chooses the appropriate skill |
| Conversations | Caller manually supplied a `session_id` | Users can create, reopen, switch, and delete recent conversations |
| User identity | Caller manually supplied a `user_id` | Anonymous device-local ID generated on first visit and reused in that browser |
| Startup | Backend services only | Backend services and frontend are started separately |
| Cross-origin access | Not required | Explicit FastAPI CORS configuration for the local frontend and configured production origins |
| Ports | Backend stack used port `3000` for Grafana | Frontend uses `3001`; Grafana remains on `3000` |

The backend endpoints remain available for scripts, benchmarks, and direct API use. In particular, `/local_query` and both batch endpoints were not removed; they are simply not exposed as modes in the main UI.

## Architecture

```text
Browser UI :3001
  |
  | POST /agent_query
  | query + user_id + session_id
  v
FastAPI :8000
  |
  |-- Intent router
  |     |-- local_medical_qa
  |     |-- web_search
  |     |-- medical_record_insert
  |     |-- medical_record_query
  |     `-- general_chat
  |
  |-- ReAct-style agent
  |     |-- search_local_knowledge
  |     |-- search_web
  |     |-- insert_medical_record
  |     `-- query_medical_records
  |
  |-- Redis
  |     |-- short-term session memory
  |     |-- global long-term memory
  |     |-- query cache
  |     `-- Celery broker/results
  |
  |-- Milvus
  |     |-- medical knowledge vectors
  |     `-- retrievable long-term memory
  |
  `-- PostgreSQL
        `-- structured medical records

Knowledge files -> Celery worker -> embeddings -> Milvus
FastAPI metrics -> Prometheus -> Grafana
Application logs -> Promtail -> Loki -> Grafana
```

## Core Capabilities

### Intelligent routing

The intent router classifies each question into one of five intents:

| Intent | Preferred behavior |
| --- | --- |
| `local_medical_qa` | Search the local medical knowledge base |
| `web_search` | Search for recent, external, or time-sensitive information |
| `medical_record_insert` | Extract and store structured medical facts |
| `medical_record_query` | Retrieve the current user's stored medical records |
| `general_chat` | Answer directly unless a tool is useful |

Confidence is converted into one of three routing policies:

- `strong` at `>= 0.85`
- `weak` at `>= 0.55`
- `uncertain` below `0.55`

The router guides the agent rather than hard-coding a route, so the agent can still choose a different skill when the question clearly requires it.

The agent, intent router, local answer generation, and memory extraction currently use the model name `gpt-5.5` configured directly in `backend/app/api/routes/query.py`. Change and validate those constants before using a different model.

### Local medical RAG

The local retrieval pipeline:

1. Loads PDF, Markdown, and XML documents from `data_base/knowledge_db`.
2. Splits content into 500-character chunks with 100-character overlap.
3. Creates OpenAI `text-embedding-3-small` vectors.
4. Retrieves candidates from Milvus and BM25.
5. Deduplicates the combined candidates.
6. Reranks candidates with Cohere `rerank-v3.5`.
7. Generates an answer from the highest-ranked context.

When chat history is present, the latest question is first rewritten into a standalone retrieval query.

### Web search

The web-search skill uses Tavily advanced search for recent or internet-dependent questions. Search results are returned to the agent with titles, URLs, excerpts, and scores.

### Medical records

The medical-record insertion skill extracts structured facts from free text and stores them in PostgreSQL. Supported item categories include:

- allergies
- symptoms
- medications
- diagnoses
- procedures
- vitals

Records are isolated by `user_id` and retain the originating `session_id` when one is provided.

## Identity, Sessions, and Storage

The application uses different identifiers for different kinds of state:

| State | Scope | Storage | Persistence |
| --- | --- | --- | --- |
| Visible recent conversations | Current browser profile | Browser storage | Until site data is cleared |
| Short-term chat context | `session_id` | Redis DB 2 | 24 hours after the latest turn |
| Global long-term memory | `user_id` | Redis DB 2 | Persistent Redis data |
| Semantically retrievable memory | `user_id` | Milvus | Persistent Milvus data |
| Medical records | `user_id` | PostgreSQL | Persistent database rows |
| Context-free query cache | Query + scope + model | Redis DB 2 | 1 hour |

Short-term memory keeps the latest 10 messages in raw form and compresses older overflow into a rolling summary. Long-term memory is loaded and updated only when an agent request contains both a Session ID and a User ID. Redis long-term lists are capped by the backend at 100 records per group.

### User and Session IDs

The frontend creates an anonymous User ID on first use and reuses it in the same browser profile. Each new conversation receives a separate Session ID. Click the `•••` button in the top-right corner of the application to view the current IDs.

### Important identity limitation

The generated User ID is an anonymous browser identifier, not authentication. It can be changed or impersonated by someone with browser access. Do not use this identity model for a production system containing real patient information.

For user-isolation testing, use separate browser profiles or an incognito window so that each test user receives separate browser data and sessions.

## Quick Start

### Prerequisites

- Docker Desktop with Docker Compose
- Node.js `>= 22.13.0`
- npm
- API credentials for the features you plan to use

Python 3.12 and the local `.venv` are useful for running scripts directly, but the backend itself runs in Docker.

### 1. Configure backend credentials

Create `.env` in the project root:

```dotenv
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key
COHERE_API_KEY=your_cohere_key

# Optional tracing
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=medical-rag-agent

# Add deployed frontend origins as a comma-separated list when needed
CORS_ORIGINS=
```

Credential usage:

- `OPENAI_API_KEY` is required for the agent, question rewriting, embeddings, and memory extraction.
- `COHERE_API_KEY` is required for local-result reranking.
- `TAVILY_API_KEY` is required only when the agent uses web search.
- LangSmith settings are optional.

Never commit `.env`.

### 2. Add knowledge-base documents

Place `.xml`, `.pdf`, or `.md` files under:

```text
data_base/knowledge_db/
```

The knowledge-base contents are intentionally ignored by Git; only `.gitkeep` is tracked.

### 3. Start the backend stack

```bash
cd backend
docker compose up --build -d
docker compose ps
```

Confirm the API is healthy:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status":"ok"}
```

`/health` is a basic API liveness check. It does not verify Redis, Milvus, PostgreSQL, the worker, or external API credentials.

### 4. Build the local knowledge index

This step is required before the local RAG skill can retrieve documents. It creates embeddings and may take time and incur API usage for a large corpus.

Submit the background task:

```bash
curl -X POST "http://localhost:8000/index?force_rebuild=false"
```

Example response:

```json
{
  "task_id": "<task-id>",
  "status": "indexing_submitted"
}
```

Poll the returned task ID:

```bash
curl "http://localhost:8000/tasks/<task-id>"
```

Use `force_rebuild=true` only when you intentionally want to replace an existing collection.

### 5. Start the frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:3001
```

The frontend defaults to `http://localhost:8000`. To use another API address, create `frontend/.env.local`:

```dotenv
NEXT_PUBLIC_API_BASE_URL=https://api.example.com
```

Restart the frontend development server after changing this value.

### Stop the backend

```bash
cd backend
docker compose down
```

This stops containers without deleting the named data volumes.

## Using the Frontend

1. Open `http://localhost:3001` after starting the backend and frontend.
2. Enter a health question and press Enter or click the send button. Use Shift+Enter to add a new line.
3. Use **New conversation** to start a separate session.
4. Use **Recents** to reopen a conversation. Use its delete button to remove it from the browser.
5. Click `•••` to view the active User ID and Session ID.
6. Check the top-bar status when you need to confirm whether recent conversations are being saved on the device.

The frontend sends all questions to the intelligent agent, which selects the appropriate backend skill automatically.

Recent conversations are stored only in the current browser profile. They are not synchronized to another device, and clearing site data removes them. Deleting a recent conversation removes the browser copy; the corresponding Redis short-term context expires separately after 24 hours.

## API Reference

### Request models

Single-query request:

```json
{
  "query": "What are the common side effects of metformin?",
  "session_id": "session-demo-1",
  "user_id": "user-demo-1"
}
```

`session_id` and `user_id` are optional at the API level. The frontend sends both.

Single-query response:

```json
{
  "answer": "...",
  "mode": "agent_memory"
}
```

Batch request:

```json
{
  "queries": [
    "What is Aarskog-Scott syndrome?",
    "What is Addison disease?"
  ]
}
```

### Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Service metadata and endpoint list |
| `GET` | `/health` | Basic API liveness check; does not check dependencies |
| `POST` | `/agent_query` | Main agent endpoint; supports session and user memory |
| `POST` | `/local_query` | Direct local RAG endpoint |
| `POST` | `/batch_agent_query` | Batch agent execution with cache support; no session or user memory |
| `POST` | `/batch_local_query` | Batch local RAG execution with cache support; no session memory |
| `DELETE` | `/cache` | Delete one cached answer by question, scope, and model |
| `POST` | `/index` | Submit a Celery indexing task |
| `GET` | `/tasks/{task_id}` | Inspect indexing task state and progress |
| `GET` | `/metrics` | Prometheus metrics; hidden from OpenAPI schema |

`DELETE /cache` expects `question`, `scope`, and `model` as query parameters. `POST /index` accepts `force_rebuild` as a query parameter.

### Agent request example

```bash
curl -X POST http://localhost:8000/agent_query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "Please remember that I am allergic to aspirin.",
    "session_id": "session-demo-1",
    "user_id": "user-demo-1"
  }'
```

Use the same User ID to query the stored record:

```bash
curl -X POST http://localhost:8000/agent_query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "What allergies are in my medical record?",
    "session_id": "session-demo-2",
    "user_id": "user-demo-1"
  }'
```

### Direct local-query example

```bash
curl -X POST http://localhost:8000/local_query \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is Aarskog-Scott syndrome?"}'
```

Context-free requests can use the one-hour query cache. Requests with a Session ID use conversation memory instead of the context-free cache path.

## CORS Configuration

The API allows the local frontend origins:

```text
http://localhost:3001
http://127.0.0.1:3001
```

Ports `3000` on localhost and `127.0.0.1` are also allowed for compatibility. For a deployed frontend, add its exact origin to the root `.env`:

```dotenv
CORS_ORIGINS=https://app.example.com,https://preview.example.com
```

Do not include paths in origins. Restart the API container after changing this value.

## Docker Services and Ports

| Service | Container | Host port(s) | Purpose |
| --- | --- | --- | --- |
| API | `medical_rag_api` | `8000` | FastAPI application |
| Worker | `medical_rag_worker` | — | Celery knowledge indexing |
| Redis | `medical_rag_redis` | `6383` | Cache, memory, broker, results |
| PostgreSQL | `medical_rag_postgres` | `5433` | Medical records |
| Milvus | `medical_rag_milvus` | `19530`, `9091` | Vector data and metrics |
| etcd | `medical_rag_etcd` | — | Milvus metadata |
| MinIO | `medical_rag_minio` | `9001` | Milvus object storage console |
| Prometheus | `medical_rag_prometheus` | `9090` | Metrics collection |
| Grafana | `medical_rag_grafana` | `3000` | Dashboards |
| Loki | `medical_rag_loki` | `3100` | Log storage |
| Promtail | `medical_rag_promtail` | — | Log forwarding |

## Observability

Useful addresses:

```text
API metrics:        http://localhost:8000/metrics
Prometheus targets: http://localhost:9090/targets
Grafana:            http://localhost:3000
```

Grafana development credentials are configured in `backend/compose.yaml`:

```text
username: admin
password: admin
```

Configure the Loki data source inside Grafana with:

```text
http://loki:3100
```

Configure the Prometheus data source with:

```text
http://prometheus:9090
```

The repository starts Grafana but does not provision data sources or dashboards automatically.

Example PromQL:

```promql
sum by (handler, method, status) (
  rate(http_requests_total[5m])
)
```

Example LogQL:

```logql
{job="medical_rag"}
```

## Development and Verification

### Frontend

```bash
cd frontend
npm run lint
npm run build
```

### Backend smoke checks

With the virtual environment active and dependencies installed:

```bash
python3 -m pip install -r requirements.txt
python3 -m compileall backend/app
PYTHONPATH=backend python3 tests/test_intent_router.py
```

The intent-router script calls the configured model and therefore requires credentials and network access.

### Benchmarks

Run benchmark scripts from the repository root while the required services are available:

```bash
python3 tests/benchmark_knowledge_base_stats.py
```

```bash
python3 tests/benchmark_memory_tokens.py \
  --mode redis \
  --turns 20 \
  --max-messages 10 \
  --session-id benchmark-memory-20
```

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

```bash
python3 tests/benchmark_api.py \
  --base-url http://localhost:8000 \
  --endpoint agent_query \
  --counts 1 5 10 \
  --mode concurrent \
  --question-set medical \
  --cache-mode uncached
```

## Recorded Local Benchmark Snapshot

These are historical measurements from the local Docker Compose environment; the original run date was not recorded. They are not guaranteed production performance and should be rerun after model, data, hardware, or dependency changes.

| Area | Recorded result |
| --- | --- |
| Knowledge-base scale | 11,274 XML files, 16,407 QA documents, and 72,877 chunks |
| Short-term memory compression | Prompt-memory tokens reduced by 61.9%, from 737 to 281 |
| Retrieval quality | Recall@5 improved by 9.4 percentage points and Precision@5 by 2.0 points over vector-only retrieval |
| Retrieval hit rate | Hit@5 improved from 84.0% to 100.0% in the contextual retrieval benchmark |
| Query cache | Repeated context-free local query latency decreased from 45.875 seconds to 0.008 seconds in a 10-query benchmark |
| Agent concurrency | Ten concurrent memory-aware requests produced approximately 4.5x the throughput of single-request execution |

## Project Structure

```text
medical_rag_agent/
  frontend/                     # Browser application
  backend/
    app/
      main.py                   # FastAPI setup, CORS, routers, metrics
      agent.py                  # ReAct-style agent loop
      rag_chain.py              # Hybrid RAG and indexing pipeline
      short_term_memory.py      # Redis session memory
      long_term_memory.py       # Redis/Milvus long-term memory
      api/routes/               # Health, query, cache, indexing routes
      services/                 # RAG, web search, records, intent routing
      skills/                   # Agent skill abstraction and registry
      db/                       # PostgreSQL schema and connection helper
      Redis_Celery/             # Cache, Celery app, indexing task
    compose.yaml                # Complete backend/observability stack
    prometheus.yml
    promtail-config.yml
  data_base/
    knowledge_db/               # Local XML, PDF, and Markdown corpus
  tests/
    benchmark_api.py
    benchmark_knowledge_base_stats.py
    benchmark_memory_tokens.py
    benchmark_retrieval_quality.py
    test_intent_router.py
    test_long_term_memory.py
  requirements.txt
  README.md
```

## Troubleshooting

### The frontend shows `Service offline`

Check the API first:

```bash
curl http://localhost:8000/health
docker logs medical_rag_api
```

Confirm that `NEXT_PUBLIC_API_BASE_URL` points to the correct API and restart the frontend after changing it.

### The browser blocks the API request

This is usually a CORS origin mismatch. Add the exact frontend origin to `CORS_ORIGINS` and restart the API container.

### Local medical questions fail

Confirm that:

1. Knowledge files exist under `data_base/knowledge_db`.
2. The worker is running.
3. `/index` completed successfully.
4. The `RAG_collection` collection exists in Milvus.

```bash
docker logs medical_rag_worker
curl http://localhost:8000/tasks/<task-id>
```

### Recent conversations disappear

Recent conversations are stored in the current browser profile. They disappear after clearing site data and normally disappear when an incognito/private session closes. Check the status text in the top bar to confirm whether writes are succeeding.

### Two test users appear to share context

Use separate browser profiles or clear the site's browser data before changing User ID manually. Short-term memory is scoped by Session ID, so reopening an old conversation also reuses its backend conversation context.

### Medical-record queries return no results

Use the same User ID for insertion and retrieval. Inspect recent stored items with:

```bash
docker exec medical_rag_postgres psql \
  -U medical_rag \
  -d medical_rag \
  -c "SELECT user_id, field_type, field_value, created_at FROM medical_record_items ORDER BY created_at DESC LIMIT 20;"
```

### Redis memory keys are missing

Short-term memory requires a Session ID. Long-term memory requires both a User ID and a memory-aware agent request.

```bash
docker exec medical_rag_redis redis-cli -p 6383 -n 2 KEYS "user:*"
```

### Containers still use old dependencies

```bash
cd backend
docker compose up --build -d
```

## Production Considerations

Before using this project beyond local development:

- replace the browser-generated User ID with authenticated server-side identity
- add authorization checks to every medical-record and memory operation
- use managed secrets instead of local `.env` files
- use HTTPS for the frontend and API
- restrict CORS to exact deployed origins
- replace development database credentials
- define retention and deletion policies for Redis, Milvus, PostgreSQL, and browser storage
- add audit logging and medical-data compliance controls
- add application-level frontend tests and end-to-end isolation tests
- perform clinical, privacy, and security review
