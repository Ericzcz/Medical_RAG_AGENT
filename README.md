# Medical RAG Agent

A full-stack medical AI assistant with local RAG, web search, conversation memory, structured medical records, background indexing, and observability.

The web app sends questions to `/agent_query`. The backend decides whether to search the local medical knowledge base, search the web, save or retrieve medical records, or answer directly.

> For general health information only. This project is not a substitute for professional diagnosis or treatment and is not a production clinical system.

## Features

- English browser chat interface
- intent-aware ReAct agent with four specialized skills
- Milvus + BM25 hybrid retrieval with Cohere reranking
- Tavily web search for recent information
- Redis short-term and long-term memory
- Milvus semantic long-term-memory recall
- PostgreSQL medical records isolated by User ID
- recent conversations stored in the current browser
- Celery background knowledge indexing
- Prometheus, Grafana, Loki, and Promtail observability stack

## Architecture

```text
Browser :3001
   |
   | POST /agent_query
   v
FastAPI :8000
   |
   |-- Intent router
   |-- ReAct agent
   |     |-- Local medical RAG
   |     |-- Web search
   |     |-- Medical-record insert
   |     `-- Medical-record query
   |
   |-- Redis       session memory, long-term memory, cache
   |-- Milvus      knowledge vectors, semantic memory
   `-- PostgreSQL  medical records

Knowledge files -> Celery worker -> Milvus
Metrics -> Prometheus -> Grafana
Logs -> Promtail -> Loki
```

## Quick Start

### Requirements

- Docker Desktop with Docker Compose
- Node.js `>= 22.13.0`
- npm

### 1. Configure credentials

Create `.env` in the project root:

```dotenv
OPENAI_API_KEY=your_openai_key
COHERE_API_KEY=your_cohere_key
TAVILY_API_KEY=your_tavily_key

# Optional
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=medical-rag-agent
CORS_ORIGINS=
```

- OpenAI is used for the agent, embeddings, routing, and memory.
- Cohere is used for local-result reranking.
- Tavily is used only for web search.

### 2. Start the backend

```bash
cd backend
docker compose up --build -d
docker compose ps
```

Check the API:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status":"ok"}
```

`/health` is a basic liveness check and does not verify every dependency.

### 3. Build the knowledge index

Place `.xml`, `.pdf`, or `.md` files in `data_base/knowledge_db`, then submit the indexing task:

```bash
curl -X POST "http://localhost:8000/index?force_rebuild=false"
```

Poll the returned task ID:

```bash
curl http://localhost:8000/tasks/<task-id>
```

Indexing creates embeddings and may take time or incur API usage for large datasets.

### 4. Start the frontend

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3001`.

The frontend uses `http://localhost:8000` by default. To use another API, create `frontend/.env.local`:

```dotenv
NEXT_PUBLIC_API_BASE_URL=https://api.example.com
```

For a deployed frontend, add its origin to the root `.env` and restart the API:

```dotenv
CORS_ORIGINS=https://app.example.com
```

### Stop the backend

```bash
cd backend
docker compose down
```

Named data volumes are preserved.

## Using the Web App

1. Enter a health question and press Enter.
2. Use Shift+Enter to add a new line.
3. Use **New conversation** to start a separate session.
4. Use **Recents** to reopen or delete a conversation.
5. Click `•••` to view the current User ID and Session ID.

Recent conversations are saved only in the current browser profile. Clearing site data removes them.

## Data and Isolation

| Data | Isolation | Storage | Retention |
| --- | --- | --- | --- |
| Visible recent conversations | Browser profile | Browser storage | Until site data is cleared |
| Short-term context | Session ID | Redis | 24 hours |
| Long-term preferences/context | User ID | Redis + Milvus | Persistent volumes |
| Medical records | User ID | PostgreSQL | Persistent volume |
| Context-free answer cache | Query + scope + model | Redis | 1 hour |

The browser-generated User ID is an anonymous identifier, not authentication. For isolation testing, use separate browser profiles or an incognito window. Do not store real patient information without adding authentication, authorization, and appropriate compliance controls.

## API

Interactive documentation: `http://localhost:8000/docs`

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Basic API liveness |
| `POST` | `/agent_query` | Main agent endpoint with optional user/session memory |
| `POST` | `/local_query` | Direct local RAG query |
| `POST` | `/batch_agent_query` | Batch agent queries |
| `POST` | `/batch_local_query` | Batch local RAG queries |
| `DELETE` | `/cache` | Delete a cached answer |
| `POST` | `/index` | Submit an indexing task |
| `GET` | `/tasks/{task_id}` | Read indexing progress/result |
| `GET` | `/metrics` | Prometheus metrics |

Example:

```bash
curl -X POST http://localhost:8000/agent_query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "What are the common side effects of metformin?",
    "session_id": "session-demo-1",
    "user_id": "user-demo-1"
  }'
```

## Services and Ports

| Service | Port |
| --- | --- |
| Frontend | `3001` |
| FastAPI | `8000` |
| Grafana | `3000` |
| Redis | `6383` |
| PostgreSQL | `5433` |
| Milvus | `19530` |
| Milvus metrics | `9091` |
| Prometheus | `9090` |
| Loki | `3100` |
| MinIO console | `9001` |

Grafana uses the local development credentials `admin` / `admin`. Data sources and dashboards are not provisioned automatically.

## Recorded Local Results

Historical local measurements; rerun after changing the model, data, dependencies, or hardware.

| Area | Result |
| --- | --- |
| Knowledge-base scale | 11,274 XML files, 16,407 QA documents, 72,877 chunks |
| Memory compression | Prompt-memory tokens reduced by 61.9%, from 737 to 281 |
| Retrieval | Recall@5 improved by 9.4 points; Hit@5 improved from 84% to 100% |
| Query cache | Repeated local-query latency decreased from 45.875s to 0.008s |
| Concurrency | Ten concurrent requests produced about 4.5x single-request throughput |

## Verification

Frontend:

```bash
cd frontend
npm run lint
npm run build
```

Backend syntax and intent-router smoke check:

```bash
python3 -m pip install -r requirements.txt
python3 -m compileall backend/app
PYTHONPATH=backend python3 tests/test_intent_router.py
```

The intent-router check calls the configured model and requires API access.

## Project Structure

```text
medical_rag_agent/
  frontend/                 browser application
  backend/
    app/
      api/routes/           FastAPI routes
      services/             RAG, web, records, intent routing
      skills/               agent skills
      Redis_Celery/         cache and indexing worker
      main.py               app setup and CORS
      agent.py              agent loop
      rag_chain.py          hybrid retrieval and indexing
      short_term_memory.py  session memory
      long_term_memory.py   long-term memory
    compose.yaml            backend and observability services
  data_base/knowledge_db/   local medical documents
  tests/                    benchmarks and smoke checks
  requirements.txt
  README.md
```

## Common Issues

- **Frontend says `Service offline`:** check `curl http://localhost:8000/health` and `docker logs medical_rag_api`.
- **Local RAG fails:** confirm documents were added and `/index` completed successfully.
- **Browser blocks requests:** add the exact frontend origin to `CORS_ORIGINS`.
- **Recent conversations disappear:** browser data was cleared, private browsing ended, or browser storage failed.
- **Test users share context:** use separate browser profiles and do not reuse the same Session ID.
