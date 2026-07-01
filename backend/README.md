# Week5 Agent API

Week5 在 Week4 的 FastAPI + Redis + Celery + Milvus 基础上加入了可观测性：

- Prometheus 采集 FastAPI 和 Milvus 指标
- Grafana 可视化指标
- JSON 日志写入 `logs/app.log`
- Promtail 采集日志并推送到 Loki
- Grafana 查询 Loki 日志

整体链路：

```text
FastAPI API
  ├─ /metrics -> Prometheus -> Grafana
  └─ logs/app.log -> Promtail -> Loki -> Grafana
```

## 服务组成

| 服务 | 作用 | 本机地址 |
| --- | --- | --- |
| `api` | FastAPI 接口服务 | http://127.0.0.1:8000 |
| `worker` | Celery 后台任务 | compose 内部服务 |
| `redis` | 缓存 + Celery broker/backend | `127.0.0.1:6383` |
| `milvus` | 向量数据库 | `127.0.0.1:19530` |
| `minio` | Milvus 对象存储 | http://127.0.0.1:9001 |
| `prometheus` | 指标采集 | http://127.0.0.1:9090 |
| `grafana` | 指标和日志可视化 | http://127.0.0.1:3000 |
| `loki` | 日志存储 | http://127.0.0.1:3100 |
| `promtail` | 日志采集 | compose 内部服务 |

Grafana 默认账号：

```text
admin / admin
```

## 启动

先确认 `Project/.env` 已配置 OpenAI、Tavily、LangSmith 等项目需要的环境变量。

```bash
cd /Users/eric_zcz/Desktop/Eric_Project/agent/Project/week5
docker compose up --build
```

后台启动：

```bash
docker compose up --build -d
```

查看服务状态：

```bash
docker compose ps
```

查看 API 日志：

```bash
docker compose logs -f api
```

停止服务：

```bash
docker compose down
```

## API 地址

常用页面：

- API root: http://127.0.0.1:8000/
- Swagger docs: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health
- Metrics: http://127.0.0.1:8000/metrics

接口列表：

| Method | Path | 说明 |
| --- | --- | --- |
| `GET` | `/health` | 健康检查 |
| `POST` | `/local_query` | 本地知识库问答 |
| `POST` | `/batch_local_query` | 批量本地知识库问答 |
| `POST` | `/agent_query` | Agent 问答 |
| `POST` | `/batch_agent_query` | 批量 Agent 问答 |
| `DELETE` | `/cache` | 删除指定缓存 |
| `POST` | `/index` | 提交 Milvus 索引构建任务 |
| `GET` | `/tasks/{task_id}` | 查询 Celery 任务状态 |

示例：

```bash
curl http://127.0.0.1:8000/health
```

```bash
curl -X POST http://127.0.0.1:8000/local_query \
  -H "Content-Type: application/json" \
  -d '{"query":"什么是 RAG？"}'
```

```bash
curl -X POST http://127.0.0.1:8000/batch_local_query \
  -H "Content-Type: application/json" \
  -d '{"queries":["什么是 RAG？","什么是梯度下降？"]}'
```

```bash
curl -X POST http://127.0.0.1:8000/index
```

## Prometheus 指标

FastAPI 使用 `prometheus-fastapi-instrumentator` 自动暴露 HTTP 指标，不需要在每个接口手动 `.inc()`。

本机验证：

```bash
curl http://127.0.0.1:8000/metrics
```

Prometheus 配置文件是 `prometheus.yml`，当前采集：

- `week5-api`: `api:80/metrics`
- `milvus`: `milvus:9091`

打开 Prometheus target 页面：

```text
http://127.0.0.1:9090/targets
```

`week5-api` 和 `milvus` 应该是 `UP`。

常用 PromQL：

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
  sum by (le) (
    rate(http_request_duration_highr_seconds_bucket[5m])
  )
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

注意：Grafana 的 Explore 里如果使用完整 PromQL，建议切到 `Code` 模式，不要把 `sum(rate(...))` 填进 Builder 的 `Metric` 输入框。

## Grafana 指标可视化

打开：

```text
http://127.0.0.1:3000
```

添加 Prometheus data source：

```text
http://prometheus:9090
```

推荐先做几个面板：

- API 请求速率：`sum by (handler) (rate(http_requests_total[5m]))`
- API 状态码：`sum by (status) (rate(http_requests_total[5m]))`
- API p95 延迟：使用 `histogram_quantile(0.95, ...)`
- 服务存活：`up`

如果图表不更新，先检查右上角时间范围是否是 `Last 5 minutes` 或 `Last 15 minutes`，再确认 Prometheus target 是 `UP`。

## 日志系统

日志配置在：

```text
app/logging_config.py
```

日志会同时输出到：

- 容器 stdout
- `Project/week5/logs/app.log`

当前关键日志点：

- `local_query` / `agent_query`: cache hit、cache miss、failed
- `batch_local_query` / `batch_agent_query`: total、cache_hits、cache_misses、failed
- `DELETE /cache`: cache delete requested
- `POST /index`: index task submitted / submit failed
- Celery `index_document`: started、completed、failed

日志不会记录完整 query，只记录 `query_length`、`scope`、`model` 等字段，避免把用户输入直接写进日志。

本机查看日志文件：

```bash
tail -f logs/app.log
```

## Loki + Promtail

Promtail 配置文件：

```text
promtail-config.yml
```

它会读取：

```text
Project/week5/logs/*.log
```

在 Promtail 容器内对应：

```text
/var/log/week5/*.log
```

然后推送到 Loki：

```text
http://loki:3100/loki/api/v1/push
```

Grafana 添加 Loki data source：

```text
http://loki:3100
```

常用 LogQL：

```logql
{job="week5"}
```

```logql
{job="week5"} | json
```

```logql
{job="week5"} | json | levelname="ERROR"
```

```logql
{job="week5"} | json | message =~ ".*cache.*"
```

## Benchmark

运行单接口 benchmark：

```bash
cd /Users/eric_zcz/Desktop/Eric_Project/agent/Project/week5
python3 tests/benchmark_api.py \
  --base-url http://127.0.0.1:8000 \
  --endpoint local_query \
  --counts 1 5 10 \
  --mode serial
```

并发 benchmark：

```bash
python3 tests/benchmark_api.py \
  --base-url http://127.0.0.1:8000 \
  --endpoint agent_query \
  --counts 1 5 10 \
  --mode concurrent
```

批量接口 benchmark：

```bash
python3 tests/benchmark_api.py \
  --base-url http://127.0.0.1:8000 \
  --endpoint batch_local_query \
  --counts 1 5 10
```

混合 workload：

```bash
python3 tests/benchmark_api.py \
  --base-url http://127.0.0.1:8000 \
  --workload mixed \
  --counts 10 \
  --mode concurrent
```

## 常见排错

Prometheus 查不到 API 指标：

1. 打开 `http://127.0.0.1:9090/targets`
2. 确认 `week5-api` 是 `UP`
3. Docker Compose 内部 target 应该是 `api:80`，不是 `localhost:8000`
4. 访问几次业务接口后再查 `http_requests_total`

Grafana 里 PromQL 查不到数据：

1. 切到 `Code` 模式输入完整 PromQL
2. 时间范围改成 `Last 5 minutes` 或 `Last 15 minutes`
3. 先查 `up`，再查 `http_requests_total`
4. `rate(...[1m])` 请求太少时可能没有明显变化，可以先用 `rate(...[5m])`

Loki 查不到日志：

1. 确认 `logs/app.log` 存在且有内容
2. 查看 Promtail 日志：`docker compose logs -f promtail`
3. Grafana Loki data source 使用 `http://loki:3100`
4. 先用 `{job="week5"}` 查询，不要一开始就加复杂过滤

容器改了 requirements 后依赖没生效：

```bash
docker compose up --build
```
