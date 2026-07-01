import argparse
import asyncio
from collections import Counter
from dataclasses import dataclass
from statistics import mean, median
import time

import httpx


DEFAULT_QUESTIONS = [
    "什么是梯度下降？",
    "什么是过拟合？",
    "什么是欠拟合？",
    "什么是注意力机制？",
    "Transformer 的核心思想是什么？",
    "什么是 RAG？",
    "向量检索和 BM25 有什么区别？",
    "什么是交叉验证？",
    "什么是偏差和方差？",
    "为什么需要 rerank？",
]

MIXED_LOCAL_QUESTIONS = [
    "什么是梯度下降？",
    "什么是过拟合？",
    "Transformer 的核心思想是什么？",
    "什么是 RAG？",
    "向量检索和 BM25 有什么区别？",
]

MIXED_WEB_QUESTIONS = [
    "请调用 search_web 工具搜索网络后回答：OpenAI 官方 API 文档首页目前主要介绍什么？",
    "请调用 search_web 工具搜索网络后回答：Python 官方网站目前推荐的最新稳定版是什么？",
    "请调用 search_web 工具搜索网络后回答：LangChain 官方文档首页目前有哪些主要模块？",
    "请调用 search_web 工具搜索网络后回答：Milvus 官方网站目前如何描述 Milvus？",
    "请调用 search_web 工具搜索网络后回答：FastAPI 官方文档首页目前展示的第一个示例是什么？",
]


ENDPOINT_CONFIG = {
    "local_query": {"scope": "local", "model": "gpt-5.5", "kind": "single"},
    "agent_query": {"scope": "agent", "model": "gpt-5.5", "kind": "single"},
    "batch_local_query": {"scope": "local", "model": "gpt-5.5", "kind": "batch"},
    "batch_agent_query": {"scope": "agent", "model": "gpt-5.5", "kind": "batch"},
}


@dataclass
class RequestResult:
    duration: float
    mode: str
    status_code: int
    route: str = "default"
    endpoint: str = ""


@dataclass
class BatchRequestResult:
    duration: float
    mode: str
    status_code: int
    item_modes: dict[str, int]
    item_count: int
    error_count: int


@dataclass(frozen=True)
class BenchmarkCase:
    query: str
    route: str
    endpoint: str
    scope: str
    model: str


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    rank = (len(ordered) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def build_mixed_cases() -> list[BenchmarkCase]:
    cases = []
    for local_query, web_query in zip(
        MIXED_LOCAL_QUESTIONS,
        MIXED_WEB_QUESTIONS,
    ):
        cases.extend(
            [
                BenchmarkCase(
                    query=local_query,
                    route="local",
                    endpoint="local_query",
                    scope="local",
                    model="gpt-5.5",
                ),
                BenchmarkCase(
                    query=web_query,
                    route="web",
                    endpoint="agent_query",
                    scope="agent",
                    model="gpt-5.5",
                ),
            ]
        )
    return cases


async def clear_cache(
    client: httpx.AsyncClient,
    *,
    question: str,
    scope: str,
    model: str,
) -> None:
    response = await client.delete(
        "/cache",
        params={
            "question": question,
            "scope": scope,
            "model": model,
        },
    )
    response.raise_for_status()


async def run_one_query(
    client: httpx.AsyncClient,
    *,
    endpoint: str,
    query: str,
) -> RequestResult:
    started = time.perf_counter()
    response = await client.post(
        f"/{endpoint}",
        json={"query": query},
    )
    elapsed = time.perf_counter() - started

    response.raise_for_status()
    payload = response.json()

    return RequestResult(
        duration=elapsed,
        mode=payload.get("mode", "unknown"),
        status_code=response.status_code,
        endpoint=endpoint,
    )


async def run_batch_query(
    client: httpx.AsyncClient,
    *,
    endpoint: str,
    queries: list[str],
) -> BatchRequestResult:
    started = time.perf_counter()
    response = await client.post(
        f"/{endpoint}",
        json={"queries": queries},
    )
    elapsed = time.perf_counter() - started

    response.raise_for_status()
    payload = response.json()
    items = payload.get("items", [])
    item_modes = Counter(item.get("mode", "unknown") for item in items)
    error_count = sum(1 for item in items if item.get("error"))

    return BatchRequestResult(
        duration=elapsed,
        mode=payload.get("mode", "unknown"),
        status_code=response.status_code,
        item_modes=dict(item_modes),
        item_count=len(items),
        error_count=error_count,
    )


async def benchmark_round(
    client: httpx.AsyncClient,
    *,
    endpoint: str,
    queries: list[str],
    scope: str,
    model: str,
    mode: str,
) -> tuple[list[RequestResult], float]:
    for query in queries:
        await clear_cache(client, question=query, scope=scope, model=model)

    started = time.perf_counter()
    if mode == "serial":
        results = []
        for query in queries:
            results.append(
                await run_one_query(client, endpoint=endpoint, query=query)
            )
    else:
        results = await asyncio.gather(
            *[
                run_one_query(client, endpoint=endpoint, query=query)
                for query in queries
            ]
        )
    wall = time.perf_counter() - started
    return results, wall


async def benchmark_batch_round(
    client: httpx.AsyncClient,
    *,
    endpoint: str,
    queries: list[str],
    scope: str,
    model: str,
) -> BatchRequestResult:
    for query in queries:
        await clear_cache(client, question=query, scope=scope, model=model)

    return await run_batch_query(client, endpoint=endpoint, queries=queries)


async def benchmark_mixed_round(
    client: httpx.AsyncClient,
    *,
    cases: list[BenchmarkCase],
    mode: str,
) -> tuple[list[RequestResult], float]:
    for case in cases:
        await clear_cache(
            client,
            question=case.query,
            scope=case.scope,
            model=case.model,
        )

    async def run_case(case: BenchmarkCase) -> RequestResult:
        result = await run_one_query(
            client,
            endpoint=case.endpoint,
            query=case.query,
        )
        result.route = case.route
        return result

    started = time.perf_counter()
    if mode == "serial":
        results = []
        for case in cases:
            results.append(await run_case(case))
    else:
        results = await asyncio.gather(*[run_case(case) for case in cases])
    wall = time.perf_counter() - started
    return results, wall


def print_single(label: str, result: RequestResult) -> None:
    print(f"{label}: {result.duration:.3f}s mode={result.mode}")


def print_concurrent(label: str, results: list[RequestResult], wall: float) -> None:
    durations = [item.duration for item in results]
    modes = Counter(item.mode for item in results)

    print(
        f"{label}: "
        f"avg={mean(durations):.3f}s "
        f"p50={median(durations):.3f}s "
        f"p95={percentile(durations, 0.95):.3f}s "
        f"max={max(durations):.3f}s "
        f"wall={wall:.3f}s "
        f"modes={dict(modes)}"
    )


def print_workload(label: str, results: list[RequestResult], wall: float) -> None:
    durations = [item.duration for item in results]
    modes = Counter(item.mode for item in results)

    print(
        f"{label}: "
        f"avg={mean(durations):.3f}s "
        f"p50={median(durations):.3f}s "
        f"p95={percentile(durations, 0.95):.3f}s "
        f"max={max(durations):.3f}s "
        f"wall={wall:.3f}s "
        f"modes={dict(modes)}"
    )


def print_batch(label: str, result: BatchRequestResult) -> None:
    print(
        f"{label}: "
        f"{result.duration:.3f}s "
        f"mode={result.mode} "
        f"items={result.item_count} "
        f"errors={result.error_count} "
        f"item_modes={result.item_modes}"
    )


def print_mixed(results: list[RequestResult], wall: float, mode: str) -> None:
    print_workload(
        f"MIXED {len(results)}-query {mode} uncached",
        results,
        wall,
    )

    for route in ("local", "web"):
        route_results = [item for item in results if item.route == route]
        if not route_results:
            continue
        route_wall = (
            sum(item.duration for item in route_results)
            if mode == "serial"
            else max(item.duration for item in route_results)
        )
        print_workload(
            f"  {route.upper()} {len(route_results)} requests",
            route_results,
            route_wall,
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Week4 query endpoints.")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Running API base URL.",
    )
    parser.add_argument(
        "--endpoint",
        choices=sorted(ENDPOINT_CONFIG),
        default="local_query",
        help="Endpoint to benchmark.",
    )
    parser.add_argument(
        "--counts",
        nargs="+",
        type=int,
        default=[1, 5, 10],
        help="Concurrency sizes to benchmark.",
    )
    parser.add_argument(
        "--mode",
        choices=["serial", "concurrent"],
        default="concurrent",
        help="Run single-query endpoint serially or concurrently. Ignored for batch endpoints.",
    )
    parser.add_argument(
        "--workload",
        choices=["default", "mixed"],
        default="default",
        help=(
            "Use the selected endpoint with default questions, or run a mixed "
            "local-query/web-agent workload."
        ),
    )
    args = parser.parse_args()

    available_questions = (
        len(build_mixed_cases())
        if args.workload == "mixed"
        else len(DEFAULT_QUESTIONS)
    )
    if max(args.counts) > available_questions:
        raise ValueError(
            f"Not enough default questions for max count={max(args.counts)}. "
            f"Maximum supported count is {available_questions}."
        )

    config = ENDPOINT_CONFIG[args.endpoint]

    async with httpx.AsyncClient(base_url=args.base_url, timeout=180.0) as client:
        for count in args.counts:
            if args.workload == "mixed":
                cases = build_mixed_cases()[:count]
                results, wall = await benchmark_mixed_round(
                    client,
                    cases=cases,
                    mode=args.mode,
                )
                print_mixed(results, wall, args.mode)
                continue

            queries = DEFAULT_QUESTIONS[:count]
            if config["kind"] == "single":
                results, wall = await benchmark_round(
                    client,
                    endpoint=args.endpoint,
                    queries=queries,
                    scope=config["scope"],
                    model=config["model"],
                    mode=args.mode,
                )

                if count == 1:
                    print_single("SINGLE uncached", results[0])
                elif args.mode == "serial":
                    print_workload(f"{count}-query serial uncached", results, wall)
                else:
                    print_concurrent(f"{count} concurrent uncached", results, wall)
            else:
                result = await benchmark_batch_round(
                    client,
                    endpoint=args.endpoint,
                    queries=queries,
                    scope=config["scope"],
                    model=config["model"],
                )
                print_batch(f"{count}-query batch uncached", result)


if __name__ == "__main__":
    asyncio.run(main())
