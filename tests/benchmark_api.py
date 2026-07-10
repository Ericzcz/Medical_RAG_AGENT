import argparse
import asyncio
from collections import Counter
from dataclasses import dataclass
from statistics import mean, median
import time

import httpx


ML_QUESTIONS = [
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

MEDICAL_QUESTIONS = [
    "What is Aarskog-Scott syndrome?",
    "How is Aarskog-Scott syndrome inherited?",
    "What are the symptoms of Noonan syndrome?",
    "How is Noonan syndrome treated?",
    "What causes celiac disease?",
    "What are the symptoms of celiac disease?",
    "What is Addison disease?",
    "How is Addison disease diagnosed?",
    "What is autoimmune hepatitis?",
    "What are the treatments for autoimmune hepatitis?",
]

QUESTION_SETS = {
    "medical": MEDICAL_QUESTIONS,
    "ml": ML_QUESTIONS,
}

MIXED_LOCAL_QUESTIONS = [
    "What is Aarskog-Scott syndrome?",
    "How is Aarskog-Scott syndrome inherited?",
    "What are the symptoms of Noonan syndrome?",
    "What causes celiac disease?",
    "What is Addison disease?",
]

MIXED_WEB_QUESTIONS = [
    "Use the search_web tool to find the current CDC overview page for celiac disease and summarize it.",
    "Use the search_web tool to find current NIH information about Addison disease and summarize it.",
    "Use the search_web tool to find current information about autoimmune hepatitis from a reliable medical source.",
    "Use the search_web tool to find current MedlinePlus information about Noonan syndrome and summarize it.",
    "Use the search_web tool to find current information about Aarskog-Scott syndrome from a reliable medical source.",
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


@dataclass
class WorkloadStats:
    count: int
    avg: float
    p50: float
    p95: float
    max_duration: float
    wall: float
    throughput: float
    modes: dict[str, int]


def make_user_session(
    *,
    base_session_id: str | None,
    base_user_id: str | None,
    index: int,
    multi_user: bool,
) -> tuple[str | None, str | None]:
    if not multi_user:
        return base_session_id, base_user_id

    session_prefix = base_session_id or "bench_session"
    user_prefix = base_user_id or "bench_user"
    return (
        f"{session_prefix}_{index + 1:03d}",
        f"{user_prefix}_{index + 1:03d}",
    )


def get_questions(question_set: str) -> list[str]:
    return QUESTION_SETS[question_set]


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


def summarize_results(results: list[RequestResult], wall: float) -> WorkloadStats:
    durations = [item.duration for item in results]
    count = len(results)
    return WorkloadStats(
        count=count,
        avg=mean(durations) if durations else 0.0,
        p50=median(durations) if durations else 0.0,
        p95=percentile(durations, 0.95),
        max_duration=max(durations) if durations else 0.0,
        wall=wall,
        throughput=(count / wall) if wall > 0 else 0.0,
        modes=dict(Counter(item.mode for item in results)),
    )


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
    session_id: str | None = None,
    user_id: str | None = None,
) -> RequestResult:
    payload = {"query": query}
    if session_id:
        payload["session_id"] = session_id
    if user_id:
        payload["user_id"] = user_id

    started = time.perf_counter()
    response = await client.post(
        f"/{endpoint}",
        json=payload,
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
    cache_mode: str,
    session_id: str | None = None,
    user_id: str | None = None,
    multi_user: bool = False,
) -> tuple[list[RequestResult], float]:
    if cache_mode == "uncached":
        for query in queries:
            await clear_cache(client, question=query, scope=scope, model=model)
    elif cache_mode == "cached":
        for idx, query in enumerate(queries):
            item_session_id, item_user_id = make_user_session(
                base_session_id=session_id,
                base_user_id=user_id,
                index=idx,
                multi_user=multi_user,
            )
            await run_one_query(
                client,
                endpoint=endpoint,
                query=query,
                session_id=item_session_id,
                user_id=item_user_id,
            )

    started = time.perf_counter()
    if mode == "serial":
        results = []
        for idx, query in enumerate(queries):
            item_session_id, item_user_id = make_user_session(
                base_session_id=session_id,
                base_user_id=user_id,
                index=idx,
                multi_user=multi_user,
            )
            results.append(
                await run_one_query(
                    client,
                    endpoint=endpoint,
                    query=query,
                    session_id=item_session_id,
                    user_id=item_user_id,
                )
            )
    else:
        request_specs = [
            (
                query,
                *make_user_session(
                    base_session_id=session_id,
                    base_user_id=user_id,
                    index=idx,
                    multi_user=multi_user,
                ),
            )
            for idx, query in enumerate(queries)
        ]
        results = await asyncio.gather(
            *[
                run_one_query(
                    client,
                    endpoint=endpoint,
                    query=query,
                    session_id=item_session_id,
                    user_id=item_user_id,
                )
                for query, item_session_id, item_user_id in request_specs
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
    cache_mode: str,
) -> BatchRequestResult:
    if cache_mode == "uncached":
        for query in queries:
            await clear_cache(client, question=query, scope=scope, model=model)
    elif cache_mode == "cached":
        await run_batch_query(client, endpoint=endpoint, queries=queries)

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
    stats = summarize_results(results, wall)

    print(
        f"{label}: "
        f"avg={stats.avg:.3f}s "
        f"p50={stats.p50:.3f}s "
        f"p95={stats.p95:.3f}s "
        f"max={stats.max_duration:.3f}s "
        f"wall={stats.wall:.3f}s "
        f"throughput={stats.throughput:.2f} req/s "
        f"modes={stats.modes}"
    )


def print_workload(label: str, results: list[RequestResult], wall: float) -> None:
    stats = summarize_results(results, wall)

    print(
        f"{label}: "
        f"avg={stats.avg:.3f}s "
        f"p50={stats.p50:.3f}s "
        f"p95={stats.p95:.3f}s "
        f"max={stats.max_duration:.3f}s "
        f"wall={stats.wall:.3f}s "
        f"throughput={stats.throughput:.2f} req/s "
        f"modes={stats.modes}"
    )


def print_batch(label: str, result: BatchRequestResult) -> None:
    throughput = result.item_count / result.duration if result.duration > 0 else 0.0
    print(
        f"{label}: "
        f"{result.duration:.3f}s "
        f"throughput={throughput:.2f} req/s "
        f"mode={result.mode} "
        f"items={result.item_count} "
        f"errors={result.error_count} "
        f"item_modes={result.item_modes}"
    )


def print_cache_comparison(
    count: int,
    uncached_results: list[RequestResult],
    uncached_wall: float,
    cached_results: list[RequestResult],
    cached_wall: float,
) -> None:
    uncached = summarize_results(uncached_results, uncached_wall)
    cached = summarize_results(cached_results, cached_wall)

    avg_reduction = (
        ((uncached.avg - cached.avg) / uncached.avg) * 100
        if uncached.avg > 0
        else 0.0
    )
    p95_reduction = (
        ((uncached.p95 - cached.p95) / uncached.p95) * 100
        if uncached.p95 > 0
        else 0.0
    )

    print(
        f"CACHE COMPARISON {count} queries: "
        f"avg {uncached.avg:.3f}s -> {cached.avg:.3f}s "
        f"({avg_reduction:.1f}% faster), "
        f"p95 {uncached.p95:.3f}s -> {cached.p95:.3f}s "
        f"({p95_reduction:.1f}% faster), "
        f"throughput {uncached.throughput:.2f} -> {cached.throughput:.2f} req/s"
    )


def print_batch_comparison(
    count: int,
    serial_results: list[RequestResult],
    serial_wall: float,
    batch_result: BatchRequestResult,
) -> None:
    serial = summarize_results(serial_results, serial_wall)
    batch_throughput = (
        batch_result.item_count / batch_result.duration
        if batch_result.duration > 0
        else 0.0
    )
    wall_reduction = (
        ((serial.wall - batch_result.duration) / serial.wall) * 100
        if serial.wall > 0
        else 0.0
    )
    throughput_gain = (
        ((batch_throughput - serial.throughput) / serial.throughput) * 100
        if serial.throughput > 0
        else 0.0
    )

    print(
        f"BATCH COMPARISON {count} queries: "
        f"wall {serial.wall:.3f}s -> {batch_result.duration:.3f}s "
        f"({wall_reduction:.1f}% faster), "
        f"throughput {serial.throughput:.2f} -> {batch_throughput:.2f} req/s "
        f"({throughput_gain:.1f}% higher), "
        f"batch_errors={batch_result.error_count}"
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
    parser = argparse.ArgumentParser(description="Benchmark Medical RAG Agent endpoints.")
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
        choices=["single", "mixed"],
        default="single",
        help=(
            "Use the selected endpoint with one question set, or run a mixed "
            "local-query/web-agent workload."
        ),
    )
    parser.add_argument(
        "--question-set",
        choices=sorted(QUESTION_SETS),
        default="medical",
        help="Question set to use for single-endpoint benchmarks.",
    )
    parser.add_argument(
        "--cache-mode",
        choices=["uncached", "cached", "both"],
        default="uncached",
        help=(
            "uncached clears cache before measuring; cached warms cache before "
            "measuring; both measures uncached and cached for comparison."
        ),
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Optional session_id to include in single-query endpoint payloads.",
    )
    parser.add_argument(
        "--user-id",
        default=None,
        help="Optional user_id to include in single-query endpoint payloads.",
    )
    parser.add_argument(
        "--compare-batch",
        action="store_true",
        help=(
            "Compare serial /local_query requests against /batch_local_query. "
            "Use with --endpoint batch_local_query."
        ),
    )
    parser.add_argument(
        "--multi-user",
        action="store_true",
        help=(
            "Generate a different session_id/user_id per request to simulate "
            "memory-aware multi-user concurrency."
        ),
    )
    args = parser.parse_args()

    available_questions = (
        len(build_mixed_cases())
        if args.workload == "mixed"
        else len(get_questions(args.question_set))
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

            queries = get_questions(args.question_set)[:count]
            if args.compare_batch:
                if args.endpoint != "batch_local_query":
                    raise ValueError("--compare-batch currently supports batch_local_query only.")

                serial_results, serial_wall = await benchmark_round(
                    client,
                    endpoint="local_query",
                    queries=queries,
                    scope=config["scope"],
                    model=config["model"],
                    mode="serial",
                    cache_mode=args.cache_mode,
                )
                batch_result = await benchmark_batch_round(
                    client,
                    endpoint=args.endpoint,
                    queries=queries,
                    scope=config["scope"],
                    model=config["model"],
                    cache_mode=args.cache_mode,
                )
                print_workload(
                    f"{count}-query serial local_query {args.cache_mode}",
                    serial_results,
                    serial_wall,
                )
                print_batch(
                    f"{count}-query batch_local_query {args.cache_mode}",
                    batch_result,
                )
                print_batch_comparison(
                    count=count,
                    serial_results=serial_results,
                    serial_wall=serial_wall,
                    batch_result=batch_result,
                )
                continue

            if config["kind"] == "single":
                if args.cache_mode == "both":
                    uncached_results, uncached_wall = await benchmark_round(
                        client,
                        endpoint=args.endpoint,
                        queries=queries,
                        scope=config["scope"],
                        model=config["model"],
                        mode=args.mode,
                        cache_mode="uncached",
                        session_id=args.session_id,
                        user_id=args.user_id,
                        multi_user=args.multi_user,
                    )
                    cached_results, cached_wall = await benchmark_round(
                        client,
                        endpoint=args.endpoint,
                        queries=queries,
                        scope=config["scope"],
                        model=config["model"],
                        mode=args.mode,
                        cache_mode="cached",
                        session_id=args.session_id,
                        user_id=args.user_id,
                        multi_user=args.multi_user,
                    )
                    label_suffix = " multi-user" if args.multi_user else ""
                    print_workload(
                        f"{count}-query {args.mode}{label_suffix} uncached",
                        uncached_results,
                        uncached_wall,
                    )
                    print_workload(
                        f"{count}-query {args.mode}{label_suffix} cached",
                        cached_results,
                        cached_wall,
                    )
                    print_cache_comparison(
                        count,
                        uncached_results,
                        uncached_wall,
                        cached_results,
                        cached_wall,
                    )
                    continue

                results, wall = await benchmark_round(
                        client,
                        endpoint=args.endpoint,
                        queries=queries,
                        scope=config["scope"],
                        model=config["model"],
                        mode=args.mode,
                        cache_mode=args.cache_mode,
                        session_id=args.session_id,
                        user_id=args.user_id,
                        multi_user=args.multi_user,
                    )

                if count == 1:
                    label_suffix = " multi-user" if args.multi_user else ""
                    print_single(f"SINGLE{label_suffix} {args.cache_mode}", results[0])
                elif args.mode == "serial":
                    label_suffix = " multi-user" if args.multi_user else ""
                    print_workload(
                        f"{count}-query serial{label_suffix} {args.cache_mode}",
                        results,
                        wall,
                    )
                else:
                    label_suffix = " multi-user" if args.multi_user else ""
                    print_concurrent(
                        f"{count} concurrent{label_suffix} {args.cache_mode}",
                        results,
                        wall,
                    )
            else:
                if args.cache_mode == "both":
                    uncached_result = await benchmark_batch_round(
                        client,
                        endpoint=args.endpoint,
                        queries=queries,
                        scope=config["scope"],
                        model=config["model"],
                        cache_mode="uncached",
                    )
                    cached_result = await benchmark_batch_round(
                        client,
                        endpoint=args.endpoint,
                        queries=queries,
                        scope=config["scope"],
                        model=config["model"],
                        cache_mode="cached",
                    )
                    print_batch(f"{count}-query batch uncached", uncached_result)
                    print_batch(f"{count}-query batch cached", cached_result)
                    continue

                result = await benchmark_batch_round(
                    client,
                    endpoint=args.endpoint,
                    queries=queries,
                    scope=config["scope"],
                    model=config["model"],
                    cache_mode=args.cache_mode,
                )
                print_batch(f"{count}-query batch {args.cache_mode}", result)


if __name__ == "__main__":
    asyncio.run(main())
