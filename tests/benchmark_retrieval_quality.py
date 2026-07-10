import argparse
import asyncio
import csv
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

from langchain_cohere import CohereRerank  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

from app.rag_chain import (  # noqa: E402
    collect_candidate_docs,
    create_bm25_retriever,
    create_vector_retriever,
    load_documents,
    merge_and_deduplicate,
    rerank_one_query,
    split_documents,
)

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"


TESTSET_DIR = Path(__file__).resolve().parent / "QA-TestSet-LiveQA-Med-Qrels-2479-Answers"
QRELS_PATH = TESTSET_DIR / "All-qrels_LiveQAMed2017-TestQuestions_2479_Judged-Answers.txt"
ANSWERS_PATH = TESTSET_DIR / "All-2479-Answers-retrieved-from-MedQuAD.csv"


@dataclass(frozen=True)
class EvalCase:
    question_id: str
    query: str
    relevant_answer_ids: set[str]
    original_query: str | None = None
    natural_query: str | None = None
    contextual_query: str | None = None


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def parse_relevance(label: str) -> int:
    return int(label.split("-", 1)[0])


def load_qrels(path: Path) -> dict[str, list[tuple[str, int]]]:
    qrels = defaultdict(list)

    with path.open() as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 3:
                continue

            question_id, label, answer_id = parts
            qrels[question_id].append((answer_id, parse_relevance(label)))

    return dict(qrels)


def load_answers(path: Path) -> dict[str, str]:
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        return {
            row["AnswerID"]: row["Answer"]
            for row in reader
            if row.get("AnswerID") and row.get("Answer")
        }


def extract_question(answer_text: str) -> str | None:
    match = re.search(r"Question:\s*(.*?)(?:\nURL:|\nAnswer:)", answer_text, re.DOTALL)
    if not match:
        return None

    return normalize_display_text(match.group(1))


def strip_aliases(question: str) -> str:
    return re.sub(r"\s*\(Also called:.*?\)\s*", " ", question).strip()


def clean_medical_topic(topic: str) -> str:
    topic = strip_aliases(topic)
    topic = topic.strip(" ?.").strip()
    return normalize_display_text(topic)


def make_query_variants(question: str) -> tuple[str, str]:
    question = normalize_display_text(strip_aliases(question))
    patterns = [
        (
            r"^What is \(are\) (.+?)\s*\?$",
            "Can you explain {topic} in simple terms?",
            "Can you explain it in simple terms?",
        ),
        (
            r"^What is (.+?)\s*\?$",
            "Can you explain {topic} in simple terms?",
            "Can you explain it in simple terms?",
        ),
        (
            r"^What are (.+?)\s*\?$",
            "Can you explain {topic} in simple terms?",
            "Can you explain it in simple terms?",
        ),
        (
            r"^What causes (.+?)\s*\?$",
            "Why do people get {topic}?",
            "Why do people get it?",
        ),
        (
            r"^What are the causes of (.+?)\s*\?$",
            "Why do people get {topic}?",
            "Why do people get it?",
        ),
        (
            r"^What are the symptoms of (.+?)\s*\?$",
            "What symptoms should someone watch for with {topic}?",
            "What symptoms should someone watch for?",
        ),
        (
            r"^What are the signs and symptoms of (.+?)\s*\?$",
            "What symptoms should someone watch for with {topic}?",
            "What symptoms should someone watch for?",
        ),
        (
            r"^How many people are affected by (.+?)\s*\?$",
            "How common is {topic}?",
            "How common is it?",
        ),
        (
            r"^Is (.+?) inherited\s*\?$",
            "Can {topic} run in families?",
            "Can this condition run in families?",
        ),
        (
            r"^What are the genetic changes related to (.+?)\s*\?$",
            "Which genes are linked to {topic}?",
            "Which genes are linked to it?",
        ),
        (
            r"^What are the treatments for (.+?)\s*\?$",
            "How do doctors treat {topic}?",
            "How do doctors treat it?",
        ),
        (
            r"^How is (.+?) treated\s*\?$",
            "How do doctors treat {topic}?",
            "How do doctors treat it?",
        ),
        (
            r"^How is (.+?) diagnosed\s*\?$",
            "How would a doctor diagnose {topic}?",
            "How would a doctor diagnose it?",
        ),
        (
            r"^How to diagnose (.+?)\s*\?$",
            "How would a doctor diagnose {topic}?",
            "How would a doctor diagnose it?",
        ),
        (
            r"^Who is at risk for (.+?)\s*\?$",
            "Who is more likely to get {topic}?",
            "Who is more likely to get it?",
        ),
        (
            r"^What are the complications of (.+?)\s*\?$",
            "What problems can {topic} cause?",
            "What problems can it cause?",
        ),
    ]

    for pattern, natural_template, contextual_template in patterns:
        match = re.match(pattern, question, flags=re.IGNORECASE)
        if match:
            topic = clean_medical_topic(match.group(1))
            if topic:
                return (
                    natural_template.format(topic=topic),
                    contextual_template.format(topic=topic),
                )

    return question, question


def naturalize_question(question: str) -> str:
    natural_query, _ = make_query_variants(question)
    return natural_query


def contextualize_question(question: str) -> str:
    _, contextual_query = make_query_variants(question)
    return contextual_query


def normalize_display_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def make_answer_id_from_metadata(metadata: dict) -> str | None:
    source = metadata.get("dataset_source")
    document_id = metadata.get("document_id")
    qid = metadata.get("qid")

    if not source or not document_id or not qid or "-" not in qid:
        return None

    section = qid.rsplit("-", 1)[-1]
    return f"{source}_{document_id}_Sec{section}.txt"


def build_answer_id_lookup(split_docs: list[Document]) -> tuple[dict[str, str], set[str]]:
    text_to_answer_id = {}
    indexed_answer_ids = set()

    for doc in split_docs:
        answer_id = make_answer_id_from_metadata(doc.metadata)
        if not answer_id:
            continue

        text_to_answer_id[normalize_text(doc.page_content)] = answer_id
        indexed_answer_ids.add(answer_id)

    return text_to_answer_id, indexed_answer_ids


def make_eval_cases(
    qrels: dict[str, list[tuple[str, int]]],
    answers: dict[str, str],
    indexed_answer_ids: set[str],
    min_relevance: int,
    limit: int,
    query_mode: str,
    rewrite_contextual: bool,
) -> list[EvalCase]:
    cases = []

    def sort_key(item: tuple[str, list[tuple[str, int]]]) -> int:
        question_id, _ = item
        return int(question_id) if question_id.isdigit() else 10**9

    for question_id, judged_answers in sorted(qrels.items(), key=sort_key):
        relevant = {
            answer_id
            for answer_id, relevance in judged_answers
            if relevance >= min_relevance and answer_id in indexed_answer_ids
        }
        if not relevant:
            continue

        original_query = None
        for answer_id, _ in sorted(
            judged_answers,
            key=lambda item: item[1],
            reverse=True,
        ):
            if answer_id not in relevant:
                continue
            original_query = extract_question(answers.get(answer_id, ""))
            if original_query:
                break

        if not original_query:
            continue

        natural_query = naturalize_question(original_query)
        contextual_query = contextualize_question(original_query)

        if query_mode == "natural":
            query = natural_query
        elif query_mode == "contextual":
            query = natural_query if rewrite_contextual else contextual_query
        else:
            query = original_query

        cases.append(
            EvalCase(
                question_id=question_id,
                query=query,
                relevant_answer_ids=relevant,
                original_query=original_query,
                natural_query=natural_query,
                contextual_query=contextual_query,
            )
        )

        if limit and len(cases) >= limit:
            break

    return cases


def doc_to_answer_id(doc: Document, text_to_answer_id: dict[str, str]) -> str | None:
    answer_id = make_answer_id_from_metadata(doc.metadata)
    if answer_id:
        return answer_id

    return text_to_answer_id.get(normalize_text(doc.page_content))


def unique_answer_ids(docs: Iterable[Document], text_to_answer_id: dict[str, str]) -> list[str]:
    answer_ids = []
    seen = set()

    for doc in docs:
        answer_id = doc_to_answer_id(doc, text_to_answer_id)
        if not answer_id or answer_id in seen:
            continue

        seen.add(answer_id)
        answer_ids.append(answer_id)

    return answer_ids


def evaluate_ranked_ids(
    ranked_answer_ids: list[str],
    relevant_answer_ids: set[str],
    top_k: int,
) -> dict[str, float]:
    top_ids = ranked_answer_ids[:top_k]
    hits = [answer_id for answer_id in top_ids if answer_id in relevant_answer_ids]
    hit_count = len(set(hits))

    reciprocal_rank = 0.0
    for rank, answer_id in enumerate(top_ids, start=1):
        if answer_id in relevant_answer_ids:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "recall": hit_count / len(relevant_answer_ids),
        "precision": hit_count / top_k,
        "hit": 1.0 if hit_count > 0 else 0.0,
        "mrr": reciprocal_rank,
    }


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0

    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * percentile_value
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


async def retrieve_docs(
    strategy: str,
    query: str,
    top_k: int,
    vector_retriever,
    bm25_retriever,
    reranker,
    rerank_sem,
) -> list[Document]:
    if strategy == "bm25":
        return bm25_retriever.invoke(query)[:top_k]

    if strategy == "vector":
        return vector_retriever(query)[:top_k]

    if strategy == "hybrid":
        vector_docs = vector_retriever(query)
        bm25_docs = bm25_retriever.invoke(query)
        for doc in bm25_docs:
            doc.metadata["retriever"] = "bm25"
        return merge_and_deduplicate(vector_docs, bm25_docs)[:top_k]

    if strategy == "hybrid_rerank":
        candidate_docs = collect_candidate_docs(
            query=query,
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
        )
        return await rerank_one_query(query, candidate_docs, reranker, rerank_sem)

    raise ValueError(f"Unknown strategy: {strategy}")


def print_strategy_result(
    strategy: str,
    metrics: list[dict[str, float]],
    latencies: list[float],
    elapsed_seconds: float,
) -> dict[str, float]:
    aggregate = {
        metric_name: sum(item[metric_name] for item in metrics) / len(metrics)
        for metric_name in ["recall", "precision", "hit", "mrr"]
    }
    aggregate["avg_latency_ms"] = (sum(latencies) / len(latencies)) * 1000
    aggregate["p50_latency_ms"] = percentile(latencies, 0.50) * 1000
    aggregate["p95_latency_ms"] = percentile(latencies, 0.95) * 1000
    aggregate["max_latency_ms"] = max(latencies) * 1000

    print(
        f"{strategy}: "
        f"recall@k={aggregate['recall']:.3f} "
        f"precision@k={aggregate['precision']:.3f} "
        f"hit@k={aggregate['hit']:.3f} "
        f"mrr={aggregate['mrr']:.3f} "
        f"avg_latency={aggregate['avg_latency_ms']:.1f}ms "
        f"p50_latency={aggregate['p50_latency_ms']:.1f}ms "
        f"p95_latency={aggregate['p95_latency_ms']:.1f}ms "
        f"max_latency={aggregate['max_latency_ms']:.1f}ms "
        f"elapsed={elapsed_seconds:.2f}s"
    )

    return aggregate


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark retrieval Recall@K and Precision@K using the LiveQA/MedQuAD qrels."
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["bm25"],
        choices=["bm25", "vector", "hybrid", "hybrid_rerank"],
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=None,
        help="Number of candidates to retrieve per route before final Top-K evaluation.",
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat each strategy benchmark to smooth latency measurements.",
    )
    parser.add_argument("--min-relevance", type=int, default=2)
    parser.add_argument(
        "--query-mode",
        choices=["reconstructed", "natural", "contextual"],
        default="reconstructed",
        help="reconstructed uses MedQuAD question text; natural uses standalone user-like wording; contextual uses follow-up wording.",
    )
    parser.add_argument(
        "--rewrite-contextual",
        action="store_true",
        help="When query-mode=contextual, use the standalone natural query for retrieval to simulate query rewriting.",
    )
    parser.add_argument("--show-examples", type=int, default=0)
    parser.add_argument(
        "--compare-to",
        default=None,
        help="Baseline strategy for delta reporting. Defaults to the first strategy.",
    )
    parser.add_argument("--collection-name", default="RAG_collection")
    parser.add_argument("--rerank-model", default="rerank-v3.5")
    parser.add_argument("--rerank-concurrency", type=int, default=1)
    args = parser.parse_args()
    args.candidate_k = args.candidate_k or args.top_k

    if args.candidate_k < args.top_k:
        raise ValueError("--candidate-k must be greater than or equal to --top-k.")

    if args.repeat <= 0:
        raise ValueError("--repeat must be positive.")

    split_docs = split_documents(load_documents())
    text_to_answer_id, indexed_answer_ids = build_answer_id_lookup(split_docs)
    qrels = load_qrels(QRELS_PATH)
    answers = load_answers(ANSWERS_PATH)
    cases = make_eval_cases(
        qrels=qrels,
        answers=answers,
        indexed_answer_ids=indexed_answer_ids,
        min_relevance=args.min_relevance,
        limit=args.limit,
        query_mode=args.query_mode,
        rewrite_contextual=args.rewrite_contextual,
    )

    if not cases:
        raise RuntimeError("No evaluation cases could be built from qrels and indexed documents.")

    print("RETRIEVAL QUALITY BENCHMARK")
    print(f"cases={len(cases)}")
    print(f"top_k={args.top_k}")
    print(f"candidate_k={args.candidate_k}")
    print(f"repeat={args.repeat}")
    print(f"min_relevance={args.min_relevance}")
    effective_query_mode = (
        "contextual_rewritten"
        if args.query_mode == "contextual" and args.rewrite_contextual
        else args.query_mode
    )
    print(f"query_mode={effective_query_mode}")
    print(f"indexed_answer_ids={len(indexed_answer_ids)}")
    print(
        "note=Queries are reconstructed from MedQuAD answer questions because this "
        "testset folder does not include the original LiveQA question text."
    )

    if args.show_examples:
        print("query_examples:")
        for case in cases[: args.show_examples]:
            if args.query_mode == "natural":
                print(f"- qid={case.question_id} original={case.original_query}")
                print(f"  natural={case.query}")
            elif args.query_mode == "contextual":
                print(f"- qid={case.question_id} original={case.original_query}")
                print(f"  contextual={case.contextual_query}")
                print(f"  rewritten={case.natural_query}")
                print(f"  used={case.query}")
            else:
                print(f"- qid={case.question_id} query={case.query}")

    bm25_retriever = None
    vector_retriever = None
    reranker = None
    rerank_sem = asyncio.Semaphore(args.rerank_concurrency)

    if any(strategy in args.strategies for strategy in ["bm25", "hybrid", "hybrid_rerank"]):
        bm25_retriever = create_bm25_retriever(top_k=args.candidate_k)

    if any(strategy in args.strategies for strategy in ["vector", "hybrid", "hybrid_rerank"]):
        vector_retriever = create_vector_retriever(
            collection_name=args.collection_name,
            top_k=args.candidate_k,
        )

    if "hybrid_rerank" in args.strategies:
        reranker = CohereRerank(
            model=args.rerank_model,
            top_n=args.top_k,
        )

    results_by_strategy = {}

    for strategy in args.strategies:
        started_at = time.perf_counter()
        metrics = []
        latencies = []

        for _ in range(args.repeat):
            for case in cases:
                query_started_at = time.perf_counter()
                docs = await retrieve_docs(
                    strategy=strategy,
                    query=case.query,
                    top_k=args.top_k,
                    vector_retriever=vector_retriever,
                    bm25_retriever=bm25_retriever,
                    reranker=reranker,
                    rerank_sem=rerank_sem,
                )
                latencies.append(time.perf_counter() - query_started_at)
                ranked_answer_ids = unique_answer_ids(docs, text_to_answer_id)
                metrics.append(
                    evaluate_ranked_ids(
                        ranked_answer_ids=ranked_answer_ids,
                        relevant_answer_ids=case.relevant_answer_ids,
                        top_k=args.top_k,
                    )
                )

        elapsed_seconds = time.perf_counter() - started_at
        results_by_strategy[strategy] = print_strategy_result(
            strategy=strategy,
            metrics=metrics,
            latencies=latencies,
            elapsed_seconds=elapsed_seconds,
        )

    if len(results_by_strategy) >= 2:
        baseline_name = args.compare_to or args.strategies[0]
        if baseline_name not in results_by_strategy:
            raise ValueError(f"Unknown comparison baseline: {baseline_name}")

        baseline = results_by_strategy[baseline_name]
        for strategy, result in results_by_strategy.items():
            if strategy == baseline_name:
                continue
            recall_delta = (result["recall"] - baseline["recall"]) * 100
            precision_delta = (result["precision"] - baseline["precision"]) * 100
            recall_relative = (
                ((result["recall"] - baseline["recall"]) / baseline["recall"]) * 100
                if baseline["recall"]
                else 0.0
            )
            precision_relative = (
                ((result["precision"] - baseline["precision"]) / baseline["precision"]) * 100
                if baseline["precision"]
                else 0.0
            )
            print(
                f"comparison={strategy}_vs_{baseline_name} "
                f"recall_delta={recall_delta:+.1f}pp "
                f"precision_delta={precision_delta:+.1f}pp "
                f"recall_relative={recall_relative:+.1f}% "
                f"precision_relative={precision_relative:+.1f}%"
            )


if __name__ == "__main__":
    asyncio.run(main())
