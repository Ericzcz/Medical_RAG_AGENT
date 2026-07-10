import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.rag_chain import (  # noqa: E402
    get_knowledge_base_dir,
    get_milvus_client,
    load_documents,
    split_documents,
)


def count_files(root: Path) -> Counter:
    counts = Counter()
    for path in root.rglob("*"):
        if path.is_file():
            counts[path.suffix.lower() or "<no_suffix>"] += 1
    return counts


def get_milvus_entity_count(collection_name: str) -> int | None:
    try:
        client = get_milvus_client()
        if not client.has_collection(collection_name=collection_name):
            return None
        stats = client.get_collection_stats(collection_name=collection_name)
    except Exception:
        return None

    row_count = stats.get("row_count")
    if row_count is None:
        return None

    return int(row_count)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report Medical RAG knowledge base and Milvus indexing scale."
    )
    parser.add_argument("--collection-name", default="RAG_collection")
    parser.add_argument(
        "--milvus-uri",
        default=None,
        help="Optional Milvus URI override, e.g. http://localhost:19530.",
    )
    args = parser.parse_args()

    if args.milvus_uri:
        os.environ["MILVUS_URI"] = args.milvus_uri

    knowledge_base_dir = get_knowledge_base_dir()
    file_counts = count_files(knowledge_base_dir)

    started_at = time.perf_counter()
    documents = load_documents()
    load_seconds = time.perf_counter() - started_at

    split_started_at = time.perf_counter()
    chunks = split_documents(documents)
    split_seconds = time.perf_counter() - split_started_at

    source_counts = Counter(
        doc.metadata.get("dataset_source") or "unknown"
        for doc in documents
    )
    qtype_counts = Counter(
        doc.metadata.get("qtype") or "unknown"
        for doc in documents
    )

    chunk_lengths = [len(chunk.page_content) for chunk in chunks]
    avg_chunk_chars = (
        sum(chunk_lengths) / len(chunk_lengths)
        if chunk_lengths
        else 0.0
    )
    max_chunk_chars = max(chunk_lengths) if chunk_lengths else 0

    milvus_entities = get_milvus_entity_count(args.collection_name)

    print("KNOWLEDGE BASE SCALE")
    print(f"knowledge_base_dir={knowledge_base_dir}")
    print(f"xml_files={file_counts.get('.xml', 0)}")
    print(f"pdf_files={file_counts.get('.pdf', 0)}")
    print(f"markdown_files={file_counts.get('.md', 0)}")
    print(f"all_files={sum(file_counts.values())}")
    print(f"qa_documents={len(documents)}")
    print(f"chunks={len(chunks)}")
    print(f"avg_chunk_chars={avg_chunk_chars:.1f}")
    print(f"max_chunk_chars={max_chunk_chars}")
    print(f"load_documents_time={load_seconds:.2f}s")
    print(f"split_documents_time={split_seconds:.2f}s")
    print(f"milvus_collection={args.collection_name}")
    print(
        "milvus_entities="
        f"{milvus_entities if milvus_entities is not None else 'unavailable'}"
    )
    print(f"dataset_sources={dict(source_counts)}")
    print(f"top_qtypes={dict(qtype_counts.most_common(10))}")

    if milvus_entities is not None:
        print(
            "resume_metric="
            f"Ingested {file_counts.get('.xml', 0):,} XML medical files into "
            f"{len(documents):,} QA documents and {milvus_entities:,} Milvus "
            f"vector records for semantic retrieval."
        )
    else:
        print(
            "resume_metric="
            f"Parsed {file_counts.get('.xml', 0):,} XML medical files into "
            f"{len(documents):,} QA documents and {len(chunks):,} retrieval chunks."
        )


if __name__ == "__main__":
    main()
