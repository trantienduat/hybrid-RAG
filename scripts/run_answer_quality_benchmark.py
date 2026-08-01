#!/usr/bin/env python3
"""Run an independent Hybrid-versus-vector answer-quality benchmark."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hybrid_rag.constants import DEFAULT_EMBED_MODEL, DEFAULT_LLM_MODEL
from hybrid_rag.eval.answer_quality import (
    AnswerQualityRunner,
    GoldAnswerDataset,
    _write_json_atomic,
    load_gold_dataset,
)
from hybrid_rag.eval.preflight import require_same_index_run, validate_index_provenance
from hybrid_rag.graph.falkordb_store import FalkorDBStore
from hybrid_rag.ingestion.ollama_embedder import OllamaEmbedder
from hybrid_rag.retrieval.hybrid_retriever import HybridRetriever
from hybrid_rag.vector.qdrant_store import QdrantStore

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "eval/gold/llama_index_core_answer_quality_v1.json"
DEFAULT_OUTPUT = ROOT / "results/answer_quality_benchmark.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _limited_dataset(dataset: GoldAnswerDataset, max_cases: int) -> GoldAnswerDataset:
    if max_cases <= 0:
        return dataset
    return GoldAnswerDataset(
        name=dataset.name,
        schema_version=dataset.schema_version,
        source_identity=dataset.source_identity,
        review_status=dataset.review_status,
        reviewer=dataset.reviewer,
        reviewer_type=dataset.reviewer_type,
        reviewed_at=dataset.reviewed_at,
        cases=dataset.cases[:max_cases],
        source_package=dataset.source_package,
        source_version=dataset.source_version,
        source_artifact_sha256=dataset.source_artifact_sha256,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Hybrid and vector-only generated-answer quality."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Default: <output>.checkpoint.json",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a strictly matching checkpoint",
    )
    parser.add_argument("--allow-draft", action="store_true")
    parser.add_argument("--max-cases", type=int, default=0, help="0 runs every case")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--context-n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repo", default="llama-core-repair")
    parser.add_argument("--graph-name", default="llama_core_repair_20260726")
    parser.add_argument("--qdrant-collection", default="llama_core_repair_20260726")
    parser.add_argument("--falkordb-host", default=os.getenv("FALKORDB_HOST", "localhost"))
    parser.add_argument(
        "--falkordb-port", type=int, default=int(os.getenv("FALKORDB_PORT", "6379"))
    )
    parser.add_argument("--qdrant-host", default=os.getenv("QDRANT_HOST", "localhost"))
    parser.add_argument("--qdrant-port", type=int, default=int(os.getenv("QDRANT_PORT", "6333")))
    parser.add_argument(
        "--ollama-url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    parser.add_argument("--answer-model", default=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL))
    parser.add_argument("--judge-model", default=os.getenv("RAGAS_JUDGE_MODEL", DEFAULT_LLM_MODEL))
    parser.add_argument("--embedding-model", default=os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL))
    parser.add_argument("--input-cost-per-million", type=float)
    parser.add_argument("--output-cost-per-million", type=float)
    args = parser.parse_args()
    if args.max_cases < 0:
        parser.error("--max-cases cannot be negative")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if args.context_n < 1 or args.context_n > args.top_k:
        parser.error("--context-n must be between 1 and --top-k")
    checkpoint_path = args.checkpoint or Path(f"{args.output}.checkpoint.json")

    dataset = _limited_dataset(
        load_gold_dataset(args.dataset, allow_draft=args.allow_draft),
        args.max_cases,
    )
    logging.info(
        "Loaded %d %s gold cases (%s)",
        len(dataset.cases),
        dataset.name,
        dataset.evidence_grade,
    )
    graph = FalkorDBStore(
        host=args.falkordb_host,
        port=args.falkordb_port,
        graph_name=args.graph_name,
    )
    vector = QdrantStore(
        host=args.qdrant_host,
        port=args.qdrant_port,
        collection=args.qdrant_collection,
    )
    index_metadata = validate_index_provenance(
        graph,
        vector,
        args.repo,
        expected_embedding_model=args.embedding_model,
    )
    logging.info("Validated index source identity %s", index_metadata["source_identity"])

    with OllamaEmbedder(
        ollama_url=args.ollama_url,
        model=args.embedding_model,
    ) as embedder:
        retriever = HybridRetriever(
            graph_store=graph,
            vector_store=vector,
            embedder=embedder,
        )
        report = AnswerQualityRunner(
            retriever,
            repository=args.repo,
            ollama_url=args.ollama_url,
            answer_model=args.answer_model,
            judge_model=args.judge_model,
            embedding_model=args.embedding_model,
            input_cost_per_million=args.input_cost_per_million,
            output_cost_per_million=args.output_cost_per_million,
        ).run(
            dataset,
            index_metadata=index_metadata,
            top_k=args.top_k,
            context_n=args.context_n,
            repeats=args.repeats,
            seed=args.seed,
            checkpoint_path=checkpoint_path,
            resume=args.resume,
        )

    final_index_metadata = validate_index_provenance(
        graph,
        vector,
        args.repo,
        expected_embedding_model=args.embedding_model,
    )
    require_same_index_run(index_metadata, final_index_metadata)

    result = report.as_dict()
    result["generated_at"] = datetime.now(UTC).isoformat()
    _write_json_atomic(args.output, result)
    print(json.dumps(result["summary"], indent=2))
    print(f"Evidence grade: {result['evidence_grade']}")
    print(f"Raw results: {args.output}")


if __name__ == "__main__":
    main()
