"""Pipeline ekstraksi topik: embedding IndoBERT -> HDBSCAN -> frekuensi term.

Mengelompokkan opini berdasarkan kepadatan fitur semantik (HDBSCAN) untuk
mengidentifikasi topik pembahasan dominan, lalu memberi label keyword tiap
topik dari term/frasa yang paling sering muncul di klaster. Input default
adalah hasil prediksi skenario
(punya kolom final_sentiment & location) sehingga tiap topik dapat dirinci
distribusi sentimennya.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

import config
from services.artifact_service import ArtifactService
from services.clustering_service import ClusteringService
from services.embedding_service import EmbeddingService
from services.topic_service import TopicService


def _read_input(input_path: Path) -> pl.DataFrame:
    if input_path.suffix.lower() == ".parquet":
        return pl.read_parquet(input_path)
    return pl.read_csv(input_path, infer_schema_length=10_000, ignore_errors=True)


def run(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    model_dir: str | Path = config.INDOBERT_MODEL_PATH,
    text_column: str = config.COL_PROCESSED,
    limit: int | None = None,
) -> dict[str, Any]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    artifact = ArtifactService()

    df = _read_input(input_path)
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit harus lebih besar dari 0")
        df = df.head(limit)
    resolved_text = text_column if text_column in df.columns else config.COL_TEXT
    if resolved_text not in df.columns:
        raise KeyError(
            f"Kolom teks tidak ditemukan (dicari: {text_column}, {config.COL_TEXT})"
        )

    # 1. Embedding semantik (mean-pool IndoBERT)
    embedder = EmbeddingService(
        backend=str(config.TOPIC_CONFIG["embedding_backend"]),
        model_path=model_dir,
        pooling=str(config.TOPIC_CONFIG["embedding_pooling"]),
    )
    vectors = embedder.encode_many(df[resolved_text].to_list())

    # 2. HDBSCAN density clustering (dengan reduksi PCA)
    clustering = ClusteringService(
        min_cluster_size=int(config.TOPIC_CONFIG["hdbscan_min_cluster_size"])
    )
    clustered = clustering.attach_hdbscan(
        df,
        vectors,
        min_cluster_size=int(config.TOPIC_CONFIG["hdbscan_min_cluster_size"]),
        min_samples=int(config.TOPIC_CONFIG["hdbscan_min_samples"]),
        metric=str(config.TOPIC_CONFIG["hdbscan_metric"]),
        pca_components=int(config.TOPIC_CONFIG["pca_components"]),
    )

    # 3. Ekstraksi keyword + ringkasan topik
    summary = TopicService().summarize(
        clustered,
        text_column="original_text" if "original_text" in clustered.columns else resolved_text,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    assignment_cols = [
        column
        for column in (
            "text_id",
            "original_text",
            resolved_text,
            config.COL_CLUSTER_ID,
            "cluster_size",
            "cluster_probability",
            "final_sentiment",
            "location",
        )
        if column in clustered.columns
    ]
    assignments = clustered.select(assignment_cols)
    artifact.save_parquet(assignments, output_dir / "topic_assignments.parquet")
    artifact.save_csv(assignments, output_dir / "topic_assignments.csv")
    artifact.save_json(
        {
            "input_path": str(input_path),
            "model_dir": str(model_dir),
            "text_column": resolved_text,
            "config": config.TOPIC_CONFIG,
            **summary,
        },
        output_dir / "topic_summary.json",
    )

    return {
        "assignments_path": output_dir / "topic_assignments.parquet",
        "summary_path": output_dir / "topic_summary.json",
        "n_topics": summary["n_topics"],
        "n_noise": summary["n_noise"],
        "summary": summary,
    }
