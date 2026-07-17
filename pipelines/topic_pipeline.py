"""Pipeline keyword langsung dari SO-CAL untuk tiap kelas sentimen final."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

import config
from services.artifact_service import ArtifactService
from services.topic_service import TopicService


def _read_input(input_path: Path) -> pl.DataFrame:
    if input_path.suffix.lower() == ".parquet":
        return pl.read_parquet(input_path)
    return pl.read_csv(input_path, infer_schema_length=10_000, ignore_errors=True)


def _validate_sentiments(df: pl.DataFrame, sentiment_column: str) -> pl.DataFrame:
    normalized = df.with_columns(
        pl.col(sentiment_column)
        .cast(pl.String)
        .str.to_lowercase()
        .str.strip_chars()
        .alias(sentiment_column)
    )
    invalid = sorted(
        {
            str(value)
            for value in normalized[sentiment_column].drop_nulls().unique().to_list()
            if value not in config.SENTIMENT_LABELS
        }
    )
    null_count = normalized[sentiment_column].null_count()
    if invalid or null_count:
        raise ValueError(
            "final_sentiment wajib berisi hanya negatif/netral/positif; "
            f"invalid={invalid}, null={null_count}"
        )
    return normalized


def run(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    model_dir: str | Path = config.INDOBERT_MODEL_PATH,
    text_column: str = config.COL_PROCESSED,
    sentiment_column: str = "final_sentiment",
    limit: int | None = None,
) -> dict[str, Any]:
    """Kelompokkan dokumen berdasarkan sentimen lalu cocokkan resource SO-CAL.

    ``model_dir`` tetap diterima oleh entry point gabungan, tetapi tidak dibaca:
    ekstraksi keyword ini tidak memuat model, embedding, UMAP, atau HDBSCAN.
    """
    del model_dir
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
    resolved_sentiment = (
        sentiment_column
        if sentiment_column in df.columns
        else config.COL_FINAL_LABEL
        if config.COL_FINAL_LABEL in df.columns
        else None
    )
    if resolved_sentiment is None:
        raise KeyError(
            f"Kolom sentimen final tidak ditemukan: {sentiment_column} "
            f"atau {config.COL_FINAL_LABEL}"
        )
    df = _validate_sentiments(df, resolved_sentiment)

    topic_service = TopicService()
    sentiment_keywords: dict[str, dict[str, Any]] = {}
    for sentiment_label in config.TOPIC_SENTIMENT_ORDER:
        subset = df.filter(pl.col(resolved_sentiment) == sentiment_label)
        if subset.is_empty():
            sentiment_keywords[sentiment_label] = topic_service.empty_summary(
                sentiment_label
            )
            continue
        sentiment_keywords[sentiment_label] = topic_service.summarize(
            subset,
            sentiment_label=sentiment_label,
            text_column=(
                "original_text" if "original_text" in subset.columns else resolved_text
            ),
        )

    assignments_full = df.with_columns(
        pl.col(resolved_sentiment).alias("topic_sentiment")
    )
    assignment_cols = [
        column
        for column in (
            "text_id",
            "original_text",
            resolved_text,
            resolved_sentiment,
            "topic_sentiment",
            "location",
        )
        if column in assignments_full.columns
    ]
    assignments = assignments_full.select(assignment_cols)

    total_keywords = sum(
        int(summary["keyword_count"])
        for summary in sentiment_keywords.values()
    )
    summary: dict[str, Any] = {
        "n_documents": df.height,
        "keyword_count": total_keywords,
        "sentiment_keywords": sentiment_keywords,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    assignments_path = output_dir / "topic_assignments.parquet"
    summary_path = output_dir / "topic_summary.json"
    artifact.save_parquet(assignments, assignments_path)
    artifact.save_csv(assignments, output_dir / "topic_assignments.csv")
    artifact.save_json(
        {
            "input_path": str(input_path),
            "text_column": resolved_text,
            "sentiment_column": resolved_sentiment,
            "config": config.TOPIC_CONFIG,
            "keyword_extraction": topic_service.extraction_metadata(),
            **summary,
        },
        summary_path,
    )

    # n_topics/n_noise dipertahankan pada return agar entry point gabungan tetap
    # dapat mencetak hasil; keduanya tidak disimpan sebagai kontrak artefak.
    return {
        "assignments_path": assignments_path,
        "summary_path": summary_path,
        "n_topics": sum(
            bool(summary["keyword_count"])
            for summary in sentiment_keywords.values()
        ),
        "n_noise": 0,
        "keyword_count": total_keywords,
        "sentiment_keywords": sentiment_keywords,
        "summary": summary,
    }
