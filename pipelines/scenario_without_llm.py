"""Runtime hybrid sentiment pipeline tanpa LLM."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import polars as pl

import config
from services.ambiguity_service import AmbiguityService
from services.artifact_service import ArtifactService
from services.fusion_service import FusionService
from services.indobert_inference_service import IndoBERTInferenceService
from services.lexicon_sentiment_service import LexiconSentimentService
from services.preprocessing_service import PreprocessingService
from services.source_blacklist_service import SourceBlacklistService


RUNTIME_REQUIRED_COLUMNS: tuple[str, ...] = (
    "text_id",
    "text",
    "source_url",
    "source_type",
    "location",
)


def run(
    *,
    input_path: str | Path = config.RAW_CANDIDATE_SCHEMA_PATH,
    model_dir: str | Path,
    calibration_artifact_path: str | Path,
    fusion_policy_path: str | Path,
    output_dir: str | Path,
    limit: int | None = None,
) -> dict[str, Any]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    artifact = ArtifactService()
    df = pl.read_csv(input_path, infer_schema_length=10_000, ignore_errors=True)
    total_input_rows = df.height
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit harus lebih besar dari 0")
        df = df.head(limit)
    validate_input(df)
    prepared = prepare_runtime_input(df)

    # Rule-based dijalankan lebih dahulu sebagai sinyal leksikal.
    ruled = LexiconSentimentService().analyze_dataframe(
        prepared,
        text_column=config.COL_PROCESSED,
    )

    # IndoBERT menambahkan sinyal kontekstual pada baris yang sama.
    inference = IndoBERTInferenceService(
        model_path=model_dir,
        calibration_artifact_path=calibration_artifact_path,
    )
    predicted = inference.predict_dataframe(
        ruled,
        text_column=config.COL_PROCESSED,
    )
    policy = artifact.load_json(fusion_policy_path)
    scored = AmbiguityService(weights=dict(policy["uncertainty_weights"])).score_dataframe(
        predicted
    )
    fused = FusionService(policy=policy).fuse_dataframe(scored)
    result = order_output_columns(fused)
    summary = build_summary(
        input_df=df,
        prepared_df=prepared,
        result_df=result,
        input_path=input_path,
        total_input_rows=total_input_rows,
        runtime_limit=limit,
        model_dir=Path(model_dir),
        calibration_artifact_path=Path(calibration_artifact_path),
        fusion_policy_path=Path(fusion_policy_path),
    )

    artifact.save_parquet(result, output_dir / "scenario_without_llm_predictions.parquet")
    artifact.save_csv(result, output_dir / "scenario_without_llm_predictions.csv")
    artifact.save_json(summary, output_dir / "scenario_without_llm_summary.json")
    artifact.save_json(
        {
            "input_path": str(input_path),
            "model_dir": str(model_dir),
            "calibration_artifact_path": str(calibration_artifact_path),
            "fusion_policy_path": str(fusion_policy_path),
            "output_dir": str(output_dir),
            "runtime_limit": limit,
            "label_mapping": config.LABEL2ID,
        },
        output_dir / "scenario_without_llm_manifest.json",
    )
    return {
        "predictions_path": output_dir / "scenario_without_llm_predictions.parquet",
        "predictions_csv_path": output_dir / "scenario_without_llm_predictions.csv",
        "summary_path": output_dir / "scenario_without_llm_summary.json",
        "manifest_path": output_dir / "scenario_without_llm_manifest.json",
        "summary": summary,
    }


def validate_input(df: pl.DataFrame) -> None:
    missing = [column for column in RUNTIME_REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(f"Kolom runtime wajib hilang: {missing}")


def prepare_runtime_input(df: pl.DataFrame) -> pl.DataFrame:
    source_service = SourceBlacklistService.from_paths()
    preprocessing = PreprocessingService()
    rows: list[dict[str, object]] = []
    for row in df.iter_rows(named=True):
        text = str(row.get("text") or "")
        processed = preprocessing.process(text)["processed_text"]
        source_url = str(row.get("source_url") or "")
        normalized_source_url = str(row.get("normalized_source_url") or "")
        if not normalized_source_url:
            normalized_source_url = source_service.normalize_url(source_url)
        rows.append(
            {
                **row,
                "original_text": text,
                config.COL_PROCESSED: processed,
                "normalized_source_url": normalized_source_url,
                "_dedupe_key": f"{normalized_source_url}::{processed}",
            }
        )
    prepared = pl.from_dicts(rows, strict=False)
    prepared = prepared.filter(pl.col(config.COL_PROCESSED).str.strip_chars().str.len_chars() > 0)
    return prepared.unique(subset=["_dedupe_key"], keep="first", maintain_order=True).drop("_dedupe_key")


def build_summary(
    *,
    input_df: pl.DataFrame,
    prepared_df: pl.DataFrame,
    result_df: pl.DataFrame,
    input_path: Path,
    total_input_rows: int,
    runtime_limit: int | None,
    model_dir: Path,
    calibration_artifact_path: Path,
    fusion_policy_path: Path,
) -> dict[str, Any]:
    final_labels = result_df["final_sentiment"].to_list() if "final_sentiment" in result_df.columns else []
    invalid_labels = sorted(set(final_labels) - set(config.SENTIMENT_LABELS))
    weak_overrides = result_df.filter(
        (pl.col("fusion_action") == "rule_confident")
        & pl.col(config.COL_RULE_STATUS).is_in(
            [config.RULE_STATUS_WEAK, config.RULE_STATUS_UNKNOWN]
        )
    ).height
    gates = {
        "all_runtime_rows_have_final_sentiment": result_df.filter(
            pl.col("final_sentiment").is_null()
            | (pl.col("final_sentiment").str.strip_chars().str.len_chars() == 0)
        ).height == 0,
        "all_labels_valid": not invalid_labels,
        "every_decision_has_fusion_reason": result_df.filter(
            pl.col("fusion_reason").is_null()
            | (pl.col("fusion_reason").str.strip_chars().str.len_chars() == 0)
        ).height == 0,
        "weak_unknown_rule_never_overrides": weak_overrides == 0,
    }
    return {
        "input_path": str(input_path),
        "model_dir": str(model_dir),
        "calibration_artifact_path": str(calibration_artifact_path),
        "fusion_policy_path": str(fusion_policy_path),
        "total_input_rows": total_input_rows,
        "runtime_limit": runtime_limit,
        "input_rows": input_df.height,
        "deduplicated_rows": prepared_df.height,
        "output_rows": result_df.height,
        "final_sentiment_distribution": dict(Counter(final_labels)),
        "needs_review_count": int(result_df.filter(pl.col("needs_review") == True).height),
        "fusion_action_distribution": dict(Counter(result_df["fusion_action"].to_list())),
        "invalid_labels": invalid_labels,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }


def order_output_columns(df: pl.DataFrame) -> pl.DataFrame:
    preferred = [
        "text_id",
        "original_text",
        config.COL_PROCESSED,
        "bert_label",
        "bert_prob_negatif",
        "bert_prob_netral",
        "bert_prob_positif",
        "bert_confidence",
        "bert_margin",
        "bert_entropy",
        *config.RULE_OUTPUT_COLUMNS,
        "cross_method_conflict",
        "confidence_uncertainty",
        "margin_uncertainty",
        "routing_uncertainty_score",
        "fusion_action",
        "fusion_reason",
        "final_sentiment",
        "needs_review",
        "source_url",
        "source_type",
        "location",
    ]
    ordered = [column for column in preferred if column in df.columns]
    extras = [column for column in df.columns if column not in ordered]
    return df.select(ordered + extras)
