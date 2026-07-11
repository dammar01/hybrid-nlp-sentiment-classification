"""Runtime hybrid sentiment pipeline dengan LLM (skenario-2).

Menjalankan pipeline yang sama dengan skenario-1 (preprocessing -> IndoBERT ->
rule SO-CAL -> ambiguity -> fusi simetris), lalu MENGADJUDIKASI hanya baris
`requires_llm` (konflik IndoBERT high-conf vs rule detected) menggunakan LLM
GGUF lokal secara selektif. Baris lain memakai hasil fusi apa adanya.
"""

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
from pipelines.scenario_without_llm import (
    order_output_columns,
    prepare_runtime_input,
    validate_input,
)


def run(
    *,
    input_path: str | Path = config.RAW_CANDIDATE_SCHEMA_PATH,
    model_dir: str | Path,
    calibration_artifact_path: str | Path,
    fusion_policy_path: str | Path,
    output_dir: str | Path,
    llm_model_path: str | Path = config.QWEN_GGUF_MODEL_PATH,
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

    inference = IndoBERTInferenceService(
        model_path=model_dir,
        calibration_artifact_path=calibration_artifact_path,
    )
    predicted = inference.predict_dataframe(prepared, text_column=config.COL_PROCESSED)
    ruled = LexiconSentimentService().analyze_dataframe(
        predicted, text_column=config.COL_PROCESSED
    )
    policy = artifact.load_json(fusion_policy_path)
    scored = AmbiguityService(weights=dict(policy["uncertainty_weights"])).score_dataframe(
        ruled
    )
    fused = FusionService(policy=policy).fuse_dataframe(scored)

    # --- Lapisan LLM selektif: hanya baris requires_llm ---
    adjudicated, llm_stats = _apply_llm_layer(fused, llm_model_path=llm_model_path)

    result = order_output_columns(adjudicated)
    summary = _build_summary(
        input_df=df,
        prepared_df=prepared,
        result_df=result,
        input_path=input_path,
        total_input_rows=total_input_rows,
        runtime_limit=limit,
        model_dir=Path(model_dir),
        calibration_artifact_path=Path(calibration_artifact_path),
        fusion_policy_path=Path(fusion_policy_path),
        llm_model_path=Path(llm_model_path),
        llm_stats=llm_stats,
    )

    artifact.save_parquet(result, output_dir / "scenario_with_llm_predictions.parquet")
    artifact.save_csv(result, output_dir / "scenario_with_llm_predictions.csv")
    artifact.save_json(summary, output_dir / "scenario_with_llm_summary.json")
    artifact.save_json(
        {
            "input_path": str(input_path),
            "model_dir": str(model_dir),
            "calibration_artifact_path": str(calibration_artifact_path),
            "fusion_policy_path": str(fusion_policy_path),
            "llm_model_path": str(llm_model_path),
            "output_dir": str(output_dir),
            "runtime_limit": limit,
            "label_mapping": config.LABEL2ID,
        },
        output_dir / "scenario_with_llm_manifest.json",
    )
    return {
        "predictions_path": output_dir / "scenario_with_llm_predictions.parquet",
        "predictions_csv_path": output_dir / "scenario_with_llm_predictions.csv",
        "summary_path": output_dir / "scenario_with_llm_summary.json",
        "manifest_path": output_dir / "scenario_with_llm_manifest.json",
        "summary": summary,
    }


def _apply_llm_layer(
    fused: pl.DataFrame, *, llm_model_path: str | Path
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Override final_sentiment baris requires_llm dengan keputusan LLM."""
    if "requires_llm" not in fused.columns:
        fused = fused.with_columns(pl.lit(False).alias("requires_llm"))

    mask = fused["requires_llm"].fill_null(False).to_list()
    target_index = [i for i, flag in enumerate(mask) if flag]

    # Kolom jejak LLM default (baris non-target tetap kosong/final fusi).
    n = fused.height
    llm_labels: list[str | None] = [None] * n
    llm_raws: list[str | None] = [None] * n
    decision_source = ["fusion"] * n

    stats = {
        "llm_invoked_count": len(target_index),
        "llm_parsed_count": 0,
        "llm_changed_count": 0,
    }

    if target_index:
        # Import lazy: hanya saat benar-benar ada baris yang butuh LLM.
        from services.llm_service import LLMService

        text_col = "original_text" if "original_text" in fused.columns else config.COL_PROCESSED
        texts = fused[text_col].to_list()
        final = fused["final_sentiment"].to_list()

        llm = LLMService(model_path=llm_model_path)
        for i in target_index:
            out = llm.classify_text(str(texts[i] or ""))
            label = str(out["llm_label"])
            llm_labels[i] = label
            llm_raws[i] = str(out["llm_raw"])
            decision_source[i] = "llm"
            if out["llm_parsed"]:
                stats["llm_parsed_count"] += 1
            if label != final[i]:
                stats["llm_changed_count"] += 1
            final[i] = label
        fused = fused.with_columns(pl.Series("final_sentiment", final))

    return (
        fused.with_columns(
            pl.Series("llm_label", llm_labels, dtype=pl.Utf8),
            pl.Series("llm_raw", llm_raws, dtype=pl.Utf8),
            pl.Series("decision_source", decision_source, dtype=pl.Utf8),
        ),
        stats,
    )


def _build_summary(
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
    llm_model_path: Path,
    llm_stats: dict[str, Any],
) -> dict[str, Any]:
    final_labels = (
        result_df["final_sentiment"].to_list()
        if "final_sentiment" in result_df.columns
        else []
    )
    invalid_labels = sorted(set(final_labels) - set(config.SENTIMENT_LABELS))
    gates = {
        "all_runtime_rows_have_final_sentiment": result_df.filter(
            pl.col("final_sentiment").is_null()
            | (pl.col("final_sentiment").str.strip_chars().str.len_chars() == 0)
        ).height
        == 0,
        "all_labels_valid": not invalid_labels,
        "llm_only_on_requires_llm": result_df.filter(
            (pl.col("decision_source") == "llm") & (pl.col("requires_llm") != True)
        ).height
        == 0,
    }
    return {
        "input_path": str(input_path),
        "model_dir": str(model_dir),
        "calibration_artifact_path": str(calibration_artifact_path),
        "fusion_policy_path": str(fusion_policy_path),
        "llm_model_path": str(llm_model_path),
        "total_input_rows": total_input_rows,
        "runtime_limit": runtime_limit,
        "input_rows": input_df.height,
        "deduplicated_rows": prepared_df.height,
        "output_rows": result_df.height,
        "final_sentiment_distribution": dict(Counter(final_labels)),
        "decision_source_distribution": dict(Counter(result_df["decision_source"].to_list())),
        "fusion_action_distribution": dict(Counter(result_df["fusion_action"].to_list())),
        "llm_stats": llm_stats,
        "invalid_labels": invalid_labels,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }
