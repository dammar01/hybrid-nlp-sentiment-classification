"""Calibrate IndoBERT confidence and select hybrid fusion policy."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import polars as pl


def find_project_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "config.py").exists() and (candidate / "services").is_dir():
            return candidate
    raise FileNotFoundError("Root proyek tidak ditemukan")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from services.artifact_service import ArtifactService
from services.ambiguity_service import AmbiguityService
from services.calibration_service import CalibrationService
from services.evaluation_service import EvaluationService
from services.fusion_service import FusionService
from services.indobert_inference_service import IndoBERTInferenceService
from services.lexicon_sentiment_service import LexiconSentimentService


LOGIT_COLUMNS = ("bert_logit_negatif", "bert_logit_netral", "bert_logit_positif")


def apply_calibrated_probabilities(
    df: pl.DataFrame,
    *,
    temperature: float,
) -> pl.DataFrame:
    calibrator = CalibrationService()
    logits = [[float(value) for value in row] for row in df.select(LOGIT_COLUMNS).rows()]
    probabilities = calibrator.apply_temperature(logits, temperature)
    rows = []
    for probs in probabilities:
        label_id = max(range(len(probs)), key=lambda index: probs[index])
        sorted_probs = sorted(probs, reverse=True)
        rows.append(
            {
                "bert_label": config.ID2LABEL[label_id],
                "bert_label_id": label_id,
                "bert_prob_negatif": float(probs[0]),
                "bert_prob_netral": float(probs[1]),
                "bert_prob_positif": float(probs[2]),
                "bert_confidence": float(sorted_probs[0]),
                "bert_margin": float(sorted_probs[0] - sorted_probs[1]),
                "bert_entropy": IndoBERTInferenceService._normalized_entropy(probs),
                "bert_temperature": temperature,
            }
        )
    replacement = pl.DataFrame(rows)
    drop_columns = [column for column in replacement.columns if column in df.columns]
    return df.drop(drop_columns).hstack(replacement)


def prepare_component_predictions(
    df: pl.DataFrame,
    *,
    model_dir: Path,
    temperature: float | None = None,
) -> pl.DataFrame:
    inference = IndoBERTInferenceService(model_path=model_dir)
    predicted = inference.predict_dataframe(df, text_column=config.COL_PROCESSED)
    if temperature is not None:
        predicted = apply_calibrated_probabilities(predicted, temperature=temperature)
    rule_service = LexiconSentimentService()
    return rule_service.analyze_dataframe(predicted, text_column=config.COL_PROCESSED)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate IndoBERT and select fusion policy.")
    parser.add_argument("--dataset", type=Path, default=config.TRAINING_DATASET_WITH_SPLIT_PATH)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir
    model_dir = experiment_dir / "model"
    df = pl.read_parquet(args.dataset)
    calibration_df = df.filter(pl.col("split") == "calibration")
    test_df = df.filter(pl.col("split") == "test")
    if calibration_df.is_empty() or test_df.is_empty():
        raise SystemExit("Calibration dan test split wajib tersedia")

    artifact = ArtifactService()
    uncalibrated_calibration = prepare_component_predictions(
        calibration_df,
        model_dir=model_dir,
        temperature=None,
    )
    calibration_artifact = CalibrationService().fit_from_dataframe(
        uncalibrated_calibration,
        label_column="label_id",
        group_column="group_id",
    )
    artifact.save_json(calibration_artifact, experiment_dir / "calibration_artifact.json")

    calibrated_calibration = apply_calibrated_probabilities(
        uncalibrated_calibration,
        temperature=float(calibration_artifact["temperature"]),
    )
    policy = FusionService().select_policy(calibrated_calibration)
    artifact.save_json(policy, experiment_dir / "fusion_policy.json")

    scored_calibration = AmbiguityService(
        weights=dict(policy["uncertainty_weights"])
    ).score_dataframe(calibrated_calibration)
    fused_calibration = FusionService(policy=policy).fuse_dataframe(scored_calibration)
    artifact.save_parquet(fused_calibration, experiment_dir / "calibration_predictions.parquet")

    test_predictions = prepare_component_predictions(
        test_df,
        model_dir=model_dir,
        temperature=float(calibration_artifact["temperature"]),
    )
    scored_test = AmbiguityService(
        weights=dict(policy["uncertainty_weights"])
    ).score_dataframe(test_predictions)
    fused_test = FusionService(policy=policy).fuse_dataframe(scored_test)
    metrics = {
        "calibration": EvaluationService().evaluate_components(
            fused_calibration,
            actual_column="sentiment_label",
        ),
        "test_after_policy_frozen": EvaluationService().evaluate_components(
            fused_test,
            actual_column="sentiment_label",
        ),
    }
    artifact.save_parquet(fused_test, experiment_dir / "test_predictions.parquet")
    artifact.save_json(metrics, experiment_dir / "metrics.json")
    print(f"Calibration artifact: {experiment_dir / 'calibration_artifact.json'}")
    print(f"Fusion policy: {experiment_dir / 'fusion_policy.json'}")
    print(f"Metrics: {experiment_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
