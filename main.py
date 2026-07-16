"""CLI entrypoint untuk pipeline tugas akhir."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import config


DEFAULT_INDOBERT_EXPERIMENT_DIR = (
    Path(config.EXPERIMENT_CONFIG["artifact_root"]) / "indobert_nlp_manual"
)
DEFAULT_FUSION_EXPERIMENT_DIR = DEFAULT_INDOBERT_EXPERIMENT_DIR
DEFAULT_HYBRID_RUNTIME_DIR = DEFAULT_INDOBERT_EXPERIMENT_DIR / "runtime_hybrid"


def summarize_fusion_facts(
    df,
    *,
    actual_column: str | None = None,
) -> dict[str, object]:
    """Ringkas perubahan label hybrid tanpa menyamakan runtime dengan evaluasi."""
    actions: dict[str, int] = {}
    comparable_rows = 0
    changed_rows = 0
    conflict_count = 0
    requires_llm_count = 0
    needs_review_count = 0
    labeled_rows = 0
    corrected_rows = 0
    worsened_rows = 0

    valid_labels = set(config.SENTIMENT_LABELS)
    for row in df.iter_rows(named=True):
        action = str(row.get("fusion_action") or "unknown")
        actions[action] = actions.get(action, 0) + 1
        conflict_count += int(bool(row.get("cross_method_conflict_final")))
        requires_llm_count += int(bool(row.get("requires_llm")))
        needs_review_count += int(bool(row.get("needs_review")))

        bert_label = str(row.get("bert_label") or "")
        final_label = str(row.get("final_sentiment") or "")
        comparable = bert_label in valid_labels and final_label in valid_labels
        changed = comparable and bert_label != final_label
        comparable_rows += int(comparable)
        changed_rows += int(changed)

        if actual_column is None:
            continue
        actual_label = str(row.get(actual_column) or "")
        if not comparable or actual_label not in valid_labels:
            continue
        labeled_rows += 1
        if changed and bert_label != actual_label and final_label == actual_label:
            corrected_rows += 1
        elif changed and bert_label == actual_label and final_label != actual_label:
            worsened_rows += 1

    net_correct_change = corrected_rows - worsened_rows
    facts: dict[str, object] = {
        "total_rows": df.height,
        "comparable_rows": comparable_rows,
        "fusion_action_distribution": dict(sorted(actions.items())),
        "changed_from_indobert_count": changed_rows,
        "changed_from_indobert_rate": round(changed_rows / comparable_rows, 6)
        if comparable_rows
        else 0.0,
        "cross_method_conflict_count": conflict_count,
        "requires_llm_count": requires_llm_count,
        "needs_review_count": needs_review_count,
        "has_actual_labels": labeled_rows > 0,
    }
    if labeled_rows:
        facts.update(
            {
                "labeled_rows": labeled_rows,
                "corrected_by_hybrid_count": corrected_rows,
                "worsened_by_hybrid_count": worsened_rows,
                "net_correct_change": net_correct_change,
                "accuracy_delta_from_indobert": round(
                    net_correct_change / labeled_rows,
                    6,
                ),
            }
        )
    return facts


def build_hybrid_report(
    *,
    predictions_path: Path,
    fusion_policy_path: Path,
    report_dir: Path,
    include_visualizations: bool = True,
) -> dict[str, Path]:
    """Bangun dataset laporan, metrik evaluasi, dan visualisasi hybrid."""
    import polars as pl

    from services.artifact_service import ArtifactService
    from services.evaluation_service import EvaluationService
    from services.visualization_service import VisualizationService

    artifact = ArtifactService()
    df = pl.read_parquet(predictions_path)
    report_dir.mkdir(parents=True, exist_ok=True)

    preferred_columns = [
        "text_id",
        "original_text",
        config.COL_PROCESSED,
        config.COL_ACTUAL_LABEL,
        "sentiment_label",
        "bert_label",
        "bert_prob_negatif",
        "bert_prob_netral",
        "bert_prob_positif",
        "bert_confidence",
        "bert_margin",
        "bert_entropy",
        *config.RULE_OUTPUT_COLUMNS,
        "cross_method_conflict",
        "cross_method_conflict_final",
        "confidence_uncertainty",
        "margin_uncertainty",
        "normalized_entropy",
        "routing_uncertainty_score",
        "fusion_action",
        "fusion_reason",
        "final_sentiment",
        "requires_llm",
        "needs_review",
        "source_url",
        "source_type",
        "location",
        "location_source",
        "location_match",
        "is_specific_location",
    ]
    report_columns = [column for column in preferred_columns if column in df.columns]
    classification_df = df.select(report_columns)
    classification_csv = artifact.save_csv(
        classification_df,
        report_dir / "hybrid_classification_report.csv",
    )
    classification_parquet = artifact.save_parquet(
        classification_df,
        report_dir / "hybrid_classification_report.parquet",
    )

    evaluator = EvaluationService()
    runtime_fusion_facts = summarize_fusion_facts(df)
    runtime_evaluation: dict[str, object] = {
        "status": "skipped",
        "reason": "Input runtime tidak memiliki label aktual yang valid.",
    }
    for label_column in (config.COL_ACTUAL_LABEL, "sentiment_label"):
        if label_column not in df.columns:
            continue
        labeled_df = df.filter(
            pl.col(label_column).cast(pl.Utf8).is_in(list(config.SENTIMENT_LABELS))
        )
        if labeled_df.is_empty():
            continue
        runtime_evaluation = {
            "status": "available",
            "label_column": label_column,
            "labeled_rows": labeled_df.height,
            "metrics": evaluator.evaluate_components(
                labeled_df,
                actual_column=label_column,
            ),
            "fusion_facts": summarize_fusion_facts(
                labeled_df,
                actual_column=label_column,
            ),
        }
        break

    experiment_metrics_path = fusion_policy_path.parent / "metrics.json"
    experiment_metrics = (
        artifact.load_json(experiment_metrics_path)
        if experiment_metrics_path.exists()
        else None
    )
    metrics_payload = {
        "predictions_path": str(predictions_path),
        "runtime_fusion_facts": runtime_fusion_facts,
        "runtime_evaluation": runtime_evaluation,
        "experiment_metrics_path": str(experiment_metrics_path),
        "experiment_metrics": experiment_metrics,
    }
    metrics_path = artifact.save_json(
        metrics_payload,
        report_dir / "hybrid_evaluation_metrics.json",
    )

    outputs = {
        "classification_csv": classification_csv,
        "classification_parquet": classification_parquet,
        "evaluation_metrics": metrics_path,
    }
    if include_visualizations:
        visualizer = VisualizationService()
        outputs["sentiment_distribution"] = visualizer.save_figure(
            visualizer.plot_hybrid_sentiment_distribution(df),
            report_dir / "hybrid_sentiment_distribution.png",
        )
        outputs["kalbar_distribution"] = visualizer.save_figure(
            visualizer.plot_kalbar_location_distribution(df),
            report_dir / "hybrid_kalbar_location_distribution.png",
        )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid sentiment pipeline tanpa LLM.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("build-training-dataset")
    raw_candidate = subparsers.add_parser("build-raw-candidate-schema")
    raw_candidate.add_argument("--output", type=Path, default=config.RAW_CANDIDATE_SCHEMA_PATH)
    raw_candidate.add_argument("--no-sentence-split", action="store_true")
    raw_candidate.add_argument(
        "--sentence-min-chars",
        type=int,
        default=config.RAW_CANDIDATE_SENTENCE_MIN_CHARS,
    )
    raw_candidate.add_argument(
        "--sentence-max-chars",
        type=int,
        default=config.RAW_CANDIDATE_SENTENCE_MAX_CHARS,
    )
    raw_candidate.add_argument("--no-write", action="store_true")
    subparsers.add_parser("create-fixed-group-split")

    train = subparsers.add_parser("train-indobert")
    train.add_argument("--dataset", type=Path, default=config.TRAINING_DATASET_WITH_SPLIT_PATH)
    train.add_argument("--experiment-id", default=None)
    train.add_argument("--base-model", type=Path, default=config.INDOBERT_MODEL_PATH)

    calibrate = subparsers.add_parser("calibrate-indobert")
    calibrate.add_argument("--dataset", type=Path, default=config.TRAINING_DATASET_WITH_SPLIT_PATH)
    calibrate.add_argument("--experiment-dir", type=Path, required=True)

    runtime = subparsers.add_parser("run-without-llm")
    runtime.add_argument("--input", type=Path, default=config.RAW_CANDIDATE_SCHEMA_PATH)
    runtime.add_argument("--model-dir", type=Path, required=True)
    runtime.add_argument("--calibration-artifact", type=Path, required=True)
    runtime.add_argument("--fusion-policy", type=Path, required=True)
    runtime.add_argument("--output-dir", type=Path, required=True)
    runtime.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Batasi jumlah row runtime yang diproses dari awal input CSV.",
    )

    hybrid_nlp = subparsers.add_parser(
        "run-hybrid-nlp",
        help="Jalankan rule-based -> IndoBERT -> hybrid fusion -> ekstraksi topik.",
    )
    hybrid_nlp.add_argument("--input", type=Path, default=config.RAW_CANDIDATE_SCHEMA_PATH)
    hybrid_nlp.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_INDOBERT_EXPERIMENT_DIR / "model",
    )
    hybrid_nlp.add_argument(
        "--calibration-artifact",
        type=Path,
        default=DEFAULT_INDOBERT_EXPERIMENT_DIR / "calibration_artifact.json",
    )
    hybrid_nlp.add_argument(
        "--fusion-policy",
        type=Path,
        default=DEFAULT_FUSION_EXPERIMENT_DIR / "fusion_policy.json",
    )
    hybrid_nlp.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_HYBRID_RUNTIME_DIR,
        help="Folder output prediksi dan ringkasan hybrid.",
    )
    hybrid_nlp.add_argument(
        "--topics-output-dir",
        type=Path,
        default=None,
        help="Folder output topik; default: <output-dir>/topics.",
    )
    hybrid_nlp.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Batasi jumlah row input sebelum rule-based dan IndoBERT.",
    )
    hybrid_nlp.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Folder laporan; default: <output-dir>/report.",
    )
    hybrid_nlp.add_argument(
        "--skip-visualizations",
        action="store_true",
        help="Lewati PNG bila matplotlib tidak tersedia.",
    )

    runtime_llm = subparsers.add_parser("run-with-llm")
    runtime_llm.add_argument("--input", type=Path, default=config.RAW_CANDIDATE_SCHEMA_PATH)
    runtime_llm.add_argument("--model-dir", type=Path, required=True)
    runtime_llm.add_argument("--calibration-artifact", type=Path, required=True)
    runtime_llm.add_argument("--fusion-policy", type=Path, required=True)
    runtime_llm.add_argument("--output-dir", type=Path, required=True)
    runtime_llm.add_argument(
        "--llm-model",
        type=Path,
        default=config.QWEN_GGUF_MODEL_PATH,
        help="Path file GGUF Qwen3-8B untuk lapisan interpretasi LLM.",
    )
    runtime_llm.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Batasi jumlah row runtime yang diproses dari awal input CSV.",
    )

    topics = subparsers.add_parser("run-topics")
    topics.add_argument("--input", type=Path, required=True,
                        help="Prediksi skenario (.parquet/.csv) atau input teks.")
    topics.add_argument("--output-dir", type=Path, required=True)
    topics.add_argument("--model-dir", type=Path, default=config.INDOBERT_MODEL_PATH,
                        help="Model IndoBERT sumber embedding topik.")
    topics.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build-training-dataset":
        from scripts.build_training_dataset import main as build_training_main

        sys.argv = ["build_training_dataset.py"]
        build_training_main()
    elif args.command == "build-raw-candidate-schema":
        from scripts.build_raw_candidate_dataset import main as build_raw_candidate_main

        sys.argv = [
            "build_raw_candidate_dataset.py",
            "--output",
            str(args.output),
            "--sentence-min-chars",
            str(args.sentence_min_chars),
            "--sentence-max-chars",
            str(args.sentence_max_chars),
        ]
        if args.no_sentence_split:
            sys.argv.append("--no-sentence-split")
        if args.no_write:
            sys.argv.append("--no-write")
        build_raw_candidate_main()
    elif args.command == "create-fixed-group-split":
        from scripts.create_fixed_group_split import main as create_split_main

        sys.argv = ["create_fixed_group_split.py"]
        create_split_main()
    elif args.command == "train-indobert":
        from scripts.train_indobert import main as train_main

        sys.argv = [
            "train_indobert.py",
            "--dataset",
            str(args.dataset),
            "--base-model",
            str(args.base_model),
        ]
        if args.experiment_id:
            sys.argv.extend(["--experiment-id", args.experiment_id])
        train_main()
    elif args.command == "calibrate-indobert":
        from scripts.calibrate_indobert import main as calibrate_main

        sys.argv = [
            "calibrate_indobert.py",
            "--dataset",
            str(args.dataset),
            "--experiment-dir",
            str(args.experiment_dir),
        ]
        calibrate_main()
    elif args.command == "run-without-llm":
        from pipelines.scenario_without_llm import run

        result = run(
            input_path=args.input,
            model_dir=args.model_dir,
            calibration_artifact_path=args.calibration_artifact,
            fusion_policy_path=args.fusion_policy,
            output_dir=args.output_dir,
            limit=args.limit,
        )
        print(f"Predictions: {result['predictions_path']}")
        print(f"Summary: {result['summary_path']}")
    elif args.command == "run-hybrid-nlp":
        from pipelines.scenario_without_llm import run as run_hybrid
        from pipelines.topic_pipeline import run as run_topics

        hybrid_result = run_hybrid(
            input_path=args.input,
            model_dir=args.model_dir,
            calibration_artifact_path=args.calibration_artifact,
            fusion_policy_path=args.fusion_policy,
            output_dir=args.output_dir,
            limit=args.limit,
        )
        report_outputs = build_hybrid_report(
            predictions_path=hybrid_result["predictions_path"],
            fusion_policy_path=args.fusion_policy,
            report_dir=args.report_dir or (args.output_dir / "report"),
            include_visualizations=not args.skip_visualizations,
        )
        topics_output_dir = args.topics_output_dir or (args.output_dir / "topics")
        topic_result = run_topics(
            input_path=hybrid_result["predictions_path"],
            output_dir=topics_output_dir,
            model_dir=args.model_dir,
        )

        print(f"Hybrid predictions: {hybrid_result['predictions_path']}")
        print(f"Hybrid summary: {hybrid_result['summary_path']}")
        print(f"Topics: {topic_result['n_topics']} (noise={topic_result['n_noise']})")
        print(f"Topic assignments: {topic_result['assignments_path']}")
        print(f"Topic summary: {topic_result['summary_path']}")
        print("Report outputs:")
        for name, path in report_outputs.items():
            print(f"- {name}: {path}")
    elif args.command == "run-with-llm":
        from pipelines.scenario_with_llm import run

        result = run(
            input_path=args.input,
            model_dir=args.model_dir,
            calibration_artifact_path=args.calibration_artifact,
            fusion_policy_path=args.fusion_policy,
            output_dir=args.output_dir,
            llm_model_path=args.llm_model,
            limit=args.limit,
        )
        print(f"Predictions: {result['predictions_path']}")
        print(f"Summary: {result['summary_path']}")
    elif args.command == "run-topics":
        from pipelines.topic_pipeline import run

        result = run(
            input_path=args.input,
            output_dir=args.output_dir,
            model_dir=args.model_dir,
            limit=args.limit,
        )
        print(f"Topics: {result['n_topics']} (noise={result['n_noise']})")
        print(f"Assignments: {result['assignments_path']}")
        print(f"Summary: {result['summary_path']}")


if __name__ == "__main__":
    main()
