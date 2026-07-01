"""CLI entrypoint untuk pipeline tugas akhir."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid sentiment pipeline tanpa LLM.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("build-training-dataset")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build-training-dataset":
        from scripts.build_training_dataset import main as build_training_main

        sys.argv = ["build_training_dataset.py"]
        build_training_main()
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
        )
        print(f"Predictions: {result['predictions_path']}")
        print(f"Summary: {result['summary_path']}")


if __name__ == "__main__":
    main()
