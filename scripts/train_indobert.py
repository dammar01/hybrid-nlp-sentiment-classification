"""Train fine-tuned IndoBERT sentiment classifier."""

from __future__ import annotations

import argparse
from datetime import datetime
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
from services.indobert_training_service import IndoBERTTrainingService


def default_experiment_id() -> str:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return f"{config.EXPERIMENT_CONFIG['experiment_prefix']}_{stamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune IndoBERT sentiment model.")
    parser.add_argument("--dataset", type=Path, default=config.TRAINING_DATASET_WITH_SPLIT_PATH)
    parser.add_argument("--experiment-id", default=default_experiment_id())
    parser.add_argument("--base-model", type=Path, default=config.INDOBERT_MODEL_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pl.read_parquet(args.dataset)
    train_rows = df.filter(pl.col("split") == "train").height
    calibration_rows = df.filter(pl.col("split") == "calibration").height
    test_rows = df.filter(pl.col("split") == "test").height
    if train_rows == 0 or calibration_rows == 0 or test_rows == 0:
        raise SystemExit("Split train/calibration/test wajib tersedia sebelum training")

    artifact = ArtifactService()
    experiment_dir = artifact.experiment_dir(args.experiment_id)
    artifact.save_json(artifact.config_snapshot(), experiment_dir / "config_snapshot.json")
    artifact.save_json(config.LABEL2ID, experiment_dir / "label_mapping.json")
    artifact.save_json(
        artifact.dataframe_manifest(df, source_path=args.dataset),
        experiment_dir / "dataset_manifest.json",
    )
    artifact.save_json(
        {
            "train_rows": train_rows,
            "calibration_rows": calibration_rows,
            "test_rows": test_rows,
            "test_usage_policy": "test set tidak dipakai untuk checkpoint, threshold, bobot, atau konfigurasi",
        },
        experiment_dir / "selected_training_configuration.json",
    )
    manifest = IndoBERTTrainingService(base_model_path=args.base_model).train(
        df,
        output_dir=experiment_dir,
    )
    print(f"Experiment: {experiment_dir}")
    print(f"Model: {manifest['model_dir']}")


if __name__ == "__main__":
    main()
