"""Create fixed StratifiedGroupKFold split untuk training IndoBERT."""

from __future__ import annotations

import argparse
from collections import Counter
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


def create_split(df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, object]]:
    required = ("text_id", "sentiment_label", "label_id", "group_id")
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Kolom split hilang: {missing}")

    try:
        from sklearn.model_selection import StratifiedGroupKFold
    except ImportError as exc:
        raise ImportError("scikit-learn wajib untuk StratifiedGroupKFold") from exc

    ordered = df.sort("text_id").with_row_index("_row_index")
    y = ordered["label_id"].to_list()
    groups = ordered["group_id"].to_list()
    splitter = StratifiedGroupKFold(
        n_splits=int(config.SPLIT_CONFIG["n_splits"]),
        shuffle=True,
        random_state=int(config.SPLIT_CONFIG["random_state"]),
    )
    fold_by_index: dict[int, int] = {}
    for fold, (_, fold_indices) in enumerate(splitter.split([[0]] * ordered.height, y, groups)):
        for index in fold_indices:
            fold_by_index[int(index)] = int(fold)

    rows: list[dict[str, object]] = []
    for index, row in enumerate(ordered.iter_rows(named=True)):
        fold = fold_by_index[index]
        split = split_name(fold)
        rows.append(
            {
                "text_id": row["text_id"],
                "base_text_id": row.get("base_text_id"),
                "group_id": row["group_id"],
                "label_id": row["label_id"],
                "sentiment_label": row["sentiment_label"],
                "split_fold": fold,
                "split": split,
            }
        )
    split_df = pl.from_dicts(rows, strict=False)
    joined = df.join(
        split_df.select(["text_id", "split_fold", "split"]),
        on="text_id",
        how="left",
    )
    manifest = build_manifest(joined, split_df)
    return joined, manifest


def split_name(fold: int) -> str:
    if fold == int(config.SPLIT_CONFIG["test_fold"]):
        return "test"
    if fold == int(config.SPLIT_CONFIG["calibration_fold"]):
        return "calibration"
    return "train"


def build_manifest(df: pl.DataFrame, split_df: pl.DataFrame) -> dict[str, object]:
    split_counts = Counter(split_df["split"].to_list())
    group_sets = {
        split: set(split_df.filter(pl.col("split") == split)["group_id"].to_list())
        for split in ("train", "calibration", "test")
    }
    overlaps = {
        "train_calibration": sorted(group_sets["train"] & group_sets["calibration"]),
        "train_test": sorted(group_sets["train"] & group_sets["test"]),
        "calibration_test": sorted(group_sets["calibration"] & group_sets["test"]),
    }
    per_split_label_distribution = {}
    for split in ("train", "calibration", "test"):
        labels = split_df.filter(pl.col("split") == split)["sentiment_label"].to_list()
        per_split_label_distribution[split] = dict(Counter(labels))
    gates = {
        "all_rows_have_split": df.filter(pl.col("split").is_null()).height == 0,
        "row_count_is_281": df.height == 281,
        "no_group_overlap": all(not values for values in overlaps.values()),
        "calibration_not_training": split_counts.get("calibration", 0) > 0,
        "test_not_training": split_counts.get("test", 0) > 0,
    }
    return {
        "split_config": config.SPLIT_CONFIG,
        "row_count": df.height,
        "group_count": df.select(pl.col("group_id").n_unique()).item(),
        "split_counts": dict(split_counts),
        "per_split_label_distribution": per_split_label_distribution,
        "overlaps": overlaps,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create fixed group-aware split.")
    parser.add_argument("--input", type=Path, default=config.TRAINING_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=config.TRAINING_DATASET_WITH_SPLIT_PATH)
    parser.add_argument(
        "--assignment-output",
        type=Path,
        default=config.FIXED_SPLIT_ASSIGNMENT_PATH,
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=config.FIXED_SPLIT_MANIFEST_PATH,
    )
    parser.add_argument("--allow-gate-failure", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pl.read_parquet(args.input)
    split_df, manifest = create_split(df)
    if not manifest["all_gates_passed"] and not args.allow_gate_failure:
        raise SystemExit(f"Split gate gagal: {manifest['gates']}")
    artifact = ArtifactService()
    artifact.save_parquet(split_df, args.output)
    assignment_rows = split_df.select(
        ["text_id", "base_text_id", "group_id", "sentiment_label", "label_id", "split_fold", "split"]
    ).to_dicts()
    artifact.save_json({"rows": assignment_rows}, args.assignment_output)
    artifact.save_json(manifest, args.manifest_output)
    print(f"Split dataset: {args.output}")
    print(f"Split assignment: {args.assignment_output}")
    print(f"Split manifest: {args.manifest_output}")


if __name__ == "__main__":
    main()
