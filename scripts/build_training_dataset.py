"""Build training_dataset.parquet dari golden sentiment dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any

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
from services.preprocessing_service import PreprocessingService
from services.source_blacklist_service import SourceBlacklistService


def base_text_id(text_id: str) -> str:
    return str(text_id or "").split("#", 1)[0].strip()


def load_golden_rows(golden_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(golden_dir.glob("*_sentiment_labeling_output.json")):
        data = json.loads(path.read_text(encoding=config.ENCODING))
        if not isinstance(data, list):
            raise ValueError(f"Golden file harus list JSON: {path}")
        for row in data:
            if not isinstance(row, dict):
                continue
            sentiment = str(row.get("sentiment") or "").strip().lower()
            rows.append(
                {
                    "text_id": str(row.get("text_id") or "").strip(),
                    "base_text_id": base_text_id(str(row.get("text_id") or "")),
                    "original_text": str(row.get("text_selected") or "").strip(),
                    "sentiment_label": sentiment,
                    "label_id": config.LABEL2ID.get(sentiment),
                    "location": str(row.get("location") or "").strip(),
                    "golden_file": path.name,
                }
            )
    return rows


def load_metadata_rows(paths: list[Path]) -> tuple[dict[str, dict], dict[str, dict]]:
    exact: dict[str, dict] = {}
    base: dict[str, dict] = {}
    for path in paths:
        if not path.exists():
            continue
        df = pl.read_csv(path, infer_schema_length=10_000, ignore_errors=True)
        if "text_id" not in df.columns:
            continue
        for row in df.iter_rows(named=True):
            text_id = str(row.get("text_id") or "").strip()
            if not text_id:
                continue
            metadata = {
                "source_url": str(row.get("source_url") or "").strip(),
                "normalized_source_url": str(
                    row.get("normalized_source_url") or ""
                ).strip(),
                "source_type": str(row.get("source_type") or "").strip(),
                "location": str(row.get("location") or "").strip(),
                "metadata_source_path": str(path),
            }
            exact.setdefault(text_id, metadata)
            base.setdefault(base_text_id(text_id), metadata)
    return exact, base


def recover_metadata(
    row: dict[str, Any],
    *,
    exact: dict[str, dict],
    base: dict[str, dict],
    source_service: SourceBlacklistService,
) -> dict[str, object]:
    text_id = str(row["text_id"])
    base_id = str(row["base_text_id"])
    metadata = exact.get(text_id)
    status = "matched_by_exact_text_id"
    if metadata is None:
        metadata = base.get(base_id)
        status = "matched_by_base_text_id" if metadata else "fallback_to_base_text_id"
    metadata = metadata or {}
    source_url = str(metadata.get("source_url") or "")
    normalized_url = str(metadata.get("normalized_source_url") or "")
    if source_url and not normalized_url:
        normalized_url = source_service.normalize_url(source_url)
    group_id = normalized_url or base_id
    group_source = "normalized_source_url" if normalized_url else "base_text_id"
    if not group_id:
        status = "unresolved"
    return {
        "source_url": source_url,
        "normalized_source_url": normalized_url,
        "source_type": str(metadata.get("source_type") or ""),
        "group_id": group_id,
        "group_source": group_source,
        "metadata_recovery_status": status,
        "metadata_source_path": str(metadata.get("metadata_source_path") or ""),
    }


def build_training_dataset(
    *,
    golden_dir: Path = config.GOLDEN_DATASET_DIR,
    metadata_paths: list[Path] | None = None,
) -> tuple[pl.DataFrame, dict[str, object]]:
    golden_rows = load_golden_rows(golden_dir)
    if metadata_paths is None:
        metadata_paths = [
            config.RAW_CANDIDATE_SCHEMA_PATH,
            *sorted(
                (config.GOLDEN_DATASET_DIR).glob("*candidate_labeling_dataset.csv")
            ),
        ]
    exact, base = load_metadata_rows(metadata_paths)
    source_service = SourceBlacklistService.from_paths()
    preprocessing = PreprocessingService()
    rows: list[dict[str, object]] = []
    for row in golden_rows:
        processed = preprocessing.process(row["original_text"])["processed_text"]
        metadata = recover_metadata(
            row,
            exact=exact,
            base=base,
            source_service=source_service,
        )
        rows.append(
            {
                **row,
                "processed_text": processed,
                **metadata,
            }
        )
    df = pl.from_dicts(rows, strict=False)
    columns = [
        "text_id",
        "base_text_id",
        "original_text",
        "processed_text",
        "sentiment_label",
        "label_id",
        "location",
        "source_url",
        "normalized_source_url",
        "source_type",
        "group_id",
        "group_source",
        "metadata_recovery_status",
        "golden_file",
        "metadata_source_path",
    ]
    df = df.select([column for column in columns if column in df.columns])
    manifest = build_manifest(df, metadata_paths=metadata_paths)
    return df, manifest


def build_manifest(
    df: pl.DataFrame, *, metadata_paths: list[Path]
) -> dict[str, object]:
    label_counts = (
        Counter(df["sentiment_label"].to_list())
        if "sentiment_label" in df.columns
        else Counter()
    )
    processed = df["processed_text"].to_list() if "processed_text" in df.columns else []
    gates = {
        "total_is_281": df.height == 281,
        "positif_is_138": label_counts.get("positif", 0) == 138,
        "negatif_is_82": label_counts.get("negatif", 0) == 82,
        "netral_is_61": label_counts.get("netral", 0) == 61,
        "unknown_label_is_0": df.filter(pl.col("label_id").is_null()).height == 0,
        "empty_text_is_0": df.filter(
            pl.col("processed_text").str.strip_chars().str.len_chars() == 0
        ).height
        == 0,
        "duplicate_normalized_text_is_0": len(processed) == len(set(processed)),
        "empty_group_id_is_0": df.filter(
            pl.col("group_id").str.strip_chars().str.len_chars() == 0
        ).height
        == 0,
    }
    return {
        "row_count": df.height,
        "label_distribution": dict(label_counts),
        "metadata_recovery_distribution": dict(
            Counter(df["metadata_recovery_status"].to_list())
        ),
        "group_count": df.select(pl.col("group_id").n_unique()).item(),
        "metadata_paths": [str(path) for path in metadata_paths],
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "label_mapping": config.LABEL2ID,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build training_dataset.parquet.")
    parser.add_argument("--golden-dir", type=Path, default=config.GOLDEN_DATASET_DIR)
    parser.add_argument("--output", type=Path, default=config.TRAINING_DATASET_PATH)
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=config.ARTIFACTS / "training_dataset_manifest.json",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        action="append",
        default=None,
        help="CSV metadata tambahan. Dapat diulang.",
    )
    parser.add_argument("--allow-gate-failure", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df, manifest = build_training_dataset(
        golden_dir=args.golden_dir,
        metadata_paths=args.metadata_csv,
    )
    if not manifest["all_gates_passed"] and not args.allow_gate_failure:
        raise SystemExit(f"Dataset gate gagal: {manifest['gates']}")
    artifact = ArtifactService()
    artifact.save_parquet(df, args.output)
    artifact.save_json(manifest, args.manifest_output)
    print(f"Training dataset: {args.output}")
    print(f"Manifest: {args.manifest_output}")


if __name__ == "__main__":
    main()
