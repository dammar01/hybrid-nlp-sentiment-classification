"""Helper penyimpanan artefak hasil eksperimen."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl

import config


class ArtifactService:
    """Simpan DataFrame dan metadata eksperimen ke folder outputs."""

    def experiment_dir(self, experiment_id: str) -> Path:
        if not str(experiment_id).strip():
            raise ValueError("experiment_id wajib diisi")
        path = Path(config.EXPERIMENT_CONFIG["artifact_root"]) / str(experiment_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_dataframe(self, df: pl.DataFrame, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_csv(path)
        return path

    def save_csv(self, df: pl.DataFrame, path: str | Path) -> Path:
        return self.save_dataframe(df, path)

    def save_parquet(self, df: pl.DataFrame, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)
        return path

    def load_parquet(self, path: str | Path) -> pl.DataFrame:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Parquet tidak ditemukan: {path}")
        return pl.read_parquet(path)

    def save_json(self, payload: dict[str, Any], path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding=config.ENCODING,
        )
        return path

    def load_json(self, path: str | Path) -> dict[str, Any]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"JSON artifact tidak ditemukan: {path}")
        data = json.loads(path.read_text(encoding=config.ENCODING))
        if not isinstance(data, dict):
            raise ValueError(f"JSON artifact harus object: {path}")
        return data

    def file_sha256(self, path: str | Path) -> str:
        path = Path(path)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def dataframe_manifest(
        self,
        df: pl.DataFrame,
        *,
        source_path: str | Path | None = None,
    ) -> dict[str, Any]:
        manifest: dict[str, Any] = {
            "row_count": df.height,
            "column_count": len(df.columns),
            "columns": df.columns,
        }
        if source_path is not None:
            path = Path(source_path)
            manifest["source_path"] = str(path)
            if path.exists():
                manifest["source_sha256"] = self.file_sha256(path)
        return manifest

    def config_snapshot(self) -> dict[str, Any]:
        return {
            "label2id": config.LABEL2ID,
            "id2label": {str(key): value for key, value in config.ID2LABEL.items()},
            "global_seed": config.GLOBAL_SEED,
            "training_config": config.TRAINING_CONFIG,
            "split_config": config.SPLIT_CONFIG,
            "calibration_config": config.CALIBRATION_CONFIG,
            "fusion_policy_grid": config.FUSION_POLICY_GRID,
            "uncertainty_weight_grid": list(config.UNCERTAINTY_WEIGHT_GRID),
            "preprocessing_resources": {
                "slang": str(config.SLANG_DICT_PATH),
                "non_standard": str(config.NON_STANDARD_DICT_PATH),
            },
            "rule_resources": {
                "metadata": str(config.SOCAL_METADATA_PATH),
                "word_rules": str(config.SOCAL_WORD_RULES_PATH),
                "phrase_rules": str(config.SOCAL_PHRASE_RULES_PATH),
                "modifiers": str(config.SOCAL_MODIFIERS_PATH),
            },
        }
