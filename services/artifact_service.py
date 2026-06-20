"""Helper penyimpanan artefak hasil eksperimen."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

import config


class ArtifactService:
    """Simpan DataFrame dan metadata eksperimen ke folder outputs."""

    def save_dataframe(self, df: pl.DataFrame, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_csv(path)
        return path

    def save_json(self, payload: dict, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding=config.ENCODING,
        )
        return path
