"""Confidence calibration untuk IndoBERT sequence classifier."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import polars as pl

import config
from services.artifact_service import ArtifactService
from services.evaluation_service import EvaluationService


@dataclass(slots=True)
class CalibrationService:
    """Scalar temperature scaling untuk logits tiga kelas."""

    ece_bins: int = int(config.CALIBRATION_CONFIG["ece_bins"])

    def fit_temperature(
        self,
        logits: Sequence[Sequence[float]],
        y_true: Sequence[int],
    ) -> dict[str, object]:
        before_probs = self.softmax(logits, temperature=1.0)
        before_argmax = self.argmax(before_probs)
        temperature = self._optimize_temperature(logits, y_true)
        after_probs = self.softmax(logits, temperature=temperature)
        after_argmax = self.argmax(after_probs)
        if before_argmax != after_argmax:
            raise RuntimeError("Temperature scaling mengubah urutan argmax")

        evaluator = EvaluationService()
        before = evaluator.evaluate_probabilities(
            y_true,
            before_probs,
            ece_bins=self.ece_bins,
        )
        after = evaluator.evaluate_probabilities(
            y_true,
            after_probs,
            ece_bins=self.ece_bins,
        )
        return {
            "temperature": temperature,
            "before": before,
            "after": after,
            "argmax_unchanged": True,
        }

    def fit_from_dataframe(
        self,
        df: pl.DataFrame,
        *,
        label_column: str = "label_id",
        logit_columns: tuple[str, str, str] = (
            "bert_logit_negatif",
            "bert_logit_netral",
            "bert_logit_positif",
        ),
        group_column: str = "group_id",
    ) -> dict[str, object]:
        missing = [
            column for column in (label_column, group_column, *logit_columns)
            if column not in df.columns
        ]
        if missing:
            raise KeyError(f"Kolom calibration hilang: {missing}")
        logits = [[float(value) for value in row] for row in df.select(logit_columns).rows()]
        y_true = [int(value) for value in df[label_column].to_list()]
        artifact = self.fit_temperature(logits, y_true)
        artifact.update(
            {
                "row_count": df.height,
                "group_count": df.select(pl.col(group_column).n_unique()).item(),
                "label_column": label_column,
                "logit_columns": list(logit_columns),
            }
        )
        return artifact

    def apply_temperature(
        self,
        logits: Sequence[Sequence[float]],
        temperature: float,
    ) -> list[list[float]]:
        return self.softmax(logits, temperature=temperature)

    def save_artifact(self, artifact: dict[str, object], path: str | Path) -> Path:
        return ArtifactService().save_json(artifact, path)

    def load_artifact(self, path: str | Path) -> dict[str, object]:
        artifact = ArtifactService().load_json(path)
        temperature = float(artifact.get("temperature") or 0.0)
        if temperature <= 0:
            raise ValueError(f"Temperature artifact tidak valid: {path}")
        return artifact

    @staticmethod
    def softmax(
        logits: Sequence[Sequence[float]],
        *,
        temperature: float = 1.0,
    ) -> list[list[float]]:
        if temperature <= 0:
            raise ValueError("temperature harus positif")
        result: list[list[float]] = []
        for row in logits:
            scaled = [float(value) / temperature for value in row]
            max_value = max(scaled)
            exps = [math.exp(value - max_value) for value in scaled]
            total = sum(exps)
            result.append([value / total for value in exps])
        return result

    @staticmethod
    def argmax(probabilities: Sequence[Sequence[float]]) -> list[int]:
        return [list(row).index(max(row)) for row in probabilities]

    def _optimize_temperature(
        self,
        logits: Sequence[Sequence[float]],
        y_true: Sequence[int],
    ) -> float:
        def objective(temperature: float) -> float:
            probs = self.softmax(logits, temperature=temperature)
            return EvaluationService.negative_log_likelihood(y_true, probs)

        try:
            from scipy.optimize import minimize_scalar
        except ImportError:
            return self._grid_search_temperature(objective)

        result = minimize_scalar(
            objective,
            bounds=(0.05, 10.0),
            method="bounded",
            options={"maxiter": int(config.CALIBRATION_CONFIG["max_iter"])},
        )
        if not result.success:
            return self._grid_search_temperature(objective)
        return round(float(result.x), 6)

    @staticmethod
    def _grid_search_temperature(objective) -> float:
        candidates = [0.05, 0.1, 0.2, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0]
        best = min(candidates, key=objective)
        return round(float(best), 6)
