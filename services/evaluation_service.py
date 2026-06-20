"""Evaluasi klasifikasi sentimen tiga kelas."""

from __future__ import annotations

from collections import Counter

import polars as pl

import config


class EvaluationService:
    """Hitung confusion matrix, precision, recall, dan balanced accuracy."""

    def __init__(self, labels: tuple[str, ...] = config.SENTIMENT_LABELS) -> None:
        self.labels = labels

    @staticmethod
    def rating_to_label(rating: int | float | str | None) -> str:
        if rating is None:
            return "netral"
        try:
            value = float(rating)
        except (TypeError, ValueError):
            return "netral"
        if value >= 4:
            return "positif"
        if value <= 2:
            return "negatif"
        return "netral"

    def add_actual_label_from_rating(
        self,
        df: pl.DataFrame,
        rating_column: str = "rating",
        output_column: str = config.COL_ACTUAL_LABEL,
    ) -> pl.DataFrame:
        if rating_column not in df.columns:
            raise KeyError(f"Kolom '{rating_column}' tidak ada pada DataFrame")
        labels = [self.rating_to_label(value) for value in df[rating_column].to_list()]
        return df.with_columns(pl.Series(output_column, labels))

    def evaluate_dataframe(
        self,
        df: pl.DataFrame,
        actual_column: str = config.COL_ACTUAL_LABEL,
        predicted_column: str = config.COL_FINAL_LABEL,
    ) -> dict[str, object]:
        missing = [
            column for column in (actual_column, predicted_column) if column not in df.columns
        ]
        if missing:
            raise KeyError(f"Kolom evaluasi hilang: {missing}")

        actual = df[actual_column].to_list()
        predicted = df[predicted_column].to_list()
        matrix = self._confusion_matrix(actual, predicted)

        per_label: dict[str, dict[str, float]] = {}
        recalls: list[float] = []
        for label in self.labels:
            tp = matrix[label][label]
            fp = sum(matrix[other][label] for other in self.labels if other != label)
            fn = sum(matrix[label][other] for other in self.labels if other != label)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            recalls.append(recall)
            per_label[label] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "support": sum(matrix[label].values()),
            }

        total = len(actual)
        correct = sum(1 for a, p in zip(actual, predicted) if a == p)
        return {
            "total": total,
            "accuracy": round(correct / total, 4) if total else 0.0,
            "balanced_accuracy": round(sum(recalls) / len(self.labels), 4),
            "per_label": per_label,
            "confusion_matrix": matrix,
            "actual_distribution": dict(Counter(actual)),
            "predicted_distribution": dict(Counter(predicted)),
        }

    def _confusion_matrix(
        self, actual: list[str], predicted: list[str]
    ) -> dict[str, dict[str, int]]:
        matrix = {
            label: {predicted_label: 0 for predicted_label in self.labels}
            for label in self.labels
        }
        for actual_label, predicted_label in zip(actual, predicted):
            if actual_label not in matrix:
                continue
            if predicted_label not in matrix[actual_label]:
                predicted_label = "netral"
            matrix[actual_label][predicted_label] += 1
        return matrix
