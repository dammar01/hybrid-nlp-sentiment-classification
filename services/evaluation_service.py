"""Evaluasi klasifikasi sentimen tiga kelas."""

from __future__ import annotations

from collections import Counter
import math
from typing import Sequence

import polars as pl

import config


class EvaluationService:
    """Hitung metric klasifikasi dan probabilitas tiga kelas."""

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
        return self.evaluate_predictions(actual, predicted)

    def evaluate_predictions(
        self,
        actual: Sequence[str],
        predicted: Sequence[str],
        *,
        probabilities: Sequence[Sequence[float]] | None = None,
    ) -> dict[str, object]:
        matrix = self._confusion_matrix(list(actual), list(predicted))
        per_label, macro_f1, weighted_f1, balanced_accuracy = self._per_label_metrics(
            matrix
        )
        total = len(actual)
        correct = sum(1 for a, p in zip(actual, predicted) if a == p)
        result: dict[str, object] = {
            "total": total,
            "accuracy": round(correct / total, 4) if total else 0.0,
            "balanced_accuracy": round(balanced_accuracy, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "per_label": per_label,
            "confusion_matrix": matrix,
            "actual_distribution": dict(Counter(actual)),
            "predicted_distribution": dict(Counter(predicted)),
        }
        if probabilities is not None:
            y_true = [config.LABEL2ID.get(label, config.LABEL2ID["netral"]) for label in actual]
            result.update(self.evaluate_probabilities(y_true, probabilities))
        return result

    def evaluate_components(
        self,
        df: pl.DataFrame,
        *,
        actual_column: str = "sentiment_label",
        bert_column: str = "bert_label",
        rule_column: str = config.COL_RULE_LABEL,
        final_column: str = "final_sentiment",
    ) -> dict[str, object]:
        missing = [
            column
            for column in (actual_column, bert_column, rule_column, final_column)
            if column not in df.columns
        ]
        if missing:
            raise KeyError(f"Kolom evaluasi komponen hilang: {missing}")
        actual = df[actual_column].to_list()
        probability_columns = [
            "bert_prob_negatif",
            "bert_prob_netral",
            "bert_prob_positif",
        ]
        probabilities = (
            df.select(probability_columns).rows()
            if all(column in df.columns for column in probability_columns)
            else None
        )
        return {
            "indobert": self.evaluate_predictions(
                actual,
                df[bert_column].to_list(),
                probabilities=probabilities,
            ),
            "rule_based": self.evaluate_predictions(actual, df[rule_column].to_list()),
            "final_hybrid": self.evaluate_predictions(actual, df[final_column].to_list()),
        }

    def evaluate_probabilities(
        self,
        y_true: Sequence[int],
        probabilities: Sequence[Sequence[float]],
        *,
        ece_bins: int | None = None,
    ) -> dict[str, float]:
        return {
            "nll": round(self.negative_log_likelihood(y_true, probabilities), 6),
            "ece": round(
                self.expected_calibration_error(
                    y_true,
                    probabilities,
                    bins=ece_bins or int(config.CALIBRATION_CONFIG["ece_bins"]),
                ),
                6,
            ),
            "multiclass_brier_score": round(
                self.multiclass_brier_score(y_true, probabilities),
                6,
            ),
        }

    @staticmethod
    def negative_log_likelihood(
        y_true: Sequence[int],
        probabilities: Sequence[Sequence[float]],
    ) -> float:
        if not y_true:
            return 0.0
        eps = 1e-12
        losses = []
        for target, probs in zip(y_true, probabilities):
            value = max(min(float(probs[int(target)]), 1.0), eps)
            losses.append(-math.log(value))
        return sum(losses) / len(losses)

    @staticmethod
    def multiclass_brier_score(
        y_true: Sequence[int],
        probabilities: Sequence[Sequence[float]],
    ) -> float:
        if not y_true:
            return 0.0
        total = 0.0
        for target, probs in zip(y_true, probabilities):
            for index, prob in enumerate(probs):
                expected = 1.0 if index == int(target) else 0.0
                total += (float(prob) - expected) ** 2
        return total / len(y_true)

    @staticmethod
    def expected_calibration_error(
        y_true: Sequence[int],
        probabilities: Sequence[Sequence[float]],
        *,
        bins: int = 10,
    ) -> float:
        if not y_true:
            return 0.0
        bin_totals = [0 for _ in range(bins)]
        bin_confidence = [0.0 for _ in range(bins)]
        bin_accuracy = [0.0 for _ in range(bins)]
        for target, probs in zip(y_true, probabilities):
            values = [float(prob) for prob in probs]
            confidence = max(values)
            prediction = values.index(confidence)
            index = min(int(confidence * bins), bins - 1)
            bin_totals[index] += 1
            bin_confidence[index] += confidence
            bin_accuracy[index] += 1.0 if prediction == int(target) else 0.0

        total = len(y_true)
        ece = 0.0
        for index, count in enumerate(bin_totals):
            if not count:
                continue
            avg_confidence = bin_confidence[index] / count
            avg_accuracy = bin_accuracy[index] / count
            ece += (count / total) * abs(avg_accuracy - avg_confidence)
        return ece

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

    def _per_label_metrics(
        self,
        matrix: dict[str, dict[str, int]],
    ) -> tuple[dict[str, dict[str, float]], float, float, float]:
        per_label: dict[str, dict[str, float]] = {}
        recalls: list[float] = []
        f1_values: list[float] = []
        weighted_f1_sum = 0.0
        total_support = 0
        for label in self.labels:
            tp = matrix[label][label]
            fp = sum(matrix[other][label] for other in self.labels if other != label)
            fn = sum(matrix[label][other] for other in self.labels if other != label)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            support = sum(matrix[label].values())
            recalls.append(recall)
            f1_values.append(f1)
            weighted_f1_sum += f1 * support
            total_support += support
            per_label[label] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": support,
            }
        macro_f1 = sum(f1_values) / len(self.labels) if self.labels else 0.0
        weighted_f1 = weighted_f1_sum / total_support if total_support else 0.0
        balanced_accuracy = sum(recalls) / len(self.labels) if self.labels else 0.0
        return per_label, macro_f1, weighted_f1, balanced_accuracy
