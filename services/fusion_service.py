"""Conditional decision-level fusion untuk IndoBERT + rule engine."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import polars as pl

import config
from services.ambiguity_service import AmbiguityService
from services.evaluation_service import EvaluationService


@dataclass(slots=True)
class FusionService:
    """Gabungkan prediksi BERT dan SO-CAL-inspired rule secara deterministik."""

    policy: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.policy is None:
            self.policy = self.default_policy()

    @staticmethod
    def default_policy() -> dict[str, Any]:
        return {
            "high_confidence_threshold": 0.80,
            "low_confidence_threshold": 0.55,
            "rule_confidence_threshold": config.RULE_WEAK_THRESHOLD,
            "uncertainty_review_threshold": 0.70,
            "uncertainty_weights": {"wc": 0.40, "wm": 0.40, "wd": 0.20},
        }

    def fuse_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        df = self._ensure_uncertainty(df)
        required = (
            "bert_label",
            "bert_confidence",
            config.COL_RULE_LABEL,
            config.COL_RULE_CONFIDENCE,
            config.COL_RULE_STATUS,
            config.COL_RULE_EVIDENCE,
            "routing_uncertainty_score",
        )
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise KeyError(f"Kolom wajib fusion hilang: {missing}")

        rows = pl.DataFrame([self.fuse_row(row) for row in df.iter_rows(named=True)])
        return df.hstack(rows)

    def fuse_row(self, row: dict[str, Any]) -> dict[str, object]:
        bert_label = str(row.get("bert_label") or "netral")
        rule_label = str(row.get(config.COL_RULE_LABEL) or "netral")
        bert_confidence = self._float(row.get("bert_confidence"))
        rule_confidence = self._float(row.get(config.COL_RULE_CONFIDENCE))
        uncertainty = self._float(row.get("routing_uncertainty_score"))
        rule_valid = AmbiguityService.rule_has_valid_evidence(row)
        high_threshold = float(self.policy["high_confidence_threshold"])
        low_threshold = float(self.policy["low_confidence_threshold"])
        rule_threshold = float(self.policy["rule_confidence_threshold"])
        review_threshold = float(self.policy["uncertainty_review_threshold"])

        if bert_label == rule_label and rule_valid:
            final_label = bert_label
            action = "agreement"
            reason = "IndoBERT dan rule valid sepakat."
        elif not rule_valid:
            final_label = bert_label
            action = "bert_rule_invalid"
            reason = "Rule weak/unknown/tanpa evidence valid; gunakan IndoBERT."
        elif bert_confidence >= high_threshold:
            final_label = bert_label
            action = "bert_high_confidence"
            reason = "Confidence IndoBERT tinggi; gunakan IndoBERT."
        elif (
            bert_confidence <= low_threshold
            and rule_confidence >= rule_threshold
            and bert_label != rule_label
            and rule_label != "netral"
        ):
            final_label = rule_label
            action = "rule_override"
            reason = "IndoBERT rendah, rule evidence kuat, dan prediksi konflik."
        else:
            final_label = bert_label
            action = "bert_default"
            reason = "Tidak memenuhi syarat override; gunakan IndoBERT."

        return {
            "fusion_action": action,
            "fusion_reason": reason,
            "final_sentiment": final_label,
            "needs_review": bool(uncertainty >= review_threshold),
        }

    def select_policy(
        self,
        calibration_df: pl.DataFrame,
        *,
        actual_column: str = "sentiment_label",
    ) -> dict[str, Any]:
        if actual_column not in calibration_df.columns:
            raise KeyError(f"Kolom label calibration hilang: {actual_column}")

        evaluator = EvaluationService()
        candidates: list[dict[str, Any]] = []
        for weights in config.UNCERTAINTY_WEIGHT_GRID:
            scored = AmbiguityService(weights=dict(weights)).score_dataframe(
                self._drop_existing_uncertainty(calibration_df)
            )
            for high, low, rule, review in product(
                config.FUSION_POLICY_GRID["high_confidence_threshold"],
                config.FUSION_POLICY_GRID["low_confidence_threshold"],
                config.FUSION_POLICY_GRID["rule_confidence_threshold"],
                config.FUSION_POLICY_GRID["uncertainty_review_threshold"],
            ):
                if low >= high:
                    continue
                policy = {
                    "high_confidence_threshold": high,
                    "low_confidence_threshold": low,
                    "rule_confidence_threshold": rule,
                    "uncertainty_review_threshold": review,
                    "uncertainty_weights": dict(weights),
                }
                fused = FusionService(policy=policy).fuse_dataframe(scored)
                metrics = evaluator.evaluate_predictions(
                    fused[actual_column].to_list(),
                    fused["final_sentiment"].to_list(),
                )
                override_count = int(
                    fused.filter(pl.col("fusion_action") == "rule_override").height
                )
                candidates.append(
                    {
                        "policy": policy,
                        "balanced_accuracy": float(metrics["balanced_accuracy"]),
                        "macro_f1": float(metrics["macro_f1"]),
                        "override_count": override_count,
                        "metrics": metrics,
                    }
                )

        if not candidates:
            raise RuntimeError("Tidak ada kandidat fusion policy yang valid")

        candidates.sort(
            key=lambda item: (
                -item["balanced_accuracy"],
                -item["macro_f1"],
                item["override_count"],
                -item["policy"]["high_confidence_threshold"],
                item["policy"]["low_confidence_threshold"],
                item["policy"]["rule_confidence_threshold"],
                item["policy"]["uncertainty_review_threshold"],
            )
        )
        selected = candidates[0]
        return {
            **selected["policy"],
            "selection_metric": {
                "balanced_accuracy": selected["balanced_accuracy"],
                "macro_f1": selected["macro_f1"],
                "override_count": selected["override_count"],
            },
            "candidate_count": len(candidates),
        }

    def _ensure_uncertainty(self, df: pl.DataFrame) -> pl.DataFrame:
        if "routing_uncertainty_score" in df.columns:
            return df
        weights = dict(self.policy.get("uncertainty_weights") or {})
        return AmbiguityService(weights=weights).score_dataframe(df)

    @staticmethod
    def _drop_existing_uncertainty(df: pl.DataFrame) -> pl.DataFrame:
        columns = [
            "cross_method_conflict",
            "confidence_uncertainty",
            "margin_uncertainty",
            "normalized_entropy",
            "routing_uncertainty_score",
            "uncertainty_weight_wc",
            "uncertainty_weight_wm",
            "uncertainty_weight_wd",
        ]
        return df.drop([column for column in columns if column in df.columns])

    @staticmethod
    def _float(value: object) -> float:
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0
