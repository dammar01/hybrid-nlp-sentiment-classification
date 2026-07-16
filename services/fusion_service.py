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
        """Fusi simetris IndoBERT <-> rule tanpa pembebanan salah satu metode.

        Prinsip: metode yang *yakin* yang menentukan. `rule_valid` (status
        `detected`) sudah dikalibrasi ke precision-on-fired 100% pada golden,
        jadi rule detected setara-yakin dengan IndoBERT high-confidence.
        - rule abstain (weak/unknown)  -> hanya IndoBERT punya sinyal
        - sepakat                       -> label yang disepakati
        - konflik & IndoBERT high-conf  -> dua-duanya yakin -> tandai review/LLM
        - konflik & IndoBERT belum yakin-> rule detected (100% precise) unggul
        """
        bert_label = str(row.get("bert_label") or "netral")
        rule_label = str(row.get(config.COL_RULE_LABEL) or "netral")
        bert_confidence = self._float(row.get("bert_confidence"))
        uncertainty = self._float(row.get("routing_uncertainty_score"))
        rule_valid = AmbiguityService.rule_has_valid_evidence(row)
        high_threshold = float(self.policy["high_confidence_threshold"])
        review_threshold = float(self.policy["uncertainty_review_threshold"])

        conflict = False
        requires_llm = False
        if not rule_valid:
            final_label = bert_label
            action = "bert_only"
            reason = "Rule abstain (weak/unknown); hanya IndoBERT bersinyal."
        elif bert_label == rule_label:
            final_label = bert_label
            action = "agreement"
            reason = "IndoBERT dan rule detected sepakat."
        else:
            conflict = True
            if bert_confidence >= high_threshold:
                final_label = bert_label
                action = "conflict_both_confident"
                reason = (
                    "IndoBERT high-confidence dan rule detected sama-sama yakin "
                    "namun berbeda; butuh adjudikasi (LLM pada skenario-2)."
                )
                requires_llm = True
            else:
                final_label = rule_label
                action = "rule_confident"
                reason = (
                    "Rule detected (kalibrasi precision 100%) unggul; "
                    "IndoBERT belum cukup yakin."
                )

        needs_review = bool(uncertainty >= review_threshold) or requires_llm
        return {
            "fusion_action": action,
            "fusion_reason": reason,
            "final_sentiment": final_label,
            "cross_method_conflict_final": conflict,
            "requires_llm": requires_llm,
            "needs_review": needs_review,
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
            for high, review in product(
                config.FUSION_POLICY_GRID["high_confidence_threshold"],
                config.FUSION_POLICY_GRID["uncertainty_review_threshold"],
            ):
                policy = {
                    "high_confidence_threshold": high,
                    "uncertainty_review_threshold": review,
                    "uncertainty_weights": dict(weights),
                }
                fused = FusionService(policy=policy).fuse_dataframe(scored)
                metrics = evaluator.evaluate_predictions(
                    fused[actual_column].to_list(),
                    fused["final_sentiment"].to_list(),
                )
                changed_from_indobert_count = int(
                    fused.filter(
                        pl.col("final_sentiment") != pl.col("bert_label")
                    ).height
                )
                error_rows = fused.filter(
                    pl.col("final_sentiment") != pl.col(actual_column)
                )
                reviewed_error_count = int(
                    error_rows.filter(pl.col("needs_review") == True).height
                )
                error_count = int(error_rows.height)
                review_count = int(
                    fused.filter(pl.col("needs_review") == True).height
                )
                candidates.append(
                    {
                        "policy": policy,
                        "balanced_accuracy": float(metrics["balanced_accuracy"]),
                        "macro_f1": float(metrics["macro_f1"]),
                        "changed_from_indobert_count": changed_from_indobert_count,
                        "error_capture_rate": (
                            reviewed_error_count / error_count if error_count else 1.0
                        ),
                        "review_rate": review_count / fused.height if fused.height else 0.0,
                        "metrics": metrics,
                    }
                )

        if not candidates:
            raise RuntimeError("Tidak ada kandidat fusion policy yang valid")

        candidates.sort(
            key=lambda item: (
                -item["balanced_accuracy"],
                -item["macro_f1"],
                -item["error_capture_rate"],
                item["review_rate"],
                item["changed_from_indobert_count"],
                -item["policy"]["high_confidence_threshold"],
                item["policy"]["uncertainty_review_threshold"],
            )
        )
        selected = candidates[0]
        return {
            **selected["policy"],
            "selection_metric": {
                "balanced_accuracy": selected["balanced_accuracy"],
                "macro_f1": selected["macro_f1"],
                "changed_from_indobert_count": selected[
                    "changed_from_indobert_count"
                ],
                "error_capture_rate": round(selected["error_capture_rate"], 6),
                "review_rate": round(selected["review_rate"], 6),
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
