"""Deteksi ambiguitas dan keputusan final tanpa LLM."""

from __future__ import annotations

import json
import math

import polars as pl

import config


class AmbiguityService:
    """Menandai opini yang belum layak diputuskan otomatis."""

    def __init__(
        self,
        alpha: float = config.RULE_CONFIDENCE_ALPHA,
        beta: float = config.SEMANTIC_SIMILARITY_BETA,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.weights = weights or {"wc": 0.40, "wm": 0.40, "wd": 0.20}

    def score_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        required = (
            "bert_label",
            "bert_confidence",
            "bert_margin",
            config.COL_RULE_LABEL,
            config.COL_RULE_STATUS,
            config.COL_RULE_EVIDENCE,
        )
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise KeyError(f"Kolom wajib uncertainty hilang: {missing}")

        rows = pl.DataFrame([self.score_row(row) for row in df.iter_rows(named=True)])
        return df.hstack(rows)

    def score_row(self, row: dict) -> dict[str, object]:
        bert_label = str(row.get("bert_label") or "")
        rule_label = str(row.get(config.COL_RULE_LABEL) or "")
        confidence = self._clamp01(row.get("bert_confidence"))
        margin = self._clamp01(row.get("bert_margin"))
        rule_has_evidence = self.rule_has_valid_evidence(row)
        disagreement = (
            1.0
            if bert_label != rule_label and rule_has_evidence
            else 0.0
        )
        confidence_uncertainty = 1.0 - confidence
        margin_uncertainty = 1.0 - margin
        entropy = self._entropy_from_row(row)
        score = (
            float(self.weights.get("wc", 0.0)) * confidence_uncertainty
            + float(self.weights.get("wm", 0.0)) * margin_uncertainty
            + float(self.weights.get("wd", 0.0)) * disagreement
        )
        return {
            "cross_method_conflict": bool(disagreement),
            "confidence_uncertainty": round(confidence_uncertainty, 6),
            "margin_uncertainty": round(margin_uncertainty, 6),
            "normalized_entropy": round(entropy, 6),
            "routing_uncertainty_score": round(score, 6),
            "uncertainty_weight_wc": float(self.weights.get("wc", 0.0)),
            "uncertainty_weight_wm": float(self.weights.get("wm", 0.0)),
            "uncertainty_weight_wd": float(self.weights.get("wd", 0.0)),
        }

    @staticmethod
    def rule_has_valid_evidence(row: dict) -> bool:
        status = str(row.get(config.COL_RULE_STATUS) or "")
        if status in {config.RULE_STATUS_WEAK, config.RULE_STATUS_UNKNOWN, ""}:
            return False
        evidence = str(row.get(config.COL_RULE_EVIDENCE) or "").strip()
        if not evidence:
            return False
        try:
            payload = json.loads(evidence)
        except json.JSONDecodeError:
            return False
        hits = payload.get("sentiment_hits") if isinstance(payload, dict) else None
        if not isinstance(hits, list):
            return False
        return any(
            isinstance(hit, dict) and float(hit.get("score") or 0.0) != 0.0
            for hit in hits
        )

    def decide_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        required = (
            config.COL_RULE_LABEL,
            config.COL_RULE_CONFIDENCE,
            config.COL_SEMANTIC_LABEL,
            config.COL_SEMANTIC_SIMILARITY,
            config.COL_CLUSTER_ID,
        )
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise KeyError(f"Kolom wajib ambiguity hilang: {missing}")

        decisions = pl.DataFrame(
            [self.decide_row(row) for row in df.iter_rows(named=True)]
        )
        return df.hstack(decisions)

    def decide_row(self, row: dict) -> dict[str, object]:
        rule_label = str(row[config.COL_RULE_LABEL])
        semantic_label = str(row[config.COL_SEMANTIC_LABEL])
        rule_confidence = float(row[config.COL_RULE_CONFIDENCE] or 0.0)
        similarity = float(row[config.COL_SEMANTIC_SIMILARITY] or 0.0)
        cluster_id = int(row[config.COL_CLUSTER_ID])

        reasons: list[str] = []
        if rule_confidence < self.alpha:
            reasons.append("low_rule_confidence")
        if similarity < self.beta:
            reasons.append("low_semantic_similarity")
        if cluster_id == -1:
            reasons.append("cluster_noise")
        if (
            rule_label != "netral"
            and semantic_label != "netral"
            and rule_label != semantic_label
        ):
            reasons.append("rule_semantic_conflict")

        is_ambiguous = bool(reasons)
        final_label = self._non_llm_final_label(rule_label, semantic_label, is_ambiguous)

        return {
            config.COL_IS_AMBIGUOUS: is_ambiguous,
            "ambiguity_reason": ", ".join(reasons),
            config.COL_FINAL_LABEL: final_label,
            "decision_source": "non_llm_ambiguous" if is_ambiguous else "automatic",
        }

    @staticmethod
    def _non_llm_final_label(
        rule_label: str, semantic_label: str, is_ambiguous: bool
    ) -> str:
        if not is_ambiguous:
            return semantic_label if semantic_label != "netral" else rule_label
        if rule_label != "netral":
            return rule_label
        if semantic_label != "netral":
            return semantic_label
        return "netral"

    @staticmethod
    def _clamp01(value: object) -> float:
        try:
            numeric = float(value or 0.0)
        except (TypeError, ValueError):
            numeric = 0.0
        return max(0.0, min(1.0, numeric))

    @classmethod
    def _entropy_from_row(cls, row: dict) -> float:
        if "bert_entropy" in row and row.get("bert_entropy") is not None:
            return cls._clamp01(row.get("bert_entropy"))
        probs = [
            cls._clamp01(row.get("bert_prob_negatif")),
            cls._clamp01(row.get("bert_prob_netral")),
            cls._clamp01(row.get("bert_prob_positif")),
        ]
        total = sum(probs)
        if total <= 0:
            return 0.0
        normalized = [prob / total for prob in probs]
        entropy = -sum(prob * math.log(prob) for prob in normalized if prob > 0)
        return entropy / math.log(len(normalized))
