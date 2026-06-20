"""Deteksi ambiguitas dan keputusan final tanpa LLM."""

from __future__ import annotations

import polars as pl

import config


class AmbiguityService:
    """Menandai opini yang belum layak diputuskan otomatis."""

    def __init__(
        self,
        alpha: float = config.RULE_CONFIDENCE_ALPHA,
        beta: float = config.SEMANTIC_SIMILARITY_BETA,
    ) -> None:
        self.alpha = alpha
        self.beta = beta

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
