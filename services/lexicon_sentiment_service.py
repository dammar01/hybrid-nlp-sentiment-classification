"""Rule-based sentiment service untuk skenario Hybrid NLP tanpa LLM."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import polars as pl

import config

_TOKEN_RE = re.compile(r"[A-Za-zÀ-ſ]+(?:['\-][A-Za-zÀ-ſ]+)*")


@dataclass(slots=True)
class LexiconSentimentService:
    """Analisis sentimen berbasis lexicon dengan penanganan negasi sederhana."""

    positive_words: set[str] = field(default_factory=set)
    negative_words: set[str] = field(default_factory=set)
    negation_words: set[str] = field(default_factory=lambda: set(config.NEGATION_WORDS))

    def __post_init__(self) -> None:
        if not self.positive_words:
            self.positive_words = {
                "aman", "baik", "bagus", "cepat", "cocok", "enak", "hebat",
                "jelas", "lancar", "mantap", "memuaskan", "murah", "oke",
                "puas", "rapi", "rekomendasi", "sesuai", "suka", "terbaik",
                "top",
            }
        if not self.negative_words:
            self.negative_words = {
                "buruk", "cacat", "gagal", "jelek", "kecewa", "kurang",
                "lambat", "mahal", "parah", "pecah", "payah", "ribet",
                "rusak", "salah", "telat", "tidak", "zonk",
            }

    def analyze_text(self, text: str) -> dict[str, object]:
        tokens = [token.lower() for token in _TOKEN_RE.findall(text or "")]
        positive = 0
        negative = 0
        hits: list[str] = []
        invert_next = False

        for token in tokens:
            if token in self.negation_words:
                invert_next = True
                continue

            polarity = 0
            if token in self.positive_words:
                polarity = 1
            elif token in self.negative_words:
                polarity = -1

            if polarity == 0:
                invert_next = False
                continue

            if invert_next:
                polarity *= -1
                invert_next = False

            if polarity > 0:
                positive += 1
            else:
                negative += 1
            hits.append(token)

        score = positive - negative
        total_hits = positive + negative
        confidence = abs(score) / total_hits if total_hits else 0.0
        if score > 0:
            label = "positif"
        elif score < 0:
            label = "negatif"
        else:
            label = "netral"

        return {
            config.COL_RULE_LABEL: label,
            "rule_score": score,
            config.COL_RULE_CONFIDENCE: round(confidence, 4),
            "rule_positive_count": positive,
            "rule_negative_count": negative,
            "rule_hits": ", ".join(hits),
        }

    def analyze_dataframe(
        self, df: pl.DataFrame, text_column: str = config.COL_PROCESSED
    ) -> pl.DataFrame:
        if text_column not in df.columns:
            raise KeyError(f"Kolom '{text_column}' tidak ada pada DataFrame")

        rows = pl.DataFrame(
            [self.analyze_text(text) for text in df[text_column].to_list()]
        )
        return df.hstack(rows)
