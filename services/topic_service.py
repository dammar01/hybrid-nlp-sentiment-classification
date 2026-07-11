"""Ekstraksi topik dari hasil klaster HDBSCAN.

Keyword topik diperoleh dari **term/frasa yang paling sering muncul** pada tiap
klaster (raw frequency n-gram). Seluruh opini dalam satu klaster digabung, lalu
dihitung frekuensi tiap term (dengan penyaringan stopword), dan top-N term
tersering menjadi keyword topik. Menyediakan pula ringkasan sentimen, lokasi
dominan, dan opini representatif per topik.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import polars as pl

import config

# Stopword bahasa Indonesia ringkas untuk menyaring keyword non-informatif.
_ID_STOPWORDS: frozenset[str] = frozenset(
    """
    yang di ke dari dan atau untuk pada dengan dalam ini itu adalah akan telah
    sudah tidak juga saja karena agar oleh sebagai para kami kita mereka saya
    anda dia nya ada tak bukan belum jika maka namun tetapi serta hingga sampai
    saat ketika setelah sebelum antara secara lebih paling sangat bisa dapat
    harus akan masih hanya pun lah kah yaitu yakni ialah tersebut sini situ
    mana apa siapa bagaimana kapan dimana kenapa mengapa nya per se
    """.split()
)


class TopicService:
    """Turunkan keyword dan ringkasan topik dari klaster."""

    def __init__(
        self,
        top_keywords: int = int(config.TOPIC_CONFIG["top_keywords"]),
        ngram_max: int = int(config.TOPIC_CONFIG["keyword_ngram_max"]),
        representative_docs: int = int(config.TOPIC_CONFIG["representative_docs"]),
    ) -> None:
        self.top_keywords = top_keywords
        self.ngram_max = ngram_max
        self.representative_docs = representative_docs

    def extract_keywords(
        self, texts: Sequence[str], cluster_ids: Sequence[int]
    ) -> dict[int, list[str]]:
        """Keyword = term/frasa paling sering muncul per klaster (kecuali noise -1)."""
        from sklearn.feature_extraction.text import CountVectorizer

        grouped: dict[int, list[str]] = {}
        for text, cluster_id in zip(texts, cluster_ids):
            if cluster_id == -1:
                continue
            grouped.setdefault(int(cluster_id), []).append(str(text or ""))
        if not grouped:
            return {}

        cluster_order = sorted(grouped)
        class_documents = [" ".join(grouped[cid]) for cid in cluster_order]

        # CountVectorizer -> frekuensi mentah tiap n-gram per klaster.
        vectorizer = CountVectorizer(
            ngram_range=(1, max(1, self.ngram_max)),
            stop_words=list(_ID_STOPWORDS),
            token_pattern=r"[A-Za-zÀ-ſ]{3,}(?:['\-][A-Za-zÀ-ſ]+)*",
            min_df=1,
        )
        matrix = vectorizer.fit_transform(class_documents)
        terms = vectorizer.get_feature_names_out()

        keywords: dict[int, list[str]] = {}
        for row_index, cluster_id in enumerate(cluster_order):
            counts = matrix.getrow(row_index).toarray().ravel()
            if counts.size == 0:
                keywords[cluster_id] = []
                continue
            # urutkan berdasarkan frekuensi tertinggi
            top_index = counts.argsort()[::-1][: self.top_keywords]
            keywords[cluster_id] = [
                str(terms[i]) for i in top_index if counts[i] > 0
            ]
        return keywords

    def summarize(
        self,
        df: pl.DataFrame,
        *,
        text_column: str = "original_text",
        cluster_column: str = config.COL_CLUSTER_ID,
        sentiment_column: str = "final_sentiment",
        location_column: str = "location",
        probability_column: str = "cluster_probability",
    ) -> dict[str, Any]:
        """Bangun ringkasan topik: keyword, sentimen, lokasi, contoh opini."""
        if cluster_column not in df.columns:
            raise KeyError(f"Kolom klaster hilang: {cluster_column}")
        text_column = text_column if text_column in df.columns else config.COL_PROCESSED

        texts = df[text_column].to_list()
        cluster_ids = [int(value) for value in df[cluster_column].to_list()]
        keywords = self.extract_keywords(texts, cluster_ids)

        sentiments = (
            df[sentiment_column].to_list()
            if sentiment_column in df.columns
            else [None] * df.height
        )
        locations = (
            df[location_column].to_list()
            if location_column in df.columns
            else [None] * df.height
        )
        probabilities = (
            [float(value or 0.0) for value in df[probability_column].to_list()]
            if probability_column in df.columns
            else [0.0] * df.height
        )

        topics: list[dict[str, Any]] = []
        for cluster_id in sorted(set(cid for cid in cluster_ids if cid != -1)):
            member_index = [i for i, cid in enumerate(cluster_ids) if cid == cluster_id]
            sent_dist = dict(
                Counter(sentiments[i] for i in member_index if sentiments[i])
            )
            dominant = (
                max(sent_dist.items(), key=lambda kv: kv[1])[0] if sent_dist else "netral"
            )
            loc_dist = Counter(locations[i] for i in member_index if locations[i])
            # opini representatif = probabilitas keanggotaan tertinggi
            ranked = sorted(member_index, key=lambda i: probabilities[i], reverse=True)
            reps = [str(texts[i]) for i in ranked[: self.representative_docs]]
            topics.append(
                {
                    "cluster_id": cluster_id,
                    "size": len(member_index),
                    "keywords": keywords.get(cluster_id, []),
                    "sentiment_distribution": sent_dist,
                    "dominant_sentiment": dominant,
                    "top_locations": loc_dist.most_common(5),
                    "representative_docs": reps,
                }
            )
        topics.sort(key=lambda item: item["size"], reverse=True)

        noise_count = sum(1 for cid in cluster_ids if cid == -1)
        return {
            "n_topics": len(topics),
            "n_documents": df.height,
            "n_noise": noise_count,
            "noise_ratio": round(noise_count / df.height, 4) if df.height else 0.0,
            "topics": topics,
        }
