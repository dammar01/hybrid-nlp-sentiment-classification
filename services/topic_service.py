"""Ekstraksi topik dari hasil klaster HDBSCAN.

Keyword topik diperoleh dari **term/frasa yang paling sering muncul** pada tiap
klaster (raw frequency n-gram). Seluruh opini dalam satu klaster digabung, lalu
dihitung frekuensi tiap term (dengan penyaringan stopword), dan top-N term
tersering menjadi keyword topik. Menyediakan pula ringkasan sentimen, lokasi
dominan, dan opini representatif per topik.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Sequence

import polars as pl

import config

_TOPIC_KEYWORD_RESOURCE_KEYS: tuple[str, ...] = (
    "hard_blacklist_tokens",
    "generic_tokens",
    "stopwords",
    "exact_blacklist",
    "platform_noise_fragments",
    "token_noise_fragments",
)


def _normalize_keyword(keyword: str) -> str:
    return " ".join(str(keyword).casefold().split())


def _load_topic_keyword_terms(raw: dict[str, Any], key: str) -> tuple[str, ...]:
    values = raw.get(key)
    if not isinstance(values, list) or not all(
        isinstance(value, str) for value in values
    ):
        raise ValueError(
            f"Resource keyword topik harus berisi list string untuk key '{key}'"
        )
    normalized_values = [_normalize_keyword(value) for value in values]
    return tuple(dict.fromkeys(value for value in normalized_values if value))


def _load_topic_keyword_resource() -> dict[str, tuple[str, ...]]:
    path = config.TOPIC_KEYWORD_BLACKLIST_PATH
    raw = json.loads(path.read_text(encoding=config.ENCODING))
    if not isinstance(raw, dict):
        raise ValueError(f"Resource keyword topik harus berupa object JSON: {path}")
    missing = [key for key in _TOPIC_KEYWORD_RESOURCE_KEYS if key not in raw]
    if missing:
        raise KeyError(f"Key resource keyword topik hilang: {missing}")
    return {
        key: _load_topic_keyword_terms(raw, key)
        for key in _TOPIC_KEYWORD_RESOURCE_KEYS
    }


_TOPIC_KEYWORD_RESOURCE = _load_topic_keyword_resource()
_TOPIC_HARD_BLACKLIST_TOKENS: frozenset[str] = frozenset(
    _TOPIC_KEYWORD_RESOURCE["hard_blacklist_tokens"]
)
_TOPIC_GENERIC_TOKENS: frozenset[str] = frozenset(
    _TOPIC_KEYWORD_RESOURCE["generic_tokens"]
)
_TOPIC_STOPWORDS: frozenset[str] = (
    frozenset(_TOPIC_KEYWORD_RESOURCE["stopwords"])
    | _TOPIC_HARD_BLACKLIST_TOKENS
    | _TOPIC_GENERIC_TOKENS
)
_TOPIC_EXACT_BLACKLIST: frozenset[str] = frozenset(
    _TOPIC_KEYWORD_RESOURCE["exact_blacklist"]
)
_PLATFORM_NOISE_FRAGMENTS: tuple[str, ...] = _TOPIC_KEYWORD_RESOURCE[
    "platform_noise_fragments"
]
_TOKEN_NOISE_FRAGMENTS: tuple[str, ...] = _TOPIC_KEYWORD_RESOURCE[
    "token_noise_fragments"
]


def _is_noise_token(token: str) -> bool:
    return (
        token in _TOPIC_HARD_BLACKLIST_TOKENS
        or not token.isascii()
        or len(token) > 24
        or "'" in token
        or token.endswith("see")
        or token.endswith("up")
        or any(fragment in token for fragment in _TOKEN_NOISE_FRAGMENTS)
    )


def _is_informative_keyword(keyword: str) -> bool:
    normalized = _normalize_keyword(keyword)
    tokens = normalized.split()
    content_tokens = [token for token in tokens if token not in _TOPIC_STOPWORDS]
    return (
        bool(tokens)
        and normalized not in _TOPIC_EXACT_BLACKLIST
        and not any(fragment in normalized for fragment in _PLATFORM_NOISE_FRAGMENTS)
        and not any(_is_noise_token(token) for token in tokens)
        and bool(content_tokens)
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
            stop_words=sorted(_TOPIC_STOPWORDS),
            token_pattern=r"(?u)\b[^\W\d_]{3,}\b",
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
            # Ambil top-N setelah noise dibuang agar keyword informatif tidak
            # kehilangan slot karena boilerplate berfrekuensi tinggi.
            selected: list[str] = []
            for term_index in counts.argsort()[::-1]:
                if counts[term_index] <= 0:
                    break
                keyword = str(terms[term_index])
                if not _is_informative_keyword(keyword):
                    continue
                selected.append(keyword)
                if len(selected) >= self.top_keywords:
                    break
            keywords[cluster_id] = selected
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
