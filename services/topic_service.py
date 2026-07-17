"""Ekstraksi keyword langsung dari resource SO-CAL per kelas sentimen."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
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


def _load_json_object(path: Path, resource_name: str) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding=config.ENCODING))
    if not isinstance(raw, dict):
        raise ValueError(f"{resource_name} harus berupa object JSON: {path}")
    return raw


def _load_topic_keyword_terms(
    raw: dict[str, Any], key: str
) -> tuple[str, ...]:
    values = raw.get(key)
    if not isinstance(values, list) or not all(
        isinstance(value, str) for value in values
    ):
        raise ValueError(
            f"Resource keyword topik harus berisi list string untuk key '{key}'"
        )
    normalized = [_normalize_keyword(value) for value in values]
    return tuple(dict.fromkeys(value for value in normalized if value))


def _load_topic_keyword_resource(
    path: Path,
) -> dict[str, tuple[str, ...]]:
    raw = _load_json_object(path, "Resource keyword topik")
    missing = [key for key in _TOPIC_KEYWORD_RESOURCE_KEYS if key not in raw]
    if missing:
        raise KeyError(f"Key resource keyword topik hilang: {missing}")
    return {
        key: _load_topic_keyword_terms(raw, key)
        for key in _TOPIC_KEYWORD_RESOURCE_KEYS
    }


def _load_socal_entries(
    path: Path,
    *,
    group: str,
    text_key: str,
    source: str,
) -> tuple[str, tuple[dict[str, Any], ...]]:
    raw = _load_json_object(path, "Resource SO-CAL")
    version = str(raw.get("version") or "").strip()
    items = raw.get(group)
    if not version:
        raise ValueError(f"Field 'version' wajib tersedia di {path}")
    if not isinstance(items, list):
        raise ValueError(f"Group '{group}' wajib berupa list di {path}")

    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"Item '{group}' wajib berupa object di {path}")
        missing = [
            key
            for key in (text_key, "label", "weight", "category")
            if key not in item
        ]
        if missing:
            raise KeyError(f"Field SO-CAL hilang di {path}: {missing}")
        keyword = _normalize_keyword(item[text_key])
        label = _normalize_keyword(item["label"])
        if not keyword:
            continue
        if label not in config.SENTIMENT_LABELS:
            raise ValueError(f"Label SO-CAL tidak valid di {path}: {label}")
        identity = (keyword, label)
        if identity in seen:
            continue
        seen.add(identity)
        entries.append(
            {
                "keyword": keyword,
                "sentiment_label": label,
                "weight": float(item["weight"]),
                "category": str(item["category"]),
                "source": source,
            }
        )

    entries.sort(
        key=lambda item: (
            -len(str(item["keyword"]).split()),
            -len(str(item["keyword"])),
            str(item["keyword"]),
        )
    )
    return version, tuple(entries)


def _count_occurrences(text: str, keyword: str) -> int:
    pattern = rf"(?<!\w){re.escape(keyword)}(?!\w)"
    return len(re.findall(pattern, text, flags=re.UNICODE))


class TopicService:
    """Cocokkan keyword SO-CAL langsung pada dokumen per sentimen."""

    def __init__(
        self,
        top_keywords: int = int(config.TOPIC_CONFIG["top_keywords"]),
        representative_docs: int = int(config.TOPIC_CONFIG["representative_docs"]),
        *,
        word_rules_path: str | Path = config.SOCAL_WORD_RULES_PATH,
        phrase_rules_path: str | Path = config.SOCAL_PHRASE_RULES_PATH,
        blacklist_path: str | Path = config.TOPIC_KEYWORD_BLACKLIST_PATH,
    ) -> None:
        self.top_keywords = int(top_keywords)
        self.representative_docs = int(representative_docs)
        self.word_rules_path = Path(word_rules_path)
        self.phrase_rules_path = Path(phrase_rules_path)
        self.blacklist_path = Path(blacklist_path)

        blacklist = _load_topic_keyword_resource(self.blacklist_path)
        self.platform_noise_fragments = blacklist["platform_noise_fragments"]
        self.token_noise_fragments = blacklist["token_noise_fragments"]
        self.lexical_blacklist = frozenset(
            blacklist["hard_blacklist_tokens"]
            + blacklist["generic_tokens"]
            + blacklist["stopwords"]
            + blacklist["exact_blacklist"]
        )

        word_version, words = _load_socal_entries(
            self.word_rules_path,
            group="word_rules",
            text_key="term",
            source="socal_word",
        )
        phrase_version, phrases = _load_socal_entries(
            self.phrase_rules_path,
            group="phrase_rules",
            text_key="phrase",
            source="socal_phrase",
        )
        self.resource_versions = {
            "socal_word_rules": word_version,
            "socal_phrase_rules": phrase_version,
        }
        self.socal_entries = phrases + words
        self.socal_by_sentiment = {
            label: tuple(
                item for item in self.socal_entries if item["sentiment_label"] == label
            )
            for label in config.SENTIMENT_LABELS
        }
        self.blacklist_overlaps = sorted(
            {
                str(item["keyword"])
                for item in self.socal_entries
                if str(item["keyword"]) in self.lexical_blacklist
            }
        )

    def _contains_noise(self, keyword: str) -> bool:
        normalized = _normalize_keyword(keyword)
        tokens = normalized.split()
        return (
            not tokens
            or any(fragment in normalized for fragment in self.platform_noise_fragments)
            or any(
                not token.isascii()
                or len(token) > 24
                or "'" in token
                or token.endswith("see")
                or token.endswith("up")
                or any(fragment in token for fragment in self.token_noise_fragments)
                for token in tokens
            )
        )

    def extract_keyword_details(
        self,
        texts: Sequence[str],
        *,
        sentiment_label: str,
    ) -> list[dict[str, Any]]:
        """Ambil keyword langsung dari SO-CAL dengan label sentimen yang sama."""
        normalized_label = _normalize_keyword(sentiment_label)
        if normalized_label not in config.SENTIMENT_LABELS:
            raise ValueError(f"Label sentimen keyword tidak valid: {sentiment_label}")

        normalized_texts = [_normalize_keyword(text) for text in texts]
        candidates: list[dict[str, Any]] = []
        for entry in self.socal_by_sentiment[normalized_label]:
            keyword = str(entry["keyword"])
            if self._contains_noise(keyword):
                continue
            occurrences = [_count_occurrences(text, keyword) for text in normalized_texts]
            frequency = sum(occurrences)
            if frequency <= 0:
                continue
            candidates.append(
                {
                    "keyword": keyword,
                    "source": entry["source"],
                    "sentiment_label": normalized_label,
                    "category": entry["category"],
                    "frequency": frequency,
                    "document_frequency": sum(count > 0 for count in occurrences),
                    "socal_weight": float(entry["weight"]),
                    "blacklist_overridden": keyword in self.lexical_blacklist,
                }
            )

        source_order = {"socal_phrase": 0, "socal_word": 1}
        candidates.sort(
            key=lambda item: (
                source_order[str(item["source"])],
                -int(item["document_frequency"]),
                -int(item["frequency"]),
                -len(str(item["keyword"]).split()),
                str(item["keyword"]),
            )
        )

        selected: list[dict[str, Any]] = []
        selected_phrases: list[str] = []
        for candidate in candidates:
            keyword = str(candidate["keyword"])
            if candidate["source"] == "socal_word" and any(
                keyword in phrase.split() for phrase in selected_phrases
            ):
                continue
            selected.append(candidate)
            if candidate["source"] == "socal_phrase":
                selected_phrases.append(keyword)
            if len(selected) >= self.top_keywords:
                break

        return [
            {**item, "rank": rank}
            for rank, item in enumerate(selected, start=1)
        ]

    def extract_keywords(
        self,
        texts: Sequence[str],
        *,
        sentiment_label: str,
    ) -> list[str]:
        return [
            str(item["keyword"])
            for item in self.extract_keyword_details(
                texts, sentiment_label=sentiment_label
            )
        ]

    def summarize(
        self,
        df: pl.DataFrame,
        *,
        sentiment_label: str,
        text_column: str = "original_text",
        location_column: str = "location",
    ) -> dict[str, Any]:
        """Ringkas keyword SO-CAL untuk tepat satu kelas sentimen."""
        normalized_label = _normalize_keyword(sentiment_label)
        if normalized_label not in config.SENTIMENT_LABELS:
            raise ValueError(f"Label sentimen keyword tidak valid: {sentiment_label}")
        text_column = text_column if text_column in df.columns else config.COL_PROCESSED
        if text_column not in df.columns:
            raise KeyError(f"Kolom teks keyword tidak ditemukan: {text_column}")

        texts = [str(value or "") for value in df[text_column].to_list()]
        details = self.extract_keyword_details(
            texts, sentiment_label=normalized_label
        )
        locations = (
            df[location_column].to_list()
            if location_column in df.columns
            else [None] * df.height
        )
        location_counts = Counter(value for value in locations if value)

        return {
            "sentiment_label": normalized_label,
            "n_documents": df.height,
            "keyword_count": len(details),
            "keywords": [str(item["keyword"]) for item in details],
            "keyword_details": details,
            "top_locations": location_counts.most_common(5),
            "representative_docs": texts[: self.representative_docs],
        }

    def empty_summary(self, sentiment_label: str) -> dict[str, Any]:
        normalized_label = _normalize_keyword(sentiment_label)
        return {
            "sentiment_label": normalized_label,
            "n_documents": 0,
            "keyword_count": 0,
            "keywords": [],
            "keyword_details": [],
            "top_locations": [],
            "representative_docs": [],
        }

    def extraction_metadata(self) -> dict[str, Any]:
        return {
            "method": "direct_socal_keyword_matching_by_sentiment",
            "sentiment_order": list(config.TOPIC_SENTIMENT_ORDER),
            "sources": ["socal_phrase", "socal_word"],
            "uses_embedding": False,
            "uses_umap": False,
            "uses_hdbscan": False,
            "uses_ngram_fallback": False,
            "blacklist_overlap_policy": "socal_wins",
            "blacklist_overlaps": self.blacklist_overlaps,
            "resource_versions": self.resource_versions,
            "resource_paths": {
                "socal_word_rules": str(self.word_rules_path),
                "socal_phrase_rules": str(self.phrase_rules_path),
                "topic_keyword_blacklist": str(self.blacklist_path),
            },
        }
