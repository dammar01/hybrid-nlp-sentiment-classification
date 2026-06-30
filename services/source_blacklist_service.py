"""Audit dan klasifikasi blacklist sumber kandidat dataset."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, unquote, urlsplit, urlunsplit

import config


class SourceBlacklistService:
    """Classifier blacklist sumber berbasis exact URL dan rule presisi tinggi."""

    def __init__(
        self,
        *,
        exact_urls: list[str] | tuple[str, ...] | None = None,
        rules: dict | None = None,
    ) -> None:
        self.rules = rules or {}
        self.tracking_params = set(self.rules.get("tracking_params") or [])
        self.commercial_domains = set(self.rules.get("commercial_domains") or [])
        self.commercial_path_markers = tuple(
            self.rules.get("commercial_path_markers") or []
        )
        self.kalbar_terms = list(self.rules.get("kalbar_terms") or [])
        self.outside_location_terms = list(
            self.rules.get("outside_location_terms") or []
        )
        self.exact_urls = tuple(exact_urls or ())
        self.exact_normalized_urls = {
            self.normalize_url(url)
            for url in self.exact_urls
            if str(url).strip()
        }

    @classmethod
    def from_paths(
        cls,
        *,
        exact_blacklist_path: str | Path = config.SOURCE_URL_BLACKLIST_PATH,
        rules_path: str | Path = config.SOURCE_BLACKLIST_RULES_PATH,
    ) -> "SourceBlacklistService":
        exact_urls = cls.load_exact_blacklist(exact_blacklist_path)
        rules = cls.load_rules(rules_path)
        return cls(exact_urls=exact_urls, rules=rules)

    @staticmethod
    def load_exact_blacklist(path: str | Path) -> tuple[str, ...]:
        path = Path(path)
        if not path.exists():
            return ()
        data = json.loads(path.read_text(encoding=config.ENCODING))
        if not isinstance(data, list):
            raise ValueError(f"Source URL blacklist harus berupa array JSON: {path}")
        return tuple(str(item).strip() for item in data if str(item).strip())

    @staticmethod
    def load_rules(path: str | Path) -> dict:
        path = Path(path)
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding=config.ENCODING))
        if not isinstance(data, dict):
            raise ValueError(f"Source blacklist rules harus berupa object JSON: {path}")
        return data

    @staticmethod
    def normalize_text(value: str | None) -> str:
        text = unicodedata.normalize("NFKC", value or "").casefold()
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def contains_phrase(cls, text: str, phrase: str) -> bool:
        normalized_phrase = cls.normalize_text(phrase)
        if not normalized_phrase:
            return False
        return bool(
            re.search(
                rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)",
                cls.normalize_text(text),
            )
        )

    @classmethod
    def find_terms(cls, text: str, terms: list[str]) -> list[str]:
        return sorted(
            {term for term in terms if cls.contains_phrase(text, term)},
            key=lambda value: (len(value), value),
        )

    def normalize_url(self, url: str | None) -> str:
        if not url:
            return ""

        parts = urlsplit(url.strip())
        host = parts.netloc.lower().removeprefix("www.")
        path = re.sub(r"/+$", "", unquote(parts.path or ""))
        query_items = []

        for key, value in parse_qsl(parts.query, keep_blank_values=True):
            lowered = key.casefold()
            if lowered in self.tracking_params:
                continue
            if lowered == "hl":
                continue
            if lowered == "page" and value.casefold() == "all":
                continue
            query_items.append((key, value))

        query_items.sort()
        return urlunsplit(
            (
                (parts.scheme or "https").lower(),
                host,
                path,
                urlencode(query_items, doseq=True),
                "",
            )
        )

    @staticmethod
    def get_domain(url: str) -> str:
        return urlsplit(url).netloc.lower().removeprefix("www.")

    def build_title_url_text(self, row: dict) -> str:
        source_url = str(row.get("source_url") or "")
        return " ".join(
            [
                str(row.get("raw_title") or ""),
                unquote(urlsplit(source_url).path),
                self.get_domain(source_url),
            ]
        )

    def is_commercial_source(self, row: dict) -> bool:
        source_url = str(row.get("source_url") or "")
        domain = self.get_domain(source_url)
        path = self.normalize_text(unquote(urlsplit(source_url).path))

        if domain in self.commercial_domains:
            return True
        return any(marker in path for marker in self.commercial_path_markers)

    def discover_reason_codes(self, row: dict) -> list[str]:
        reasons: list[str] = []
        title_url_text = self.build_title_url_text(row)

        if self.is_commercial_source(row):
            reasons.append("commercial_self_promotion")

        outside = self.find_terms(title_url_text, self.outside_location_terms)
        kalbar = self.find_terms(title_url_text, self.kalbar_terms)
        if outside and not kalbar:
            reasons.append("explicit_non_kalbar_location_in_url_or_title")

        return reasons

    def classify(self, row: dict) -> dict:
        source_url = str(row.get("source_url") or "")
        normalized = self.normalize_url(source_url)

        if normalized in self.exact_normalized_urls:
            return {
                "blacklist_status": "original_exact_blacklist",
                "blacklist_reason_codes": ["exact_url_match"],
                "normalized_source_url": normalized,
                "blacklist_is_excluded": True,
            }

        reasons = self.discover_reason_codes(row)
        if reasons:
            return {
                "blacklist_status": "discovered_pattern_blacklist",
                "blacklist_reason_codes": reasons,
                "normalized_source_url": normalized,
                "blacklist_is_excluded": True,
            }

        return {
            "blacklist_status": "clear",
            "blacklist_reason_codes": [],
            "normalized_source_url": normalized,
            "blacklist_is_excluded": False,
        }
