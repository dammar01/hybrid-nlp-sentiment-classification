"""Rule-based sentiment service berbasis resource lexicon.

Service ini memakai tiga resource JSON sebagai source of truth:
word lexicon, phrase rules, dan modifier rules. Outputnya dibuat eksplisit
agar layer lexicon bisa berdiri sebagai komponen independen dalam pipeline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

import config

_TOKEN_RE = re.compile(config.RULE_TOKEN_PATTERN)


@dataclass(slots=True)
class _Token:
    text: str
    start: int
    end: int
    index: int


@dataclass(slots=True)
class _Hit:
    term: str
    label: str
    weight: float
    source: str
    category: str
    reason: str
    start: int
    end: int
    score: float = 0.0
    modifiers: list[str] = field(default_factory=list)
    clause_weight: float = 1.0


@dataclass(slots=True)
class LexiconSentimentService:
    """Analisis sentimen rule-based dari resource JSON."""

    lexicon_words_path: str | Path = config.LEXICON_WORDS_PATH
    phrase_rules_path: str | Path = config.PHRASE_RULES_PATH
    modifier_rules_path: str | Path = config.MODIFIER_RULES_PATH
    weak_threshold: float = config.RULE_WEAK_THRESHOLD
    word_rules: dict[str, dict[str, Any]] = field(init=False)
    phrase_rules: list[dict[str, Any]] = field(init=False)
    modifier_rules: dict[str, Any] = field(init=False)
    resource_versions: dict[str, str] = field(init=False)

    def __post_init__(self) -> None:
        self.lexicon_words_path = Path(self.lexicon_words_path)
        self.phrase_rules_path = Path(self.phrase_rules_path)
        self.modifier_rules_path = Path(self.modifier_rules_path)

        self.word_rules = self._load_word_rules(self.lexicon_words_path)
        self.phrase_rules = self._load_phrase_rules(self.phrase_rules_path)
        self.modifier_rules = self._load_modifier_rules(self.modifier_rules_path)
        self.resource_versions = {
            "lexicon_words": self._resource_version(self.lexicon_words_path),
            "phrase_rules": self._resource_version(self.phrase_rules_path),
            "modifier_rules": self._resource_version(self.modifier_rules_path),
        }

    # ------------------------------------------------------------------
    # Resource loading
    # ------------------------------------------------------------------
    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Resource lexicon tidak ditemukan: {path}")
        data = json.loads(path.read_text(encoding=config.ENCODING))
        if not isinstance(data, dict):
            raise ValueError(f"Resource harus berupa object JSON: {path}")
        return data

    def _resource_version(self, path: Path) -> str:
        data = self._read_json(path)
        version = data.get("version", config.RULE_RESOURCE_VERSION)
        if not isinstance(version, str) or not version.strip():
            raise ValueError(f"Version resource tidak valid di {path}: {version!r}")
        return version.strip()

    def _load_word_rules(self, path: Path) -> dict[str, dict[str, Any]]:
        data = self._read_json(path)
        required_groups = ("positive", "negative", "neutral")
        self._require_groups(data, required_groups, path)

        rules: dict[str, dict[str, Any]] = {}
        for group, label in (
            ("positive", "positif"),
            ("negative", "negatif"),
            ("neutral", "netral"),
        ):
            for item in data[group]:
                self._require_fields(item, ("term", "weight", "category", "reason"), path)
                term = str(item["term"]).strip().lower()
                if not term:
                    continue
                self._validate_label(label, path)
                self._validate_weight(item["weight"], path, term)
                rules[term] = {
                    "label": label,
                    "weight": float(item["weight"]),
                    "category": str(item["category"]),
                    "reason": str(item["reason"]),
                }
        return rules

    def _load_phrase_rules(self, path: Path) -> list[dict[str, Any]]:
        data = self._read_json(path)
        required_groups = ("positive_phrases", "negative_phrases", "neutral_phrases")
        self._require_groups(data, required_groups, path)

        rules: list[dict[str, Any]] = []
        for group in required_groups:
            for item in data[group]:
                self._require_fields(
                    item, ("phrase", "label", "weight", "category", "reason"), path
                )
                phrase = str(item["phrase"]).strip().lower()
                label = str(item["label"]).strip().lower()
                if not phrase:
                    continue
                self._validate_label(label, path)
                self._validate_weight(item["weight"], path, phrase)
                rules.append(
                    {
                        "phrase": phrase,
                        "label": label,
                        "weight": float(item["weight"]),
                        "category": str(item["category"]),
                        "reason": str(item["reason"]),
                    }
                )
        return sorted(rules, key=lambda item: len(item["phrase"]), reverse=True)

    def _load_modifier_rules(self, path: Path) -> dict[str, Any]:
        data = self._read_json(path)
        required_groups = ("negations", "intensifiers", "downtoners", "contrast_markers")
        self._require_groups(data, required_groups, path)
        for item in data["negations"]:
            self._require_fields(item, ("term", "scope", "effect", "reason"), path)
            self._validate_term(item["term"], path)
            self._validate_scope(item["scope"], path, item["term"])
            if str(item["effect"]) not in (
                config.RULE_EFFECT_INVERT,
                config.RULE_EFFECT_WEAKEN,
            ):
                raise ValueError(f"Effect negation tidak valid di {path}: {item!r}")
        for group in ("intensifiers", "downtoners"):
            for item in data[group]:
                self._require_fields(item, ("term", "multiplier", "reason"), path)
                self._validate_term(item["term"], path)
                self._validate_multiplier(item["multiplier"], path, item["term"])
        for item in data["contrast_markers"]:
            self._require_fields(item, ("term", "effect", "reason"), path)
            self._validate_term(item["term"], path)
            if str(item["effect"]) not in (
                config.RULE_EFFECT_PREFER_AFTER_CLAUSE,
                config.RULE_EFFECT_AMPLIFY_NEGATIVE_AFTER,
                config.RULE_EFFECT_AMPLIFY_POSITIVE_AFTER,
                config.RULE_EFFECT_SUBORDINATE_BEFORE_CLAUSE,
                config.RULE_EFFECT_CONTRADICT_EXPECTATION,
            ):
                raise ValueError(f"Effect contrast tidak valid di {path}: {item!r}")
        return {
            "negations": {str(item["term"]).lower(): item for item in data["negations"]},
            "intensifiers": {
                str(item["term"]).lower(): item for item in data["intensifiers"]
            },
            "downtoners": {str(item["term"]).lower(): item for item in data["downtoners"]},
            "contrast_markers": [
                {**item, "term": str(item["term"]).lower()}
                for item in data["contrast_markers"]
            ],
        }

    @staticmethod
    def _require_groups(data: dict[str, Any], groups: tuple[str, ...], path: Path) -> None:
        missing = [group for group in groups if group not in data]
        if missing:
            raise ValueError(f"Group wajib hilang di {path}: {missing}")
        invalid = [group for group in groups if not isinstance(data[group], list)]
        if invalid:
            raise ValueError(f"Group wajib berupa list di {path}: {invalid}")

    @staticmethod
    def _require_fields(item: Any, fields: tuple[str, ...], path: Path) -> None:
        if not isinstance(item, dict):
            raise ValueError(f"Item rule harus object di {path}: {item!r}")
        missing = [field for field in fields if field not in item]
        if missing:
            raise ValueError(f"Field wajib hilang di {path}: {missing} pada {item!r}")

    @staticmethod
    def _validate_label(label: str, path: Path) -> None:
        if label not in config.SENTIMENT_LABELS:
            raise ValueError(f"Label rule tidak valid di {path}: {label}")

    @staticmethod
    def _validate_weight(raw_weight: Any, path: Path, term: str) -> None:
        weight = float(raw_weight)
        if weight < 0 or weight > 2:
            raise ValueError(f"Weight rule di luar rentang 0..2 di {path}: {term}={weight}")

    @staticmethod
    def _validate_term(raw_term: Any, path: Path) -> None:
        if not str(raw_term).strip():
            raise ValueError(f"Term kosong tidak valid di {path}")

    @staticmethod
    def _validate_scope(raw_scope: Any, path: Path, term: Any) -> None:
        scope = int(raw_scope)
        if scope <= 0:
            raise ValueError(f"Scope modifier harus positif di {path}: {term}={scope}")

    @staticmethod
    def _validate_multiplier(raw_multiplier: Any, path: Path, term: Any) -> None:
        multiplier = float(raw_multiplier)
        if multiplier <= 0:
            raise ValueError(
                f"Multiplier modifier harus positif di {path}: {term}={multiplier}"
            )

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    def analyze_text(self, text: str) -> dict[str, object]:
        original = "" if text is None else str(text)
        normalized = original.lower()
        tokens = self._tokenize(normalized)
        clause_weights = self._clause_weights(normalized)
        hits, neutral_hits = self._collect_evidence(normalized, tokens, clause_weights)

        positive_score = round(sum(hit.score for hit in hits if hit.score > 0), 4)
        negative_score = round(abs(sum(hit.score for hit in hits if hit.score < 0)), 4)
        score = round(positive_score - negative_score, 4)
        total_score = positive_score + negative_score
        confidence = round(abs(score) / total_score, 4) if total_score else 0.0

        if score > 0:
            label = "positif"
        elif score < 0:
            label = "negatif"
        else:
            label = "netral"

        if not hits:
            status = config.RULE_STATUS_UNKNOWN
        elif confidence < self.weak_threshold:
            status = config.RULE_STATUS_WEAK
        else:
            status = config.RULE_STATUS_DETECTED

        phrase_hits = [hit for hit in hits if hit.source == "phrase"]
        word_hits = [hit for hit in hits if hit.source == "word"]
        phrase_terms = [hit.term for hit in phrase_hits]
        word_terms = [hit.term for hit in word_hits]
        all_hit_terms = phrase_terms + word_terms
        modifier_terms = self._unique(
            modifier for hit in hits for modifier in hit.modifiers
        )
        explanation = self._build_explanation(
            label=label,
            status=status,
            score=score,
            positive_score=positive_score,
            negative_score=negative_score,
            phrase_hits=phrase_terms,
            word_hits=word_terms,
            modifier_hits=modifier_terms,
        )

        result = {
            config.COL_RULE_CONTRACT_VERSION: config.RULE_CONTRACT_VERSION,
            config.COL_RULE_RESOURCE_VERSION: self._resource_version_label(),
            config.COL_RULE_LABEL: label,
            config.COL_RULE_SCORE: score,
            config.COL_RULE_CONFIDENCE: confidence,
            config.COL_RULE_STATUS: status,
            config.COL_RULE_POSITIVE_SCORE: positive_score,
            config.COL_RULE_NEGATIVE_SCORE: negative_score,
            config.COL_RULE_POSITIVE_COUNT: sum(1 for hit in hits if hit.score > 0),
            config.COL_RULE_NEGATIVE_COUNT: sum(1 for hit in hits if hit.score < 0),
            config.COL_RULE_HITS: ", ".join(all_hit_terms),
            config.COL_RULE_NEUTRAL_HITS: ", ".join(neutral_hits),
            config.COL_RULE_PHRASE_HITS: ", ".join(phrase_terms),
            config.COL_RULE_WORD_HITS: ", ".join(word_terms),
            config.COL_RULE_MODIFIER_HITS: ", ".join(modifier_terms),
            config.COL_RULE_EVIDENCE: self._structured_evidence(hits, neutral_hits),
            config.COL_RULE_EXPLANATION: explanation,
        }
        self._validate_output_contract(result)
        return result

    def analyze_dataframe(
        self, df: pl.DataFrame, text_column: str = config.COL_PROCESSED
    ) -> pl.DataFrame:
        if text_column not in df.columns:
            raise KeyError(f"Kolom '{text_column}' tidak ada pada DataFrame")

        rows = pl.DataFrame(
            [self.analyze_text(text) for text in df[text_column].to_list()]
        )
        return df.hstack(rows)

    def _tokenize(self, text: str) -> list[_Token]:
        return [
            _Token(match.group(0), match.start(), match.end(), index)
            for index, match in enumerate(_TOKEN_RE.finditer(text))
        ]

    def _collect_evidence(
        self,
        text: str,
        tokens: list[_Token],
        clause_weights: list[tuple[int, float]],
    ) -> tuple[list[_Hit], list[str]]:
        phrase_hits, consumed_spans = self._match_phrases(text, clause_weights)
        word_hits, neutral_hits = self._match_words(tokens, consumed_spans, clause_weights)
        return phrase_hits + word_hits, neutral_hits

    def _match_phrases(
        self, text: str, clause_weights: list[tuple[int, float]]
    ) -> tuple[list[_Hit], list[tuple[int, int]]]:
        hits: list[_Hit] = []
        consumed: list[tuple[int, int]] = []

        for rule in self.phrase_rules:
            phrase = rule["phrase"]
            pattern = re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)")
            for match in pattern.finditer(text):
                span = (match.start(), match.end())
                if self._overlaps(span, consumed):
                    continue
                score = self._signed_score(rule["label"], rule["weight"])
                clause_weight = self._weight_at(span[0], clause_weights)
                score *= clause_weight
                hit = _Hit(
                    term=phrase,
                    label=rule["label"],
                    weight=rule["weight"],
                    source="phrase",
                    category=rule["category"],
                    reason=rule["reason"],
                    start=span[0],
                    end=span[1],
                    score=round(score, 4),
                    clause_weight=clause_weight,
                )
                hits.append(hit)
                consumed.append(span)

        return hits, consumed

    def _match_words(
        self,
        tokens: list[_Token],
        consumed_spans: list[tuple[int, int]],
        clause_weights: list[tuple[int, float]],
    ) -> tuple[list[_Hit], list[str]]:
        hits: list[_Hit] = []
        neutral_hits: list[str] = []

        for token in tokens:
            if self._inside_any((token.start, token.end), consumed_spans):
                continue
            rule = self.word_rules.get(token.text)
            if not rule:
                continue

            label = rule["label"]
            if label == "netral":
                neutral_hits.append(token.text)
                continue

            score = self._signed_score(label, rule["weight"])
            modifiers = self._modifiers_for_token(token, tokens)
            score = self._apply_modifiers(score, modifiers)
            clause_weight = self._weight_at(token.start, clause_weights)
            score *= clause_weight

            hits.append(
                _Hit(
                    term=token.text,
                    label="positif" if score > 0 else "negatif",
                    weight=rule["weight"],
                    source="word",
                    category=rule["category"],
                    reason=rule["reason"],
                    start=token.start,
                    end=token.end,
                    score=round(score, 4),
                    modifiers=[item["term"] for item in modifiers],
                    clause_weight=clause_weight,
                )
            )

        return hits, neutral_hits

    def _modifiers_for_token(
        self, token: _Token, tokens: list[_Token]
    ) -> list[dict[str, Any]]:
        modifiers: list[dict[str, Any]] = []
        before = [item for item in tokens if item.index < token.index]
        after = [item for item in tokens if item.index > token.index]

        for prev in reversed(before[-3:]):
            negation = self.modifier_rules["negations"].get(prev.text)
            if negation and token.index - prev.index <= int(negation.get("scope", 1)):
                modifiers.append({**negation, "term": prev.text})
                break

        for prev in reversed(before[-2:]):
            intensifier = self.modifier_rules["intensifiers"].get(prev.text)
            if intensifier:
                modifiers.append({**intensifier, "term": prev.text})
                break
            downtoner = self.modifier_rules["downtoners"].get(prev.text)
            if downtoner:
                modifiers.append({**downtoner, "term": prev.text})
                break

        if after:
            next_token = after[0]
            intensifier = self.modifier_rules["intensifiers"].get(next_token.text)
            if intensifier:
                modifiers.append({**intensifier, "term": next_token.text})

        return modifiers

    @staticmethod
    def _apply_modifiers(score: float, modifiers: list[dict[str, Any]]) -> float:
        adjusted = score
        for modifier in modifiers:
            effect = str(modifier.get("effect", ""))
            if effect == config.RULE_EFFECT_INVERT:
                adjusted *= -1
            elif effect == config.RULE_EFFECT_WEAKEN:
                adjusted *= 0.5
            if "multiplier" in modifier:
                adjusted *= float(modifier["multiplier"])
        return adjusted

    def _clause_weights(self, text: str) -> list[tuple[int, float]]:
        weights = [(0, 1.0)]
        for marker in self.modifier_rules["contrast_markers"]:
            term = marker["term"]
            effect = str(marker.get("effect", ""))
            pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)")
            match = pattern.search(text)
            if not match:
                continue
            if effect == config.RULE_EFFECT_PREFER_AFTER_CLAUSE:
                weights.append((0, config.RULE_CONTRAST_BEFORE_WEIGHT))
                weights.append((match.end(), config.RULE_CONTRAST_AFTER_WEIGHT))
            elif effect == config.RULE_EFFECT_AMPLIFY_NEGATIVE_AFTER:
                weights.append((match.end(), config.RULE_AMPLIFY_AFTER_WEIGHT))
            elif effect == config.RULE_EFFECT_AMPLIFY_POSITIVE_AFTER:
                weights.append((match.end(), config.RULE_AMPLIFY_AFTER_WEIGHT))
            elif effect == config.RULE_EFFECT_SUBORDINATE_BEFORE_CLAUSE:
                weights.append((0, config.RULE_CONCESSION_BEFORE_WEIGHT))
                weights.append((match.end(), config.RULE_CONCESSION_AFTER_WEIGHT))
            elif effect == config.RULE_EFFECT_CONTRADICT_EXPECTATION:
                weights.append((match.end(), config.RULE_CONCESSION_AFTER_WEIGHT))
        return sorted(weights, key=lambda item: item[0])

    @staticmethod
    def _weight_at(position: int, weights: list[tuple[int, float]]) -> float:
        selected = 1.0
        for start, weight in weights:
            if position >= start:
                selected = weight
            else:
                break
        return selected

    @staticmethod
    def _signed_score(label: str, weight: float) -> float:
        if label == "positif":
            return float(weight)
        if label == "negatif":
            return -float(weight)
        return 0.0

    @staticmethod
    def _overlaps(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
        return any(span[0] < other[1] and other[0] < span[1] for other in spans)

    @staticmethod
    def _inside_any(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
        return any(other[0] <= span[0] and span[1] <= other[1] for other in spans)

    @staticmethod
    def _unique(items: Any) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for item in items:
            text = str(item)
            if text and text not in seen:
                seen.add(text)
                result.append(text)
        return result

    @staticmethod
    def _validate_output_contract(result: dict[str, object]) -> None:
        missing = [column for column in config.RULE_OUTPUT_COLUMNS if column not in result]
        if missing:
            raise RuntimeError(f"Output contract rule-based tidak lengkap: {missing}")

    def _resource_version_label(self) -> str:
        return ";".join(
            f"{name}={version}" for name, version in self.resource_versions.items()
        )

    @staticmethod
    def _structured_evidence(hits: list[_Hit], neutral_hits: list[str]) -> str:
        payload: dict[str, Any] = {
            "sentiment_hits": [
                {
                    "term": hit.term,
                    "label": hit.label,
                    "source": hit.source,
                    "score": hit.score,
                    "weight": hit.weight,
                    "category": hit.category,
                    "reason": hit.reason,
                    "span": [hit.start, hit.end],
                    "modifiers": hit.modifiers,
                    "clause_weight": hit.clause_weight,
                }
                for hit in hits
            ],
            "neutral_hits": neutral_hits,
        }
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _build_explanation(
        *,
        label: str,
        status: str,
        score: float,
        positive_score: float,
        negative_score: float,
        phrase_hits: list[str],
        word_hits: list[str],
        modifier_hits: list[str],
    ) -> str:
        if status == "unknown":
            return "Tidak ada rule sentimen yang terdeteksi."

        evidence = []
        if phrase_hits:
            evidence.append(f"frasa: {', '.join(phrase_hits)}")
        if word_hits:
            evidence.append(f"kata: {', '.join(word_hits)}")
        if modifier_hits:
            evidence.append(f"modifier: {', '.join(modifier_hits)}")
        evidence_text = "; ".join(evidence) if evidence else "tanpa evidence detail"
        return (
            f"Label {label} ({status}) dari skor {score:.4f}; "
            f"positif={positive_score:.4f}, negatif={negative_score:.4f}; {evidence_text}."
        )
