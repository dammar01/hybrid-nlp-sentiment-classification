"""SO-CAL-inspired lexicon rule engine for sentiment analysis.

The implementation follows the bounded parts of Taboada et al. (2011):
prior polarity, strength, multi-word precedence, percentage modifiers,
shift negation, contrast weighting, irrealis blocking, negative weighting,
and repetition weighting. It keeps the existing DataFrame/output contract.
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
    repetition_weight: float = 1.0
    blocked_by: str = ""


@dataclass(slots=True)
class LexiconSentimentService:
    """Analyze sentiment using split SO-CAL-inspired rule resources."""

    metadata_path: str | Path = config.SOCAL_METADATA_PATH
    word_rules_path: str | Path = config.SOCAL_WORD_RULES_PATH
    phrase_rules_path: str | Path = config.SOCAL_PHRASE_RULES_PATH
    modifiers_path: str | Path = config.SOCAL_MODIFIERS_PATH
    weak_threshold: float = config.RULE_WEAK_THRESHOLD
    neutral_cutoff: float = config.RULE_NEUTRAL_CUTOFF
    word_rules: dict[str, dict[str, Any]] = field(init=False)
    phrase_rules: list[dict[str, Any]] = field(init=False)
    modifier_rules: dict[str, Any] = field(init=False)
    method_reference: str = field(init=False)
    rules_version: str = field(init=False)
    resource_versions: dict[str, str] = field(init=False)
    score_scale: dict[str, Any] = field(init=False)
    negation_shift: float = field(init=False)
    negative_weight: float = field(init=False)
    repetition_weighting: bool = field(init=False)

    def __post_init__(self) -> None:
        self.metadata_path = Path(self.metadata_path)
        self.word_rules_path = Path(self.word_rules_path)
        self.phrase_rules_path = Path(self.phrase_rules_path)
        self.modifiers_path = Path(self.modifiers_path)

        metadata = self._read_json(self.metadata_path)
        word_rules = self._read_json(self.word_rules_path)
        phrase_rules = self._read_json(self.phrase_rules_path)
        modifiers = self._read_json(self.modifiers_path)

        self.rules_version = self._required_text(metadata, "version", self.metadata_path)
        self.method_reference = self._required_text(
            metadata, "method_reference", self.metadata_path
        )
        self.score_scale = self._required_object(
            metadata, "score_scale", self.metadata_path
        )
        self.negation_shift = float(
            metadata.get("negation_shift", config.RULE_NEGATION_SHIFT)
        )
        self.negative_weight = float(
            metadata.get("negative_weight", config.RULE_NEGATIVE_WEIGHT)
        )
        self.repetition_weighting = bool(metadata.get("repetition_weighting", True))
        self.resource_versions = {
            "metadata": self.rules_version,
            "word_rules": self._required_text(word_rules, "version", self.word_rules_path),
            "phrase_rules": self._required_text(
                phrase_rules, "version", self.phrase_rules_path
            ),
            "modifiers": self._required_text(modifiers, "version", self.modifiers_path),
        }
        self.word_rules = self._load_entries(
            word_rules, "word_rules", "term", self.word_rules_path
        )
        self.phrase_rules = sorted(
            self._load_entries(
                phrase_rules, "phrase_rules", "phrase", self.phrase_rules_path
            ),
            key=lambda item: len(str(item["phrase"])),
            reverse=True,
        )
        self.modifier_rules = self._load_modifier_rules(modifiers, self.modifiers_path)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Resource lexicon tidak ditemukan: {path}")
        data = json.loads(path.read_text(encoding=config.ENCODING))
        if not isinstance(data, dict):
            raise ValueError(f"Resource harus berupa object JSON: {path}")
        return data

    @staticmethod
    def _required_text(data: dict[str, Any], key: str, path: Path) -> str:
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Field '{key}' wajib berupa teks di {path}")
        return value.strip()

    @staticmethod
    def _required_object(data: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
        value = data.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"Field '{key}' wajib berupa object di {path}")
        return value

    def _load_entries(
        self, data: dict[str, Any], group: str, text_key: str, path: Path
    ) -> dict[str, dict[str, Any]] | list[dict[str, Any]]:
        raw_items = data.get(group)
        if not isinstance(raw_items, list):
            raise ValueError(f"Group '{group}' wajib berupa list di {path}")

        entries: list[dict[str, Any]] = []
        for item in raw_items:
            self._require_fields(
                item, (text_key, "label", "weight", "category", "reason"), path
            )
            text = str(item[text_key]).strip().lower()
            label = str(item["label"]).strip().lower()
            if not text:
                continue
            self._validate_label(label, path)
            self._validate_weight(item["weight"], path, text)
            entries.append(
                {
                    text_key: text,
                    "label": label,
                    "weight": float(item["weight"]),
                    "category": str(item["category"]),
                    "reason": str(item["reason"]),
                }
            )

        if text_key == "term":
            return {str(item["term"]): item for item in entries}
        return entries

    def _load_modifier_rules(self, rules: dict[str, Any], path: Path) -> dict[str, Any]:
        modifiers = self._required_object(rules, "modifiers", path)
        required = (
            "negations",
            "intensifiers",
            "downtoners",
            "irrealis_blockers",
            "contrast_markers",
        )
        for group in required:
            if not isinstance(modifiers.get(group), list):
                raise ValueError(
                    f"Modifier group '{group}' wajib berupa list di {path}"
                )

        for item in modifiers["negations"]:
            self._require_fields(item, ("term", "scope", "effect", "reason"), path)
            self._validate_term(item["term"], path)
            self._validate_scope(item["scope"], path, item["term"])
            if str(item["effect"]) != config.RULE_EFFECT_SHIFT:
                raise ValueError(
                    f"Negasi SO-CAL-inspired harus memakai effect shift: {item!r}"
                )

        for group in ("intensifiers", "downtoners"):
            for item in modifiers[group]:
                self._require_fields(item, ("term", "multiplier", "reason"), path)
                self._validate_term(item["term"], path)
                self._validate_multiplier(item["multiplier"], path, item["term"])

        for item in modifiers["irrealis_blockers"]:
            self._require_fields(item, ("term", "scope", "reason"), path)
            self._validate_term(item["term"], path)
            self._validate_scope(item["scope"], path, item["term"])

        for item in modifiers["contrast_markers"]:
            self._require_fields(item, ("term", "effect", "reason"), path)
            self._validate_term(item["term"], path)
            if str(item["effect"]) not in (
                config.RULE_EFFECT_PREFER_AFTER_CLAUSE,
                config.RULE_EFFECT_AMPLIFY_NEGATIVE_AFTER,
                config.RULE_EFFECT_AMPLIFY_POSITIVE_AFTER,
                config.RULE_EFFECT_SUBORDINATE_BEFORE_CLAUSE,
                config.RULE_EFFECT_CONTRADICT_EXPECTATION,
            ):
                raise ValueError(f"Effect contrast tidak valid: {item!r}")

        return {
            "negations": self._index_by_term(modifiers["negations"]),
            "intensifiers": self._index_by_term(modifiers["intensifiers"]),
            "downtoners": self._index_by_term(modifiers["downtoners"]),
            "irrealis_blockers": self._index_by_term(modifiers["irrealis_blockers"]),
            "contrast_markers": [
                {**item, "term": str(item["term"]).strip().lower()}
                for item in modifiers["contrast_markers"]
            ],
        }

    @staticmethod
    def _index_by_term(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {str(item["term"]).strip().lower(): item for item in items}

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

    def analyze_text(self, text: str) -> dict[str, object]:
        original = "" if text is None else str(text)
        normalized = original.lower()
        tokens = self._tokenize(normalized)
        clause_weights = self._clause_weights(normalized)
        hits, neutral_hits = self._collect_evidence(normalized, tokens, clause_weights)
        if self.repetition_weighting:
            self._apply_repetition_weights(hits)

        scored_hits = [hit for hit in hits if hit.score != 0]
        positive_score = round(sum(hit.score for hit in scored_hits if hit.score > 0), 4)
        negative_score = round(abs(sum(hit.score for hit in scored_hits if hit.score < 0)), 4)
        score = round(positive_score - negative_score, 4)
        max_abs_score = float(self.score_scale.get("max_abs", 2.0)) or 2.0
        confidence = round(min(abs(score) / max_abs_score, 1.0), 4)

        if score > self.neutral_cutoff:
            label = "positif"
        elif score < -self.neutral_cutoff:
            label = "negatif"
        else:
            label = "netral"

        if not scored_hits:
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
            config.COL_RULE_POSITIVE_COUNT: sum(1 for hit in scored_hits if hit.score > 0),
            config.COL_RULE_NEGATIVE_COUNT: sum(1 for hit in scored_hits if hit.score < 0),
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
        phrase_hits, consumed_spans = self._match_phrases(text, tokens, clause_weights)
        word_hits, neutral_hits = self._match_words(tokens, consumed_spans, clause_weights)
        return phrase_hits + word_hits, neutral_hits

    def _match_phrases(
        self,
        text: str,
        tokens: list[_Token],
        clause_weights: list[tuple[int, float]],
    ) -> tuple[list[_Hit], list[tuple[int, int]]]:
        hits: list[_Hit] = []
        consumed: list[tuple[int, int]] = []

        for rule in self.phrase_rules:
            phrase = str(rule["phrase"])
            pattern = re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)")
            for match in pattern.finditer(text):
                span = (match.start(), match.end())
                if self._overlaps(span, consumed):
                    continue
                span_tokens = self._tokens_for_span(tokens, span)
                if not span_tokens:
                    continue
                score = self._signed_score(rule["label"], rule["weight"])
                modifiers = self._modifiers_for_span(
                    span_tokens[0].index, span_tokens[-1].index, tokens
                )
                blocked_by = self._irrealis_blocker_for_span(span_tokens[0].index, tokens)
                if blocked_by:
                    score = 0.0
                else:
                    score = self._apply_modifiers(score, modifiers)
                    score *= self._weight_at(span[0], clause_weights)
                    score = self._apply_negative_weight(score)
                clause_weight = self._weight_at(span[0], clause_weights)
                hit = _Hit(
                    term=phrase,
                    label=self._score_label(score, str(rule["label"])),
                    weight=float(rule["weight"]),
                    source="phrase",
                    category=str(rule["category"]),
                    reason=str(rule["reason"]),
                    start=span[0],
                    end=span[1],
                    score=round(score, 4),
                    modifiers=[str(item["term"]) for item in modifiers],
                    clause_weight=clause_weight,
                    blocked_by=blocked_by,
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

            label = str(rule["label"])
            if label == "netral":
                neutral_hits.append(token.text)
                continue

            score = self._signed_score(label, rule["weight"])
            modifiers = self._modifiers_for_span(token.index, token.index, tokens)
            blocked_by = self._irrealis_blocker_for_span(token.index, tokens)
            if blocked_by:
                score = 0.0
            else:
                score = self._apply_modifiers(score, modifiers)
                score *= self._weight_at(token.start, clause_weights)
                score = self._apply_negative_weight(score)

            hits.append(
                _Hit(
                    term=token.text,
                    label=self._score_label(score, label),
                    weight=float(rule["weight"]),
                    source="word",
                    category=str(rule["category"]),
                    reason=str(rule["reason"]),
                    start=token.start,
                    end=token.end,
                    score=round(score, 4),
                    modifiers=[str(item["term"]) for item in modifiers],
                    clause_weight=self._weight_at(token.start, clause_weights),
                    blocked_by=blocked_by,
                )
            )

        return hits, neutral_hits

    def _modifiers_for_span(
        self, start_index: int, end_index: int, tokens: list[_Token]
    ) -> list[dict[str, Any]]:
        modifiers: list[dict[str, Any]] = []
        before = [item for item in tokens if item.index < start_index]
        after = [item for item in tokens if item.index > end_index]

        for prev in reversed(before[-4:]):
            negation = self.modifier_rules["negations"].get(prev.text)
            if negation and start_index - prev.index <= int(negation.get("scope", 1)):
                modifiers.append({**negation, "term": prev.text})
                break

        for prev in reversed(before[-3:]):
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
            downtoner = self.modifier_rules["downtoners"].get(next_token.text)
            if downtoner:
                modifiers.append({**downtoner, "term": next_token.text})

        return modifiers

    def _irrealis_blocker_for_span(
        self, start_index: int, tokens: list[_Token]
    ) -> str:
        before = [item for item in tokens if item.index < start_index]
        for prev in reversed(before[-4:]):
            blocker = self.modifier_rules["irrealis_blockers"].get(prev.text)
            if blocker and start_index - prev.index <= int(blocker.get("scope", 1)):
                return prev.text
        return ""

    def _apply_modifiers(self, score: float, modifiers: list[dict[str, Any]]) -> float:
        adjusted = score
        has_shift_negation = False
        for modifier in modifiers:
            if "multiplier" in modifier:
                adjusted *= float(modifier["multiplier"])
            if str(modifier.get("effect", "")) == config.RULE_EFFECT_SHIFT:
                has_shift_negation = True
        if has_shift_negation:
            adjusted = self._shift_negation(adjusted)
        return adjusted

    def _shift_negation(self, score: float) -> float:
        if score > 0:
            return score - self.negation_shift
        if score < 0:
            return score + self.negation_shift
        return 0.0

    def _apply_negative_weight(self, score: float) -> float:
        if score < 0:
            return score * self.negative_weight
        return score

    def _apply_repetition_weights(self, hits: list[_Hit]) -> None:
        counts: dict[str, int] = {}
        for hit in sorted(hits, key=lambda item: (item.start, item.end)):
            if hit.score == 0 or hit.modifiers:
                continue
            key = f"{hit.source}:{hit.term}"
            counts[key] = counts.get(key, 0) + 1
            if counts[key] <= 1:
                continue
            hit.repetition_weight = round(1 / counts[key], 4)
            hit.score = round(hit.score * hit.repetition_weight, 4)

    def _clause_weights(self, text: str) -> list[tuple[int, float]]:
        weights = [(0, 1.0)]
        for marker in self.modifier_rules["contrast_markers"]:
            term = str(marker["term"])
            effect = str(marker.get("effect", ""))
            pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)")
            for match in pattern.finditer(text):
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
    def _score_label(score: float, fallback: str) -> str:
        if score > 0:
            return "positif"
        if score < 0:
            return "negatif"
        return fallback

    @staticmethod
    def _tokens_for_span(tokens: list[_Token], span: tuple[int, int]) -> list[_Token]:
        return [token for token in tokens if token.start >= span[0] and token.end <= span[1]]

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
            f"socal_{name}={version}"
            for name, version in self.resource_versions.items()
        )

    def _structured_evidence(self, hits: list[_Hit], neutral_hits: list[str]) -> str:
        payload: dict[str, Any] = {
            "method": "SO-CAL-inspired",
            "method_reference": self.method_reference,
            "resource_version": self._resource_version_label(),
            "resource_versions": self.resource_versions,
            "score_scale": self.score_scale,
            "negation_shift": self.negation_shift,
            "negative_weight": self.negative_weight,
            "repetition_weighting": self.repetition_weighting,
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
                    "repetition_weight": hit.repetition_weight,
                    "blocked_by": hit.blocked_by,
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
        if status == config.RULE_STATUS_UNKNOWN:
            return "Tidak ada rule sentimen yang dihitung."

        evidence = []
        if phrase_hits:
            evidence.append(f"frasa: {', '.join(phrase_hits)}")
        if word_hits:
            evidence.append(f"kata: {', '.join(word_hits)}")
        if modifier_hits:
            evidence.append(f"modifier: {', '.join(modifier_hits)}")
        evidence_text = "; ".join(evidence) if evidence else "tanpa evidence detail"
        return (
            f"Label {label} ({status}) dari skor SO {score:.4f}; "
            f"positif={positive_score:.4f}, negatif={negative_score:.4f}; {evidence_text}."
        )
