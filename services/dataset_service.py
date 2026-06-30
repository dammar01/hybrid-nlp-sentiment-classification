"""DatasetService — pemuatan, validasi, deduplikasi, dan ringkasan dataset.

Menangani data opini masyarakat mengenai Solar Home System (SHS) di
Kalimantan Barat. Seluruh operasi DataFrame menggunakan Polars (bukan pandas)
sesuai rancangan tugas akhir.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

import polars as pl

import config
from services.source_blacklist_service import SourceBlacklistService

logger = logging.getLogger(__name__)

MAX_CSV_FIELD_SIZE = sys.maxsize
while True:
    try:
        csv.field_size_limit(MAX_CSV_FIELD_SIZE)
        break
    except OverflowError:
        MAX_CSV_FIELD_SIZE //= 10

V1_SHS_DATASET_COLUMNS: tuple[str, ...] = (
    "text_id",
    "text",
    "subjectivity_type",
    "speaker_type",
    "public_opinion_scope",
    "corpus_role",
    "aspect",
    "location",
    "sentiment_label",
    "label_status",
    "source_id",
    "source_type",
    "source_url",
    "dataset_tier",
    "inclusion_status",
    "verification_status",
    "evidence_support_score",
    "parent_text_id",
    "decision_note",
)

CANDIDATE_COMBINATION_COLUMNS: tuple[str, ...] = (
    "source_type",
    "aspect",
)

CANDIDATE_LABELING_DATASET_COLUMNS: tuple[str, ...] = (
    "text_id",
    "source_url",
    "source_type",
    "aspect",
    "dataset_tier",
    "location",
    "location_source",
    "location_match",
    "is_specific_location",
    "labeling_bucket_column",
    "labeling_bucket_value",
    "source_id",
    "raw_source_file",
    "raw_domain",
    "content_status",
    "raw_title",
    "raw_text_length",
    "query_group",
    "query",
    "subjectivity_type",
    "speaker_type",
    "public_opinion_scope",
    "corpus_role",
    "inclusion_status",
    "verification_status",
    "evidence_support_score",
    "parent_text_id",
    "decision_note",
    "sentiment_label",
    "label_status",
    "text",
)


class DatasetService:
    """Layanan pengelolaan dataset mentah hingga siap diproses."""

    def __init__(
        self,
        required_columns: tuple[str, ...] = config.REQUIRED_COLUMNS,
        encoding: str = config.ENCODING,
        encoding_fallback: str = config.ENCODING_FALLBACK,
    ) -> None:
        self.required_columns = required_columns
        self.encoding = encoding
        self.encoding_fallback = encoding_fallback

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    def load(self, path: str | Path) -> pl.DataFrame:
        """Muat dataset CSV menjadi ``pl.DataFrame``.

        Mencoba decode dengan ``utf-8`` lalu fallback ke ``latin-1`` untuk
        data informal berbahasa Indonesia yang sering tidak konsisten
        encoding-nya. Kolom identitas/teks dipaksa bertipe string agar
        inferensi skema Polars tidak salah menebak (mis. teks numerik).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")

        try:
            df = pl.read_csv(path, encoding=self.encoding, infer_schema_length=10_000)
        except (UnicodeDecodeError, pl.exceptions.ComputeError):
            logger.warning(
                "Decode %s gagal, fallback ke %s", self.encoding, self.encoding_fallback
            )
            import io

            raw = path.read_bytes().decode(self.encoding_fallback)
            df = pl.read_csv(io.StringIO(raw), infer_schema_length=10_000)

        # Pastikan kolom kunci bertipe string bila ada.
        casts = [
            pl.col(c).cast(pl.Utf8, strict=False)
            for c in self.required_columns
            if c in df.columns
        ]
        if casts:
            df = df.with_columns(casts)

        logger.info("Dataset dimuat: %s (%d baris)", path, df.height)
        return df

    # ------------------------------------------------------------------
    # URL discovery raw dataset
    # ------------------------------------------------------------------
    def load_research_config(self, path: str | Path) -> dict:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Research config tidak ditemukan: {path}")
        data = json.loads(path.read_text(encoding=self.encoding))
        if not isinstance(data, dict):
            raise ValueError(f"Research config harus berupa object JSON: {path}")
        return data

    def load_source_url_blacklist(self, path: str | Path) -> tuple[str, ...]:
        path = Path(path)
        if not path.exists():
            return ()
        data = json.loads(path.read_text(encoding=self.encoding))
        if not isinstance(data, list):
            raise ValueError(f"Source URL blacklist harus berupa array JSON: {path}")
        return tuple(
            str(item).strip()
            for item in data
            if str(item).strip()
        )

    def load_used_candidate_source_urls(
        self,
        folder: str | Path,
        *,
        pattern: str = "v*_candidate_labeling_dataset.csv",
        exclude_paths: tuple[str | Path, ...] | list[str | Path] = (),
    ) -> tuple[str, ...]:
        folder = Path(folder)
        if not folder.exists():
            return ()

        excluded = {
            Path(path).resolve()
            for path in exclude_paths
            if str(path).strip()
        }
        source_urls: dict[str, None] = {}

        for path in sorted(folder.glob(pattern)):
            if path.resolve() in excluded:
                continue

            try:
                with path.open("r", encoding=self.encoding, newline="") as handle:
                    reader = csv.DictReader(handle)
                    if "source_url" not in (reader.fieldnames or []):
                        continue
                    for row in reader:
                        source_url = str(row.get("source_url") or "").strip()
                        if source_url:
                            source_urls.setdefault(source_url, None)
            except UnicodeDecodeError:
                with path.open(
                    "r",
                    encoding=config.ENCODING_FALLBACK,
                    newline="",
                ) as handle:
                    reader = csv.DictReader(handle)
                    if "source_url" not in (reader.fieldnames or []):
                        continue
                    for row in reader:
                        source_url = str(row.get("source_url") or "").strip()
                        if source_url:
                            source_urls.setdefault(source_url, None)

        return tuple(source_urls)

    def apply_source_url_blacklist(
        self,
        df: pl.DataFrame,
        source_url_blacklist: tuple[str, ...] | list[str],
    ) -> pl.DataFrame:
        if not source_url_blacklist or "source_url" not in df.columns:
            return df
        return df.filter(~pl.col("source_url").is_in(list(source_url_blacklist)))

    def enrich_source_blacklist_status(
        self,
        df: pl.DataFrame,
        blacklist_service: SourceBlacklistService,
    ) -> pl.DataFrame:
        if df.is_empty():
            return df

        rows: list[dict] = []
        for row in df.iter_rows(named=True):
            decision = blacklist_service.classify(row)
            rows.append(
                {
                    **row,
                    **decision,
                    "blacklist_reason_codes": json.dumps(
                        decision["blacklist_reason_codes"],
                        ensure_ascii=False,
                    ),
                }
            )
        return pl.from_dicts(rows, strict=False)

    def filter_clear_source_candidates(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty() or "blacklist_status" not in df.columns:
            return df
        return df.filter(pl.col("blacklist_status") == "clear")

    def filter_blacklist_audit_candidates(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty() or "blacklist_status" not in df.columns:
            return pl.DataFrame()
        return df.filter(pl.col("blacklist_status") != "clear")

    def load_url_discovery_meta(self, folder: str | Path) -> pl.DataFrame:
        rows: list[dict] = []
        for path in self._url_discovery_files(folder):
            data = self._read_url_discovery_file(path)
            meta = data.get("meta", {})
            if isinstance(meta, dict):
                rows.append({"source_file": path.name, **meta})
        return pl.from_dicts(rows, strict=False) if rows else pl.DataFrame()

    def load_url_discovery_queries(self, folder: str | Path) -> pl.DataFrame:
        rows: list[dict] = []
        for path in self._url_discovery_files(folder):
            data = self._read_url_discovery_file(path)
            for query in data.get("queries", []) or []:
                if isinstance(query, dict):
                    rows.append({"source_file": path.name, **query})
        return pl.from_dicts(rows, strict=False) if rows else pl.DataFrame()

    def load_url_discovery_records(self, folder: str | Path) -> pl.DataFrame:
        rows: list[dict] = []
        for path in self._url_discovery_files(folder):
            data = self._read_url_discovery_file(path)
            queries = {
                str(item.get("query_id")): item
                for item in data.get("queries", []) or []
                if isinstance(item, dict) and item.get("query_id")
            }
            records = data.get("records", [])
            if not isinstance(records, list):
                raise ValueError(f"Key 'records' pada {path.name} harus berupa list")

            for record in records:
                if not isinstance(record, dict):
                    continue
                content = record.get("content") or {}
                if not isinstance(content, dict):
                    content = {}

                found_by = record.get("found_by") or []
                first_found = found_by[0] if found_by and isinstance(found_by[0], dict) else {}
                query_id = str(first_found.get("query_id") or "")
                query = queries.get(query_id, {})

                text = str(content.get("text") or "")
                rows.append(
                    {
                        "source_file": path.name,
                        "canonical_url": record.get("canonical_url") or "",
                        "url": record.get("url") or "",
                        "domain": record.get("domain") or self._domain(
                            record.get("canonical_url") or record.get("url") or ""
                        ),
                        "raw_title": record.get("title") or "",
                        "content_title": content.get("title") or "",
                        "snippet": record.get("snippet") or "",
                        "published_date_hint": record.get("published_date_hint") or "",
                        "first_found_at": record.get("first_found_at") or "",
                        "last_found_at": record.get("last_found_at") or "",
                        "is_blacklisted": bool(record.get("is_blacklisted")),
                        "content_status": content.get("status") or "",
                        "content_text": text,
                        "content_text_length": len(text),
                        "content_description": content.get("description") or "",
                        "content_published_at": content.get("published_at") or "",
                        "content_type": content.get("content_type") or "",
                        "http_status": content.get("http_status"),
                        "extraction_method": content.get("extraction_method") or "",
                        "query_id": query_id,
                        "query_group": first_found.get("query_group")
                        or query.get("group")
                        or "",
                        "query": first_found.get("query") or query.get("query") or "",
                        "query_status": query.get("status") or "",
                        "found_by_count": len(found_by),
                    }
                )

        if not rows:
            return pl.DataFrame()
        return pl.from_dicts(rows, strict=False).unique(
            subset=["canonical_url"], keep="first", maintain_order=True
        )

    def build_v1_candidate_rows(
        self,
        records_df: pl.DataFrame,
        research_config: dict,
    ) -> pl.DataFrame:
        rows: list[dict] = []
        if records_df.is_empty():
            return pl.DataFrame()

        for index, row in enumerate(records_df.iter_rows(named=True), start=1):
            candidate = self._build_v1_candidate_row(
                index=index,
                row=row,
                research_config=research_config,
            )
            rows.append(candidate)

        return pl.from_dicts(rows, strict=False)

    @staticmethod
    def v1_dataset_columns() -> tuple[str, ...]:
        return V1_SHS_DATASET_COLUMNS

    @staticmethod
    def candidate_combination_columns() -> tuple[str, ...]:
        return CANDIDATE_COMBINATION_COLUMNS

    def build_candidate_labeling_dataset(
        self,
        candidate_df: pl.DataFrame,
        max_rows_per_value: int = 10,
    ) -> pl.DataFrame:
        """Ambil kandidat siap labeling per nilai unik tiap kolom kombinasi."""
        if candidate_df.is_empty():
            return candidate_df
        if max_rows_per_value <= 0:
            return pl.DataFrame()

        eligible_df = self.filter_labeling_ready_candidates(candidate_df)
        if eligible_df.is_empty():
            return eligible_df

        groups = self._candidate_labeling_groups(eligible_df)
        ordered_keys = sorted(groups, key=lambda key: (len(groups[key]), key[0], key[1]))
        sorted_rows_by_key = {
            key: sorted(rows, key=self._candidate_labeling_sort_key)
            for key, rows in groups.items()
        }

        selected: list[dict] = []
        blacklisted_ids: set[str] = set()
        for column, value in ordered_keys:
            selected_count = 0
            for row in sorted_rows_by_key[(column, value)]:
                selection_id = self._candidate_selection_id(row, len(selected))
                if selection_id in blacklisted_ids:
                    continue
                selected_row = {
                    **row,
                    "labeling_bucket_column": column,
                    "labeling_bucket_value": value,
                }
                selected.append(selected_row)
                blacklisted_ids.add(selection_id)
                selected_count += 1
                if selected_count >= max_rows_per_value:
                    break

        if not selected:
            return pl.DataFrame()
        return self._order_candidate_labeling_columns(
            pl.from_dicts(selected, strict=False)
        )

    def _candidate_labeling_groups(
        self,
        eligible_df: pl.DataFrame,
    ) -> dict[tuple[str, str], list[dict]]:
        groups: dict[tuple[str, str], list[dict]] = {}
        for row in eligible_df.iter_rows(named=True):
            for column in CANDIDATE_COMBINATION_COLUMNS:
                value = str(row.get(column) or "").strip()
                if value:
                    groups.setdefault((column, value), []).append(row)
        return groups

    @staticmethod
    def _order_candidate_labeling_columns(labeling_df: pl.DataFrame) -> pl.DataFrame:
        ordered_columns = [
            column
            for column in CANDIDATE_LABELING_DATASET_COLUMNS
            if column in labeling_df.columns
        ]
        extra_columns = [
            column
            for column in labeling_df.columns
            if column not in ordered_columns and column != "text"
        ]
        final_columns = ordered_columns + extra_columns
        if "text" in labeling_df.columns and final_columns[-1:] != ["text"]:
            final_columns = [column for column in final_columns if column != "text"]
            final_columns.append("text")
        return labeling_df.select(final_columns)

    def filter_labeling_ready_candidates(self, candidate_df: pl.DataFrame) -> pl.DataFrame:
        """Saring kandidat yang punya lokasi Kalbar spesifik dari text atau query."""
        if candidate_df.is_empty():
            return candidate_df
        required = {
            *CANDIDATE_COMBINATION_COLUMNS,
            "dataset_tier",
            "text",
            "location",
            "location_source",
            "is_specific_location",
        }
        missing = required.difference(candidate_df.columns)
        if missing:
            raise ValueError(f"Kolom kandidat siap labeling belum lengkap: {sorted(missing)}")

        return candidate_df.filter(
            pl.col("dataset_tier").is_not_null()
            & (pl.col("dataset_tier").str.strip_chars().str.len_chars() > 0)
            & pl.col("dataset_tier").is_in(["A_candidate_core", "B_review_queue"])
            & pl.col("source_type").is_not_null()
            & (pl.col("source_type").str.strip_chars().str.len_chars() > 0)
            & pl.col("aspect").is_not_null()
            & (pl.col("aspect").str.strip_chars().str.len_chars() > 0)
            & pl.col("text").is_not_null()
            & (pl.col("text").str.strip_chars().str.len_chars() > 0)
            & pl.col("location").is_not_null()
            & (pl.col("location").str.strip_chars().str.len_chars() > 0)
            & (pl.col("is_specific_location") == True)
            & pl.col("location_source").is_in(["text", "query"])
        )

    def summarize_candidate_combinations(self, candidate_df: pl.DataFrame) -> pl.DataFrame:
        if candidate_df.is_empty():
            return pl.DataFrame()
        missing = [
            column
            for column in CANDIDATE_COMBINATION_COLUMNS
            if column not in candidate_df.columns
        ]
        if missing:
            raise ValueError(f"Kolom kombinasi kandidat belum lengkap: {missing}")

        rows: list[dict] = []
        for column in CANDIDATE_COMBINATION_COLUMNS:
            grouped = (
                candidate_df.group_by(column)
                .agg(pl.len().alias("jumlah"))
                .sort(column)
            )
            for row in grouped.iter_rows(named=True):
                value = str(row.get(column) or "").strip()
                if not value:
                    continue
                rows.append(
                    {
                        "combination_column": column,
                        "combination_value": value,
                        "jumlah": int(row.get("jumlah") or 0),
                    }
                )

        if not rows:
            return pl.DataFrame()
        return pl.from_dicts(rows, strict=False)

    def summarize_labeling_selection_buckets(
        self,
        labeling_df: pl.DataFrame,
    ) -> pl.DataFrame:
        if labeling_df.is_empty():
            return pl.DataFrame()
        required = {"labeling_bucket_column", "labeling_bucket_value"}
        if missing := required.difference(labeling_df.columns):
            raise ValueError(f"Kolom bucket labeling belum lengkap: {sorted(missing)}")

        rows: list[dict] = []
        grouped = (
            labeling_df.group_by(["labeling_bucket_column", "labeling_bucket_value"])
            .agg(pl.len().alias("jumlah"))
            .sort(["labeling_bucket_column", "labeling_bucket_value"])
        )
        for row in grouped.iter_rows(named=True):
            column = str(row.get("labeling_bucket_column") or "").strip()
            value = str(row.get("labeling_bucket_value") or "").strip()
            if not column or not value:
                continue
            rows.append(
                {
                    "combination_column": column,
                    "combination_value": value,
                    "jumlah": int(row.get("jumlah") or 0),
                }
            )

        if not rows:
            return pl.DataFrame()
        return pl.from_dicts(rows, strict=False)

    def _build_v1_candidate_row(
        self,
        index: int,
        row: dict,
        research_config: dict,
    ) -> dict:
        raw_text = str(row.get("content_text") or "")
        content_title = str(row.get("content_title") or "")
        raw_title = str(row.get("raw_title") or "")
        snippet = str(row.get("snippet") or "")
        description = str(row.get("content_description") or "")
        domain = str(row.get("domain") or "")
        source_url = str(row.get("canonical_url") or row.get("url") or "")
        content_status = str(row.get("content_status") or "")
        cleaned_raw_text = self._clean_content_text(raw_text, source_url)
        url_title = self._title_from_url(source_url)
        text = self._candidate_text(
            content_title=content_title,
            raw_title=raw_title,
            snippet=snippet,
            description=description,
            url_title=url_title,
            raw_text=cleaned_raw_text,
        )
        combined = text

        is_success = content_status == "success" and bool(raw_text.strip())
        is_excluded_domain = self._matches_domain(
            domain, source_url, research_config.get("excluded_domains", [])
        )
        is_social = self._matches_domain(
            domain, source_url, research_config.get("social_domains", [])
        )
        is_relevant = self._contains_any(
            " ".join([source_url, text, cleaned_raw_text[:2000]]),
            research_config.get("url_required_keywords", []),
        )
        location_match = self._infer_location_match(
            [
                ("text", combined),
                ("query", str(row.get("query") or "")),
                ("source_url", source_url),
            ],
            research_config=research_config,
        )
        location = location_match["location"]
        aspect = self._infer_aspect(combined, research_config)
        source_type = self._infer_source_type(domain, source_url, is_social)

        if not is_success or is_excluded_domain:
            dataset_tier = "C_holdout_excluded"
            inclusion_status = "held_out_not_for_sentiment_core"
            corpus_role = "excluded"
        elif is_relevant and not is_social:
            dataset_tier = "A_candidate_core"
            inclusion_status = "candidate_analysis_ready"
            corpus_role = "core_public_opinion"
        else:
            dataset_tier = "B_review_queue"
            inclusion_status = "review_required_before_core"
            corpus_role = "contextual_evidence"

        support_score = self._evidence_support_score(
            is_success=is_success,
            is_relevant=is_relevant,
            location=location,
            is_social=is_social,
            is_excluded_domain=is_excluded_domain,
            text=text,
        )

        final_row = {
            "text_id": f"RAW-{index:04d}",
            "text": text,
            "subjectivity_type": self._infer_subjectivity_type(
                combined, source_type, dataset_tier
            ),
            "speaker_type": self._infer_speaker_type(source_type),
            "public_opinion_scope": self._infer_opinion_scope(source_type, dataset_tier),
            "corpus_role": corpus_role,
            "aspect": aspect,
            "location": location,
            "sentiment_label": "",
            "label_status": "unlabeled",
            "source_id": f"RAW-SRC-{index:04d}",
            "source_type": source_type,
            "source_url": source_url,
            "dataset_tier": dataset_tier,
            "inclusion_status": inclusion_status,
            "verification_status": "perlu_verifikasi",
            "evidence_support_score": support_score,
            "parent_text_id": "",
            "decision_note": self._decision_note(
                is_success=is_success,
                is_relevant=is_relevant,
                is_social=is_social,
                is_excluded_domain=is_excluded_domain,
            ),
            "raw_source_file": row.get("source_file") or "",
            "raw_domain": domain,
            "content_status": content_status,
            "query_group": row.get("query_group") or "",
            "query": row.get("query") or "",
            "raw_title": raw_title or content_title,
            "raw_text_length": int(row.get("content_text_length") or 0),
            "location_source": location_match["source"],
            "location_match": location_match["match"],
            "is_specific_location": location_match["is_specific"],
        }
        return final_row

    @staticmethod
    def _url_discovery_files(folder: str | Path) -> list[Path]:
        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError(f"Folder raw dataset tidak ditemukan: {folder}")
        return sorted(path for path in folder.iterdir() if path.suffix.lower() == ".json")

    def _read_url_discovery_file(self, path: Path) -> dict:
        data = json.loads(path.read_text(encoding=self.encoding))
        if not isinstance(data, dict):
            raise ValueError(f"Raw dataset harus berupa object JSON: {path}")
        return data

    @staticmethod
    def _domain(url: str) -> str:
        parsed = urlparse(url)
        return parsed.netloc.lower()

    @staticmethod
    def _contains_any(text: str, terms: list[str]) -> bool:
        normalized = text.lower()
        return any(str(term).lower() in normalized for term in terms if str(term).strip())

    @classmethod
    def _first_matching_term(cls, text: str, terms: list[str]) -> str:
        normalized = text.lower()
        matches = [
            str(term).strip()
            for term in terms
            if str(term).strip() and str(term).lower() in normalized
        ]
        if not matches:
            return ""
        return max(matches, key=len)

    @classmethod
    def _infer_location(cls, texts: list[str], research_config: dict) -> str:
        match = cls._infer_location_match(
            [(f"text_{index}", text) for index, text in enumerate(texts, start=1)],
            research_config=research_config,
        )
        return match["location"]

    @classmethod
    def _infer_location_match(
        cls,
        text_sources: list[tuple[str, str]],
        research_config: dict,
    ) -> dict:
        location_terms = cls._location_terms(research_config)
        specific_terms = [term for term in location_terms if term[3]]
        province_terms = [term for term in location_terms if not term[3]]

        for source, text in text_sources:
            match = cls._first_matching_location(text, specific_terms)
            if match:
                return {
                    "location": match["canonical"],
                    "source": source,
                    "match": match["alias"],
                    "is_specific": True,
                }
        for source, text in text_sources:
            match = cls._first_matching_location(text, province_terms)
            if match:
                return {
                    "location": match["canonical"],
                    "source": source,
                    "match": match["alias"],
                    "is_specific": False,
                }
        return {"location": "", "source": "", "match": "", "is_specific": False}

    @classmethod
    def _location_terms(cls, research_config: dict) -> list[tuple[str, str, bool, bool]]:
        terms: dict[str, str] = {}
        primary_location = str(research_config.get("primary_location") or "").strip()
        specific_locations = cls._kalbar_specific_location_names()
        specific_location_keys = {location.casefold() for location in specific_locations}

        for canonical, aliases in (
            research_config.get("location_aliases") or {}
        ).items():
            canonical_text = str(canonical).strip()
            if not canonical_text:
                continue
            terms.setdefault(canonical_text.casefold(), canonical_text)
            if not isinstance(aliases, list):
                continue
            for alias in aliases:
                alias_text = str(alias).strip()
                if alias_text:
                    terms.setdefault(alias_text.casefold(), canonical_text)

        for location in research_config.get("locations") or []:
            location_text = str(location).strip()
            if location_text:
                terms.setdefault(location_text.casefold(), location_text)

        for entity in research_config.get("known_entities") or []:
            entity_text = str(entity).strip()
            if entity_text:
                terms.setdefault(entity_text.casefold(), entity_text)

        for location in specific_locations:
            terms.setdefault(location.casefold(), location)

        return sorted(
            (
                (
                    alias,
                    canonical,
                    bool(primary_location)
                    and canonical.casefold() == primary_location.casefold(),
                    canonical.casefold() in specific_location_keys
                    or alias in specific_location_keys,
                )
                for alias, canonical in terms.items()
            ),
            key=lambda item: (item[3], len(item[0])),
            reverse=True,
        )

    @classmethod
    def _kalbar_specific_location_names(cls) -> set[str]:
        province_code = cls._province_code("KALIMANTAN BARAT")
        if not province_code:
            return set()

        regencies: dict[str, str] = {}
        for row in cls._read_wilayah_csv(config.RESOURCES / "wilayah" / "kabupaten.csv"):
            if row.get("parent_code") == province_code:
                code = str(row.get("code") or "").strip()
                name = cls._canonical_location_name(row.get("name") or "")
                if code and name:
                    regencies[code] = name

        locations = set(regencies.values())
        for row in cls._read_wilayah_csv(config.RESOURCES / "wilayah" / "kecamatan.csv"):
            if row.get("parent_code") in regencies:
                name = cls._canonical_location_name(row.get("name") or "")
                if name:
                    locations.add(name)
        return locations

    @classmethod
    def _province_code(cls, province_name: str) -> str:
        target = province_name.casefold()
        for row in cls._read_wilayah_csv(config.RESOURCES / "wilayah" / "provinsi.csv"):
            if str(row.get("name") or "").casefold() == target:
                return str(row.get("code") or "").strip()
        return ""

    @staticmethod
    def _read_wilayah_csv(path: Path) -> list[dict[str, str]]:
        if not path.exists():
            return []
        with path.open(encoding=config.ENCODING, newline="") as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def _canonical_location_name(name: str) -> str:
        value = re.sub(r"^(KABUPATEN|KOTA|KECAMATAN)\s+", "", str(name or "").strip())
        return value.title()

    @staticmethod
    def _first_matching_location(
        text: str, location_terms: list[tuple[str, str, bool, bool]]
    ) -> dict[str, str]:
        normalized = str(text or "").casefold()
        if not normalized:
            return {}
        for alias, canonical, _is_primary_scope, _is_specific in location_terms:
            if DatasetService._contains_location_alias(normalized, alias):
                return {"canonical": canonical, "alias": alias}
        return {}

    @staticmethod
    def _contains_location_alias(normalized_text: str, alias: str) -> bool:
        escaped = re.escape(str(alias or "").casefold())
        if not escaped:
            return False
        pattern = rf"(?<![0-9A-Za-zÀ-ſ]){escaped}(?![0-9A-Za-zÀ-ſ])"
        return re.search(pattern, normalized_text) is not None

    @staticmethod
    def _candidate_labeling_sort_key(row: dict) -> tuple:
        score = float(row.get("evidence_support_score") or 0.0)
        text = str(row.get("text") or "").strip()
        location = str(row.get("location") or "").strip()
        text_id = str(row.get("text_id") or "")
        raw_text_length = int(row.get("raw_text_length") or 0)
        return (-score, -int(bool(text)), -int(bool(location)), -raw_text_length, text_id)

    @staticmethod
    def _candidate_selection_id(
        row: dict,
        fallback_index: int,
    ) -> str:
        text_id = str(row.get("text_id") or "")
        return text_id or f"candidate:{fallback_index}"

    @staticmethod
    def _matches_domain(domain: str, url: str, patterns: list[str]) -> bool:
        haystack = f"{domain} {url}".lower()
        return any(str(pattern).lower() in haystack for pattern in patterns)

    @classmethod
    def _infer_aspect(cls, text: str, research_config: dict) -> str:
        for group_name, terms in (research_config.get("issue_groups") or {}).items():
            if cls._contains_any(text, terms):
                return str(group_name)
        for group_name, terms in (research_config.get("topic_groups") or {}).items():
            if cls._contains_any(text, terms):
                return str(group_name)
        return "general_shs"

    @staticmethod
    def _infer_source_type(domain: str, url: str, is_social: bool) -> str:
        value = f"{domain} {url}".lower()
        if is_social:
            return "social_media"
        if any(term in value for term in ("repository", "journal", "researchgate", "ac.id")):
            return "academic_repository"
        if any(term in value for term in ("go.id", "esdm.go.id", "pln.co.id")):
            return "government_official"
        if any(term in value for term in ("company", "corporate")):
            return "corporate_official"
        return "online_news"

    @staticmethod
    def _infer_speaker_type(source_type: str) -> str:
        if source_type == "academic_repository":
            return "researcher"
        if source_type == "government_official":
            return "public_official"
        if source_type == "corporate_official":
            return "company_representative"
        if source_type == "social_media":
            return "public_user"
        return "community_representative"

    @staticmethod
    def _infer_subjectivity_type(text: str, source_type: str, dataset_tier: str) -> str:
        normalized = text.lower()
        if dataset_tier == "C_holdout_excluded":
            return "contextual_source"
        if source_type == "academic_repository":
            return "researcher_assessment"
        if source_type in ("government_official", "corporate_official"):
            return "institutional_claim"
        if any(term in normalized for term in ("harap", "dambakan", "ingin", "butuh")):
            return "public_expectation"
        return "public_experience"

    @staticmethod
    def _infer_opinion_scope(source_type: str, dataset_tier: str) -> str:
        if dataset_tier == "C_holdout_excluded":
            return "contextual_reference"
        if source_type == "academic_repository":
            return "researcher_or_author_assessment"
        if source_type in ("government_official", "corporate_official"):
            return "stakeholder_opinion"
        return "public_opinion"

    @classmethod
    def _candidate_text(
        cls,
        *,
        content_title: str,
        raw_title: str,
        snippet: str,
        description: str,
        url_title: str,
        raw_text: str,
    ) -> str:
        segments = [
            content_title,
            raw_title,
            snippet,
            description,
            url_title,
            raw_text,
        ]
        return " ".join(cls._unique_text_segments(segments))

    @classmethod
    def _unique_text_segments(cls, segments: list[str]) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for segment in segments:
            normalized = cls._normalize_text(segment)
            if not normalized:
                continue
            key = normalized.casefold()
            if key in seen:
                continue
            seen.add(key)
            unique.append(normalized)
        return unique

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip()

    @classmethod
    def _clean_content_text(cls, raw_text: str, source_url: str) -> str:
        text = cls._normalize_text(raw_text)
        if not text:
            return ""

        normalized = text.casefold()
        compact = re.sub(r"\s+", "", normalized)
        domain = urlparse(source_url).netloc.casefold()

        instagram_patterns = (
            "loginsignupnevermissapostfrom",
            "signupforinstagramtostayintheloop",
            "masukdaftarjanganpernahlewatkanpostingan",
            "post isn't available",
            "the link may be broken, or the profile may have been removed",
            "see everyday moments from your close friends",
            "log into instagram",
        )
        tiktok_patterns = (
            "drag the slider to fit the puzzle",
            "access to www.tiktok.com was denied",
            "you don't have authorization to view this page",
        )

        if "instagram.com" in domain and len(text) <= 1_200:
            if any(
                pattern in compact or pattern in normalized
                for pattern in instagram_patterns
            ):
                return ""
        if "tiktok.com" in domain and len(text) <= 800:
            if any(pattern in normalized for pattern in tiktok_patterns):
                return ""
        return text

    @classmethod
    def _title_from_url(cls, source_url: str) -> str:
        parsed = urlparse(source_url)
        path_parts = [
            unquote(part).strip()
            for part in parsed.path.split("/")
            if unquote(part).strip()
        ]
        if not path_parts:
            return ""

        if "instagram.com" in parsed.netloc.casefold() and len(path_parts) >= 2:
            return cls._normalize_url_title(" ".join(path_parts[:2]))
        if "tiktok.com" in parsed.netloc.casefold():
            relevant_parts = [
                part for part in path_parts if part not in ("video", "photo")
            ]
            return cls._normalize_url_title(" ".join(relevant_parts or path_parts))
        return cls._normalize_url_title(" ".join(path_parts[-2:]))

    @staticmethod
    def _normalize_url_title(text: str) -> str:
        text = re.sub(r"[_-]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _evidence_support_score(
        *,
        is_success: bool,
        is_relevant: bool,
        location: str,
        is_social: bool,
        is_excluded_domain: bool,
        text: str,
    ) -> float:
        score = 0.0
        if is_success:
            score += 0.35
        if is_relevant:
            score += 0.25
        if location:
            score += 0.2
        if len(text) >= 80:
            score += 0.1
        if not is_social and not is_excluded_domain:
            score += 0.1
        return round(min(score, 1.0), 2)

    @staticmethod
    def _decision_note(
        *,
        is_success: bool,
        is_relevant: bool,
        is_social: bool,
        is_excluded_domain: bool,
    ) -> str:
        reasons: list[str] = []
        if not is_success:
            reasons.append("content_not_success")
        if not is_relevant:
            reasons.append("keyword_relevance_needs_review")
        if is_social:
            reasons.append("social_domain_review")
        if is_excluded_domain:
            reasons.append("excluded_domain")
        if not reasons:
            reasons.append("candidate_from_raw_discovery")
        return "; ".join(reasons)

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------
    def validate(self, df: pl.DataFrame) -> dict:
        """Periksa kolom wajib dan integritas dasar ``text`` / ``source``.

        Mengembalikan ringkasan hasil validasi (bukan melempar exception)
        agar pemanggil dapat memutuskan tindakan lanjutan.
        """
        issues: list[str] = []
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            issues.append(f"Kolom wajib hilang: {missing}")

        null_text = empty_text = empty_source = 0

        if config.COL_TEXT in df.columns:
            null_text = int(df.select(pl.col(config.COL_TEXT).is_null().sum()).item())
            empty_text = int(
                df.filter(
                    pl.col(config.COL_TEXT).is_not_null()
                    & (pl.col(config.COL_TEXT).str.strip_chars().str.len_chars() == 0)
                ).height
            )
            if null_text:
                issues.append(f"{null_text} baris memiliki text null")
            if empty_text:
                issues.append(f"{empty_text} baris memiliki text kosong")

        if config.COL_SOURCE in df.columns:
            empty_source = int(
                df.filter(
                    pl.col(config.COL_SOURCE).is_null()
                    | (pl.col(config.COL_SOURCE).str.strip_chars().str.len_chars() == 0)
                ).height
            )
            if empty_source:
                issues.append(f"{empty_source} baris memiliki source kosong/null")

        return {
            "total_rows": df.height,
            "required_columns": list(self.required_columns),
            "missing_columns": missing,
            "null_text": null_text,
            "empty_text": empty_text,
            "empty_source": empty_source,
            "is_valid": not issues,
            "issues": issues,
        }

    # ------------------------------------------------------------------
    # Deduplicate
    # ------------------------------------------------------------------
    def deduplicate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Hapus duplikat berdasarkan (``source``, ``text``).

        Mengembalikan DataFrame bersih. Jumlah awal / duplikat / akhir
        dicatat melalui logger; hitungan yang sama juga tersedia pada
        :meth:`build_summary`.
        """
        subset = [c for c in (config.COL_SOURCE, config.COL_TEXT) if c in df.columns]
        if not subset:
            logger.warning(
                "Kolom dedup tidak tersedia; DataFrame dikembalikan apa adanya"
            )
            return df

        before = df.height
        clean = df.unique(subset=subset, keep="first", maintain_order=True)
        after = clean.height
        duplicates = before - after
        logger.info(
            "Deduplikasi: awal=%d duplikat=%d akhir=%d", before, duplicates, after
        )
        return clean

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def build_summary(self, df: pl.DataFrame) -> dict:
        total = df.height

        # Distribusi sumber data
        distribusi: dict[str, int] = {}
        if config.COL_SOURCE in df.columns and total > 0:
            grouped = df.group_by(config.COL_SOURCE).agg(pl.len().alias("count"))
            for row in grouped.iter_rows(named=True):
                key = row[config.COL_SOURCE]
                distribusi[str(key) if key is not None else "null"] = int(row["count"])

        # Jumlah data kosong (text null atau hanya whitespace)
        jumlah_kosong = 0
        rata_panjang = 0.0
        if config.COL_TEXT in df.columns and total > 0:
            jumlah_kosong = int(
                df.filter(
                    pl.col(config.COL_TEXT).is_null()
                    | (pl.col(config.COL_TEXT).str.strip_chars().str.len_chars() == 0)
                ).height
            )
            mean_len = df.select(pl.col(config.COL_TEXT).str.len_chars().mean()).item()
            rata_panjang = round(float(mean_len), 2) if mean_len is not None else 0.0

        # Jumlah duplikat berdasarkan (source, text)
        jumlah_duplikat = 0
        subset = [c for c in (config.COL_SOURCE, config.COL_TEXT) if c in df.columns]
        if subset and total > 0:
            jumlah_duplikat = total - df.unique(subset=subset).height

        return {
            "total_data": total,
            "distribusi_sumber": distribusi,
            "jumlah_data_kosong": jumlah_kosong,
            "jumlah_duplikat": jumlah_duplikat,
            "rata_rata_panjang_teks": rata_panjang,
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export_summary(
        self, summary: dict, path: str | Path = config.DATASET_SUMMARY_PATH
    ) -> Path:
        """Simpan ringkasan dataset ke file JSON (default outputs/artifacts/)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding=config.ENCODING
        )
        logger.info("Ringkasan dataset diekspor: %s", path)
        return path
