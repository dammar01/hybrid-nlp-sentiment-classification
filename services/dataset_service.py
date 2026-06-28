"""DatasetService — pemuatan, validasi, deduplikasi, dan ringkasan dataset.

Menangani data opini masyarakat mengenai Solar Home System (SHS) di
Kalimantan Barat. Seluruh operasi DataFrame menggunakan Polars (bukan pandas)
sesuai rancangan tugas akhir.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from urllib.parse import urlparse

import polars as pl

import config

logger = logging.getLogger(__name__)

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
        max_text_chars: int = 260,
    ) -> pl.DataFrame:
        rows: list[dict] = []
        if records_df.is_empty():
            return pl.DataFrame()

        for index, row in enumerate(records_df.iter_rows(named=True), start=1):
            candidate = self._build_v1_candidate_row(
                index=index,
                row=row,
                research_config=research_config,
                max_text_chars=max_text_chars,
            )
            rows.append(candidate)

        return pl.from_dicts(rows, strict=False)

    @staticmethod
    def v1_dataset_columns() -> tuple[str, ...]:
        return V1_SHS_DATASET_COLUMNS

    def _build_v1_candidate_row(
        self,
        index: int,
        row: dict,
        research_config: dict,
        max_text_chars: int,
    ) -> dict:
        raw_text = str(row.get("content_text") or "")
        title = str(row.get("content_title") or row.get("raw_title") or "")
        snippet = str(row.get("snippet") or "")
        description = str(row.get("content_description") or "")
        combined = " ".join([title, snippet, description, raw_text])
        domain = str(row.get("domain") or "")
        source_url = str(row.get("canonical_url") or row.get("url") or "")
        content_status = str(row.get("content_status") or "")

        is_success = content_status == "success" and bool(raw_text.strip())
        is_excluded_domain = self._matches_domain(
            domain, source_url, research_config.get("excluded_domains", [])
        )
        is_social = self._matches_domain(
            domain, source_url, research_config.get("social_domains", [])
        )
        is_relevant = self._contains_any(
            " ".join([source_url, title, snippet, description, raw_text[:2000]]),
            research_config.get("url_required_keywords", []),
        )
        location = self._first_matching_term(
            combined,
            list(research_config.get("known_entities", []))
            + list(research_config.get("locations", [])),
        )
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

        text = self._candidate_text(raw_text, title, snippet, max_text_chars)
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
            "raw_title": title,
            "raw_text_length": int(row.get("content_text_length") or 0),
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

    @staticmethod
    def _candidate_text(
        raw_text: str, title: str, snippet: str, max_text_chars: int
    ) -> str:
        source = raw_text.strip() or snippet.strip() or title.strip()
        source = re.sub(r"\s+", " ", source)
        if len(source) <= max_text_chars:
            return source
        trimmed = source[:max_text_chars].rsplit(" ", 1)[0].strip()
        return trimmed or source[:max_text_chars].strip()

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
