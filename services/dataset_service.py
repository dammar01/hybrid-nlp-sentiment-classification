"""DatasetService — pemuatan, validasi, deduplikasi, dan ringkasan dataset.

Menangani data opini masyarakat mengenai Solar Home System (SHS) di
Kalimantan Barat. Seluruh operasi DataFrame menggunakan Polars (bukan pandas)
sesuai rancangan tugas akhir.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import polars as pl

import config

logger = logging.getLogger(__name__)


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
