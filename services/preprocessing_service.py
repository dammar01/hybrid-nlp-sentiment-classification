"""PreprocessingService — pembersihan teks opini SHS Kalimantan Barat.

Preprocessing dirancang konservatif: menjaga makna asli opini dan TIDAK
melakukan stemming, lemmatization, stopword removal, POS tagging, NER, atau
penyaringan token agresif. Tahapan mengikuti laporan tugas akhir.

Urutan pipeline:
    URL Removal
    -> Symbol Normalization
    -> Case Folding
    -> Slang Normalization
    -> Non-Standard Word Normalization
    -> Negation Preservation
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import polars as pl

import config

logger = logging.getLogger(__name__)

# Bentuk slang/singkatan negasi -> bentuk baku. Dijaga terpisah agar makna
# negasi tetap utuh walau kamus slang umum belum lengkap.
NEGATION_MAP: dict[str, str] = {
    "gak": "tidak", "ga": "tidak", "gk": "tidak", "nggak": "tidak",
    "enggak": "tidak", "ngga": "tidak", "tdk": "tidak", "kagak": "tidak",
    "ndak": "tidak", "tak": "tidak",
    "bkn": "bukan", "bukn": "bukan",
    "blm": "belum", "blom": "belum", "belom": "belum",
    "jgn": "jangan", "jngn": "jangan",
}

# Regex modul-level (dikompilasi sekali).
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Sisakan huruf (termasuk beraksen), angka, spasi, dan tanda baca dasar.
_ALLOWED_RE = re.compile(r"[^0-9A-Za-zÀ-ſ\s.,!?'\-]")
_MULTI_PUNCT_RE = re.compile(r"([.,!?'\-])\1+")
_MULTI_SPACE_RE = re.compile(r"\s+")
# Token kata (mendukung apostrof/hubung di tengah).
_WORD_RE = re.compile(r"[A-Za-zÀ-ſ]+(?:['\-][A-Za-zÀ-ſ]+)*")


class PreprocessingService:
    """Pipeline preprocessing teks berbahasa Indonesia informal."""

    def __init__(
        self,
        slang_path: str | Path = config.SLANG_DICT_PATH,
        non_standard_path: str | Path = config.NON_STANDARD_DICT_PATH,
    ) -> None:
        self.slang = self._load_dict(slang_path)
        self.non_standard = self._load_dict(non_standard_path)

    @staticmethod
    def _load_dict(path: str | Path) -> dict[str, str]:
        """Muat kamus JSON {kata: bentuk_baku} dengan kunci di-lowercase."""
        path = Path(path)
        if not path.exists():
            logger.warning("Kamus tidak ditemukan: %s (dilewati)", path)
            return {}
        data = json.loads(path.read_text(encoding=config.ENCODING))
        return {str(k).lower(): str(v) for k, v in data.items()}

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    @staticmethod
    def _replace_words(text: str, mapping: dict[str, str]) -> str:
        """Ganti token kata sesuai ``mapping`` (lookup case-insensitive)."""
        if not text or not mapping:
            return text

        def repl(match: re.Match[str]) -> str:
            word = match.group(0)
            return mapping.get(word.lower(), word)

        return _WORD_RE.sub(repl, text)

    @staticmethod
    def _collapse_whitespace(text: str) -> str:
        return _MULTI_SPACE_RE.sub(" ", text).strip()

    # ------------------------------------------------------------------
    # Tahapan
    # ------------------------------------------------------------------
    def remove_url(self, text: str) -> str:
        """Hapus URL (http/https/www) dari teks."""
        if not text:
            return ""
        return _URL_RE.sub(" ", text)

    def normalize_symbols(self, text: str) -> str:
        """Bersihkan karakter kontrol & simbol tak relevan, jaga isi opini."""
        if not text:
            return ""
        t = _CONTROL_RE.sub(" ", text)
        t = _ALLOWED_RE.sub(" ", t)
        t = _MULTI_PUNCT_RE.sub(r"\1", t)  # "!!!" -> "!"
        return self._collapse_whitespace(t)

    def case_fold(self, text: str) -> str:
        """Ubah teks menjadi huruf kecil."""
        if not text:
            return ""
        return text.lower()

    def normalize_slang(self, text: str) -> str:
        """Normalisasi kata slang ke bentuk baku via kamus slang."""
        return self._replace_words(text, self.slang)

    def normalize_non_standard_words(self, text: str) -> str:
        """Normalisasi kata tidak baku ke bentuk standar via kamus non-baku."""
        return self._replace_words(text, self.non_standard)

    def preserve_negation(self, text: str) -> str:
        """Pertahankan makna negasi.

        Menormalkan varian slang/singkatan negasi (mis. ``gak``, ``tdk``) ke
        bentuk baku (``tidak``) sehingga kata negasi tetap eksplisit untuk
        analisis sentimen berikutnya. Kata negasi baku tidak pernah dihapus
        atau diubah maknanya.
        """
        return self._replace_words(text, NEGATION_MAP)

    # ------------------------------------------------------------------
    # Pipeline penuh
    # ------------------------------------------------------------------
    def process(self, text: str) -> dict[str, str]:
        """Jalankan seluruh tahapan preprocessing pada satu teks."""
        original = "" if text is None else str(text)
        t = self.remove_url(original)
        t = self.normalize_symbols(t)
        t = self.case_fold(t)
        t = self.normalize_slang(t)
        t = self.normalize_non_standard_words(t)
        t = self.preserve_negation(t)
        t = self._collapse_whitespace(t)
        return {"original_text": original, "processed_text": t}

    def process_dataframe(
        self, df: pl.DataFrame, text_column: str = config.COL_TEXT
    ) -> pl.DataFrame:
        """Tambah kolom ``processed_text`` tanpa menghapus kolom asli."""
        if text_column not in df.columns:
            raise KeyError(f"Kolom '{text_column}' tidak ada pada DataFrame")

        processed = df.select(
            pl.col(text_column)
            .map_elements(
                lambda s: self.process(s)["processed_text"],
                return_dtype=pl.Utf8,
            )
            .alias(config.COL_PROCESSED)
        )
        return df.with_columns(processed)
