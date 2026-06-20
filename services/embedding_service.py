"""Embedding teks untuk skenario non-LLM.

Default service memakai hashed bag-of-words agar notebook tetap dapat berjalan
tanpa instalasi dependency model berat. Jika ``backend='indobert'`` dipilih dan
``transformers`` serta ``torch`` tersedia, service memakai model lokal IndoBERT.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Iterable

import polars as pl

import config

_TOKEN_RE = re.compile(r"[A-Za-zÀ-ſ]+(?:['\-][A-Za-zÀ-ſ]+)*")


class EmbeddingService:
    """Menghasilkan representasi vektor dari teks opini."""

    def __init__(
        self,
        backend: str = "hashing",
        dimension: int = config.EMBEDDING_DIMENSION,
        model_path: str | Path = config.INDOBERT_MODEL_PATH,
    ) -> None:
        self.backend = backend
        self.dimension = dimension
        self.model_path = Path(model_path)
        self._tokenizer = None
        self._model = None

        if backend == "indobert":
            self._load_indobert()

    def _load_indobert(self) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Backend indobert membutuhkan dependency transformers dan torch"
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModel.from_pretrained(self.model_path)
        self._model.eval()

    def encode_text(self, text: str) -> list[float]:
        if self.backend == "indobert":
            return self._encode_indobert(text)
        return self._encode_hashing(text)

    def encode_many(self, texts: Iterable[str]) -> list[list[float]]:
        return [self.encode_text(text) for text in texts]

    def transform_dataframe(
        self, df: pl.DataFrame, text_column: str = config.COL_PROCESSED
    ) -> tuple[pl.DataFrame, list[list[float]]]:
        if text_column not in df.columns:
            raise KeyError(f"Kolom '{text_column}' tidak ada pada DataFrame")
        vectors = self.encode_many(df[text_column].to_list())
        return df.with_columns(pl.Series("embedding_backend", [self.backend] * df.height)), vectors

    def _encode_hashing(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = _TOKEN_RE.findall((text or "").lower())
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.blake2b(token.encode(config.ENCODING), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [round(value / norm, 6) for value in vector]

    def _encode_indobert(self, text: str) -> list[float]:
        import torch

        if self._tokenizer is None or self._model is None:
            raise RuntimeError("Model IndoBERT belum dimuat")

        inputs = self._tokenizer(
            text or "",
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
        vector = outputs.last_hidden_state[:, 0, :].squeeze(0).tolist()
        norm = math.sqrt(sum(float(value) * float(value) for value in vector))
        if norm == 0:
            return [float(value) for value in vector]
        return [round(float(value) / norm, 6) for value in vector]
