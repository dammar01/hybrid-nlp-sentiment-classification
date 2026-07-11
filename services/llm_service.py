"""Lapisan interpretasi LLM (skenario-2) via GGUF/llama-cpp-python.

Model Qwen3-8B (GGUF Q4_K_M) dijalankan lokal untuk mengadjudikasi opini
ambigu yang tidak terselesaikan oleh fusi IndoBERT+rule. Dipanggil selektif
(hanya baris `requires_llm`), bukan seluruh dataset, demi efisiensi.

Dependency `llama-cpp-python` diimpor secara lazy agar modul lain (skenario-1)
tetap berjalan tanpa LLM terpasang.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import config

_LABEL_PATTERN = re.compile(r"\b(positif|negatif|netral)\b", re.IGNORECASE)

_SYSTEM_PROMPT = (
    "Anda adalah pengklasifikasi sentimen opini masyarakat tentang Solar Home "
    "System (SHS) / listrik tenaga surya di Kalimantan Barat. Klasifikasikan "
    "sentimen sebuah opini ke dalam TEPAT SATU kategori: positif, negatif, atau "
    "netral. Jawab HANYA dengan satu kata kategori tersebut, tanpa penjelasan."
)

_USER_TEMPLATE = (
    "Opini: \"{text}\"\n"
    "Kategori sentimen (positif/negatif/netral):"
)


@dataclass(slots=True)
class LLMService:
    """Klasifikasi sentimen selektif menggunakan LLM GGUF lokal."""

    model_path: str | Path = config.QWEN_GGUF_MODEL_PATH
    n_ctx: int = int(config.LLM_CONFIG["n_ctx"])
    n_gpu_layers: int = int(config.LLM_CONFIG["n_gpu_layers"])
    n_batch: int = int(config.LLM_CONFIG["n_batch"])
    temperature: float = float(config.LLM_CONFIG["temperature"])
    max_tokens: int = int(config.LLM_CONFIG["max_tokens"])
    seed: int = int(config.LLM_CONFIG["seed"])
    verbose: bool = bool(config.LLM_CONFIG["verbose"])
    llm: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.model_path = Path(self.model_path)
        self._load_model()

    def _load_model(self) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover - dependency opsional
            raise ImportError(
                "Dependency LLM belum tersedia. Install 'llama-cpp-python' "
                "(disarankan wheel prebuilt CUDA) untuk menjalankan skenario-2."
            ) from exc

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model GGUF tidak ditemukan: {self.model_path}. Unduh "
                "Qwen3-8B-Q4_K_M.gguf ke direktori tersebut terlebih dahulu."
            )

        self.llm = Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            n_batch=self.n_batch,
            seed=self.seed,
            verbose=self.verbose,
        )

    def classify_text(self, text: str) -> dict[str, object]:
        """Klasifikasikan satu opini. Kembalikan label + output mentah."""
        cleaned = "" if text is None else str(text).strip()
        if not cleaned:
            return {"llm_label": "netral", "llm_raw": "", "llm_parsed": False}

        # "/no_think" menonaktifkan mode reasoning Qwen3 -> output langsung label.
        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT + " /no_think"},
                {"role": "user", "content": _USER_TEMPLATE.format(text=cleaned[:2000])},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
        )
        raw = str(response["choices"][0]["message"]["content"] or "")
        label, parsed = self._parse_label(raw)
        return {"llm_label": label, "llm_raw": raw.strip(), "llm_parsed": parsed}

    def classify_texts(self, texts: Sequence[str]) -> list[dict[str, object]]:
        return [self.classify_text(text) for text in texts]

    @staticmethod
    def _parse_label(raw: str) -> tuple[str, bool]:
        match = _LABEL_PATTERN.search(raw or "")
        if match:
            return match.group(1).lower(), True
        return "netral", False  # fallback aman bila LLM tak patuh format
