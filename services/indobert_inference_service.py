"""Inferensi IndoBERT sequence classifier tanpa dependency LLM."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import polars as pl

import config
from services.calibration_service import CalibrationService


@dataclass(slots=True)
class IndoBERTInferenceService:
    """Load model fine-tuned IndoBERT dan hasilkan probabilitas terkalibrasi."""

    model_path: str | Path
    calibration_artifact_path: str | Path | None = None
    max_length: int = int(config.TRAINING_CONFIG["max_length"])
    device: str | None = None
    model: Any = field(init=False, default=None)
    tokenizer: Any = field(init=False, default=None)
    torch: Any = field(init=False, default=None)
    temperature: float = field(init=False, default=1.0)

    def __post_init__(self) -> None:
        self.model_path = Path(self.model_path)
        if self.calibration_artifact_path is not None:
            artifact = CalibrationService().load_artifact(self.calibration_artifact_path)
            self.temperature = float(artifact["temperature"])
        self._load_model()

    def predict_dataframe(
        self,
        df: pl.DataFrame,
        *,
        text_column: str = config.COL_PROCESSED,
    ) -> pl.DataFrame:
        if text_column not in df.columns:
            raise KeyError(f"Kolom teks inferensi hilang: {text_column}")
        predictions = self.predict_texts(df[text_column].to_list())
        return df.hstack(pl.DataFrame(predictions))

    def predict_texts(self, texts: Sequence[str]) -> list[dict[str, object]]:
        logits = self._predict_logits(texts)
        probabilities = CalibrationService.softmax(logits, temperature=self.temperature)
        rows: list[dict[str, object]] = []
        for logit_row, prob_row in zip(logits, probabilities):
            sorted_probs = sorted(prob_row, reverse=True)
            label_id = int(max(range(len(prob_row)), key=lambda index: prob_row[index]))
            confidence = float(sorted_probs[0])
            margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0
            entropy = self._normalized_entropy(prob_row)
            rows.append(
                {
                    "bert_label": config.ID2LABEL[label_id],
                    "bert_label_id": label_id,
                    "bert_logit_negatif": float(logit_row[0]),
                    "bert_logit_netral": float(logit_row[1]),
                    "bert_logit_positif": float(logit_row[2]),
                    "bert_prob_negatif": float(prob_row[0]),
                    "bert_prob_netral": float(prob_row[1]),
                    "bert_prob_positif": float(prob_row[2]),
                    "bert_confidence": confidence,
                    "bert_margin": margin,
                    "bert_entropy": entropy,
                    "bert_temperature": self.temperature,
                }
            )
        return rows

    def _predict_logits(self, texts: Sequence[str]) -> list[list[float]]:
        self.model.eval()
        rows: list[list[float]] = []
        batch_size = int(config.TRAINING_CONFIG["eval_batch_size"])
        with self.torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_texts = ["" if text is None else str(text) for text in texts[start:start + batch_size]]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                output = self.model(**encoded)
                rows.extend(output.logits.detach().cpu().tolist())
        return rows

    def _load_model(self) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Dependency IndoBERT belum tersedia. Install torch dan transformers."
            ) from exc

        self.torch = torch
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)

    @staticmethod
    def _normalized_entropy(probabilities: Sequence[float]) -> float:
        entropy = -sum(float(prob) * math.log(float(prob)) for prob in probabilities if prob > 0)
        return entropy / math.log(len(probabilities)) if probabilities else 0.0
