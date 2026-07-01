"""Fine-tuning IndoBERT untuk klasifikasi sentimen tiga kelas."""

from __future__ import annotations

import random
import inspect
import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

import config
from services.artifact_service import ArtifactService


@dataclass(slots=True)
class IndoBERTTrainingService:
    """Orkestrasi fine-tuning IndoBERT lokal tanpa memuat LLM resolver."""

    base_model_path: str | Path = config.INDOBERT_MODEL_PATH
    training_config: dict[str, object] | None = None

    def __post_init__(self) -> None:
        self.base_model_path = Path(self.base_model_path)
        self.training_config = dict(self.training_config or config.TRAINING_CONFIG)

    def train(
        self,
        training_df: pl.DataFrame,
        *,
        output_dir: str | Path,
        text_column: str = config.COL_PROCESSED,
        label_column: str = "label_id",
    ) -> dict[str, object]:
        required = (text_column, label_column, "split")
        missing = [column for column in required if column not in training_df.columns]
        if missing:
            raise KeyError(f"Kolom training hilang: {missing}")

        train_df = training_df.filter(pl.col("split") == "train")
        if train_df.is_empty():
            raise ValueError("Training fold kosong")

        train_rows, eval_rows = self._internal_train_eval_split(train_df)
        class_weights = self._class_weights(
            [int(row[label_column]) for row in train_rows]
        )
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        torch, transformers = self._load_backend()
        self._set_seed(torch, int(self.training_config["seed"]))
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.base_model_path)
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            self.base_model_path,
            num_labels=len(config.LABEL2ID),
            id2label={str(key): value for key, value in config.ID2LABEL.items()},
            label2id=config.LABEL2ID,
            ignore_mismatched_sizes=True,
        )

        train_dataset = _TextDataset(
            rows=train_rows,
            tokenizer=tokenizer,
            text_column=text_column,
            label_column=label_column,
            max_length=int(self.training_config["max_length"]),
        )
        eval_dataset = _TextDataset(
            rows=eval_rows,
            tokenizer=tokenizer,
            text_column=text_column,
            label_column=label_column,
            max_length=int(self.training_config["max_length"]),
        )

        use_cuda = torch.cuda.is_available()
        args = self._training_arguments(
            transformers=transformers,
            output_dir=output_dir / "trainer",
            use_cuda=use_cuda,
        )
        trainer = _WeightedTrainer(
            class_weights=torch.tensor(class_weights, dtype=torch.float),
            **self._trainer_kwargs(
                tokenizer=tokenizer,
                compute_metrics=self._build_compute_metrics(),
                model=model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                callbacks=[
                    transformers.EarlyStoppingCallback(
                        early_stopping_patience=int(
                            self.training_config["early_stopping_patience"]
                        )
                    )
                ],
            ),
        )
        trainer.train()

        model_dir = output_dir / "model"
        trainer.model.save_pretrained(model_dir, safe_serialization=True)
        tokenizer.save_pretrained(model_dir)
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        manifest = {
            "base_model_path": str(self.base_model_path),
            "model_dir": str(model_dir),
            "train_rows": len(train_rows),
            "internal_eval_rows": len(eval_rows),
            "class_weights": class_weights,
            "training_config": self.training_config,
            "metrics": {
                key: float(value)
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            },
        }
        ArtifactService().save_json(manifest, output_dir / "model_manifest.json")
        return manifest

    def _trainer_kwargs(
        self,
        *,
        tokenizer: Any,
        compute_metrics: Any,
        model: Any,
        args: Any,
        train_dataset: Any,
        eval_dataset: Any,
        callbacks: list[Any],
    ) -> dict[str, Any]:
        import transformers

        _disable_local_datasets_shadow(transformers)
        from transformers import Trainer

        parameters = inspect.signature(Trainer.__init__).parameters
        kwargs: dict[str, Any] = {
            "compute_metrics": compute_metrics,
            "model": model,
            "args": args,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "callbacks": callbacks,
        }
        if "processing_class" in parameters:
            kwargs["processing_class"] = tokenizer
        elif "tokenizer" in parameters:
            kwargs["tokenizer"] = tokenizer
        return {key: value for key, value in kwargs.items() if key in parameters}

    @staticmethod
    def _load_backend() -> tuple[Any, Any]:
        try:
            import torch
            import transformers
        except ImportError as exc:
            raise ImportError(
                "Dependency fine-tuning belum tersedia. Install torch, transformers, accelerate, scikit-learn."
            ) from exc
        _disable_local_datasets_shadow(transformers)
        return torch, transformers

    def _training_arguments(
        self,
        *,
        transformers: Any,
        output_dir: Path,
        use_cuda: bool,
    ) -> Any:
        parameters = inspect.signature(
            transformers.TrainingArguments.__init__
        ).parameters
        kwargs: dict[str, object] = {
            "output_dir": str(output_dir),
            "learning_rate": float(self.training_config["learning_rate"]),
            "per_device_train_batch_size": int(
                self.training_config["train_batch_size"]
            ),
            "per_device_eval_batch_size": int(
                self.training_config["eval_batch_size"]
            ),
            "num_train_epochs": int(self.training_config["max_epochs"]),
            "warmup_ratio": float(self.training_config["warmup_ratio"]),
            "weight_decay": float(self.training_config["weight_decay"]),
            "load_best_model_at_end": True,
            "metric_for_best_model": str(
                self.training_config["metric_for_best_model"]
            ),
            "greater_is_better": True,
            "seed": int(self.training_config["seed"]),
            "fp16": bool(use_cuda),
            "report_to": [],
            "save_total_limit": 2,
        }
        if "evaluation_strategy" in parameters:
            kwargs["evaluation_strategy"] = "epoch"
        elif "eval_strategy" in parameters:
            kwargs["eval_strategy"] = "epoch"

        if "save_strategy" in parameters:
            kwargs["save_strategy"] = "epoch"

        supported_kwargs = {
            key: value for key, value in kwargs.items() if key in parameters
        }
        return transformers.TrainingArguments(**supported_kwargs)

    @staticmethod
    def _set_seed(torch: Any, seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _class_weights(labels: list[int]) -> list[float]:
        counts = {label_id: labels.count(label_id) for label_id in config.ID2LABEL}
        total = len(labels)
        return [
            total / (len(config.ID2LABEL) * max(counts[label_id], 1))
            for label_id in sorted(config.ID2LABEL)
        ]

    def _internal_train_eval_split(
        self, train_df: pl.DataFrame
    ) -> tuple[list[dict], list[dict]]:
        rows = train_df.to_dicts()
        labels = [int(row["label_id"]) for row in rows]
        try:
            from sklearn.model_selection import train_test_split

            train_rows, eval_rows = train_test_split(
                rows,
                test_size=0.2,
                random_state=int(self.training_config["seed"]),
                stratify=labels,
            )
            return list(train_rows), list(eval_rows)
        except Exception:
            random.Random(int(self.training_config["seed"])).shuffle(rows)
            split_at = max(1, int(len(rows) * 0.8))
            return rows[:split_at], rows[split_at:] or rows[-1:]

    @staticmethod
    def _build_compute_metrics():
        def compute_metrics(eval_pred):
            try:
                from sklearn.metrics import balanced_accuracy_score, f1_score
                import numpy as np
            except ImportError as exc:
                raise ImportError(
                    "scikit-learn dan numpy wajib untuk metric training"
                ) from exc

            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return {
                "balanced_accuracy": balanced_accuracy_score(labels, predictions),
                "macro_f1": f1_score(
                    labels, predictions, average="macro", zero_division=0
                ),
            }

        return compute_metrics


class _TextDataset:
    def __init__(
        self,
        *,
        rows: list[dict],
        tokenizer: Any,
        text_column: str,
        label_column: str,
        max_length: int,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.label_column = label_column
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        encoded = self.tokenizer(
            str(row.get(self.text_column) or ""),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        encoded["labels"] = int(row[self.label_column])
        return encoded


class _WeightedTrainer:
    """Wrapper dinamis agar subclass dibuat hanya setelah transformers tersedia."""

    def __new__(cls, *args, class_weights, **kwargs):
        import torch
        import transformers

        _disable_local_datasets_shadow(transformers)
        from transformers import Trainer

        class WeightedTrainer(Trainer):
            def __init__(self, *inner_args, **inner_kwargs):
                self._class_weights = class_weights
                super().__init__(*inner_args, **inner_kwargs)

            def compute_loss(self, model, inputs, return_outputs=False, **_kwargs):
                labels = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs.get("logits")
                weights = self._class_weights.to(logits.device)
                loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
                loss = loss_fn(
                    logits.view(-1, model.config.num_labels), labels.view(-1)
                )
                return (loss, outputs) if return_outputs else loss

        return WeightedTrainer(*args, **kwargs)


def _disable_local_datasets_shadow(transformers: Any) -> None:
    """Avoid treating the repo's data folder as HuggingFace datasets package."""
    spec = importlib.util.find_spec("datasets")
    loaded = sys.modules.get("datasets")
    if spec is None and loaded is None:
        return

    local_datasets_dir = config.DATASETS.resolve()
    candidate_paths = []
    if spec and spec.origin:
        candidate_paths.append(Path(spec.origin))
    if spec:
        candidate_paths.extend(
            Path(path) for path in spec.submodule_search_locations or []
        )
    loaded_file = getattr(loaded, "__file__", None)
    if loaded_file:
        candidate_paths.append(Path(loaded_file))
    loaded_path = getattr(loaded, "__path__", None)
    if loaded_path:
        candidate_paths.extend(Path(path) for path in loaded_path)

    is_local_data_folder = any(
        path.resolve() == local_datasets_dir
        or local_datasets_dir in path.resolve().parents
        for path in candidate_paths
    )
    if not is_local_data_folder:
        return

    if loaded is not None and hasattr(loaded, "Dataset"):
        if getattr(loaded, "__spec__", None) is None:
            loaded.__spec__ = importlib.util.spec_from_loader("datasets", loader=None)
        return

    class _UnavailableHFDataset:
        pass

    stub = types.ModuleType("datasets")
    stub.Dataset = _UnavailableHFDataset
    stub.IterableDataset = _UnavailableHFDataset
    stub.__file__ = str(local_datasets_dir)
    stub.__path__ = []
    stub.__spec__ = importlib.util.spec_from_loader("datasets", loader=None)
    sys.modules["datasets"] = stub

    import_utils = getattr(getattr(transformers, "utils", None), "import_utils", None)
    if import_utils is not None and hasattr(import_utils, "_datasets_available"):
        import_utils._datasets_available = False
    utils = getattr(transformers, "utils", None)
    if utils is not None and hasattr(utils, "_datasets_available"):
        utils._datasets_available = False
