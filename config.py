"""Konfigurasi terpusat untuk pipeline penelitian.

Berisi konstanta path (pathlib), nama kolom dataset, pengaturan I/O, dan
kata negasi. Modul lain mengimpor dari sini agar tidak ada "magic string"
yang tersebar di banyak file.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT: Path = Path(__file__).resolve().parent

DATASETS: Path = ROOT / "datasets"          # sumber data mentah (input)
RESOURCES: Path = ROOT / "resources"        # kamus statis (slang, non-baku)

OUTPUTS: Path = ROOT / "outputs"
ARTIFACTS: Path = OUTPUTS / "artifacts"     # ringkasan / metadata hasil olah
RESULTS: Path = OUTPUTS / "results"
FIGURES: Path = OUTPUTS / "figures"
REPORTS: Path = OUTPUTS / "reports"
OUTPUT_DATASETS: Path = OUTPUTS / "datasets"

# File kamus
SLANG_DICT_PATH: Path = RESOURCES / "slang.json"
NON_STANDARD_DICT_PATH: Path = RESOURCES / "non_standard.json"
SOCAL_METADATA_PATH: Path = RESOURCES / "socal_metadata.json"
SOCAL_WORD_RULES_PATH: Path = RESOURCES / "socal_word_rules.json"
SOCAL_PHRASE_RULES_PATH: Path = RESOURCES / "socal_phrase_rules.json"
SOCAL_MODIFIERS_PATH: Path = RESOURCES / "socal_modifiers.json"
RESEARCH_CONFIG_PATH: Path = RESOURCES / "research_config.json"
SOURCE_URL_BLACKLIST_PATH: Path = RESOURCES / "source_url_blacklist.json"
SOURCE_BLACKLIST_RULES_PATH: Path = RESOURCES / "source_blacklist_rules.json"

# Output artefak
DATASET_SUMMARY_PATH: Path = ARTIFACTS / "dataset_summary.json"
CALIBRATION_DATASET_PATH: Path = OUTPUTS / "datasets" / "tokopedia_calibration.csv"
CALIBRATION_SUMMARY_PATH: Path = ARTIFACTS / "tokopedia_calibration_summary.json"
NON_LLM_RESULTS_PATH: Path = RESULTS / "scenario_without_llm.csv"
NON_LLM_METRICS_PATH: Path = ARTIFACTS / "scenario_without_llm_metrics.json"
RAW_CANDIDATE_SCHEMA_PATH: Path = DATASETS / "raw_candidate_schema.csv"
RAW_CANDIDATE_SENTENCE_MIN_CHARS: int = 35
RAW_CANDIDATE_SENTENCE_MAX_CHARS: int = 320
GOLDEN_DATASET_DIR: Path = DATASETS / "golden"
TRAINING_DATASET_PATH: Path = OUTPUT_DATASETS / "training_dataset.parquet"
TRAINING_DATASET_WITH_SPLIT_PATH: Path = OUTPUT_DATASETS / "training_dataset_with_split.parquet"
FIXED_SPLIT_ASSIGNMENT_PATH: Path = ARTIFACTS / "fixed_group_split_assignment.json"
FIXED_SPLIT_MANIFEST_PATH: Path = ARTIFACTS / "fixed_group_split_manifest.json"

# Model lokal
INDOBERT_MODEL_PATH: Path = ROOT / "model" / "indobert-base-p2"

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
ENCODING: str = "utf-8"
ENCODING_FALLBACK: str = "latin-1"

# ---------------------------------------------------------------------------
# Kolom dataset
# ---------------------------------------------------------------------------
COL_ID: str = "id"
COL_SOURCE: str = "source"
COL_TEXT: str = "text"
COL_PROCESSED: str = "processed_text"
COL_ACTUAL_LABEL: str = "actual_label"
COL_RULE_LABEL: str = "rule_label"
COL_RULE_CONFIDENCE: str = "rule_confidence"
COL_RULE_CONTRACT_VERSION: str = "rule_contract_version"
COL_RULE_RESOURCE_VERSION: str = "rule_resource_version"
COL_RULE_SCORE: str = "rule_score"
COL_RULE_STATUS: str = "rule_status"
COL_RULE_POSITIVE_SCORE: str = "rule_positive_score"
COL_RULE_NEGATIVE_SCORE: str = "rule_negative_score"
COL_RULE_POSITIVE_COUNT: str = "rule_positive_count"
COL_RULE_NEGATIVE_COUNT: str = "rule_negative_count"
COL_RULE_HITS: str = "rule_hits"
COL_RULE_NEUTRAL_HITS: str = "rule_neutral_hits"
COL_RULE_PHRASE_HITS: str = "rule_phrase_hits"
COL_RULE_WORD_HITS: str = "rule_word_hits"
COL_RULE_MODIFIER_HITS: str = "rule_modifier_hits"
COL_RULE_EVIDENCE: str = "rule_evidence"
COL_RULE_EXPLANATION: str = "rule_explanation"
COL_SEMANTIC_LABEL: str = "semantic_label"
COL_SEMANTIC_SIMILARITY: str = "semantic_similarity"
COL_CLUSTER_ID: str = "cluster_id"
COL_IS_AMBIGUOUS: str = "is_ambiguous"
COL_FINAL_LABEL: str = "final_label"

REQUIRED_COLUMNS: tuple[str, ...] = (COL_ID, COL_SOURCE, COL_TEXT)

# ---------------------------------------------------------------------------
# Reproducibility dan label sentimen
# ---------------------------------------------------------------------------
GLOBAL_SEED: int = 42
LABEL2ID: dict[str, int] = {"negatif": 0, "netral": 1, "positif": 2}
ID2LABEL: dict[int, str] = {value: key for key, value in LABEL2ID.items()}
SENTIMENT_LABELS: tuple[str, ...] = ("negatif", "netral", "positif")

# ---------------------------------------------------------------------------
# Negasi
# ---------------------------------------------------------------------------
# Bentuk baku kata negasi. WAJIB dipertahankan oleh preprocessing dan TIDAK
# boleh dipetakan ke kata lain di kamus slang / non-standar.
NEGATION_WORDS: tuple[str, ...] = ("tidak", "bukan", "belum", "jangan")

# ---------------------------------------------------------------------------
# Skenario non-LLM
# ---------------------------------------------------------------------------
RULE_CONFIDENCE_ALPHA: float = 0.35
SEMANTIC_SIMILARITY_BETA: float = 0.35
EMBEDDING_DIMENSION: int = 128
CLUSTER_SIMILARITY_THRESHOLD: float = 0.72
MIN_CLUSTER_SIZE: int = 3

# ---------------------------------------------------------------------------
# IndoBERT hybrid experiment
# ---------------------------------------------------------------------------
EXPERIMENT_CONFIG: dict[str, object] = {
    "experiment_prefix": "hybrid_indobert_socal",
    "artifact_root": str(ARTIFACTS / "experiments"),
}

SPLIT_CONFIG: dict[str, object] = {
    "n_splits": 6,
    "test_fold": 0,
    "calibration_fold": 1,
    "train_folds": (2, 3, 4, 5),
    "random_state": GLOBAL_SEED,
}

TRAINING_CONFIG: dict[str, object] = {
    "seed": GLOBAL_SEED,
    "learning_rate": 2e-5,
    "train_batch_size": 8,
    "eval_batch_size": 16,
    "max_length": 256,
    "max_epochs": 10,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "early_stopping_patience": 2,
    "metric_for_best_model": "balanced_accuracy",
}

CALIBRATION_CONFIG: dict[str, object] = {
    "initial_temperature": 1.0,
    "max_iter": 200,
    "ece_bins": 10,
}

UNCERTAINTY_WEIGHT_GRID: tuple[dict[str, float], ...] = (
    {"wc": 0.50, "wm": 0.30, "wd": 0.20},
    {"wc": 0.40, "wm": 0.40, "wd": 0.20},
    {"wc": 0.45, "wm": 0.25, "wd": 0.30},
    {"wc": 0.34, "wm": 0.33, "wd": 0.33},
)

FUSION_POLICY_GRID: dict[str, tuple[float, ...]] = {
    "high_confidence_threshold": (0.70, 0.75, 0.80, 0.85),
    "low_confidence_threshold": (0.45, 0.50, 0.55, 0.60),
    "rule_confidence_threshold": (0.35, 0.45, 0.55, 0.65),
    "uncertainty_review_threshold": (0.50, 0.60, 0.70, 0.80),
}

# ---------------------------------------------------------------------------
# Rule-based lexicon contract
# ---------------------------------------------------------------------------
RULE_CONTRACT_VERSION: str = "2.0.0"
RULE_RESOURCE_VERSION: str = "1.0.0"
RULE_WEAK_THRESHOLD: float = 0.35
RULE_TOKEN_PATTERN: str = r"[A-Za-zÀ-ſ]+(?:['\-][A-Za-zÀ-ſ]+)*"
RULE_STATUS_DETECTED: str = "detected"
RULE_STATUS_WEAK: str = "weak"
RULE_STATUS_UNKNOWN: str = "unknown"
RULE_EFFECT_SHIFT: str = "shift"
RULE_EFFECT_INVERT: str = "invert"
RULE_EFFECT_WEAKEN: str = "weaken"
RULE_EFFECT_PREFER_AFTER_CLAUSE: str = "prefer_after_clause"
RULE_EFFECT_AMPLIFY_NEGATIVE_AFTER: str = "amplify_negative_after"
RULE_EFFECT_AMPLIFY_POSITIVE_AFTER: str = "amplify_positive_after"
RULE_EFFECT_SUBORDINATE_BEFORE_CLAUSE: str = "subordinate_before_clause"
RULE_EFFECT_CONTRADICT_EXPECTATION: str = "contradict_expectation"
RULE_CONTRAST_BEFORE_WEIGHT: float = 0.75
RULE_CONTRAST_AFTER_WEIGHT: float = 1.25
RULE_CONCESSION_BEFORE_WEIGHT: float = 0.85
RULE_CONCESSION_AFTER_WEIGHT: float = 1.15
RULE_AMPLIFY_AFTER_WEIGHT: float = 1.2
RULE_NEGATION_SHIFT: float = 1.6
RULE_NEGATIVE_WEIGHT: float = 1.5
RULE_NEUTRAL_CUTOFF: float = 0.15
RULE_OUTPUT_COLUMNS: tuple[str, ...] = (
    COL_RULE_CONTRACT_VERSION,
    COL_RULE_RESOURCE_VERSION,
    COL_RULE_LABEL,
    COL_RULE_SCORE,
    COL_RULE_CONFIDENCE,
    COL_RULE_STATUS,
    COL_RULE_POSITIVE_SCORE,
    COL_RULE_NEGATIVE_SCORE,
    COL_RULE_POSITIVE_COUNT,
    COL_RULE_NEGATIVE_COUNT,
    COL_RULE_HITS,
    COL_RULE_NEUTRAL_HITS,
    COL_RULE_PHRASE_HITS,
    COL_RULE_WORD_HITS,
    COL_RULE_MODIFIER_HITS,
    COL_RULE_EVIDENCE,
    COL_RULE_EXPLANATION,
)
