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

# File kamus
SLANG_DICT_PATH: Path = RESOURCES / "slang.json"
NON_STANDARD_DICT_PATH: Path = RESOURCES / "non_standard.json"
LEXICON_WORDS_PATH: Path = RESOURCES / "lexicon_words.json"
PHRASE_RULES_PATH: Path = RESOURCES / "phrase_rules.json"
MODIFIER_RULES_PATH: Path = RESOURCES / "modifier_rules.json"

# Output artefak
DATASET_SUMMARY_PATH: Path = ARTIFACTS / "dataset_summary.json"
CALIBRATION_DATASET_PATH: Path = OUTPUTS / "datasets" / "tokopedia_calibration.csv"
CALIBRATION_SUMMARY_PATH: Path = ARTIFACTS / "tokopedia_calibration_summary.json"
NON_LLM_RESULTS_PATH: Path = RESULTS / "scenario_without_llm.csv"
NON_LLM_METRICS_PATH: Path = ARTIFACTS / "scenario_without_llm_metrics.json"

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
COL_SEMANTIC_LABEL: str = "semantic_label"
COL_SEMANTIC_SIMILARITY: str = "semantic_similarity"
COL_CLUSTER_ID: str = "cluster_id"
COL_IS_AMBIGUOUS: str = "is_ambiguous"
COL_FINAL_LABEL: str = "final_label"

REQUIRED_COLUMNS: tuple[str, ...] = (COL_ID, COL_SOURCE, COL_TEXT)

# ---------------------------------------------------------------------------
# Negasi
# ---------------------------------------------------------------------------
# Bentuk baku kata negasi. WAJIB dipertahankan oleh preprocessing dan TIDAK
# boleh dipetakan ke kata lain di kamus slang / non-standar.
NEGATION_WORDS: tuple[str, ...] = ("tidak", "bukan", "belum", "jangan")

# ---------------------------------------------------------------------------
# Skenario non-LLM
# ---------------------------------------------------------------------------
SENTIMENT_LABELS: tuple[str, ...] = ("positif", "negatif", "netral")
RULE_CONFIDENCE_ALPHA: float = 0.35
SEMANTIC_SIMILARITY_BETA: float = 0.35
EMBEDDING_DIMENSION: int = 128
CLUSTER_SIMILARITY_THRESHOLD: float = 0.72
MIN_CLUSTER_SIZE: int = 3
