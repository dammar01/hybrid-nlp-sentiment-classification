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
