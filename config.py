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

# Output artefak
DATASET_SUMMARY_PATH: Path = ARTIFACTS / "dataset_summary.json"

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

REQUIRED_COLUMNS: tuple[str, ...] = (COL_ID, COL_SOURCE, COL_TEXT)

# ---------------------------------------------------------------------------
# Negasi
# ---------------------------------------------------------------------------
# Bentuk baku kata negasi. WAJIB dipertahankan oleh preprocessing dan TIDAK
# boleh dipetakan ke kata lain di kamus slang / non-standar.
NEGATION_WORDS: tuple[str, ...] = ("tidak", "bukan", "belum", "jangan")
