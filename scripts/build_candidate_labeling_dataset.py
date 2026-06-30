"""Build candidate dataset siap labeling dari raw url discovery."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import polars as pl


def find_project_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "config.py").exists() and (candidate / "services").is_dir():
            return candidate
    raise FileNotFoundError("Root proyek tidak ditemukan")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from services.dataset_service import DatasetService


def build_frames(
    *,
    raw_folder: Path = config.DATASETS / "url_discovery",
    research_config_path: Path = config.RESEARCH_CONFIG_PATH,
    source_url_blacklist_path: Path = config.SOURCE_URL_BLACKLIST_PATH,
    max_rows_per_value: int = 10,
) -> dict[str, pl.DataFrame]:
    service = DatasetService()
    research_config = service.load_research_config(research_config_path)
    source_url_blacklist = service.load_source_url_blacklist(source_url_blacklist_path)
    raw_records_df = service.load_url_discovery_records(raw_folder)
    candidate_df = service.build_v1_candidate_rows(
        records_df=raw_records_df,
        research_config=research_config,
    )
    candidate_df = service.apply_source_url_blacklist(
        candidate_df,
        source_url_blacklist,
    )
    labeling_df = service.build_candidate_labeling_dataset(
        candidate_df,
        max_rows_per_value=max_rows_per_value,
    )

    return {
        "raw_records": raw_records_df,
        "candidate": candidate_df,
        "labeling": labeling_df,
        "candidate_combinations": service.summarize_candidate_combinations(
            candidate_df
        ),
        "labeling_combinations": service.summarize_labeling_selection_buckets(
            labeling_df
        ),
    }


def apply_filter_conditions(df: pl.DataFrame, conditions: list) -> pl.DataFrame:
    if not conditions:
        return df
    expression = conditions[0]
    for condition in conditions[1:]:
        expression = expression & condition
    return df.filter(expression)


def build_filtered_frames(
    *,
    raw_folder: Path = config.DATASETS / "url_discovery",
    research_config_path: Path = config.RESEARCH_CONFIG_PATH,
    source_url_blacklist_path: Path = config.SOURCE_URL_BLACKLIST_PATH,
    record_filter_conditions: list | None = None,
    candidate_filter_conditions: list | None = None,
    max_rows_per_value: int = 10,
) -> dict[str, pl.DataFrame]:
    service = DatasetService()
    research_config = service.load_research_config(research_config_path)
    source_url_blacklist = service.load_source_url_blacklist(source_url_blacklist_path)
    meta_df = service.load_url_discovery_meta(raw_folder)
    queries_df = service.load_url_discovery_queries(raw_folder)
    raw_records_df = service.load_url_discovery_records(raw_folder)
    records_df = apply_filter_conditions(raw_records_df, record_filter_conditions or [])
    candidate_df = service.build_v1_candidate_rows(
        records_df=records_df,
        research_config=research_config,
    )
    candidate_df = service.apply_source_url_blacklist(
        candidate_df,
        source_url_blacklist,
    )
    filtered_candidate_df = apply_filter_conditions(
        candidate_df,
        candidate_filter_conditions or [],
    )
    labeling_df = service.build_candidate_labeling_dataset(
        filtered_candidate_df,
        max_rows_per_value=max_rows_per_value,
    )

    return {
        "meta": meta_df,
        "queries": queries_df,
        "raw_records": raw_records_df,
        "records": records_df,
        "candidate": candidate_df,
        "filtered_candidate": filtered_candidate_df,
        "labeling": labeling_df,
        "candidate_combinations": service.summarize_candidate_combinations(
            filtered_candidate_df
        ),
        "labeling_combinations": service.summarize_labeling_selection_buckets(
            labeling_df
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build candidate dataset siap labeling per nilai unik source/aspect/tier."
    )
    parser.add_argument(
        "--raw-folder",
        type=Path,
        default=config.DATASETS / "url_discovery",
        help="Folder raw url_discovery JSON.",
    )
    parser.add_argument(
        "--research-config",
        type=Path,
        default=config.RESEARCH_CONFIG_PATH,
        help="Path research_config.json.",
    )
    parser.add_argument(
        "--source-url-blacklist",
        type=Path,
        default=config.SOURCE_URL_BLACKLIST_PATH,
        help="Path JSON array URL yang dikeluarkan dari kandidat.",
    )
    parser.add_argument(
        "--max-per-value",
        type=int,
        default=10,
        help="Maksimal row per nilai unik source_type/aspect/dataset_tier.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.OUTPUTS / "datasets" / "candidate_labeling_dataset.csv",
        help="Path CSV output.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Bangun frame dan tampilkan ringkasan tanpa menulis CSV.",
    )
    return parser.parse_args()


def print_combination_list(combinations_df: pl.DataFrame) -> None:
    print("Kombinasi final per kolom:")
    if combinations_df.is_empty():
        print("- tidak ada kombinasi siap labeling")
        return

    for row in combinations_df.iter_rows(named=True):
        print(
            "- "
            f"{row.get('combination_column')}={row.get('combination_value')}; "
            f"jumlah={row.get('jumlah')}"
        )


def main() -> None:
    args = parse_args()
    frames = build_frames(
        raw_folder=args.raw_folder,
        research_config_path=args.research_config,
        source_url_blacklist_path=args.source_url_blacklist,
        max_rows_per_value=args.max_per_value,
    )
    labeling_df = frames["labeling"]
    print(f"Raw records: {frames['raw_records'].height:,}")
    print(f"Candidate rows: {frames['candidate'].height:,}")
    print(f"Labeling rows: {labeling_df.height:,}")
    print(f"Labeling combinations: {frames['labeling_combinations'].height:,}")

    if args.no_write:
        print_combination_list(frames["labeling_combinations"])
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    labeling_df.write_csv(args.output)
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
