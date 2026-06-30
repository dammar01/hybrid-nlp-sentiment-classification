"""Build seluruh candidate dataset dari raw url discovery tanpa sampling."""

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
) -> dict[str, pl.DataFrame]:
    service = DatasetService()
    research_config = service.load_research_config(research_config_path)
    source_url_blacklist = service.load_source_url_blacklist(source_url_blacklist_path)

    meta_df = service.load_url_discovery_meta(raw_folder)
    queries_df = service.load_url_discovery_queries(raw_folder)
    raw_records_df = service.load_url_discovery_records(raw_folder)
    candidate_df = service.build_v1_candidate_rows(
        records_df=raw_records_df,
        research_config=research_config,
    )
    candidate_df = service.apply_source_url_blacklist(
        candidate_df,
        source_url_blacklist,
    )

    return {
        "meta": meta_df,
        "queries": queries_df,
        "raw_records": raw_records_df,
        "candidate": candidate_df,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build semua candidate row dari raw url_discovery tanpa sampling."
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
        "--output",
        type=Path,
        default=config.DATASETS / "candidate_datasets.csv",
        help="Path CSV output untuk seluruh candidate row.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Bangun frame dan tampilkan ringkasan tanpa menulis CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = build_frames(
        raw_folder=args.raw_folder,
        research_config_path=args.research_config,
        source_url_blacklist_path=args.source_url_blacklist,
    )
    candidate_df = frames["candidate"]
    print(f"Raw records: {frames['raw_records'].height:,}")
    print(f"Candidate rows: {candidate_df.height:,}")

    if args.no_write:
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    candidate_df.write_csv(args.output)
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
