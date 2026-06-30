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
from services.source_blacklist_service import SourceBlacklistService


def build_source_blacklist_service(
    *,
    service: DatasetService,
    source_url_blacklist_path: Path,
    source_blacklist_rules_path: Path,
    used_candidate_blacklist_folder: Path | None,
    used_candidate_blacklist_pattern: str,
) -> tuple[SourceBlacklistService, int]:
    exact_urls = list(
        SourceBlacklistService.load_exact_blacklist(source_url_blacklist_path)
    )
    used_candidate_source_urls: tuple[str, ...] = ()

    if used_candidate_blacklist_folder is not None:
        used_candidate_source_urls = service.load_used_candidate_source_urls(
            used_candidate_blacklist_folder,
            pattern=used_candidate_blacklist_pattern,
        )
        exact_urls.extend(used_candidate_source_urls)

    return (
        SourceBlacklistService(
            exact_urls=exact_urls,
            rules=SourceBlacklistService.load_rules(source_blacklist_rules_path),
        ),
        len(used_candidate_source_urls),
    )


def build_frames(
    *,
    raw_folder: Path = config.DATASETS / "url_discovery",
    research_config_path: Path = config.RESEARCH_CONFIG_PATH,
    source_url_blacklist_path: Path = config.SOURCE_URL_BLACKLIST_PATH,
    source_blacklist_rules_path: Path = config.SOURCE_BLACKLIST_RULES_PATH,
    used_candidate_blacklist_folder: Path | None = config.OUTPUTS / "datasets",
    used_candidate_blacklist_pattern: str = "v*_candidate_labeling_dataset.csv",
) -> dict[str, object]:
    service = DatasetService()
    research_config = service.load_research_config(research_config_path)
    blacklist_service, used_candidate_source_url_count = build_source_blacklist_service(
        service=service,
        source_url_blacklist_path=source_url_blacklist_path,
        source_blacklist_rules_path=source_blacklist_rules_path,
        used_candidate_blacklist_folder=used_candidate_blacklist_folder,
        used_candidate_blacklist_pattern=used_candidate_blacklist_pattern,
    )

    meta_df = service.load_url_discovery_meta(raw_folder)
    queries_df = service.load_url_discovery_queries(raw_folder)
    raw_records_df = service.load_url_discovery_records(raw_folder)
    candidate_df = service.build_v1_candidate_rows(
        records_df=raw_records_df,
        research_config=research_config,
    )
    candidate_df = service.enrich_source_blacklist_status(
        candidate_df,
        blacklist_service,
    )
    blacklist_audit_df = service.filter_blacklist_audit_candidates(candidate_df)
    clear_candidate_df = service.filter_clear_source_candidates(candidate_df)

    return {
        "meta": meta_df,
        "queries": queries_df,
        "raw_records": raw_records_df,
        "candidate": candidate_df,
        "clear_candidate": clear_candidate_df,
        "blacklist_audit": blacklist_audit_df,
        "used_candidate_source_url_count": used_candidate_source_url_count,
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
        "--source-blacklist-rules",
        type=Path,
        default=config.SOURCE_BLACKLIST_RULES_PATH,
        help="Path JSON object rule blacklist sumber.",
    )
    parser.add_argument(
        "--used-candidate-blacklist-folder",
        type=Path,
        default=config.OUTPUTS / "datasets",
        help=(
            "Folder CSV v*_candidate_labeling_dataset.csv yang source_url-nya "
            "ditandai sebagai blacklist pemakaian sebelumnya."
        ),
    )
    parser.add_argument(
        "--used-candidate-blacklist-pattern",
        default="v*_candidate_labeling_dataset.csv",
        help="Glob pattern CSV kandidat lama untuk blacklist pemakaian sebelumnya.",
    )
    parser.add_argument(
        "--disable-used-candidate-blacklist",
        action="store_true",
        help="Matikan blacklist otomatis dari CSV kandidat versi lama.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.DATASETS / "raw_candidate_schema.csv",
        help="Path CSV output untuk candidate row yang lolos blacklist.",
    )
    parser.add_argument(
        "--blacklist-audit-output",
        type=Path,
        default=None,
        help="Path CSV audit row yang dikeluarkan oleh source blacklist.",
    )
    parser.add_argument(
        "--include-blacklisted",
        action="store_true",
        help="Tulis seluruh candidate row, termasuk yang terkena blacklist.",
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
        source_blacklist_rules_path=args.source_blacklist_rules,
        used_candidate_blacklist_folder=(
            None
            if args.disable_used_candidate_blacklist
            else args.used_candidate_blacklist_folder
        ),
        used_candidate_blacklist_pattern=args.used_candidate_blacklist_pattern,
    )
    output_df = frames["candidate"] if args.include_blacklisted else frames["clear_candidate"]
    print(f"Raw records: {frames['raw_records'].height:,}")
    print(f"Candidate rows: {frames['candidate'].height:,}")
    print(
        "Used candidate source URLs: "
        f"{frames['used_candidate_source_url_count']:,}"
    )
    print(f"Clear candidate rows: {frames['clear_candidate'].height:,}")
    print(f"Blacklist excluded rows: {frames['blacklist_audit'].height:,}")
    print(f"Output rows: {output_df.height:,}")

    if args.no_write:
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.write_csv(args.output)
    print(f"Output: {args.output}")

    if args.blacklist_audit_output:
        args.blacklist_audit_output.parent.mkdir(parents=True, exist_ok=True)
        frames["blacklist_audit"].write_csv(args.blacklist_audit_output)
        print(f"Blacklist audit: {args.blacklist_audit_output}")


if __name__ == "__main__":
    main()
