"""Tambahkan satu row manual ke candidate labeling dataset CSV."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse


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
from services.source_blacklist_service import SourceBlacklistService


MAX_CSV_FIELD_SIZE = sys.maxsize
while True:
    try:
        csv.field_size_limit(MAX_CSV_FIELD_SIZE)
        break
    except OverflowError:
        MAX_CSV_FIELD_SIZE //= 10


DEFAULT_DATASET_PATH = config.OUTPUTS / "datasets" / "v5_candidate_labeling_dataset.csv"
DEFAULT_BACKUP_DIR = config.OUTPUTS / "backups" / "manual_candidate_dataset"
MANUAL_TEXT_ID_PREFIX = "MANUAL-"
MANUAL_SOURCE_ID_PREFIX = "MANUAL-SRC-"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tambahkan satu dataset manual ke candidate labeling dataset CSV."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path CSV candidate labeling dataset yang akan ditambahkan row.",
    )
    parser.add_argument("--source-url", default="", help="URL sumber dataset.")
    parser.add_argument("--text", default="", help="Teks dataset.")
    parser.add_argument(
        "--text-file",
        type=Path,
        default=None,
        help="File teks untuk isi kolom text. Dipakai jika --text kosong.",
    )
    parser.add_argument("--raw-title", default="", help="Judul mentah sumber.")
    parser.add_argument("--query", default="", help="Query asal, bila ada.")
    parser.add_argument("--query-group", default="manual", help="Query group.")
    parser.add_argument(
        "--source-type",
        default="",
        choices=[
            "",
            "academic_repository",
            "corporate_official",
            "government_official",
            "online_news",
            "social_media",
        ],
        help="Jenis sumber.",
    )
    parser.add_argument("--aspect", default="", help="Aspect dataset.")
    parser.add_argument(
        "--dataset-tier",
        default="B_review_queue",
        choices=["A_candidate_core", "B_review_queue"],
        help="Tier dataset manual. Candidate labeling hanya menerima A/B.",
    )
    parser.add_argument("--location", default="", help="Lokasi spesifik Kalbar.")
    parser.add_argument(
        "--location-source",
        default="manual",
        help="Sumber lokasi, default manual.",
    )
    parser.add_argument(
        "--location-match",
        default="",
        help="Match lokasi. Default memakai --location lowercase.",
    )
    parser.add_argument(
        "--labeling-bucket-column",
        default="aspect",
        choices=["source_type", "aspect"],
        help="Bucket kombinasi labeling untuk row manual.",
    )
    parser.add_argument(
        "--subjectivity-type",
        default="public_expectation",
        help="Subjectivity type awal.",
    )
    parser.add_argument(
        "--speaker-type",
        default="community_representative",
        help="Speaker type awal.",
    )
    parser.add_argument(
        "--public-opinion-scope",
        default="public_opinion",
        help="Scope opini awal.",
    )
    parser.add_argument(
        "--corpus-role",
        default="core_public_opinion",
        help="Corpus role awal.",
    )
    parser.add_argument(
        "--verification-status",
        default="perlu_verifikasi",
        help="Status verifikasi awal.",
    )
    parser.add_argument(
        "--decision-note",
        default="manual_added",
        help="Catatan keputusan awal.",
    )
    parser.add_argument(
        "--allow-duplicate",
        action="store_true",
        help="Izinkan source_url yang sudah ada di dataset target.",
    )
    parser.add_argument(
        "--allow-blacklisted",
        action="store_true",
        help="Izinkan source yang terdeteksi blacklist rules.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Jangan membuat backup sebelum menulis CSV.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Tampilkan row yang akan ditambahkan tanpa menulis CSV.",
    )
    return parser.parse_args()


def prompt_required(label: str, current: str) -> str:
    value = current.strip()
    if value:
        return value
    while not value:
        value = input(f"{label}: ").strip()
    return value


def read_text(args: argparse.Namespace) -> str:
    if args.text.strip():
        return args.text.strip()
    if args.text_file:
        return args.text_file.read_text(encoding=config.ENCODING).strip()

    print("Masukkan text. Akhiri dengan satu baris berisi titik: .")
    lines: list[str] = []
    while True:
        line = input()
        if line == ".":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset target tidak ditemukan: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]

    if not fieldnames:
        raise ValueError(f"Dataset target tidak memiliki header CSV: {path}")
    return fieldnames, rows


def next_manual_number(rows: list[dict[str, str]], column: str, prefix: str) -> int:
    highest = 0
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    for row in rows:
        match = pattern.match(str(row.get(column) or "").strip())
        if match:
            highest = max(highest, int(match.group(1)))
    return highest + 1


def build_backup_path(dataset_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return DEFAULT_BACKUP_DIR / timestamp / dataset_path.name


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=config.ENCODING, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def inclusion_status(dataset_tier: str) -> str:
    if dataset_tier == "A_candidate_core":
        return "candidate_analysis_ready"
    return "review_required_before_core"


def evidence_support_score(dataset_tier: str) -> str:
    if dataset_tier == "A_candidate_core":
        return "1.0"
    return "0.9"


def build_manual_row(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, str]],
    fieldnames: list[str],
    blacklist_service: SourceBlacklistService,
) -> dict[str, str]:
    source_url = prompt_required("source_url", args.source_url)
    source_type = prompt_required("source_type", args.source_type)
    aspect = prompt_required("aspect", args.aspect)
    location = prompt_required("location", args.location)
    text = read_text(args)
    if not text:
        raise ValueError("Text manual tidak boleh kosong.")

    text_number = next_manual_number(rows, "text_id", MANUAL_TEXT_ID_PREFIX)
    source_number = next_manual_number(rows, "source_id", MANUAL_SOURCE_ID_PREFIX)
    normalized_source_url = blacklist_service.normalize_url(source_url)
    raw_title = args.raw_title.strip()
    location_match = args.location_match.strip() or location.casefold()
    bucket_column = args.labeling_bucket_column
    bucket_value = source_type if bucket_column == "source_type" else aspect
    raw_domain = urlparse(source_url).netloc

    row = {
        "text_id": f"{MANUAL_TEXT_ID_PREFIX}{text_number:04d}",
        "source_url": source_url,
        "source_type": source_type,
        "aspect": aspect,
        "dataset_tier": args.dataset_tier,
        "location": location,
        "location_source": args.location_source.strip() or "manual",
        "location_match": location_match,
        "is_specific_location": "true",
        "labeling_bucket_column": bucket_column,
        "labeling_bucket_value": bucket_value,
        "source_id": f"{MANUAL_SOURCE_ID_PREFIX}{source_number:04d}",
        "raw_source_file": "manual",
        "raw_domain": raw_domain,
        "content_status": "manual_success",
        "raw_title": raw_title,
        "raw_text_length": str(len(text)),
        "query_group": args.query_group.strip() or "manual",
        "query": args.query.strip(),
        "subjectivity_type": args.subjectivity_type.strip(),
        "speaker_type": args.speaker_type.strip(),
        "public_opinion_scope": args.public_opinion_scope.strip(),
        "corpus_role": args.corpus_role.strip(),
        "inclusion_status": inclusion_status(args.dataset_tier),
        "verification_status": args.verification_status.strip(),
        "evidence_support_score": evidence_support_score(args.dataset_tier),
        "parent_text_id": "",
        "decision_note": args.decision_note.strip(),
        "sentiment_label": "",
        "label_status": "unlabeled",
        "blacklist_status": "clear",
        "blacklist_reason_codes": "[]",
        "normalized_source_url": normalized_source_url,
        "blacklist_is_excluded": "false",
        "text": text,
    }

    return {column: row.get(column, "") for column in fieldnames}


def validate_row(
    *,
    row: dict[str, str],
    rows: list[dict[str, str]],
    args: argparse.Namespace,
    blacklist_service: SourceBlacklistService,
) -> None:
    normalized_source_url = row.get("normalized_source_url") or ""
    if not args.allow_duplicate:
        for existing in rows:
            existing_url = existing.get("normalized_source_url") or (
                blacklist_service.normalize_url(existing.get("source_url") or "")
            )
            if existing_url and existing_url == normalized_source_url:
                raise ValueError(
                    "source_url sudah ada di dataset target. "
                    "Gunakan --allow-duplicate jika tetap ingin menambahkan."
                )

    decision = blacklist_service.classify(row)
    if decision["blacklist_status"] != "clear" and not args.allow_blacklisted:
        raise ValueError(
            "source_url terdeteksi blacklist: "
            f"{decision['blacklist_status']} {decision['blacklist_reason_codes']}. "
            "Gunakan --allow-blacklisted jika tetap ingin menambahkan."
        )


def print_row_summary(row: dict[str, str]) -> None:
    print("Manual row:")
    for key in [
        "text_id",
        "source_url",
        "source_type",
        "aspect",
        "dataset_tier",
        "location",
        "labeling_bucket_column",
        "labeling_bucket_value",
        "raw_text_length",
    ]:
        print(f"- {key}: {row.get(key, '')}")


def main() -> None:
    args = parse_args()
    fieldnames, rows = read_rows(args.dataset)
    blacklist_service = SourceBlacklistService.from_paths()
    row = build_manual_row(
        args=args,
        rows=rows,
        fieldnames=fieldnames,
        blacklist_service=blacklist_service,
    )
    validate_row(
        row=row,
        rows=rows,
        args=args,
        blacklist_service=blacklist_service,
    )
    print_row_summary(row)

    if args.dry_run:
        print("Dry run: CSV tidak ditulis.")
        return

    if not args.no_backup:
        backup_path = build_backup_path(args.dataset)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.dataset, backup_path)
        print(f"Backup: {backup_path}")

    write_rows(args.dataset, fieldnames, [*rows, row])
    print(f"Dataset updated: {args.dataset}")


if __name__ == "__main__":
    main()
