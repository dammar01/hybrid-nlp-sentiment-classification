"""Review dan update langsung record pada dataset url_discovery."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from clean_html_to_clipboard import clean_html
except ImportError:
    clean_html = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "datasets" / "url_discovery"
DEFAULT_BACKUP_DIR = PROJECT_ROOT / "outputs" / "backups" / "url_discovery"
END_MARKER = "::end"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cari record url_discovery berdasarkan URL, pilih record yang cocok, "
            "lalu update langsung content.text pada JSON. Backup dibuat sebelum "
            "file pertama kali diedit."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Folder dataset url_discovery.",
    )
    parser.add_argument(
        "--file",
        action="append",
        type=Path,
        default=[],
        help="Batasi review ke file JSON tertentu. Bisa diulang.",
    )
    parser.add_argument(
        "--include-blacklisted",
        action="store_true",
        help="Tetap tampilkan record yang sudah is_blacklisted=true.",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=DEFAULT_BACKUP_DIR,
        help="Folder backup JSON sebelum diedit.",
    )
    return parser.parse_args()


def dataset_files(args: argparse.Namespace) -> list[Path]:
    if args.file:
        return [path.resolve() for path in args.file]
    return sorted(args.dataset_dir.glob("*.json"))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def make_backup_once(
    path: Path,
    backup_dir: Path,
    backed_up: set[Path],
    session_id: str,
) -> Path:
    resolved = path.resolve()
    if resolved in backed_up:
        return backup_path(path, backup_dir, session_id)

    target = backup_path(path, backup_dir, session_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    backed_up.add(resolved)
    return target


def backup_path(path: Path, backup_dir: Path, session_id: str) -> Path:
    return backup_dir / session_id / path.name


def record_matches_url(
    record: dict[str, Any],
    query: str,
    include_blacklisted: bool,
) -> bool:
    content = safe_content(record)
    if not include_blacklisted and bool(record.get("is_blacklisted")):
        return False

    needle = query.strip().casefold()
    if not needle:
        return False
    url_values = [
        str(record.get("canonical_url") or ""),
        str(record.get("url") or ""),
        str(content.get("canonical_url") or ""),
        str(content.get("source_url") or ""),
    ]
    return any(needle in value.casefold() for value in url_values)


def safe_content(record: dict[str, Any]) -> dict[str, Any]:
    content = record.get("content")
    if not isinstance(content, dict):
        content = {}
        record["content"] = content
    return content


def preview(text: str, limit: int = 700) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def print_record(path: Path, index: int, total: int, record: dict[str, Any]) -> None:
    content = safe_content(record)
    text = str(content.get("text") or "")
    print("\n" + "=" * 88)
    print(f"File    : {path}")
    print(f"Record  : {index + 1}/{total}")
    print(f"URL     : {record.get('canonical_url') or record.get('url') or content.get('source_url')}")
    print(f"Title   : {content.get('title') or record.get('title') or ''}")
    print(f"Status  : {content.get('status') or ''}")
    print(f"HTTP    : {content.get('http_status') or ''}")
    print(f"Black   : {bool(record.get('is_blacklisted'))}")
    print(f"Text len: {len(text):,}")
    print("-" * 88)
    print(preview(text) or "<kosong>")
    print("-" * 88)


def edit_with_editor(initial_text: str) -> str:
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "nano"
    with tempfile.NamedTemporaryFile(
        mode="w+",
        suffix=".txt",
        encoding="utf-8",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        handle.write(initial_text)

    try:
        subprocess.run([editor, str(temp_path)], check=False)
        return temp_path.read_text(encoding="utf-8").strip()
    finally:
        temp_path.unlink(missing_ok=True)


def read_multiline(label: str) -> str:
    print(f"Tempel {label}. Akhiri dengan baris berisi {END_MARKER}")
    lines: list[str] = []
    while True:
        line = input()
        if line == END_MARKER:
            break
        lines.append(line)
    return "\n".join(lines).strip()


def apply_content_update(record: dict[str, Any]) -> bool:
    content = safe_content(record)
    print("Input konten: [m] teks multiline | [h] HTML multiline | [e] editor | [q] batal")
    action = input("> ").strip().lower()
    if action in {"", "m"}:
        content["text"] = read_multiline("teks bersih")
    elif action == "h":
        if clean_html is None:
            print("clean_html_to_clipboard.py tidak bisa diimport.")
            return False
        content["text"] = clean_html(read_multiline("HTML"))
    elif action == "e":
        content["text"] = edit_with_editor(str(content.get("text") or ""))
    elif action == "q":
        return False
    else:
        print(f"Aksi tidak dikenal: {action}")
        return False

    status = input("Status content [success]: ").strip() or "success"
    content["status"] = status
    if "http_status" not in content or content.get("http_status") in ("", None):
        content["http_status"] = 200 if status == "success" else content.get("http_status")
    return True


def load_datasets(files: list[Path]) -> list[dict[str, Any]]:
    datasets: list[dict[str, Any]] = []
    for path in files:
        data = load_json(path)
        records = data.get("records")
        if not isinstance(records, list):
            print(f"Skip file tanpa records list: {path}")
            continue
        datasets.append({"path": path, "data": data, "records": records})
    return datasets


def find_record_matches(
    datasets: list[dict[str, Any]],
    query: str,
    include_blacklisted: bool,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for dataset in datasets:
        path = dataset["path"]
        records = dataset["records"]
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                continue
            if record_matches_url(record, query, include_blacklisted):
                matches.append(
                    {
                        "path": path,
                        "data": dataset["data"],
                        "records": records,
                        "index": index,
                        "record": record,
                    }
                )
    return matches


def print_match_list(matches: list[dict[str, Any]]) -> None:
    print(f"\nDitemukan {len(matches):,} record cocok.")
    for number, match in enumerate(matches, start=1):
        record = match["record"]
        content = safe_content(record)
        url = record.get("canonical_url") or record.get("url") or content.get("source_url")
        title = content.get("title") or record.get("title") or ""
        text_len = len(str(content.get("text") or ""))
        print(
            f"{number}. {match['path'].name}#{match['index'] + 1} "
            f"| black={bool(record.get('is_blacklisted'))} "
            f"| text={text_len:,} | {preview(url, 120)}"
        )
        if title:
            print(f"   {preview(title, 140)}")


def prompt_match_choice(matches: list[dict[str, Any]]) -> int | None:
    while True:
        raw = input("Pilih nomor record, [Enter] cari URL lain, q keluar: ").strip()
        if raw == "":
            return None
        if raw.lower() == "q":
            raise KeyboardInterrupt
        if raw.isdigit() and 1 <= int(raw) <= len(matches):
            return int(raw) - 1
        print("Pilihan tidak valid.")


def main() -> None:
    args = parse_args()
    files = dataset_files(args)
    if not files:
        raise SystemExit(f"Tidak ada dataset JSON: {args.dataset_dir}")

    datasets = load_datasets(files)
    if not datasets:
        raise SystemExit("Tidak ada dataset valid untuk direview.")

    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backed_up: set[Path] = set()
    updated_count = 0

    print(f"Loaded dataset files: {len(datasets):,}")
    print("Input URL atau bagian URL yang ingin direview. Kosong/q untuk selesai.")
    while True:
        query = input("\nURL search> ").strip()
        if not query or query.lower() == "q":
            break

        matches = find_record_matches(
            datasets=datasets,
            query=query,
            include_blacklisted=args.include_blacklisted,
        )
        if not matches:
            print("Tidak ada record cocok.")
            continue

        print_match_list(matches)
        choice = prompt_match_choice(matches)
        if choice is None:
            continue

        match = matches[choice]
        record = match["record"]
        print_record(
            path=match["path"],
            index=match["index"],
            total=len(match["records"]),
            record=record,
        )
        if not apply_content_update(record):
            print("Batal update.")
            continue

        backup = make_backup_once(
            match["path"],
            args.backup_dir,
            backed_up,
            session_id,
        )
        write_json(match["path"], match["data"])
        updated_count += 1
        print(f"Updated: {match['path']}#{match['index'] + 1}")
        print(f"Backup : {backup}")

    print("\nSelesai.")
    print(f"Updated: {updated_count:,}")
    if backed_up:
        print(f"Backup dir: {args.backup_dir / session_id}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDihentikan oleh user.", file=sys.stderr)
        raise SystemExit(130)
