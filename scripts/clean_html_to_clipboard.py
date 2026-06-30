"""Bersihkan HTML menjadi teks dan salin hasilnya ke clipboard."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from html import unescape
from html.parser import HTMLParser
from pathlib import Path


BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "dd",
    "div",
    "dl",
    "dt",
    "figcaption",
    "figure",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "tbody",
    "td",
    "tfoot",
    "th",
    "thead",
    "tr",
    "ul",
}

IGNORED_TAGS = {"script", "style", "noscript", "svg", "canvas"}


class HtmlTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in IGNORED_TAGS:
            self._ignored_depth += 1
            return
        if tag in BLOCK_TAGS:
            self._add_newline()

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in IGNORED_TAGS and self._ignored_depth:
            self._ignored_depth -= 1
            return
        if tag in BLOCK_TAGS:
            self._add_newline()

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        text = " ".join(unescape(data).split())
        if text:
            self._parts.append(text)

    def get_text(self) -> str:
        lines: list[str] = []
        current: list[str] = []
        for part in self._parts:
            if part == "\n":
                if current:
                    lines.append(" ".join(current).strip())
                    current = []
                continue
            current.append(part)
        if current:
            lines.append(" ".join(current).strip())

        cleaned_lines: list[str] = []
        for line in lines:
            if line and (not cleaned_lines or cleaned_lines[-1] != line):
                cleaned_lines.append(line)
        return "\n".join(cleaned_lines).strip()

    def _add_newline(self) -> None:
        if self._parts and self._parts[-1] != "\n":
            self._parts.append("\n")


def clean_html(html: str) -> str:
    parser = HtmlTextExtractor()
    parser.feed(html)
    parser.close()
    return parser.get_text()


def copy_to_clipboard(text: str) -> None:
    commands = [
        ["wl-copy"],
        ["xclip", "-selection", "clipboard"],
        ["xsel", "--clipboard", "--input"],
        ["pbcopy"],
        ["clip.exe"],
        ["powershell.exe", "-NoProfile", "-Command", "Set-Clipboard"],
    ]
    for command in commands:
        if shutil.which(command[0]) is None:
            continue
        try:
            subprocess.run(
                command,
                input=text,
                text=True,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        except (OSError, subprocess.CalledProcessError):
            continue
    raise RuntimeError(
        "Clipboard command tidak ditemukan. Install wl-copy/xclip/xsel, "
        "atau jalankan di WSL dengan clip.exe/powershell.exe tersedia."
    )


def read_input(path: Path | None) -> str:
    if path is None:
        return sys.stdin.read()
    return path.read_text(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bersihkan HTML dari file/stdin dan salin teks ke clipboard."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="Path file HTML. Jika kosong, HTML dibaca dari stdin.",
    )
    parser.add_argument(
        "--no-print",
        action="store_true",
        help="Jangan tampilkan teks hasil pembersihan ke stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = clean_html(read_input(args.input))
    copy_to_clipboard(text)
    if not args.no_print:
        print(text)


if __name__ == "__main__":
    main()
