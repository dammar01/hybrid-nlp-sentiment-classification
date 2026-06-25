# Changelog

Template ini dipakai untuk mencatat perubahan penting pada project.

Format mengikuti prinsip [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) dan versi dapat memakai pola `MAJOR.MINOR.PATCH`.

## [Unreleased]

### Added

-

### Changed

-

### Fixed

-

### Removed

-

### Notes

-

## [0.1.0] - 2026-06-20

### Added

- Inisialisasi semua kode awal dan struktur

### Changed

-

### Fixed

-

### Removed

-

### Notes

-

## [0.1.1] - 2026-06-25

### Added

- Beberapa datasets lexicon untuk rule based, datasets kalimat spesifik, dan dataset bantuan speerti negasi, dan score multiplier
- - Path resource lexicon baru di `config.py`.
- Output detail untuk rule-based sentiment: `rule_status`, `rule_positive_score`, `rule_negative_score`, `rule_neutral_hits`, `rule_phrase_hits`, `rule_word_hits`, `rule_modifier_hits`, dan `rule_explanation`.

### Changed

- Refactor `LexiconSentimentService` agar menggunakan `lexicon_words.json`, `phrase_rules.json`, dan `modifier_rules.json` sebagai source of truth.
- Mengubah scoring rule-based dari hit-count sederhana menjadi weighted scoring berbasis word, phrase, dan modifier.
- Menambahkan phrase-first matching untuk memprioritaskan frasa spesifik sebelum token kata.
- Menambahkan dukungan negation, intensifier, downtoner, dan contrast marker dalam proses scoring.
- Menyesuaikan `VisualizationService` agar membaca `rule_phrase_hits` dan `rule_word_hits`.

### Fixed

-

### Removed

- Hardcoded daftar kata positif/negatif dari `LexiconSentimentService`.

### Notes

- Penyesuaian algoritma dilakukan dalam meningkatkan rule based/lexicon agar lebih kuat menjadi independen (berdiri sendiri) didalam pipeline hybrid NLP ini dengan menambahkan dataset modifier_rules.json dan phrase_rules.json yang berguna dalam memproses data umum.
- Lexicon_words.json digunakan sebagai datasets kata apa saja yang berhubungan dengan sentimen secara langsung
- Modifier_rules.json digunakan sebagai datasets kata negasi dan kata penguat/pelemah score
- Phrase_rules.json digunakan sebagai datasets kalimat utuh yang secara spesifik menggambarkan sentimen dari data yang di proses
- 3 file ini akan digunakan sebagai source of truth dalam pemrosesan pipeline rule based
- `rule_confidence` masih berupa heuristic berbasis dominasi skor, bukan probabilitas statistik.

## [0.1.2] - 2026-06-25

### Added

- Output contract resmi untuk rule-based lexicon di `config.py` melalui `RULE_OUTPUT_COLUMNS`.
- Kolom versi kontrak dan versi resource: `rule_contract_version` dan `rule_resource_version`.
- Kolom evidence terstruktur `rule_evidence` untuk menyimpan detail hit dalam format JSON.
- Konstanta rule-based lexicon di `config.py`, termasuk token pattern, status, effect modifier, threshold, dan bobot contrast/concession.
- Jupyter notebook untuk melakukan analisis dataset yang telah dibuat

### Changed

- `LexiconSentimentService` sekarang memakai `config.RULE_TOKEN_PATTERN`, `RULE_WEAK_THRESHOLD`, status constants, dan effect constants.
- Pipeline internal phrase dan word disatukan melalui proses evidence collection sebelum scoring final.
- Output `analyze_text()` sekarang divalidasi terhadap `config.RULE_OUTPUT_COLUMNS`.
- `VisualizationService` sekarang membaca kolom hit rule melalui konstanta `config`, bukan literal string.

### Fixed

- Resource validation diperketat untuk label, weight, term kosong, scope modifier, multiplier, dan effect contrast/negation.
- Structured evidence menyimpan `source`, `score`, `weight`, `category`, `reason`, `span`, `modifiers`, dan `clause_weight` untuk setiap sentiment hit.

### Removed

-

### Notes

- Versi kontrak rule-based lexicon saat ini adalah `2.0.0`.
- Versi resource default saat ini adalah `1.0.0` jika file JSON belum memiliki field `version`.
- `rule_evidence` disimpan sebagai string JSON agar tetap kompatibel dengan output tabular Polars/CSV.
