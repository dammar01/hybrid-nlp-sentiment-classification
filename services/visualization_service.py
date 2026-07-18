"""Visualisasi evaluasi skenario sentimen."""

from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse

import polars as pl

import config


class VisualizationService:
    """Buat chart evaluasi dari hasil klasifikasi tanpa menyimpan logic di notebook."""

    def __init__(self, labels: tuple[str, ...] = config.SENTIMENT_LABELS) -> None:
        self.labels = labels

    def plot_dataset_overview(self, df: pl.DataFrame, summary: dict[str, Any]):
        """Tampilkan ringkasan dataset berbasis teks."""
        plt = self._load_pyplot()
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        fig.suptitle("Ringkasan Dataset Kalibrasi", fontsize=13, fontweight="bold")

        if config.COL_SOURCE in df.columns:
            source_counts = Counter(str(value or "kosong") for value in df[config.COL_SOURCE].to_list())
            source_items = source_counts.most_common(10)
            axes[0].bar(
                [item[0] for item in source_items],
                [item[1] for item in source_items],
                color="#4C78A8",
            )
            axes[0].set_title("Sebaran Sumber Data")
            axes[0].set_xlabel("Sumber")
            axes[0].set_ylabel("Jumlah Data")
            axes[0].tick_params(axis="x", rotation=25)
        else:
            self._empty_axis(axes[0], "Kolom source tidak tersedia")

        text_lengths = self._text_lengths(df, config.COL_TEXT)
        if text_lengths:
            axes[1].hist(text_lengths, bins=30, color="#59A14F")
            axes[1].set_title("Distribusi Panjang Teks")
            axes[1].set_xlabel("Jumlah Karakter")
            axes[1].set_ylabel("Frekuensi")
        else:
            self._empty_axis(axes[1], "Kolom text tidak tersedia")

        quality_values = [
            int(summary.get("total_data", 0)),
            int(summary.get("jumlah_data_kosong", 0)),
            int(summary.get("jumlah_duplikat", 0)),
        ]
        quality_labels = ["Total", "Kosong", "Duplikat"]
        bars = axes[2].bar(quality_labels, quality_values, color=["#4C78A8", "#E15759", "#F28E2B"])
        axes[2].set_title("Kualitas Dataset")
        axes[2].set_ylabel("Jumlah Data")
        self._annotate_bars(axes[2], bars, quality_values)

        fig.tight_layout(rect=(0, 0, 1, 0.9))
        return fig

    def plot_preprocessing_overview(
        self,
        before_df: pl.DataFrame,
        after_df: pl.DataFrame,
        text_column: str = config.COL_TEXT,
        processed_column: str = config.COL_PROCESSED,
    ):
        """Bandingkan jumlah data dan panjang teks sebelum/sesudah preprocessing."""
        plt = self._load_pyplot()
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        fig.suptitle("Dampak Cleaning dan Preprocessing", fontsize=13, fontweight="bold")

        counts = [before_df.height, after_df.height]
        bars = axes[0].bar(["Sebelum", "Sesudah"], counts, color=["#4C78A8", "#59A14F"])
        axes[0].set_title("Jumlah Data")
        axes[0].set_ylabel("Baris")
        self._annotate_bars(axes[0], bars, counts)

        before_lengths = self._text_lengths(before_df, text_column)
        after_lengths = self._text_lengths(after_df, processed_column)
        axes[1].hist(before_lengths, bins=30, alpha=0.65, label="Sebelum", color="#4C78A8")
        axes[1].hist(after_lengths, bins=30, alpha=0.65, label="Sesudah", color="#F28E2B")
        axes[1].set_title("Distribusi Panjang Teks")
        axes[1].set_xlabel("Jumlah Karakter")
        axes[1].set_ylabel("Frekuensi")
        axes[1].legend()

        average_values = [
            sum(before_lengths) / len(before_lengths) if before_lengths else 0,
            sum(after_lengths) / len(after_lengths) if after_lengths else 0,
        ]
        bars = axes[2].bar(["Sebelum", "Sesudah"], average_values, color=["#4C78A8", "#F28E2B"])
        axes[2].set_title("Rata-Rata Panjang Teks")
        axes[2].set_ylabel("Karakter")
        self._annotate_bars(axes[2], bars, average_values, precision=1)

        fig.tight_layout(rect=(0, 0, 1, 0.9))
        return fig

    def plot_rule_sentiment_overview(self, df: pl.DataFrame, top_n: int = 15):
        """Tampilkan distribusi rule label, confidence, dan rule terbanyak tertrigger."""
        plt = self._load_pyplot()
        fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
        fig.suptitle("Analisis Rule-Based Sentiment", fontsize=13, fontweight="bold")

        self._plot_single_label_distribution(
            axes[0],
            df,
            column=config.COL_RULE_LABEL,
            title="Sebaran Label Rule",
            color="#4C78A8",
        )

        if config.COL_RULE_CONFIDENCE in df.columns:
            axes[1].hist(df[config.COL_RULE_CONFIDENCE].to_list(), bins=20, color="#59A14F")
            axes[1].set_title("Distribusi Confidence Rule")
            axes[1].set_xlabel("Confidence")
            axes[1].set_ylabel("Jumlah Data")
        else:
            self._empty_axis(axes[1], "Kolom confidence tidak tersedia")

        hit_counts = self._rule_hit_counts(df)
        if hit_counts:
            top_hits = hit_counts.most_common(top_n)
            words = [item[0] for item in top_hits]
            values = [item[1] for item in top_hits]
            axes[2].barh(words[::-1], values[::-1], color="#F28E2B")
            axes[2].set_title("Rule/Kata Terbanyak Tertrigger")
            axes[2].set_xlabel("Jumlah Trigger")
        else:
            self._empty_axis(axes[2], "Belum ada rule hit")

        fig.tight_layout(rect=(0, 0, 1, 0.9))
        return fig

    def plot_semantic_overview(self, df: pl.DataFrame):
        """Tampilkan ringkasan cluster, similarity, dan heatmap rule vs semantic."""
        plt = self._load_pyplot()
        fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
        fig.suptitle("Analisis Semantic Clustering", fontsize=13, fontweight="bold")

        if config.COL_CLUSTER_ID in df.columns:
            counter = Counter(df[config.COL_CLUSTER_ID].to_list())
            top_clusters = counter.most_common(12)
            labels = [str(item[0]) for item in top_clusters]
            values = [item[1] for item in top_clusters]
            axes[0].bar(labels, values, color="#4C78A8")
            axes[0].set_title("Top Cluster")
            axes[0].set_xlabel("Cluster ID (-1 = noise)")
            axes[0].set_ylabel("Jumlah Data")
        else:
            self._empty_axis(axes[0], "Kolom cluster tidak tersedia")

        if config.COL_SEMANTIC_SIMILARITY in df.columns:
            axes[1].hist(df[config.COL_SEMANTIC_SIMILARITY].to_list(), bins=20, color="#59A14F")
            axes[1].set_title("Distribusi Semantic Similarity")
            axes[1].set_xlabel("Similarity")
            axes[1].set_ylabel("Jumlah Data")
        else:
            self._empty_axis(axes[1], "Kolom similarity tidak tersedia")

        self._plot_cross_label_heatmap(
            axes[2],
            df,
            row_column=config.COL_RULE_LABEL,
            col_column=config.COL_SEMANTIC_LABEL,
            title="Heatmap Rule vs Semantic",
        )

        fig.tight_layout(rect=(0, 0, 1, 0.9))
        return fig

    def plot_ambiguity_overview(self, df: pl.DataFrame, top_n: int = 10):
        """Tampilkan sebaran ambiguous row, alasan ambiguity, dan final label."""
        plt = self._load_pyplot()
        fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
        fig.suptitle("Analisis Ambiguity Detection Tanpa LLM", fontsize=13, fontweight="bold")

        self._plot_ambiguity_distribution(axes[0], df, config.COL_IS_AMBIGUOUS)

        reason_counts = self._ambiguity_reason_counts(df)
        if reason_counts:
            top_reasons = reason_counts.most_common(top_n)
            reasons = [item[0] for item in top_reasons]
            values = [item[1] for item in top_reasons]
            axes[1].barh(reasons[::-1], values[::-1], color="#E15759")
            axes[1].set_title("Alasan Ambiguity Terbanyak")
            axes[1].set_xlabel("Jumlah Data")
        else:
            self._empty_axis(axes[1], "Tidak ada alasan ambiguity")

        self._plot_single_label_distribution(
            axes[2],
            df,
            column=config.COL_FINAL_LABEL,
            title="Sebaran Final Label",
            color="#F28E2B",
        )

        fig.tight_layout(rect=(0, 0, 1, 0.9))
        return fig

    def plot_topic_overview(self, topic_summary: dict[str, Any], top_n: int = 10):
        """Visualisasi frekuensi keyword SO-CAL per kelas sentimen."""
        plt = self._load_pyplot()
        sentiment_keywords = topic_summary.get("sentiment_keywords") or {}
        labels = list(config.TOPIC_SENTIMENT_ORDER)
        fig, axes = plt.subplots(3, 1, figsize=(14, 15), squeeze=False)
        fig.suptitle(
            "Keyword SO-CAL per Sentimen",
            fontsize=13,
            fontweight="bold",
        )
        palette = {"negatif": "#E15759", "netral": "#BFBFBF", "positif": "#59A14F"}

        for row_index, label in enumerate(labels):
            axis = axes[row_index][0]
            details = list(
                sentiment_keywords.get(label, {}).get("keyword_details", [])
            )[:top_n][::-1]
            if not details:
                self._empty_axis(axis, f"Keyword SO-CAL {label} tidak tersedia")
                continue

            keywords = [str(item["keyword"]) for item in details]
            frequencies = [int(item.get("frequency", 0)) for item in details]
            axis.barh(keywords, frequencies, color=palette.get(label, "#4C78A8"))
            axis.set_title(f"Keyword {label.title()}")
            axis.set_xlabel("Jumlah Kemunculan")

        fig.tight_layout(rect=(0, 0, 1, 0.95))
        return fig

    def plot_hybrid_sentiment_distribution(self, df: pl.DataFrame):
        """Bandingkan label IndoBERT, rule-based, final hybrid, dan aksi fusion."""
        plt = self._load_pyplot()
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle("Distribusi Sentimen Hybrid NLP", fontsize=14, fontweight="bold")

        self._plot_single_label_distribution(
            axes[0][0], df, "bert_label", "Sentimen IndoBERT", "#4C78A8"
        )
        self._plot_single_label_distribution(
            axes[0][1], df, config.COL_RULE_LABEL, "Sentimen Rule-Based", "#59A14F"
        )
        self._plot_single_label_distribution(
            axes[1][0], df, "final_sentiment", "Sentimen Final Hybrid", "#F28E2B"
        )

        if "fusion_action" in df.columns:
            actions = Counter(str(value or "unknown") for value in df["fusion_action"].to_list())
            labels = list(actions)
            values = [actions[label] for label in labels]
            bars = axes[1][1].bar(labels, values, color="#B279A2")
            axes[1][1].set_title("Aksi Fusion")
            axes[1][1].set_ylabel("Jumlah Data")
            axes[1][1].tick_params(axis="x", rotation=25)
            self._annotate_bars(axes[1][1], bars, values)
        else:
            self._empty_axis(axes[1][1], "Kolom fusion_action tidak tersedia")

        fig.tight_layout(rect=(0, 0, 1, 0.95))
        return fig

    def plot_kalbar_location_distribution(self, df: pl.DataFrame):
        """Plot distribusi final sentiment untuk seluruh kabupaten/kota Kalbar."""
        plt = self._load_pyplot()
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.suptitle(
            "Persebaran Sentimen Hybrid di Kalimantan Barat",
            fontsize=14,
            fontweight="bold",
        )

        if "location" not in df.columns or "final_sentiment" not in df.columns:
            self._empty_axis(ax, "Kolom location atau final_sentiment tidak tersedia")
            return fig

        kalbar_locations = self._kalbar_regency_names()
        lookup = {self._canonical_location(name).casefold(): name for name in kalbar_locations}
        general_label = "Kalimantan Barat (umum)"
        empty_label = "Lokasi tidak spesifik"
        other_label = "Lokasi lainnya"
        counts = {
            label: {sentiment: 0 for sentiment in self.labels}
            for label in (*kalbar_locations, general_label, empty_label, other_label)
        }

        for row in df.select("location", "final_sentiment").iter_rows(named=True):
            raw_location = str(row.get("location") or "").strip()
            sentiment = str(row.get("final_sentiment") or "")
            canonical = self._canonical_location(raw_location)
            if not raw_location:
                label = empty_label
            elif canonical.casefold() in {"kalimantan barat", "kalbar"}:
                label = general_label
            else:
                label = lookup.get(canonical.casefold(), other_label)
            if sentiment in self.labels:
                counts[label][sentiment] += 1

        labels = list(kalbar_locations)
        for extra in (general_label, empty_label, other_label):
            if sum(counts[extra].values()) > 0:
                labels.append(extra)

        y_positions = list(range(len(labels)))
        left = [0] * len(labels)
        colors = {"negatif": "#E15759", "netral": "#BFBFBF", "positif": "#59A14F"}
        for sentiment in self.labels:
            values = [counts[label][sentiment] for label in labels]
            ax.barh(
                y_positions,
                values,
                left=left,
                label=sentiment,
                color=colors.get(sentiment, "#4C78A8"),
            )
            left = [current + value for current, value in zip(left, values)]

        ax.set_yticks(y_positions, labels)
        ax.invert_yaxis()
        ax.set_xlabel("Jumlah Data")
        ax.set_ylabel("Kabupaten/Kota")
        ax.legend(title="Sentimen")
        for index, total in enumerate(left):
            ax.text(total, index, f" {total}", va="center", fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        return fig

    def plot_indobert_to_hybrid_transition(self, df: pl.DataFrame):
        """Tampilkan transisi label IndoBERT ke keputusan final hybrid."""
        plt = self._load_pyplot()
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        self._plot_cross_label_heatmap(
            ax,
            df,
            row_column="bert_label",
            col_column="final_sentiment",
            title="Transisi IndoBERT -> Hybrid (bukan evaluasi)",
        )
        ax.set_xlabel("Label Final Hybrid")
        ax.set_ylabel("Label IndoBERT")
        fig.tight_layout()
        return fig

    def plot_held_out_method_comparison(self, metrics: dict[str, Any]):
        """Bandingkan metrik rule, IndoBERT, dan hybrid pada held-out test."""
        plt = self._load_pyplot()
        methods = ("rule_based", "indobert", "final_hybrid")
        names = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
        labels = {"rule_based": "Rule-Based", "indobert": "IndoBERT", "final_hybrid": "Hybrid"}
        colors = {"rule_based": "#59A14F", "indobert": "#4C78A8", "final_hybrid": "#F28E2B"}
        fig, ax = plt.subplots(1, 1, figsize=(11, 5))
        x_positions = list(range(len(names)))
        width = 0.24
        for offset, method in enumerate(methods):
            values = [float((metrics.get(method) or {}).get(name, 0.0)) for name in names]
            positions = [value + (offset - 1) * width for value in x_positions]
            bars = ax.bar(positions, values, width=width, label=labels[method], color=colors[method])
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center", fontsize=8)
        ax.set_xticks(x_positions, ["Accuracy", "Balanced Accuracy", "Macro F1", "Weighted F1"])
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Skor")
        ax.set_title("Perbandingan Metode pada Held-Out Test")
        ax.legend()
        fig.tight_layout()
        return fig

    def plot_held_out_per_label(self, metrics: dict[str, Any]):
        """Tampilkan metrik per label dan confusion matrix hybrid held-out."""
        plt = self._load_pyplot()
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        per_label = metrics.get("per_label") or {}
        x_positions = list(range(len(self.labels)))
        width = 0.25
        for offset, name in enumerate(("precision", "recall", "f1")):
            values = [float((per_label.get(label) or {}).get(name, 0.0)) for label in self.labels]
            axes[0].bar([value + (offset - 1) * width for value in x_positions], values, width=width, label=name.upper())
        axes[0].set_xticks(x_positions, self.labels)
        axes[0].set_ylim(0, 1.05)
        axes[0].set_title("Performa Hybrid pada Held-Out Test")
        axes[0].set_ylabel("Skor")
        axes[0].legend()
        self._plot_confusion_matrix(axes[1], metrics)
        axes[1].set_title("Confusion Matrix Hybrid - Held-Out Test")
        fig.tight_layout()
        return fig

    def plot_split_and_runtime_distribution(self, split_df: pl.DataFrame, runtime_df: pl.DataFrame):
        """Bandingkan sebaran label fixed split dan sentimen runtime."""
        plt = self._load_pyplot()
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        if {"split", "sentiment_label"}.issubset(split_df.columns):
            grouped = Counter(split_df.select("split", "sentiment_label").iter_rows())
            splits = ("train", "calibration", "test")
            x_positions = list(range(len(splits)))
            width = 0.24
            colors = {"negatif": "#E15759", "netral": "#72B7B2", "positif": "#59A14F"}
            for offset, label in enumerate(self.labels):
                axes[0].bar(
                    [value + (offset - 1) * width for value in x_positions],
                    [grouped.get((split, label), 0) for split in splits],
                    width=width,
                    label=label,
                    color=colors.get(label),
                )
            axes[0].set_xticks(x_positions, splits)
            axes[0].set_title("Sebaran Label per Split")
            axes[0].set_ylabel("Jumlah")
            axes[0].legend()
        else:
            self._empty_axis(axes[0], "Fixed split tidak tersedia")
        self._plot_single_label_distribution(axes[1], runtime_df, "final_sentiment", "Sebaran Sentimen Final Hybrid", "#F28E2B")
        fig.tight_layout()
        return fig

    def plot_domain_and_source_type_distribution(self, df: pl.DataFrame, top_n: int = 15):
        """Tampilkan top domain dan source type, dikelompokkan per sentimen."""
        plt = self._load_pyplot()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        domains = self._domain_values(df)
        sources = self._column_values(df, "source_type")
        sentiments = self._column_values(df, "final_sentiment")
        self._plot_category_by_sentiment(axes[0], domains, sentiments, "Top Domain per Sentimen", top_n)
        self._plot_category_by_sentiment(axes[1], sources, sentiments, "Sebaran Source Type per Sentimen", top_n)
        fig.tight_layout()
        return fig

    def plot_aspect_distribution(self, df: pl.DataFrame, top_n: int = 15):
        """Tampilkan sebaran aspect yang dikelompokkan per sentimen."""
        plt = self._load_pyplot()
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        self._plot_category_by_sentiment(
            ax,
            self._column_values(df, "aspect"),
            self._column_values(df, "final_sentiment"),
            "Sebaran Aspect per Sentimen",
            top_n,
        )
        fig.tight_layout()
        return fig

    def save_figure(self, figure, path: str | Path, *, dpi: int = 160) -> Path:
        """Simpan figure dan tutup resource matplotlib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        self._load_pyplot().close(figure)
        return path

    def plot_evaluation_dashboard(
        self,
        df: pl.DataFrame,
        metrics: dict[str, Any],
        actual_column: str = config.COL_ACTUAL_LABEL,
        predicted_column: str = config.COL_FINAL_LABEL,
        ambiguity_column: str = config.COL_IS_AMBIGUOUS,
    ):
        """Tampilkan metrik utama, distribusi label, ambiguitas, dan confusion matrix."""
        plt = self._load_pyplot()
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle("Evaluasi Skenario Hybrid NLP Tanpa LLM", fontsize=14, fontweight="bold")

        self._plot_metrics(axes[0][0], metrics)
        self._plot_label_distribution(
            axes[0][1],
            df,
            actual_column=actual_column,
            predicted_column=predicted_column,
        )
        self._plot_ambiguity_distribution(axes[1][0], df, ambiguity_column)
        self._plot_confusion_matrix(axes[1][1], metrics)

        fig.tight_layout(rect=(0, 0, 1, 0.95))
        return fig

    def _plot_metrics(self, ax, metrics: dict[str, Any]) -> None:
        names = ["accuracy", "balanced_accuracy"]
        values = [float(metrics.get(name, 0.0)) for name in names]
        labels = ["Accuracy", "Balanced Accuracy"]

        bars = ax.bar(labels, values, color=["#4C78A8", "#59A14F"])
        ax.set_title("Metrik Utama")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Skor")
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(value + 0.03, 0.98),
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

    def _plot_label_distribution(
        self,
        ax,
        df: pl.DataFrame,
        actual_column: str,
        predicted_column: str,
    ) -> None:
        actual = Counter(df[actual_column].to_list()) if actual_column in df.columns else {}
        predicted = (
            Counter(df[predicted_column].to_list()) if predicted_column in df.columns else {}
        )

        x_positions = range(len(self.labels))
        width = 0.36
        actual_values = [actual.get(label, 0) for label in self.labels]
        predicted_values = [predicted.get(label, 0) for label in self.labels]

        ax.bar([x - width / 2 for x in x_positions], actual_values, width, label="Aktual", color="#4C78A8")
        ax.bar([x + width / 2 for x in x_positions], predicted_values, width, label="Prediksi", color="#F28E2B")
        ax.set_title("Sebaran Label Aktual vs Prediksi")
        ax.set_xticks(list(x_positions), self.labels)
        ax.set_ylabel("Jumlah Data")
        ax.legend()

    def _plot_single_label_distribution(
        self,
        ax,
        df: pl.DataFrame,
        column: str,
        title: str,
        color: str,
    ) -> None:
        if column not in df.columns:
            self._empty_axis(ax, f"Kolom {column} tidak tersedia")
            return

        counts = Counter(df[column].to_list())
        values = [counts.get(label, 0) for label in self.labels]
        bars = ax.bar(self.labels, values, color=color)
        ax.set_title(title)
        ax.set_ylabel("Jumlah Data")
        self._annotate_bars(ax, bars, values)

    def _plot_ambiguity_distribution(
        self, ax, df: pl.DataFrame, ambiguity_column: str
    ) -> None:
        if ambiguity_column not in df.columns:
            counts = {"Tidak Ambigu": 0, "Ambigu": 0}
        else:
            counter = Counter(bool(value) for value in df[ambiguity_column].to_list())
            counts = {"Tidak Ambigu": counter.get(False, 0), "Ambigu": counter.get(True, 0)}

        if sum(counts.values()) == 0:
            self._empty_axis(ax, "Belum ada data ambiguitas")
            return

        colors = ["#59A14F", "#E15759"]
        ax.pie(
            list(counts.values()),
            labels=list(counts.keys()),
            autopct=lambda pct: f"{pct:.1f}%" if pct > 0 else "",
            startangle=90,
            colors=colors,
        )
        ax.set_title("Sebaran Ambiguitas")

    def _plot_confusion_matrix(self, ax, metrics: dict[str, Any]) -> None:
        matrix = metrics.get("confusion_matrix", {})
        values = [
            [int(matrix.get(actual, {}).get(predicted, 0)) for predicted in self.labels]
            for actual in self.labels
        ]

        image = ax.imshow(values, cmap="Blues")
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Label Prediksi")
        ax.set_ylabel("Label Aktual")
        ax.set_xticks(range(len(self.labels)), self.labels)
        ax.set_yticks(range(len(self.labels)), self.labels)

        max_value = max((value for row in values for value in row), default=0)
        for row_index, row in enumerate(values):
            for col_index, value in enumerate(row):
                color = "white" if max_value and value > max_value / 2 else "black"
                ax.text(col_index, row_index, str(value), ha="center", va="center", color=color)
        ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    def _plot_cross_label_heatmap(
        self,
        ax,
        df: pl.DataFrame,
        row_column: str,
        col_column: str,
        title: str,
    ) -> None:
        if row_column not in df.columns or col_column not in df.columns:
            self._empty_axis(ax, "Kolom heatmap tidak lengkap")
            return

        matrix = [[0 for _ in self.labels] for _ in self.labels]
        row_index = {label: index for index, label in enumerate(self.labels)}
        col_index = {label: index for index, label in enumerate(self.labels)}
        for row in df.select(row_column, col_column).iter_rows(named=True):
            row_label = row.get(row_column)
            col_label = row.get(col_column)
            if row_label in row_index and col_label in col_index:
                matrix[row_index[row_label]][col_index[col_label]] += 1

        image = ax.imshow(matrix, cmap="YlGnBu")
        ax.set_title(title)
        ax.set_xlabel(col_column)
        ax.set_ylabel(row_column)
        ax.set_xticks(range(len(self.labels)), self.labels)
        ax.set_yticks(range(len(self.labels)), self.labels)
        max_value = max((value for row in matrix for value in row), default=0)
        for row_pos, row_values in enumerate(matrix):
            for col_pos, value in enumerate(row_values):
                color = "white" if max_value and value > max_value / 2 else "black"
                ax.text(col_pos, row_pos, str(value), ha="center", va="center", color=color)
        ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    def _plot_category_by_sentiment(self, ax, categories: list[str], sentiments: list[str], title: str, top_n: int) -> None:
        if not categories or not sentiments:
            self._empty_axis(ax, "Data tidak tersedia")
            return
        totals = Counter(categories)
        labels = [name for name, _count in totals.most_common(top_n)]
        y_positions = list(range(len(labels)))
        left = [0] * len(labels)
        colors = {"negatif": "#E15759", "netral": "#BFBFBF", "positif": "#59A14F"}
        pairs = Counter(zip(categories, sentiments))
        for sentiment in self.labels:
            values = [pairs.get((label, sentiment), 0) for label in labels]
            ax.barh(y_positions, values, left=left, label=sentiment, color=colors.get(sentiment))
            left = [current + value for current, value in zip(left, values)]
        ax.set_yticks(y_positions, labels)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("Jumlah")
        ax.legend()

    @staticmethod
    def _column_values(df: pl.DataFrame, column: str) -> list[str]:
        if column not in df.columns:
            return []
        return [str(value or "(kosong)").strip() or "(kosong)" for value in df[column].to_list()]

    @classmethod
    def _domain_values(cls, df: pl.DataFrame) -> list[str]:
        for column in ("raw_domain", "domain"):
            values = cls._column_values(df, column)
            if values and any(value != "(kosong)" for value in values):
                return values
        if "source_url" not in df.columns:
            return []
        return [urlparse(str(value or "")).netloc.lower() or "(kosong)" for value in df["source_url"].to_list()]

    @staticmethod
    def _text_lengths(df: pl.DataFrame, column: str) -> list[int]:
        if column not in df.columns:
            return []
        return [len(str(value or "")) for value in df[column].to_list()]

    @staticmethod
    def _canonical_location(name: str) -> str:
        value = re.sub(r"\s+", " ", str(name or "").strip())
        value = re.sub(r"^(kabupaten|kab\.?|kota)\s+", "", value, flags=re.IGNORECASE)
        return value.title()

    @classmethod
    def _kalbar_regency_names(cls) -> list[str]:
        province_code = ""
        with (config.RESOURCES / "wilayah" / "provinsi.csv").open(
            encoding=config.ENCODING,
            newline="",
        ) as handle:
            for row in csv.DictReader(handle):
                if str(row.get("name") or "").casefold() == "kalimantan barat":
                    province_code = str(row.get("code") or "").strip()
                    break

        names: list[str] = []
        seen: set[str] = set()
        with (config.RESOURCES / "wilayah" / "kabupaten.csv").open(
            encoding=config.ENCODING,
            newline="",
        ) as handle:
            for row in csv.DictReader(handle):
                if str(row.get("parent_code") or "").strip() != province_code:
                    continue
                name = cls._canonical_location(row.get("name") or "")
                if name and name.casefold() not in seen:
                    names.append(name)
                    seen.add(name.casefold())
        return names

    @staticmethod
    def _rule_hit_counts(df: pl.DataFrame) -> Counter:
        hit_columns = [
            column
            for column in (
                config.COL_RULE_PHRASE_HITS,
                config.COL_RULE_WORD_HITS,
                config.COL_RULE_HITS,
            )
            if column in df.columns
        ]
        if not hit_columns:
            return Counter()

        counter: Counter = Counter()
        for column in hit_columns:
            for raw_hits in df[column].to_list():
                for hit in str(raw_hits or "").split(","):
                    hit = hit.strip()
                    if hit:
                        counter[hit] += 1
        return counter

    @staticmethod
    def _ambiguity_reason_counts(df: pl.DataFrame) -> Counter:
        if "ambiguity_reason" not in df.columns:
            return Counter()

        counter: Counter = Counter()
        for raw_reasons in df["ambiguity_reason"].to_list():
            for reason in str(raw_reasons or "").split(","):
                reason = reason.strip()
                if reason:
                    counter[reason] += 1
        return counter

    @staticmethod
    def _annotate_bars(ax, bars, values: list[float], precision: int = 0) -> None:
        for bar, value in zip(bars, values):
            label = f"{value:.{precision}f}" if precision else f"{int(value)}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                label,
                ha="center",
                va="bottom",
            )

    @staticmethod
    def _empty_axis(ax, message: str) -> None:
        ax.axis("off")
        ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)

    @staticmethod
    def _load_pyplot():
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError(
                "Visualisasi membutuhkan matplotlib. Install matplotlib untuk menjalankan chart."
            ) from exc
        return plt
