"""Clustering dan similarity untuk skenario non-LLM."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import polars as pl

import config


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


class ClusteringService:
    """Kelompokkan opini berdasarkan kedekatan vektor tanpa dependency wajib."""

    def __init__(
        self,
        similarity_threshold: float = config.CLUSTER_SIMILARITY_THRESHOLD,
        min_cluster_size: int = config.MIN_CLUSTER_SIZE,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size

    def cluster_dataframe(
        self,
        df: pl.DataFrame,
        vectors: list[list[float]],
        rule_label_column: str = config.COL_RULE_LABEL,
    ) -> pl.DataFrame:
        if len(vectors) != df.height:
            raise ValueError("Jumlah vector harus sama dengan jumlah baris DataFrame")
        if rule_label_column not in df.columns:
            raise KeyError(f"Kolom '{rule_label_column}' tidak ada pada DataFrame")

        cluster_ids = self._connected_components(vectors)
        cluster_sizes = Counter(cluster_ids)
        cluster_ids = [
            cluster_id if cluster_sizes[cluster_id] >= self.min_cluster_size else -1
            for cluster_id in cluster_ids
        ]

        centroids = self._centroids(vectors, cluster_ids)
        similarities = [
            round(cosine_similarity(vector, centroids.get(cluster_id, [])), 4)
            if cluster_id != -1 else 0.0
            for vector, cluster_id in zip(vectors, cluster_ids)
        ]
        semantic_labels = self._majority_labels(
            df[rule_label_column].to_list(), cluster_ids
        )

        return df.with_columns(
            pl.Series(config.COL_CLUSTER_ID, cluster_ids),
            pl.Series("cluster_size", [Counter(cluster_ids)[cid] for cid in cluster_ids]),
            pl.Series(config.COL_SEMANTIC_SIMILARITY, similarities),
            pl.Series(
                config.COL_SEMANTIC_LABEL,
                [semantic_labels.get(cid, "netral") for cid in cluster_ids],
            ),
        )

    def _connected_components(self, vectors: list[list[float]]) -> list[int]:
        centroids: list[list[float]] = []
        counts: list[int] = []
        cluster_ids: list[int] = []

        for vector in vectors:
            best_cluster = -1
            best_similarity = -1.0
            for cluster_id, centroid in enumerate(centroids):
                similarity = cosine_similarity(vector, centroid)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster = cluster_id

            if best_similarity >= self.similarity_threshold:
                cluster_ids.append(best_cluster)
                counts[best_cluster] += 1
                count = counts[best_cluster]
                centroids[best_cluster] = [
                    old + ((new - old) / count)
                    for old, new in zip(centroids[best_cluster], vector)
                ]
            else:
                cluster_ids.append(len(centroids))
                centroids.append(vector[:])
                counts.append(1)

        return cluster_ids

    @staticmethod
    def _centroids(
        vectors: list[list[float]], cluster_ids: list[int]
    ) -> dict[int, list[float]]:
        grouped: dict[int, list[list[float]]] = defaultdict(list)
        for vector, cluster_id in zip(vectors, cluster_ids):
            if cluster_id != -1:
                grouped[cluster_id].append(vector)

        centroids: dict[int, list[float]] = {}
        for cluster_id, items in grouped.items():
            dimension = len(items[0])
            centroids[cluster_id] = [
                sum(item[i] for item in items) / len(items) for i in range(dimension)
            ]
        return centroids

    @staticmethod
    def _majority_labels(labels: list[str], cluster_ids: list[int]) -> dict[int, str]:
        grouped: dict[int, list[str]] = defaultdict(list)
        for label, cluster_id in zip(labels, cluster_ids):
            if cluster_id != -1:
                grouped[cluster_id].append(label)

        return {
            cluster_id: Counter(items).most_common(1)[0][0]
            for cluster_id, items in grouped.items()
        }
