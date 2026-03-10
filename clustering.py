"""
clustering.py — Online Speaker Clustering
==========================================

GOAL:
  Given a stream of speaker embeddings (one per audio chunk),
  assign each embedding to a speaker label: "Speaker 1", "Speaker 2", etc.

CHALLENGE:
  We don't know in advance how many speakers there are.
  We need to discover them in real time as the conversation unfolds.

ALGORITHM:
  We implement Online Agglomerative Clustering — FROM SCRATCH.

  AGGLOMERATIVE = "bottom-up"
    Start with each embedding as its own cluster.
    Merge the two closest clusters if they're similar enough.

  ONLINE = "streaming"
    Process embeddings one at a time as they arrive.
    Keep a running set of cluster centroids.

  COSINE SIMILARITY = our distance metric
    Measures angle between two vectors.
    1.0 = identical direction (same speaker)
    0.0 = orthogonal (unrelated)
    -1.0 = opposite (very different)
    Threshold ~0.75 works well for speaker matching.

WHAT WE IMPLEMENT FROM SCRATCH:
  - Cosine similarity computation
  - Centroid update rule
  - Online cluster assignment
  - Speaker label management
  - Cluster merging logic
  - Memory decay for old clusters

STUDENT NOTE:
  This is the core intelligence of the diarization system.
  Understanding this file is essential for understanding the whole system.
"""

import numpy as np
import logging
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
import time

logger = logging.getLogger("clustering")


# ──────────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────────

@dataclass
class SpeakerCluster:
    """
    Represents one discovered speaker cluster.

    A cluster is a group of embeddings that all belong
    to the same speaker (according to our algorithm).
    """
    speaker_id: int                        # Internal ID (0, 1, 2, ...)
    label: str                             # Display label ("Speaker 1", "Speaker 2", ...)
    centroid: np.ndarray                   # Average embedding of all members
    n_samples: int = 0                     # Number of embeddings in this cluster
    last_seen: float = 0.0                 # Timestamp of most recent activity
    total_duration: float = 0.0            # Total speaking time in seconds
    color: str = "#4CAF50"                 # Display color for frontend

    def __post_init__(self):
        self.last_seen = time.time()


# ══════════════════════════════════════════════════════════════
# ONLINE AGGLOMERATIVE CLUSTERING — FROM SCRATCH
# ══════════════════════════════════════════════════════════════

class OnlineSpeakerClusterer:
    """
    Online agglomerative clustering for real-time speaker diarization.

    ALGORITHM OVERVIEW:
      For each new embedding:
        1. Compute cosine similarity with all existing cluster centroids
        2. If max similarity > threshold → assign to that cluster, update centroid
        3. If max similarity < threshold → create a new cluster (new speaker!)
        4. Optionally: periodically merge very similar clusters

    CENTROID UPDATE:
      When we add embedding e to cluster C with n existing samples:
        new_centroid = (n * old_centroid + e) / (n + 1)
      Then L2-normalize.

      This is an online/incremental mean, no need to store all embeddings.

    PARAMETERS:
      similarity_threshold  : float — How similar two vectors must be to match
                                       (0.75 is typical for speaker matching)
      max_speakers          : int   — Hard cap on speaker count
      min_samples_to_keep   : int   — Don't keep clusters with too few samples
    """

    # Speaker display colors (one per possible speaker)
    SPEAKER_COLORS = [
        "#4FC3F7",  # Speaker 1: Sky Blue
        "#AED581",  # Speaker 2: Light Green
        "#FFB74D",  # Speaker 3: Orange
        "#F48FB1",  # Speaker 4: Pink
        "#CE93D8",  # Speaker 5: Purple
        "#80DEEA",  # Speaker 6: Cyan
        "#FFCC02",  # Speaker 7: Yellow
        "#FF8A65",  # Speaker 8: Deep Orange
    ]

    def __init__(
        self,
        similarity_threshold: float = 0.40,
        # WHY 0.40: In live 1.5-second chunks with overlap, same-speaker scores on
        # laptop microphones often land around 0.40–0.60 while true speaker changes
        # drop into the 0.20–0.35 range.  0.45 was too conservative and caused
        # persistent under-segmentation.
        max_speakers: int = 8,
        min_samples_to_keep: int = 2,
        merge_threshold: float = 0.75,
        # WHY 0.75: Only merge clusters that are virtually identical (an accidentally
        # split single speaker).  Two real speakers never score this high.
        max_inactive_seconds: float = 300.0,
        new_speaker_patience: int = 4
        # WHY 4: 4 consecutive below-threshold chunks = ≈6 seconds of sustained
        # mismatch at the default 1.5-second chunk size.  This is still resistant
        # to one-off noisy misses but responds fast enough to real speaker turns in
        # live conversation.
    ):
        self.similarity_threshold = similarity_threshold
        self.max_speakers = max_speakers
        self.min_samples_to_keep = min_samples_to_keep
        self.merge_threshold = merge_threshold
        self.max_inactive_seconds = max_inactive_seconds
        self.new_speaker_patience = new_speaker_patience

        # Active clusters (one per discovered speaker)
        self.clusters: List[SpeakerCluster] = []

        # Counter for assigning human-readable labels
        self._next_speaker_number: int = 1

        # History: list of (speaker_id, timestamp) for timeline
        self._assignment_history: List[Tuple[int, float]] = []

        # NEW: Consecutive miss counter — prevents creating a new speaker
        # from a single noisy embedding. Must see N consecutive misses
        # (below threshold) before actually creating a new speaker.
        self._consecutive_misses: int = 0
        self._miss_embeddings: List[np.ndarray] = []  # buffer of missed embeddings

        logger.info(
            f"OnlineSpeakerClusterer: threshold={similarity_threshold}, "
            f"max_speakers={max_speakers}, merge_threshold={merge_threshold}, "
            f"new_speaker_patience={new_speaker_patience}"
        )

    # ──────────────────────────────────────────────────────────
    # Main Assignment (FROM SCRATCH)
    # ──────────────────────────────────────────────────────────

    def assign_speaker(
        self,
        embedding: np.ndarray,
        duration: float = 1.0,
        timestamp: float = 0.0
    ) -> Tuple[int, str, float]:
        """
        Assign a speaker label to a new embedding — FROM SCRATCH.

        This is the core of the diarization algorithm.

        Parameters:
        -----------
        embedding  : np.ndarray — L2-normalized speaker embedding (192-dim)
        duration   : float      — Duration of this segment in seconds
        timestamp  : float      — Time in session (seconds)

        Returns:
        --------
        (speaker_id, speaker_label, confidence)
          speaker_id    : int   — Internal ID (0-indexed)
          speaker_label : str   — "Speaker 1", "Speaker 2", etc.
          confidence    : float — Similarity score of the match (0-1)
        """
        embedding = self._ensure_normalized(embedding)

        # Case 1: No clusters yet → create first speaker
        if len(self.clusters) == 0:
            return self._create_new_cluster(embedding, duration, timestamp)

        # Step 1: Compute similarity to all existing clusters
        similarities = self._compute_all_similarities(embedding)

        # Step 2: Find best match
        best_idx = int(np.argmax(similarities))
        best_similarity = float(similarities[best_idx])

        # Always log at INFO level for diagnosability
        sim_strs = ", ".join(
            f"{self.clusters[i].label}={similarities[i]:.3f}"
            for i in range(len(self.clusters))
        )
        logger.info(
            f"[Clustering] Similarities: [{sim_strs}] | "
            f"best={best_similarity:.3f} vs threshold={self.similarity_threshold:.3f} | "
            f"misses={self._consecutive_misses}/{self.new_speaker_patience}"
        )

        # Step 3: Decide: existing speaker or new speaker?
        if best_similarity >= self.similarity_threshold:
            # Existing speaker found — reset miss counter
            self._consecutive_misses = 0
            self._miss_embeddings.clear()

            cluster = self.clusters[best_idx]
            self._update_centroid(cluster, embedding, duration)
            cluster.last_seen = time.time()

            self._assignment_history.append((cluster.speaker_id, timestamp))
            logger.debug(f"Assigned to existing {cluster.label} (sim={best_similarity:.3f})")
            return cluster.speaker_id, cluster.label, best_similarity
        else:
            # Below threshold — possible new speaker, but don't be trigger-happy.
            # Require `new_speaker_patience` consecutive misses before creating
            # a new cluster.  This avoids fragmenting on one noisy embedding.
            self._consecutive_misses += 1
            self._miss_embeddings.append(embedding)

            if self._consecutive_misses < self.new_speaker_patience:
                # Not enough evidence yet — assign to best existing cluster anyway
                cluster = self.clusters[best_idx]
                # Do NOT update centroid (don't contaminate with possibly-different speaker)
                self._assignment_history.append((cluster.speaker_id, timestamp))
                logger.debug(
                    f"Below threshold (sim={best_similarity:.3f}), "
                    f"miss {self._consecutive_misses}/{self.new_speaker_patience}, "
                    f"tentatively assigning to {cluster.label}"
                )
                return cluster.speaker_id, cluster.label, best_similarity

            # Enough consecutive misses — genuinely new speaker
            self._consecutive_misses = 0

            if len(self.clusters) >= self.max_speakers:
                # Hit speaker cap: assign to closest existing speaker
                logger.warning(f"Max speakers ({self.max_speakers}) reached. Assigning to closest.")
                cluster = self.clusters[best_idx]
                self._update_centroid(cluster, embedding, duration)
                self._miss_embeddings.clear()
                return cluster.speaker_id, cluster.label, best_similarity

            # Create new speaker with the AVERAGE of accumulated miss embeddings
            # for a more stable initial centroid
            avg_embedding = np.mean(self._miss_embeddings, axis=0)
            avg_embedding = self._l2_normalize(avg_embedding)
            self._miss_embeddings.clear()

            return self._create_new_cluster(avg_embedding, duration, timestamp)

    # ──────────────────────────────────────────────────────────
    # FROM SCRATCH: Cosine Similarity
    # ──────────────────────────────────────────────────────────

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Cosine similarity between two vectors — FROM SCRATCH.

        Formula: cos(θ) = (a · b) / (||a|| × ||b||)

        For L2-normalized vectors, ||a|| = ||b|| = 1,
        so this simplifies to just the dot product:
        cos(θ) = a · b

        Range: [-1, 1] where 1 = same direction, 0 = orthogonal

        STUDENT EXPLANATION:
          Imagine each embedding as an arrow in 192-dimensional space.
          Two arrows pointing in the same direction = same speaker.
          Cosine similarity measures the angle between them.
          Threshold 0.75 means the angle must be < ~41 degrees.
        """
        norm_a = np.sqrt(np.sum(a ** 2))
        norm_b = np.sqrt(np.sum(b ** 2))

        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0

        dot_product = float(np.dot(a, b))
        return dot_product / (norm_a * norm_b)

    def _compute_all_similarities(self, embedding: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity against all cluster centroids — FROM SCRATCH.

        Returns:
        --------
        np.ndarray — shape (n_clusters,), each entry is cosine similarity
        """
        similarities = np.zeros(len(self.clusters), dtype=np.float32)
        for i, cluster in enumerate(self.clusters):
            similarities[i] = self._cosine_similarity(embedding, cluster.centroid)
        return similarities

    # ──────────────────────────────────────────────────────────
    # FROM SCRATCH: Centroid Update
    # ──────────────────────────────────────────────────────────

    def _update_centroid(
        self,
        cluster: SpeakerCluster,
        new_embedding: np.ndarray,
        duration: float
    ):
        """
        Update cluster centroid with a new embedding — FROM SCRATCH.

        INCREMENTAL MEAN UPDATE:
          Instead of storing all embeddings (expensive),
          we maintain a running average.

          Formula: new_centroid = (n * old + new_embedding) / (n + 1)
          Then normalize to unit length.

          This is mathematically equivalent to computing the mean
          of all embeddings seen so far.

        STUDENT NOTE:
          This is also called "online mean estimation" or
          "Welford's online algorithm" (simplified version).
        """
        n = cluster.n_samples

        # Exponential moving average (EMA) centroid update
        # For early samples, use simple mean to build a stable centroid.
        # After enough samples, switch to EMA so recent speech has more
        # weight — this tracks voice drift over a long session.
        if n < 20:
            # Simple incremental mean for first 20 samples — build a stable initial
            # centroid before switching to EMA.  20 samples = ~40 seconds of speech,
            # enough to capture sentence-to-sentence pitch variation for that speaker.
            new_centroid = (n * cluster.centroid + new_embedding) / (n + 1)
        else:
            # EMA: alpha=0.25 gives recent speech 25% weight.
            # WHY 0.25 not 0.1: With 0.1 the centroid barely moves; a speaker's
            # voice sounds different across different sentences / loudness levels
            # and the frozen centroid diverges from actual speech, producing
            # artificially low similarity scores.  0.25 keeps the centroid
            # tracking the speaker without over-reacting to a single outlier.
            alpha = 0.25
            new_centroid = (1 - alpha) * cluster.centroid + alpha * new_embedding

        # L2 normalize to keep centroid on unit sphere
        new_centroid = self._l2_normalize(new_centroid)

        cluster.centroid = new_centroid
        cluster.n_samples += 1
        cluster.total_duration += duration

    # ──────────────────────────────────────────────────────────
    # FROM SCRATCH: Cluster Management
    # ──────────────────────────────────────────────────────────

    def _create_new_cluster(
        self,
        embedding: np.ndarray,
        duration: float,
        timestamp: float
    ) -> Tuple[int, str, float]:
        """
        Create a new cluster for a newly detected speaker — FROM SCRATCH.
        """
        speaker_id = len(self.clusters)
        speaker_number = self._next_speaker_number
        self._next_speaker_number += 1

        label = f"Speaker {speaker_number}"
        color_idx = speaker_id % len(self.SPEAKER_COLORS)
        color = self.SPEAKER_COLORS[color_idx]

        new_cluster = SpeakerCluster(
            speaker_id=speaker_id,
            label=label,
            centroid=embedding.copy(),
            n_samples=1,
            last_seen=time.time(),
            total_duration=duration,
            color=color
        )
        self.clusters.append(new_cluster)
        self._assignment_history.append((speaker_id, timestamp))

        logger.info(f"New speaker detected: {label} (total speakers: {len(self.clusters)})")
        return speaker_id, label, 1.0

    def log_diagnostic_matrix(self):
        """
        Log the full pairwise cosine similarity matrix between all clusters.

        This is the key feedback metric after the embedding preprocessing fix:
          - Same-speaker pairs should score > 0.45 within a session
          - Different-speaker pairs should score < 0.35 after enough samples

        Also logs per-cluster stats: n_samples, total_duration, last_seen age.
        """
        import time as _time
        if len(self.clusters) < 1:
            return

        now = _time.time()
        logger.info("=" * 60)
        logger.info("[DIAGNOSTIC] Speaker Cluster Matrix")
        logger.info(f"  Active clusters: {len(self.clusters)}")

        for c in self.clusters:
            age_sec = now - c.last_seen
            logger.info(
                f"  {c.label}: n_samples={c.n_samples}, "
                f"duration={c.total_duration:.1f}s, "
                f"last_seen={age_sec:.1f}s ago"
            )

        if len(self.clusters) >= 2:
            logger.info("  Pairwise similarity matrix:")
            header = "         " + "".join(f"{c.label:>12}" for c in self.clusters)
            logger.info(header)
            for i, ci in enumerate(self.clusters):
                row = f"  {ci.label:>6} |"
                for j, cj in enumerate(self.clusters):
                    sim = self._cosine_similarity(ci.centroid, cj.centroid)
                    marker = " ←SAME?" if (i != j and sim > self.merge_threshold) else ""
                    row += f"  {sim:+.3f}{marker:6s}"
                logger.info(row)
        logger.info("=" * 60)

    def merge_similar_clusters(self):
        """
        Merge clusters that have become too similar — FROM SCRATCH.

        This handles the case where a speaker gets accidentally
        split into two clusters (e.g., if they changed their voice).

        ALGORITHM:
          1. Build pairwise similarity matrix for all clusters
          2. Find pairs with similarity > merge_threshold
          3. Merge the smaller cluster into the larger one
          4. Repeat until no more merges needed

        This is called "post-hoc refinement" in research literature.
        """
        if len(self.clusters) < 2:
            return

        merged = True
        while merged:
            merged = False
            n = len(self.clusters)

            for i in range(n):
                for j in range(i + 1, n):
                    sim = self._cosine_similarity(
                        self.clusters[i].centroid,
                        self.clusters[j].centroid
                    )

                    if sim > self.merge_threshold:
                        # Merge j into i (keep larger cluster)
                        larger_idx  = i if self.clusters[i].n_samples >= self.clusters[j].n_samples else j
                        smaller_idx = j if larger_idx == i else i

                        # ── CRASH FIX ─────────────────────────────────────────────────
                        # Capture labels BEFORE calling _merge_clusters().
                        # _merge_clusters() calls self.clusters.pop(remove_idx), which
                        # invalidates all indices >= remove_idx immediately.
                        # Accessing self.clusters[smaller_idx] in the log message AFTER
                        # the pop causes IndexError: list index out of range.
                        keep_label   = self.clusters[larger_idx].label
                        remove_label = self.clusters[smaller_idx].label

                        self._merge_clusters(larger_idx, smaller_idx)
                        logger.info(
                            f"Merged '{remove_label}' into '{keep_label}' (sim={sim:.3f})"
                        )
                        merged = True
                        break
                if merged:
                    break

    def _merge_clusters(self, keep_idx: int, remove_idx: int):
        """
        Merge cluster at remove_idx into cluster at keep_idx — FROM SCRATCH.

        The merged centroid is a weighted average of both centroids,
        weighted by number of samples.
        """
        keep = self.clusters[keep_idx]
        remove = self.clusters[remove_idx]

        total_n = keep.n_samples + remove.n_samples
        merged_centroid = (keep.n_samples * keep.centroid + remove.n_samples * remove.centroid) / total_n
        merged_centroid = self._l2_normalize(merged_centroid)

        keep.centroid = merged_centroid
        keep.n_samples = total_n
        keep.total_duration += remove.total_duration

        # Remove the merged cluster
        self.clusters.pop(remove_idx)

        # Update speaker IDs to keep them consecutive
        for idx, cluster in enumerate(self.clusters):
            cluster.speaker_id = idx

    def prune_inactive_clusters(self):
        """
        Remove clusters that haven't been active for a long time — FROM SCRATCH.

        In long meetings, many speakers may come and go.
        We drop clusters with very few samples after a long inactivity.
        """
        now = time.time()
        to_remove = []

        for idx, cluster in enumerate(self.clusters):
            inactive_sec = now - cluster.last_seen
            if (inactive_sec > self.max_inactive_seconds and
                    cluster.n_samples < self.min_samples_to_keep):
                to_remove.append(idx)
                logger.info(f"Pruning inactive cluster: {cluster.label}")

        # Remove in reverse order to preserve indices
        for idx in sorted(to_remove, reverse=True):
            self.clusters.pop(idx)

        # Reassign IDs
        for idx, cluster in enumerate(self.clusters):
            cluster.speaker_id = idx


    def consolidate_to(self, target_n: int):
        """
        Force-merge clusters down to exactly target_n speakers — FROM SCRATCH.

        Called after processing a full file when you KNOW the real speaker count.
        Repeatedly merges the two most-similar clusters until target_n remain.

        Example: 8 clusters found but --speakers 3 passed → merge down to 3.

        WHY THIS WORKS:
          Spurious extra clusters (e.g. Speaker 5 with 1 segment) are usually
          acoustic variations of an existing speaker. Their centroids will be
          closest to the real speaker they belong to, so merging by similarity
          is correct.
        """
        if target_n <= 0 or len(self.clusters) <= target_n:
            return

        logger.info(f"Consolidating {len(self.clusters)} clusters → {target_n}")

        while len(self.clusters) > target_n:
            n = len(self.clusters)
            best_sim = -1.0
            merge_i, merge_j = 0, 1

            # Find the two most similar clusters
            for i in range(n):
                for j in range(i + 1, n):
                    sim = self._cosine_similarity(
                        self.clusters[i].centroid,
                        self.clusters[j].centroid
                    )
                    if sim > best_sim:
                        best_sim = sim
                        merge_i, merge_j = i, j

            logger.info(
                f"  Merging '{self.clusters[merge_j].label}' into "
                f"'{self.clusters[merge_i].label}' (sim={best_sim:.3f})"
            )
            self._merge_clusters(merge_i, merge_j)

    def drop_noise_clusters(self, min_segments: int = 2):
        """
        Remove clusters with very few segments — FROM SCRATCH.

        Any cluster with fewer than min_segments is almost certainly a
        mis-detection (noise spike, cough) not a real speaker.
        We absorb it into the nearest real cluster.
        """
        changed = True
        while changed:
            changed = False
            for idx, cluster in enumerate(self.clusters):
                if cluster.n_samples < min_segments and len(self.clusters) > 1:
                    # Find nearest other cluster
                    best_sim  = -1.0
                    best_other = -1
                    for j, other in enumerate(self.clusters):
                        if j == idx:
                            continue
                        sim = self._cosine_similarity(cluster.centroid, other.centroid)
                        if sim > best_sim:
                            best_sim  = sim
                            best_other = j

                    if best_other >= 0:
                        keep   = min(idx, best_other)
                        remove = max(idx, best_other)
                        logger.info(
                            f"  Noise cluster '{self.clusters[remove].label}' "
                            f"({self.clusters[remove].n_samples} seg) absorbed into "
                            f"'{self.clusters[keep].label}'"
                        )
                        self._merge_clusters(keep, remove)
                        changed = True
                        break  # restart after mutation

    # ──────────────────────────────────────────────────────────
    # FROM SCRATCH: Utility Methods
    # ──────────────────────────────────────────────────────────

    def _l2_normalize(self, v: np.ndarray) -> np.ndarray:
        """L2 normalization — FROM SCRATCH."""
        norm = np.sqrt(np.sum(v ** 2))
        if norm < 1e-10:
            return v
        return v / norm

    def _ensure_normalized(self, embedding: np.ndarray) -> np.ndarray:
        """Ensure embedding is L2-normalized before use."""
        return self._l2_normalize(embedding.astype(np.float32))

    # ──────────────────────────────────────────────────────────
    # Analytics
    # ──────────────────────────────────────────────────────────

    def get_speaker_stats(self) -> List[dict]:
        """Return statistics for each discovered speaker."""
        stats = []
        for cluster in self.clusters:
            stats.append({
                "speaker_id": cluster.speaker_id,
                "label": cluster.label,
                "color": cluster.color,
                "n_segments": cluster.n_samples,
                "total_duration_sec": round(cluster.total_duration, 1),
                "centroid_norm": round(float(np.linalg.norm(cluster.centroid)), 4),
            })
        return stats

    def get_similarity_matrix(self) -> Optional[np.ndarray]:
        """
        Compute pairwise similarity matrix — FROM SCRATCH.

        Useful for understanding how distinct the speakers are.
        Returns None if fewer than 2 speakers.
        """
        n = len(self.clusters)
        if n < 2:
            return None

        matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                matrix[i, j] = self._cosine_similarity(
                    self.clusters[i].centroid,
                    self.clusters[j].centroid
                )
        return matrix

    def reset(self):
        """Clear all clusters. Call between sessions."""
        self.clusters.clear()
        self._next_speaker_number = 1
        self._assignment_history.clear()
        self._consecutive_misses = 0
        self._miss_embeddings.clear()
        logger.info("Clustering state reset.")

    @property
    def n_speakers(self) -> int:
        return len(self.clusters)


# ──────────────────────────────────────────────────────────────
# Batch Agglomerative Clustering (for post-processing)
# ──────────────────────────────────────────────────────────────

class AgglomerativeClusterer:
    """
    Batch agglomerative clustering — FROM SCRATCH.

    This is for post-hoc analysis when you have ALL embeddings
    and want to cluster them together (unlike the online version above).

    Used when re-processing or improving diarization accuracy.

    ALGORITHM (Ward linkage):
      1. Start: each embedding is its own cluster
      2. Compute distance between all pairs
      3. Merge the two closest clusters
      4. Recompute distances
      5. Repeat until desired number of clusters

    This is O(n^2 log n) — slow for large n, but good quality.
    """

    def __init__(
        self,
        n_clusters: Optional[int] = None,
        distance_threshold: float = 0.3,  # 1 - cosine_similarity
        linkage: str = "average"           # "average", "complete", "single"
    ):
        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold
        self.linkage = linkage

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Cluster a batch of embeddings — FROM SCRATCH.

        Parameters:
        -----------
        embeddings : np.ndarray — shape (n_samples, embedding_dim)

        Returns:
        --------
        np.ndarray — shape (n_samples,), integer cluster labels
        """
        n = len(embeddings)
        if n == 0:
            return np.array([], dtype=int)
        if n == 1:
            return np.array([0])

        # Normalize embeddings
        normed = self._batch_normalize(embeddings)

        # Compute pairwise distance matrix
        # distance = 1 - cosine_similarity
        dist_matrix = self._compute_distance_matrix(normed)

        # Agglomerative clustering loop
        labels = self._agglomerative_cluster(dist_matrix, n)

        return labels

    def _batch_normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize each row — FROM SCRATCH."""
        norms = np.sqrt(np.sum(embeddings ** 2, axis=1, keepdims=True))
        norms = np.maximum(norms, 1e-10)
        return embeddings / norms

    def _compute_distance_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute pairwise cosine distances — FROM SCRATCH.

        dist(a, b) = 1 - cosine_similarity(a, b)

        For normalized vectors: cosine_similarity = a · b^T
        """
        n = len(embeddings)
        dot_matrix = np.dot(embeddings, embeddings.T)  # n×n cosine similarity
        dot_matrix = np.clip(dot_matrix, -1.0, 1.0)
        dist_matrix = 1.0 - dot_matrix  # Convert to distance

        # Distance to self should be 0
        np.fill_diagonal(dist_matrix, 0.0)

        return dist_matrix.astype(np.float32)

    def _agglomerative_cluster(
        self,
        dist_matrix: np.ndarray,
        n: int
    ) -> np.ndarray:
        """
        Agglomerative clustering loop — FROM SCRATCH.

        Uses the "complete linkage" criterion:
          Distance between two clusters = max pairwise distance between members

        This tends to create compact, well-separated clusters.
        """
        # Each sample starts in its own cluster
        labels = list(range(n))
        current_dist = dist_matrix.copy()

        # Set diagonal to inf (don't merge with self)
        np.fill_diagonal(current_dist, np.inf)

        # Track which original samples belong to each active cluster
        cluster_members = {i: [i] for i in range(n)}
        active_clusters = list(range(n))
        next_cluster_id = n

        target_k = self.n_clusters if self.n_clusters is not None else 1

        while len(active_clusters) > max(target_k, 1):
            # Find the two closest clusters
            # Only look at active clusters
            min_dist = np.inf
            merge_i, merge_j = -1, -1

            for ii in range(len(active_clusters)):
                for jj in range(ii + 1, len(active_clusters)):
                    ci = active_clusters[ii]
                    cj = active_clusters[jj]
                    d = self._cluster_distance(ci, cj, cluster_members, dist_matrix)
                    if d < min_dist:
                        min_dist = d
                        merge_i, merge_j = ci, cj

            # Stop if distance exceeds threshold
            if self.n_clusters is None and min_dist >= self.distance_threshold:
                break

            if merge_i == -1:
                break

            # Merge merge_j into merge_i
            new_members = cluster_members[merge_i] + cluster_members[merge_j]
            cluster_members[next_cluster_id] = new_members

            # Update labels for all members of merged clusters
            for m in cluster_members[merge_j]:
                labels[m] = next_cluster_id
            for m in cluster_members[merge_i]:
                labels[m] = next_cluster_id

            active_clusters.remove(merge_i)
            active_clusters.remove(merge_j)
            active_clusters.append(next_cluster_id)
            next_cluster_id += 1

        # Relabel to consecutive integers 0, 1, 2, ...
        unique_labels = list(dict.fromkeys(labels))  # preserve order, unique
        label_map = {old: new for new, old in enumerate(unique_labels)}
        return np.array([label_map[l] for l in labels])

    def _cluster_distance(
        self,
        c1: int,
        c2: int,
        cluster_members: dict,
        dist_matrix: np.ndarray
    ) -> float:
        """
        Compute distance between two clusters — FROM SCRATCH.

        Supports "average", "complete", "single" linkage:
          average  = mean of all pairwise distances
          complete = max of all pairwise distances
          single   = min of all pairwise distances
        """
        members_1 = cluster_members.get(c1, [c1])
        members_2 = cluster_members.get(c2, [c2])

        distances = []
        for m1 in members_1:
            for m2 in members_2:
                if m1 < dist_matrix.shape[0] and m2 < dist_matrix.shape[1]:
                    distances.append(dist_matrix[m1, m2])

        if not distances:
            return np.inf

        if self.linkage == "average":
            return float(np.mean(distances))
        elif self.linkage == "complete":
            return float(np.max(distances))
        elif self.linkage == "single":
            return float(np.min(distances))
        else:
            return float(np.mean(distances))


# ──────────────────────────────────────────────────────────────
# Standalone Test
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Clustering Self-Test ===\n")

    np.random.seed(42)

    # Simulate 3 speakers with different embedding "voices"
    dim = 192

    # Speaker 1: centered around direction A
    center_1 = np.random.randn(dim).astype(np.float32)
    center_1 /= np.linalg.norm(center_1)

    # Speaker 2: centered around direction B (orthogonal to A)
    center_2 = np.random.randn(dim).astype(np.float32)
    center_2 -= np.dot(center_2, center_1) * center_1  # Remove component along center_1
    center_2 /= np.linalg.norm(center_2)

    # Speaker 3: centered around direction C
    center_3 = np.random.randn(dim).astype(np.float32)
    center_3 /= np.linalg.norm(center_3)

    def make_embedding(center, noise=0.15):
        """Generate a noisy embedding around a center direction."""
        e = center + np.random.randn(dim).astype(np.float32) * noise
        e /= np.linalg.norm(e)
        return e

    print("=== Online Clusterer Test ===")
    clusterer = OnlineSpeakerClusterer(similarity_threshold=0.75)

    # Simulate alternating speakers
    schedule = [
        (center_1, "expected: Speaker 1"),
        (center_2, "expected: Speaker 2"),
        (center_1, "expected: Speaker 1"),
        (center_3, "expected: Speaker 3"),
        (center_2, "expected: Speaker 2"),
        (center_1, "expected: Speaker 1"),
    ]

    for i, (center, expected) in enumerate(schedule):
        emb = make_embedding(center)
        speaker_id, label, confidence = clusterer.assign_speaker(emb, duration=1.5, timestamp=float(i))
        print(f"  Chunk {i+1}: {label} (confidence={confidence:.3f}) — {expected}")

    print(f"\n  Total speakers discovered: {clusterer.n_speakers}")
    print(f"\n  Speaker stats:")
    for stat in clusterer.get_speaker_stats():
        print(f"    {stat['label']}: {stat['n_segments']} segments, {stat['total_duration_sec']}s")

    sim_matrix = clusterer.get_similarity_matrix()
    if sim_matrix is not None:
        print(f"\n  Similarity matrix between clusters:")
        print(f"  (diagonal=1.0 always, off-diagonal should be low for distinct speakers)")
        print(np.round(sim_matrix, 3))

    print("\n=== Batch Agglomerative Test ===")
    # Create 9 embeddings: 3 groups of 3
    embeddings = np.array(
        [make_embedding(center_1) for _ in range(3)] +
        [make_embedding(center_2) for _ in range(3)] +
        [make_embedding(center_3) for _ in range(3)]
    )

    batch_clusterer = AgglomerativeClusterer(n_clusters=3)
    batch_labels = batch_clusterer.fit_predict(embeddings)
    print(f"  Input: 9 embeddings (3 speakers × 3 chunks)")
    print(f"  Predicted labels: {batch_labels}")
    print(f"  (Should show 3 groups of 3 with same labels)")

    print("\nSelf-test complete.")