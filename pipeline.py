"""
pipeline.py — Full Speech Processing Pipeline
===============================================

This is the ORCHESTRATOR. It ties together all modules:
  audio_stream → vad → diarization → clustering → whisper_asr → output

PIPELINE FLOWCHART:
  ┌─────────────────────────────────────────────────────────┐
  │  audio chunk (numpy float32, 16kHz)                     │
  └────────────────────────┬────────────────────────────────┘
                           ▼
  ┌─────────────────────────────────────────────────────────┐
  │  VAD (CombinedVAD)                                      │
  │  Is there speech in this chunk? (yes/no)                │
  └────────────────────────┬────────────────────────────────┘
                    (yes) ▼  (no) → skip
  ┌─────────────────────────────────────────────────────────┐
  │  Speaker Embedding (ECAPA-TDNN or fallback)             │
  │  Extract 192-dim voice fingerprint                      │
  └────────────────────────┬────────────────────────────────┘
                           ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Online Clustering (cosine similarity)                  │
  │  Assign "Speaker 1", "Speaker 2", etc.                  │
  └────────────────────────┬────────────────────────────────┘
                           ▼
  ┌─────────────────────────────────────────────────────────┐
  │  ASR (Faster-Whisper)                                   │
  │  Transcribe speech to text + word timestamps            │
  └────────────────────────┬────────────────────────────────┘
                           ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Merge diarization + transcription                      │
  │  → {timestamp, speaker, text}                           │
  └────────────────────────┬────────────────────────────────┘
                           ▼
  ┌─────────────────────────────────────────────────────────┐
  │  WebSocket → Frontend                                   │
  └─────────────────────────────────────────────────────────┘

STUDENT NOTE:
  This file also shows the "from scratch" subtitle merging algorithm.
  The subtitle buffer collects results and merges consecutive segments
  from the same speaker into one subtitle line.
"""

import numpy as np
import logging
import asyncio
import time
from collections import deque
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

# Import our modules
from vad import CombinedVAD
from diarization import SpeakerEmbeddingExtractor
from clustering import OnlineSpeakerClusterer, AgglomerativeClusterer
from whisper_asr import WhisperASR, TranscriptionResult

logger = logging.getLogger("pipeline")


# ──────────────────────────────────────────────────────────────
# Output Format
# ──────────────────────────────────────────────────────────────

@dataclass
class SubtitleResult:
    """A single subtitle line to display on frontend."""
    timestamp: str       # "MM:SS" format
    speaker: str         # "Speaker 1", "Speaker 2", etc.
    speaker_id: int      # Numeric ID (for color coding)
    text: str            # Transcribed text
    confidence: float    # ASR confidence
    is_partial: bool = False  # True if still being refined


# ──────────────────────────────────────────────────────────────
# Subtitle Buffer — FROM SCRATCH
# ──────────────────────────────────────────────────────────────

class SubtitleBuffer:
    """
    Merges consecutive speech segments from the same speaker — FROM SCRATCH.

    PROBLEM:
      Each chunk (1.5s) produces a separate result.
      If Speaker 1 speaks for 30 seconds, we'd get 20 separate results.
      That's ugly. We want to merge them into one running line.

    ALGORITHM:
      1. Keep a "current" subtitle for each speaker
      2. If new text arrives from same speaker → append to current subtitle
      3. If new text from different speaker → emit the old one, start new
      4. If speaker resumes after a pause → start fresh subtitle

    MERGE CONDITION:
      - Same speaker as last result
      - Within merge_window_sec of last result

    This is similar to how live captions work on YouTube/Zoom.
    """

    def __init__(
        self,
        merge_window_sec: float = 3.0,  # Merge same-speaker if within 3s
        max_chars_per_subtitle: int = 150  # Force new line if too long
    ):
        self.merge_window_sec = merge_window_sec
        self.max_chars_per_subtitle = max_chars_per_subtitle

        # Current "open" subtitle per speaker
        # key: speaker_id → {text, timestamp, last_time}
        self._open_subtitles: Dict[int, dict] = {}

        # Finalized subtitles (ready to send)
        self._finalized: List[SubtitleResult] = []

    def add(self, result: SubtitleResult) -> List[SubtitleResult]:
        """
        Add a new subtitle result and return any finalized subtitles — FROM SCRATCH.

        When a different speaker starts talking, all open subtitles from
        OTHER speakers are flushed immediately.  This ensures real-time
        speaker-change visibility instead of holding subtitles for seconds.

        Returns:
        --------
        List[SubtitleResult] — Subtitles ready to display
        """
        now = time.time()
        sid = result.speaker_id
        finalized = []

        # When a new (different) speaker speaks, flush all OTHER open subtitles immediately
        for other_sid in list(self._open_subtitles.keys()):
            if other_sid != sid:
                f = self._finalize_subtitle(other_sid)
                if f:
                    finalized.append(f)

        # Check if we have an open subtitle for this speaker
        if sid in self._open_subtitles:
            existing = self._open_subtitles[sid]
            time_gap = now - existing["last_time"]

            should_merge = (
                time_gap <= self.merge_window_sec and
                len(existing["text"]) + len(result.text) < self.max_chars_per_subtitle
            )

            if should_merge:
                # Append to existing subtitle
                existing["text"] = existing["text"] + " " + result.text
                existing["last_time"] = now
                return finalized  # Return any flushed other-speaker subtitles
            else:
                # Finalize the existing subtitle and start fresh
                f = self._finalize_subtitle(sid)
                if f:
                    finalized.append(f)
                self._start_new_subtitle(result, now)
                return finalized
        else:
            # New speaker or first entry
            self._start_new_subtitle(result, now)
            return finalized

    def flush_all(self) -> List[SubtitleResult]:
        """
        Finalize all open subtitles — call when recording stops — FROM SCRATCH.
        """
        results = []
        for sid in list(self._open_subtitles.keys()):
            finalized = self._finalize_subtitle(sid)
            if finalized:
                results.append(finalized)
        return results

    def _start_new_subtitle(self, result: SubtitleResult, now: float):
        """Open a new subtitle entry for this speaker."""
        self._open_subtitles[result.speaker_id] = {
            "text": result.text,
            "timestamp": result.timestamp,
            "speaker": result.speaker,
            "speaker_id": result.speaker_id,
            "confidence": result.confidence,
            "last_time": now
        }

    def _finalize_subtitle(self, speaker_id: int) -> Optional[SubtitleResult]:
        """Close and return the open subtitle for a speaker."""
        if speaker_id not in self._open_subtitles:
            return None
        existing = self._open_subtitles.pop(speaker_id)
        return SubtitleResult(
            timestamp=existing["timestamp"],
            speaker=existing["speaker"],
            speaker_id=existing["speaker_id"],
            text=existing["text"].strip(),
            confidence=existing["confidence"]
        )

    def _check_stale_subtitles(self, now: float) -> List[SubtitleResult]:
        """Finalize subtitles that haven't been updated for a while."""
        stale_threshold = self.merge_window_sec * 2
        results = []
        stale_sids = [
            sid for sid, data in self._open_subtitles.items()
            if now - data["last_time"] > stale_threshold
        ]
        for sid in stale_sids:
            finalized = self._finalize_subtitle(sid)
            if finalized:
                results.append(finalized)
        return results


# ══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════

class SpeechPipeline:
    """
    Full end-to-end speech processing pipeline.

    Orchestrates:
      VAD → Embedding → Clustering → ASR → Subtitle Merging

    This is instantiated once per WebSocket session (see main.py).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        whisper_model: Optional[str] = "medium",
        fallback_whisper_model: Optional[str] = None,
        vad_threshold: float = 0.45,
        speaker_threshold: float = 0.40,  # Matches OnlineSpeakerClusterer live default
        max_speakers: int = 3
    ):
        self.sample_rate = sample_rate
        self._initialized = False

        auto_whisper_model, auto_fallback_whisper_model, asr_device, asr_compute_type, fallback_compute_type = self._resolve_asr_runtime()
        whisper_model = whisper_model or auto_whisper_model
        fallback_whisper_model = fallback_whisper_model or auto_fallback_whisper_model

        # Component instances
        self.vad = CombinedVAD(sample_rate=sample_rate, threshold=vad_threshold)
        self.embedder = SpeakerEmbeddingExtractor(sample_rate=sample_rate)
        self.clusterer = OnlineSpeakerClusterer(
            similarity_threshold=speaker_threshold,
            max_speakers=max_speakers,
            merge_threshold=0.75,
            new_speaker_patience=4
        )
        self.asr = WhisperASR(
            model_size=whisper_model,
            device=asr_device,
            compute_type=asr_compute_type,
            language="en"
        )
        self.asr_fallback = None
        self._fallback_loaded = False
        self.subtitle_buffer = SubtitleBuffer(merge_window_sec=3.0)

        # Session state
        self._session_start: float = time.time()
        self._chunk_count: int = 0
        self._speech_chunk_count: int = 0
        self._segment_counter: int = 0

        # Runtime pressure signals from websocket queue.
        self._runtime_pressure = {
            "queue_depth": 0,
            "queue_max": 1,
            "queue_pressure": 0.0,
            "dropped_chunks": 0,
            "chunk_seconds": 1.5,
            "overloaded": False,
        }

        # Turn-level hysteresis and smoothing.
        self._active_speaker_id: Optional[int] = None
        self._active_speaker_since: float = 0.0
        self._candidate_speaker_id: Optional[int] = None
        self._candidate_speaker_since: float = 0.0
        self._candidate_hits: int = 0
        self.min_turn_duration_sec = 1.0
        self.new_speaker_hold_sec = 0.9
        self.reassign_smoothing_hits = 2

        # Rolling delayed refinement window (mini-batch reclustering).
        self._recent_segments = deque(maxlen=64)
        self._refine_window_sec = 15.0
        self._refine_interval_sec = 10.0
        self._last_refine_sec = 0.0

        # Observability.
        self._last_embedding: Optional[np.ndarray] = None
        self._last_asr_mode: str = "primary"
        self._stage_latency_ms = {
            "vad": 0.0,
            "embedding": 0.0,
            "assignment": 0.0,
            "refinement": 0.0,
            "asr": 0.0,
            "merge": 0.0,
            "total": 0.0,
        }
        self._stage_latency_n = 0
        self._embedding_drift_ema: Optional[float] = None
        self._boundary_score_ema: Optional[float] = None
        self._revision_count = 0

        logger.info(
            "SpeechPipeline created. ASR primary=%s fallback=%s device=%s",
            whisper_model,
            fallback_whisper_model or "disabled",
            asr_device,
        )

    def _resolve_asr_runtime(self):
        """Force live ASR to medium on CUDA only."""
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Live transcription is configured for Whisper medium on CUDA only. "
                "Install a CUDA-enabled torch build and run on the NVIDIA GPU."
            )

        return "medium", None, "cuda", "float16", None

    async def initialize(self):
        """
        Load all ML models.
        Called once when the WebSocket connects.
        """
        if self._initialized:
            return

        logger.info("Initializing pipeline models...")

        # Load VAD
        self.vad.initialize()

        # Load speaker embedding model
        self.embedder.load()

        # Load Whisper ASR
        self.asr.load()

        # Live mode is locked to a single ASR model.
        self._fallback_loaded = False

        self._initialized = True
        self._session_start = time.time()
        logger.info("Pipeline fully initialized and ready.")

    # ──────────────────────────────────────────────────────────
    # Main Processing Entry Point
    # ──────────────────────────────────────────────────────────

    def process_chunk(
        self,
        audio: np.ndarray,
        timestamp: str,
        meta: Optional[Dict[str, Any]] = None
    ) -> List[dict]:
        """
        Process a single audio chunk through the full pipeline.

        This is called synchronously from an executor thread in main.py.

        Parameters:
        -----------
        audio     : np.ndarray — float32, (N,), 16kHz
        timestamp : str        — "MM:SS" formatted session timestamp

        Returns:
        --------
        List[dict] — Zero or more subtitle JSONs to send to frontend
            {
              "timestamp": "00:03",
              "speaker": "Speaker 1",
              "speaker_id": 0,
              "text": "Hello everyone",
              "confidence": 0.89
            }
        """
        self._chunk_count += 1
        chunk_start_time = time.time()
        meta = meta or {}
        self.set_runtime_pressure(
            queue_depth=int(meta.get("queue_depth", self._runtime_pressure["queue_depth"])),
            queue_max=int(meta.get("queue_max", self._runtime_pressure["queue_max"])),
            chunk_seconds=float(meta.get("chunk_seconds", self._runtime_pressure["chunk_seconds"])),
            dropped_chunks=int(meta.get("dropped_chunks", self._runtime_pressure["dropped_chunks"])),
        )

        stage = {}

        # Stage 1: VAD / segmentation gate.
        t0 = time.time()
        is_speech, vad_conf = self.vad.detect(audio)
        stage["vad"] = (time.time() - t0) * 1000

        if not is_speech:
            stage["total"] = (time.time() - chunk_start_time) * 1000
            self._update_stage_latency(stage)
            return self._maybe_metrics_event(vad_conf=vad_conf)

        self._speech_chunk_count += 1
        elapsed_sec = self._elapsed_seconds()
        duration_sec = len(audio) / self.sample_rate

        # Stage 2: Speaker embedding extraction.
        t0 = time.time()
        embedding_result = self.embedder.extract(audio)
        stage["embedding"] = (time.time() - t0) * 1000

        embedding = embedding_result.vector if embedding_result is not None else None
        embedding_drift = None
        if embedding is not None and self._last_embedding is not None:
            embedding_drift = float(np.dot(embedding, self._last_embedding))
            if self._embedding_drift_ema is None:
                self._embedding_drift_ema = embedding_drift
            else:
                self._embedding_drift_ema = 0.9 * self._embedding_drift_ema + 0.1 * embedding_drift
        if embedding is not None:
            self._last_embedding = embedding.copy()

        # Stage 3: Online assignment (low-latency provisional labels).
        t0 = time.time()
        speaker_id = 0
        speaker_label = "Speaker 1"
        speaker_conf = 1.0

        if embedding is not None:
            proposed_id, proposed_label, speaker_conf = self.clusterer.assign_speaker(
                embedding=embedding,
                duration=duration_sec,
                timestamp=elapsed_sec,
            )
            boundary_score = self._compute_boundary_score(vad_conf, embedding_drift)
            speaker_id = self._apply_turn_hysteresis(proposed_id, elapsed_sec, boundary_score)
            speaker_label = self._speaker_label(speaker_id)

            try:
                n_samp = (
                    self.clusterer.clusters[speaker_id].n_samples
                    if speaker_id < len(self.clusterer.clusters)
                    else 1
                )
                self.embedder.profile_store.update(
                    label=speaker_label,
                    embedding=embedding,
                    n_samples=n_samp,
                )
            except Exception as pe:
                logger.debug("Profile store update skipped: %s", pe)

            if self._speech_chunk_count % 10 == 0:
                self.clusterer.log_diagnostic_matrix()
            if self._speech_chunk_count % 15 == 0 and self._speech_chunk_count >= 20:
                self.clusterer.merge_similar_clusters()
        elif self._active_speaker_id is not None:
            speaker_id = self._active_speaker_id
            speaker_label = self._speaker_label(speaker_id)

        stage["assignment"] = (time.time() - t0) * 1000

        segment_record = {
            "timestamp": timestamp,
            "elapsed_sec": elapsed_sec,
            "embedding": embedding,
            "speaker_id": speaker_id,
            "speaker": speaker_label,
            "segment_id": None,
            "emitted": False,
        }
        self._recent_segments.append(segment_record)

        # Stage 4: Delayed refinement (rolling mini-batch reclustering).
        t0 = time.time()
        revision_events = self._run_delayed_refinement(elapsed_sec)
        stage["refinement"] = (time.time() - t0) * 1000

        # Stage 5: ASR routing under pressure.
        asr_mode, asr_engine, asr_profile = self._select_asr_strategy(vad_conf)
        self._last_asr_mode = asr_mode

        t0 = time.time()
        asr_result = None
        if asr_mode != "skip":
            asr_result = asr_engine.transcribe(
                audio,
                sample_rate=self.sample_rate,
                profile=asr_profile,
            )
        stage["asr"] = (time.time() - t0) * 1000

        # Stage 6: ASR + speaker merge into subtitle events.
        t0 = time.time()
        results = list(revision_events)

        if asr_result is not None and asr_result.text.strip():
            self._segment_counter += 1
            segment_id = self._segment_counter
            segment_record["segment_id"] = segment_id
            segment_record["emitted"] = True

            subtitle = SubtitleResult(
                timestamp=timestamp,
                speaker=speaker_label,
                speaker_id=speaker_id,
                text=asr_result.text,
                confidence=asr_result.confidence,
            )

            finalized_subtitles = self.subtitle_buffer.add(subtitle)
            if finalized_subtitles:
                for sub in finalized_subtitles:
                    d = self._subtitle_to_dict(sub)
                    d["segment_id"] = segment_id
                    results.append(d)
            else:
                d = self._subtitle_to_dict(subtitle)
                d["is_partial"] = True
                d["segment_id"] = segment_id
                results.append(d)

        stage["merge"] = (time.time() - t0) * 1000
        stage["total"] = (time.time() - chunk_start_time) * 1000
        self._update_stage_latency(stage)

        results.extend(self._maybe_metrics_event(vad_conf=vad_conf, embedding_drift=embedding_drift))

        logger.info(
            "Chunk %d processed | q=%d/%d pressure=%.2f | asr=%s | total=%.0fms",
            self._chunk_count,
            self._runtime_pressure["queue_depth"],
            self._runtime_pressure["queue_max"],
            self._runtime_pressure["queue_pressure"],
            asr_mode,
            stage["total"],
        )

        return results

    def process_flush(self) -> List[dict]:
        """
        Flush remaining buffered subtitles.
        Call this when recording stops.
        """
        final_subs = self.subtitle_buffer.flush_all()
        out = [self._subtitle_to_dict(s) for s in final_subs]
        out.extend(self._maybe_metrics_event(force=True))
        return out

    def set_runtime_pressure(
        self,
        queue_depth: int,
        queue_max: int,
        chunk_seconds: float,
        dropped_chunks: int = 0,
    ):
        queue_max = max(1, int(queue_max))
        queue_depth = max(0, int(queue_depth))
        pressure = min(1.0, queue_depth / queue_max)
        self._runtime_pressure = {
            "queue_depth": queue_depth,
            "queue_max": queue_max,
            "queue_pressure": pressure,
            "dropped_chunks": max(0, int(dropped_chunks)),
            "chunk_seconds": float(chunk_seconds),
            "overloaded": pressure >= 0.75,
        }

    def _select_asr_strategy(self, vad_conf: float):
        """
        Queue-pressure aware ASR policy.

        Order of protection under overload:
          1) skip ASR for low-confidence speech chunks,
          2) run fast decode profile on the primary model.
        """
        pressure = self._runtime_pressure["queue_pressure"]
        if pressure >= 0.80:
            if vad_conf < 0.62:
                return "skip", self.asr, "fast"
            return "primary_fast", self.asr, "fast"
        if pressure >= 0.60:
            return "primary_fast", self.asr, "fast"
        return "primary", self.asr, "quality"

    def _compute_boundary_score(self, vad_conf: float, embedding_drift: Optional[float]) -> float:
        """
        Pyannote-style boundary evidence score.

        Uses embedding similarity drift + VAD confidence as a proxy for
        speaker-change likelihood when true SCD models are unavailable.
        """
        if embedding_drift is None:
            score = max(0.0, min(1.0, 0.35 * vad_conf))
        else:
            sim = float(np.clip(embedding_drift, -1.0, 1.0))
            change_score = (1.0 - sim) * 0.5
            score = max(0.0, min(1.0, 0.7 * change_score + 0.3 * vad_conf))

        if self._boundary_score_ema is None:
            self._boundary_score_ema = score
        else:
            self._boundary_score_ema = 0.9 * self._boundary_score_ema + 0.1 * score
        return score

    def _apply_turn_hysteresis(
        self,
        proposed_speaker_id: int,
        elapsed_sec: float,
        boundary_score: float,
    ) -> int:
        """
        Stabilize speaker turns with minimum turn duration + hold time.
        """
        if self._active_speaker_id is None:
            self._active_speaker_id = proposed_speaker_id
            self._active_speaker_since = elapsed_sec
            return proposed_speaker_id

        if proposed_speaker_id == self._active_speaker_id:
            self._candidate_speaker_id = None
            self._candidate_hits = 0
            return self._active_speaker_id

        # Do not allow rapid ping-pong speaker switches.
        if elapsed_sec - self._active_speaker_since < self.min_turn_duration_sec:
            return self._active_speaker_id

        if self._candidate_speaker_id != proposed_speaker_id:
            self._candidate_speaker_id = proposed_speaker_id
            self._candidate_speaker_since = elapsed_sec
            self._candidate_hits = 1
            return self._active_speaker_id

        self._candidate_hits += 1
        hold_elapsed = elapsed_sec - self._candidate_speaker_since
        strong_boundary = boundary_score >= 0.45
        if strong_boundary and (
            hold_elapsed >= self.new_speaker_hold_sec or
            self._candidate_hits >= self.reassign_smoothing_hits
        ):
            self._active_speaker_id = proposed_speaker_id
            self._active_speaker_since = elapsed_sec
            self._candidate_speaker_id = None
            self._candidate_hits = 0

        return self._active_speaker_id

    def _speaker_label(self, speaker_id: int) -> str:
        if 0 <= speaker_id < len(self.clusterer.clusters):
            return self.clusterer.clusters[speaker_id].label
        return f"Speaker {speaker_id + 1}"

    def _speaker_color(self, speaker_id: int) -> str:
        if 0 <= speaker_id < len(self.clusterer.clusters):
            return self.clusterer.clusters[speaker_id].color
        return "#4FC3F7"

    def _run_delayed_refinement(self, elapsed_sec: float) -> List[dict]:
        """
        Recluster recent embeddings in a short rolling window and emit
        revision events when labels should be corrected.
        """
        if elapsed_sec - self._last_refine_sec < self._refine_interval_sec:
            return []
        self._last_refine_sec = elapsed_sec

        window = [
            s for s in self._recent_segments
            if s["embedding"] is not None and (elapsed_sec - s["elapsed_sec"]) <= self._refine_window_sec
        ]
        if len(window) < 4:
            return []

        embeddings = np.array([w["embedding"] for w in window], dtype=np.float32)
        recluster = AgglomerativeClusterer(n_clusters=None, distance_threshold=0.55, linkage="average")
        labels = recluster.fit_predict(embeddings)

        # Map local clusters to dominant online speaker IDs.
        label_to_target = {}
        unique_labels = sorted(set(int(l) for l in labels))
        for lab in unique_labels:
            ids = [window[i]["speaker_id"] for i, x in enumerate(labels) if int(x) == lab]
            counts = {}
            for sid in ids:
                counts[sid] = counts.get(sid, 0) + 1
            label_to_target[lab] = max(counts.items(), key=lambda kv: kv[1])[0]

        revisions = []
        for i, lab in enumerate(labels):
            target = label_to_target[int(lab)]
            rec = window[i]
            old_sid = rec["speaker_id"]
            if old_sid == target:
                continue

            rec["speaker_id"] = target
            rec["speaker"] = self._speaker_label(target)
            self._revision_count += 1

            if rec.get("emitted") and rec.get("segment_id") is not None:
                revisions.append({
                    "type": "revision",
                    "segment_id": rec["segment_id"],
                    "timestamp": rec["timestamp"],
                    "old_speaker_id": old_sid,
                    "new_speaker_id": target,
                    "new_speaker": rec["speaker"],
                    "color": self._speaker_color(target),
                    "reason": "rolling_recluster",
                })

        return revisions

    def _update_stage_latency(self, stage_ms: Dict[str, float]):
        self._stage_latency_n += 1
        alpha = 0.2
        for k, v in stage_ms.items():
            prev = self._stage_latency_ms.get(k, 0.0)
            self._stage_latency_ms[k] = v if self._stage_latency_n == 1 else (1 - alpha) * prev + alpha * v

    def _maybe_metrics_event(
        self,
        vad_conf: float = 0.0,
        embedding_drift: Optional[float] = None,
        force: bool = False,
    ) -> List[dict]:
        if not force and (self._chunk_count % 4 != 0):
            return []
        return [{
            "type": "metrics",
            "timestamp": f"{self._elapsed_seconds():.1f}s",
            "queue_depth": self._runtime_pressure["queue_depth"],
            "queue_max": self._runtime_pressure["queue_max"],
            "queue_pressure": round(self._runtime_pressure["queue_pressure"], 3),
            "dropped_chunks": self._runtime_pressure["dropped_chunks"],
            "chunk_seconds": self._runtime_pressure["chunk_seconds"],
            "asr_mode": self._last_asr_mode,
            "vad_conf": round(float(vad_conf), 3),
            "embedding_drift": (
                round(float(embedding_drift), 4)
                if embedding_drift is not None
                else (round(float(self._embedding_drift_ema), 4) if self._embedding_drift_ema is not None else None)
            ),
            "boundary_score": (
                round(float(self._boundary_score_ema), 4)
                if self._boundary_score_ema is not None
                else None
            ),
            "stage_latency_ms": {k: round(float(v), 1) for k, v in self._stage_latency_ms.items()},
            "revisions": self._revision_count,
        }]

    # ──────────────────────────────────────────────────────────
    # FROM SCRATCH: Subtitle Merging & Formatting
    # ──────────────────────────────────────────────────────────

    def _subtitle_to_dict(self, subtitle: SubtitleResult) -> dict:
        """
        Convert SubtitleResult to JSON-serializable dict — FROM SCRATCH.

        Also looks up the speaker's color from the clustering module.
        """
        color = self._speaker_color(subtitle.speaker_id)

        return {
            "timestamp": subtitle.timestamp,
            "speaker": subtitle.speaker,
            "speaker_id": subtitle.speaker_id,
            "text": subtitle.text,
            "confidence": round(subtitle.confidence, 3),
            "color": color,
            "is_partial": subtitle.is_partial
        }

    def _elapsed_seconds(self) -> float:
        """Seconds since this pipeline session started."""
        return time.time() - self._session_start

    # ──────────────────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────────────────

    def get_session_stats(self) -> dict:
        """Return full session statistics."""
        return {
            "elapsed_sec": round(self._elapsed_seconds(), 1),
            "chunks_processed": self._chunk_count,
            "speech_chunks": self._speech_chunk_count,
            "silence_ratio": round(
                1.0 - self._speech_chunk_count / max(1, self._chunk_count), 2
            ),
            "n_speakers_found": self.clusterer.n_speakers,
            "speakers": self.clusterer.get_speaker_stats(),
            "asr": self.asr.get_stats(),
            "asr_fallback_loaded": self._fallback_loaded,
            "asr_mode": self._last_asr_mode,
            "runtime": self._runtime_pressure,
            "stage_latency_ms": {k: round(v, 1) for k, v in self._stage_latency_ms.items()},
            "embedding_drift_ema": (
                round(float(self._embedding_drift_ema), 4)
                if self._embedding_drift_ema is not None
                else None
            ),
            "boundary_score_ema": (
                round(float(self._boundary_score_ema), 4)
                if self._boundary_score_ema is not None
                else None
            ),
            "revisions": self._revision_count,
        }

    def reset(self):
        """Reset pipeline state for a new session."""
        self.clusterer.reset()
        self.asr.reset()
        self.subtitle_buffer = SubtitleBuffer(merge_window_sec=3.0)
        self._session_start = time.time()
        self._chunk_count = 0
        self._speech_chunk_count = 0
        self._segment_counter = 0
        self._active_speaker_id = None
        self._active_speaker_since = 0.0
        self._candidate_speaker_id = None
        self._candidate_speaker_since = 0.0
        self._candidate_hits = 0
        self._recent_segments.clear()
        self._last_refine_sec = 0.0
        self._last_embedding = None
        self._embedding_drift_ema = None
        self._boundary_score_ema = None
        self._last_asr_mode = "primary"
        self._revision_count = 0
        self._runtime_pressure = {
            "queue_depth": 0,
            "queue_max": 1,
            "queue_pressure": 0.0,
            "dropped_chunks": 0,
            "chunk_seconds": 1.5,
            "overloaded": False,
        }
        for k in self._stage_latency_ms:
            self._stage_latency_ms[k] = 0.0
        self._stage_latency_n = 0
        logger.info("Pipeline reset for new session.")


# ──────────────────────────────────────────────────────────────
# Standalone Test
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio

    print("=== Pipeline Self-Test ===\n")

    async def test():
        pipeline = SpeechPipeline(whisper_model="base")
        await pipeline.initialize()

        sr = 16000
        t = np.linspace(0, 1.5, int(sr * 1.5))

        # Simulate 2 speakers alternating
        for i in range(4):
            # Alternate between two "voices" (different pitches)
            freq = 150 if i % 2 == 0 else 200
            speaker_name = "Speaker A" if i % 2 == 0 else "Speaker B"

            audio = (
                0.4 * np.sin(2 * np.pi * freq * t) +
                0.1 * np.random.randn(len(t))
            ).astype(np.float32)

            timestamp = f"00:0{i*2}"
            results = pipeline.process_chunk(audio, timestamp)
            print(f"Chunk {i+1} ({speaker_name}): {len(results)} result(s)")
            for r in results:
                print(f"  → [{r['timestamp']}] {r['speaker']}: {r['text']}")

        # Flush
        final = pipeline.process_flush()
        print(f"\nFlushed: {len(final)} subtitle(s)")
        for r in final:
            print(f"  → [{r['timestamp']}] {r['speaker']}: {r['text']}")

        # Stats
        print("\nSession stats:")
        stats = pipeline.get_session_stats()
        for k, v in stats.items():
            if k not in ("speakers", "asr"):
                print(f"  {k}: {v}")

    asyncio.run(test())
    print("\nSelf-test complete.")