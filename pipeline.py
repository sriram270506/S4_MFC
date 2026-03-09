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
from typing import List, Optional, Dict
from dataclasses import dataclass, field

# Import our modules
from vad import CombinedVAD
from diarization import SpeakerEmbeddingExtractor
from clustering import OnlineSpeakerClusterer
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
        whisper_model: str = "base",
        vad_threshold: float = 0.45,
        speaker_threshold: float = 0.45,  # Matches OnlineSpeakerClusterer default
        max_speakers: int = 8
    ):
        self.sample_rate = sample_rate
        self._initialized = False

        # Component instances
        self.vad = CombinedVAD(sample_rate=sample_rate, threshold=vad_threshold)
        self.embedder = SpeakerEmbeddingExtractor(sample_rate=sample_rate)
        self.clusterer = OnlineSpeakerClusterer(
            similarity_threshold=speaker_threshold,
            max_speakers=max_speakers,
            merge_threshold=0.75,
            new_speaker_patience=10
        )
        self.asr = WhisperASR(
            model_size=whisper_model,
            device="cpu",
            compute_type="int8",
            language="en"
        )
        self.subtitle_buffer = SubtitleBuffer(merge_window_sec=3.0)

        # Session state
        self._session_start: float = time.time()
        self._chunk_count: int = 0
        self._speech_chunk_count: int = 0

        logger.info("SpeechPipeline created.")

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

        self._initialized = True
        self._session_start = time.time()
        logger.info("Pipeline fully initialized and ready.")

    # ──────────────────────────────────────────────────────────
    # Main Processing Entry Point
    # ──────────────────────────────────────────────────────────

    def process_chunk(
        self,
        audio: np.ndarray,
        timestamp: str
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

        # ── Step 1: VAD ──────────────────────────────────────
        is_speech, vad_conf = self.vad.detect(audio)

        # Diagnostic: audio RMS and VAD result
        audio_rms = float(np.sqrt(np.mean(audio ** 2)))
        logger.info(
            f"[Diag] chunk={self._chunk_count} | rms={audio_rms:.5f} | "
            f"vad={'SPEECH' if is_speech else 'SILENCE'} (conf={vad_conf:.3f})"
        )

        if not is_speech:
            logger.debug(f"Chunk {self._chunk_count}: silence (VAD conf={vad_conf:.2f}), skipping.")
            return []

        self._speech_chunk_count += 1
        logger.debug(f"Chunk {self._chunk_count}: speech detected (VAD conf={vad_conf:.2f})")

        # ── Step 2: Speaker Embedding ────────────────────────
        speaker_label = "Speaker 1"
        speaker_id = 0
        speaker_confidence = 1.0

        embedding_result = self.embedder.extract(audio)
        if embedding_result is not None:
            # ── Step 3: Clustering ───────────────────────────
            elapsed_sec = self._elapsed_seconds()
            speaker_id, speaker_label, speaker_confidence = self.clusterer.assign_speaker(
                embedding=embedding_result.vector,
                duration=len(audio) / self.sample_rate,
                timestamp=elapsed_sec
            )

            # Diagnostic: embedding norm
            emb_norm = float(np.linalg.norm(embedding_result.vector))
            logger.info(
                f"[Diag] emb_norm={emb_norm:.4f} | "
                f"assigned={speaker_label} (conf={speaker_confidence:.3f}) | "
                f"n_clusters={self.clusterer.n_speakers}"
            )

            # Update persistent speaker profile store
            if embedding_result is not None:
                try:
                    n_samp = (
                        self.clusterer.clusters[speaker_id].n_samples
                        if speaker_id < len(self.clusterer.clusters)
                        else 1
                    )
                    self.embedder.profile_store.update(
                        label=speaker_label,
                        embedding=embedding_result.vector,
                        n_samples=n_samp
                    )
                except Exception as _pe:
                    logger.debug(f"Profile store update skipped: {_pe}")

            # Log diagnostic similarity matrix every 10 speech chunks
            if self._speech_chunk_count % 10 == 0 and self._speech_chunk_count > 0:
                self.clusterer.log_diagnostic_matrix()

            # Merge similar clusters every 15 speech chunks (≈30s) after 20 chunks.
            # Merging too early collapses clusters before centroids have stabilised.
            if self._speech_chunk_count % 15 == 0 and self._speech_chunk_count >= 20:
                self.clusterer.merge_similar_clusters()
        else:
            # Embedding failed (audio too short/quiet)
            # Use last known speaker or default
            if self.clusterer.n_speakers > 0:
                last_cluster = self.clusterer.clusters[-1]
                speaker_id = last_cluster.speaker_id
                speaker_label = last_cluster.label
            logger.debug("Embedding extraction failed, using last known speaker.")

        # ── Step 4: ASR ──────────────────────────────────────
        asr_result = self.asr.transcribe(audio, sample_rate=self.sample_rate)

        if asr_result is None or not asr_result.text.strip():
            logger.debug(f"ASR returned empty text for chunk {self._chunk_count}")
            return []

        # ── Step 5: Build Subtitle Result ───────────────────
        subtitle = SubtitleResult(
            timestamp=timestamp,
            speaker=speaker_label,
            speaker_id=speaker_id,
            text=asr_result.text,
            confidence=asr_result.confidence
        )

        # ── Step 6: Merge via SubtitleBuffer ─────────────────
        # (from scratch — merges same-speaker consecutive results)
        finalized_subtitles = self.subtitle_buffer.add(subtitle)

        # Log timing
        proc_ms = (time.time() - chunk_start_time) * 1000
        logger.info(
            f"Chunk processed in {proc_ms:.0f}ms | "
            f"{speaker_label}: '{asr_result.text[:50]}...'" if len(asr_result.text) > 50
            else f"Chunk processed in {proc_ms:.0f}ms | {speaker_label}: '{asr_result.text}'"
        )

        # Convert to JSON-serializable dicts
        results = []
        for sub in finalized_subtitles:
            results.append(self._subtitle_to_dict(sub))

        # Also send the current (possibly partial) subtitle immediately
        # This gives real-time feedback even if not yet "finalized"
        if not finalized_subtitles:
            # Send as partial (will be merged into next result)
            d = self._subtitle_to_dict(subtitle)
            d["is_partial"] = True
            results.append(d)

        return results

    def process_flush(self) -> List[dict]:
        """
        Flush remaining buffered subtitles.
        Call this when recording stops.
        """
        final_subs = self.subtitle_buffer.flush_all()
        return [self._subtitle_to_dict(s) for s in final_subs]

    # ──────────────────────────────────────────────────────────
    # FROM SCRATCH: Subtitle Merging & Formatting
    # ──────────────────────────────────────────────────────────

    def _subtitle_to_dict(self, subtitle: SubtitleResult) -> dict:
        """
        Convert SubtitleResult to JSON-serializable dict — FROM SCRATCH.

        Also looks up the speaker's color from the clustering module.
        """
        # Get color for this speaker
        color = "#4FC3F7"  # Default blue
        if subtitle.speaker_id < len(self.clusterer.clusters):
            color = self.clusterer.clusters[subtitle.speaker_id].color

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
        }

    def reset(self):
        """Reset pipeline state for a new session."""
        self.clusterer.reset()
        self.asr.reset()
        self.subtitle_buffer = SubtitleBuffer(merge_window_sec=3.0)
        self._session_start = time.time()
        self._chunk_count = 0
        self._speech_chunk_count = 0
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