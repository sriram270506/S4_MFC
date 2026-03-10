"""
audio_stream.py — Audio Chunking Handler
=========================================

This module handles the low-level audio buffering logic.

WHY WE NEED THIS:
  The browser sends audio in small bursts (e.g., every 250ms).
  But our ML models need larger chunks (1–2 seconds) to work well.
  This class accumulates audio and cuts it into fixed-size chunks.

STUDENT EXPLANATION:
  Think of this like filling a cup with water drip by drip.
  We keep adding audio until the cup is full (chunk_seconds),
  then we hand the full cup to the ML pipeline.

IMPLEMENTED FROM SCRATCH:
  - Ring buffer logic for audio accumulation
  - Overlap handling for context preservation
  - Pre-emphasis filter (our own implementation)
  - Energy-based clipping detection
"""

import numpy as np
import logging
from typing import List, Optional
from collections import deque

logger = logging.getLogger("audio_stream")


class AudioStreamHandler:
    """
    Accumulates incoming audio bytes and yields fixed-size chunks.

    Parameters:
    -----------
    sample_rate   : int   — Audio sample rate (16000 Hz standard for ASR)
    chunk_seconds : float — How many seconds per chunk to send to pipeline
    overlap_ratio : float — How much of previous chunk to prepend (for context)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_seconds: float = 1.5,
        overlap_ratio: float = 0.2,
        enable_slow_agc: bool = True,
        agc_target_rms: float = 0.07,
        agc_smoothing: float = 0.05
    ):
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.overlap_ratio = overlap_ratio
        self.enable_slow_agc = enable_slow_agc
        self.agc_target_rms = agc_target_rms
        self.agc_smoothing = agc_smoothing

        # How many samples per chunk
        self.chunk_size = int(sample_rate * chunk_seconds)

        # How many samples of overlap (context from previous chunk)
        self.overlap_size = int(self.chunk_size * overlap_ratio)

        # Internal buffer: deque is efficient for left-pop operations
        self._buffer: deque = deque()
        self._buffer_len: int = 0

        # Keep last N samples from previous chunk for overlap
        self._prev_tail: Optional[np.ndarray] = None

        # Session-level slow AGC state.
        # We intentionally avoid per-burst peak normalization because it
        # distorts speaker identity cues between adjacent packets.
        self._agc_gain: float = 1.0

        logger.info(
            f"AudioStreamHandler init: "
            f"sample_rate={sample_rate}, "
            f"chunk_size={self.chunk_size} samples ({chunk_seconds}s), "
            f"overlap={self.overlap_size} samples, "
            f"slow_agc={self.enable_slow_agc}"
        )

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def add_audio(self, audio: np.ndarray) -> List[np.ndarray]:
        """
        Add new audio samples to the buffer.

        Returns a list of complete chunks ready for processing.
        If not enough audio yet, returns empty list.

        Parameters:
        -----------
        audio : np.ndarray — float32, shape (N,), range [-1.0, 1.0]

        Returns:
        --------
        List[np.ndarray] — Zero or more chunks of shape (chunk_size,)
        """
        # Basic validation
        audio = self._preprocess(audio)

        # Add to internal buffer
        self._buffer.append(audio)
        self._buffer_len += len(audio)

        # Extract complete chunks
        ready_chunks = []
        while self._buffer_len >= self.chunk_size:
            chunk = self._extract_chunk()
            ready_chunks.append(chunk)

        return ready_chunks

    def flush(self) -> Optional[np.ndarray]:
        """
        Force-extract whatever audio is left in the buffer.
        Call this when the user stops recording.

        If less than 0.5 seconds, returns None (not worth transcribing).
        """
        min_samples = int(self.sample_rate * 0.5)
        if self._buffer_len < min_samples:
            logger.debug(f"Flush: only {self._buffer_len} samples, skipping.")
            return None

        # Concatenate all remaining samples
        remaining = np.concatenate(list(self._buffer)) if self._buffer else np.array([], dtype=np.float32)
        self.reset()

        if len(remaining) > 0:
            # Pad to chunk_size if needed
            padded = self._pad_to_chunk(remaining)
            return padded
        return None

    def reset(self):
        """Clear the buffer. Call between sessions."""
        self._buffer.clear()
        self._buffer_len = 0
        self._prev_tail = None
        self._agc_gain = 1.0

    def set_chunk_seconds(self, chunk_seconds: float):
        """Adjust chunk duration at runtime for adaptive backpressure control."""
        chunk_seconds = float(max(0.5, min(3.0, chunk_seconds)))
        self.chunk_seconds = chunk_seconds
        self.chunk_size = int(self.sample_rate * self.chunk_seconds)
        self.overlap_size = int(self.chunk_size * self.overlap_ratio)
        logger.info(
            "AudioStreamHandler chunk resized: %.2fs (%d samples), overlap=%d",
            self.chunk_seconds,
            self.chunk_size,
            self.overlap_size,
        )

    # ──────────────────────────────────────────────────────────
    # Internal Methods (implemented from scratch)
    # ──────────────────────────────────────────────────────────

    def _extract_chunk(self) -> np.ndarray:
        """
        Extract exactly chunk_size samples from the buffer.

        We prepend overlap samples from the previous chunk
        so the models have context at segment boundaries.

        OUR ALGORITHM:
          1. Flatten buffer into array
          2. Slice first chunk_size samples
          3. Store last overlap_size samples as tail
          4. Prepend tail from PREVIOUS chunk to current chunk
          5. Update buffer with leftover
        """
        # Flatten buffer deque into single array
        flat = np.concatenate(list(self._buffer))

        # Take the first chunk
        chunk = flat[:self.chunk_size].copy()

        # Save tail of this chunk for next chunk's context
        current_tail = chunk[-self.overlap_size:].copy() if self.overlap_size > 0 else None

        # Prepend previous tail to give context at boundaries
        if self._prev_tail is not None and len(self._prev_tail) > 0:
            chunk_with_context = np.concatenate([self._prev_tail, chunk])
            # Trim back to chunk_size (context is for model, not extra length)
            chunk_with_context = chunk_with_context[:self.chunk_size]
        else:
            chunk_with_context = chunk

        self._prev_tail = current_tail

        # Update buffer: put back the leftover
        leftover = flat[self.chunk_size:]
        self._buffer.clear()
        if len(leftover) > 0:
            self._buffer.append(leftover)
        self._buffer_len = len(leftover)

        return self._apply_slow_agc(chunk_with_context)

    def _preprocess(self, audio: np.ndarray) -> np.ndarray:
        """
        Preprocess incoming audio chunk.

        Steps (all implemented from scratch):
          1. Ensure float32 dtype
          2. Clip hard peaks (avoid NaN/Inf from bad mic input)
          3. Keep original loudness envelope (identity-safe)

        NOTE: Pre-emphasis (high-pass filter) is intentionally NOT applied here.
          Pre-emphasis is applied per-burst (every ~250ms browser packet), which
          creates spectral discontinuities at burst boundaries within a 2-second
          chunk.  Those per-burst artifacts corrupt ECAPA-TDNN speaker embeddings
          because the model sees a different spectral shape each chunk even for
          the same speaker.

          Pre-emphasis is instead applied only inside WhisperASR._run_whisper(),
          where it runs once over the complete continuous chunk — exactly what
          it was designed for.
        """
        # Step 1: dtype safety
        audio = np.array(audio, dtype=np.float32)

        # Step 2: Clip extreme values (corrupted mic data)
        audio = np.clip(audio, -1.0, 1.0)

        # Step 3: Do NOT normalize per incoming packet.
        # Per-packet normalization causes speaker embedding drift.

        return audio

    def _apply_slow_agc(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply slow, session-level AGC to stabilize loudness without
        creating packet-to-packet identity distortion.
        """
        if not self.enable_slow_agc or len(audio) == 0:
            return audio

        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 1e-5:
            return audio

        desired_gain = self.agc_target_rms / rms
        desired_gain = float(np.clip(desired_gain, 0.25, 4.0))

        # Exponential smoothing keeps gain changes gradual across chunks.
        self._agc_gain = (
            (1.0 - self.agc_smoothing) * self._agc_gain
            + self.agc_smoothing * desired_gain
        )
        return np.clip(audio * self._agc_gain, -1.0, 1.0).astype(np.float32)

    def _pre_emphasis(self, audio: np.ndarray, alpha: float = 0.97) -> np.ndarray:
        """
        Pre-emphasis filter — implemented from scratch.

        This is a first-order high-pass filter commonly used in ASR.
        It compensates for the natural roll-off in speech and mic hardware.

        Formula: y[n] = x[n] - alpha * x[n-1]

        Parameters:
        -----------
        audio : np.ndarray — Input signal
        alpha : float      — Pre-emphasis coefficient (typically 0.95–0.97)

        Returns:
        --------
        np.ndarray — Filtered signal, same shape as input
        """
        if len(audio) == 0:
            return audio

        # Initialize output array
        emphasized = np.zeros_like(audio)

        # First sample stays the same (no previous sample)
        emphasized[0] = audio[0]

        # Apply filter: each sample minus alpha * previous sample
        for i in range(1, len(audio)):
            emphasized[i] = audio[i] - alpha * audio[i - 1]

        # Vectorized version (same result, faster):
        # emphasized = np.append(audio[0], audio[1:] - alpha * audio[:-1])

        return emphasized

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """
        Amplitude normalization — implemented from scratch.

        Divides by max absolute value to scale signal to [-1, 1].
        Avoids division by zero for silent segments.
        """
        max_val = np.max(np.abs(audio))
        if max_val > 1e-8:  # Avoid division by near-zero
            return audio / max_val
        return audio  # Already silent, return as-is

    def _pad_to_chunk(self, audio: np.ndarray) -> np.ndarray:
        """
        Pad audio to exactly chunk_size with zeros.
        Used for the final flush when audio is shorter than chunk_size.
        """
        if len(audio) >= self.chunk_size:
            return audio[:self.chunk_size]

        padding = np.zeros(self.chunk_size - len(audio), dtype=np.float32)
        return np.concatenate([audio, padding])

    # ──────────────────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────────────────

    @property
    def buffer_seconds(self) -> float:
        """How many seconds of audio are currently buffered."""
        return self._buffer_len / self.sample_rate

    def get_stats(self) -> dict:
        """Return current buffer statistics."""
        return {
            "buffer_samples": self._buffer_len,
            "buffer_seconds": round(self.buffer_seconds, 3),
            "chunk_size": self.chunk_size,
            "chunk_seconds": round(self.chunk_seconds, 3),
            "sample_rate": self.sample_rate,
            "agc_gain": round(self._agc_gain, 4),
        }


# ──────────────────────────────────────────────────────────────
# Standalone Test
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """Quick self-test for the AudioStreamHandler."""
    import time

    print("=== AudioStreamHandler Self-Test ===\n")

    handler = AudioStreamHandler(sample_rate=16000, chunk_seconds=1.5)

    # Simulate browser sending 250ms audio bursts
    burst_size = int(16000 * 0.25)  # 4000 samples per burst
    total_bursts = 12  # 12 * 250ms = 3 seconds of audio

    for i in range(total_bursts):
        # Simulate speech: 440 Hz sine wave with some noise
        t = np.linspace(0, 0.25, burst_size)
        audio_burst = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
        audio_burst += np.random.randn(burst_size).astype(np.float32) * 0.05

        chunks = handler.add_audio(audio_burst)
        print(f"Burst {i+1:2d}: sent {len(audio_burst)} samples → got {len(chunks)} chunk(s). Buffer: {handler.get_stats()['buffer_seconds']:.3f}s")

        for j, chunk in enumerate(chunks):
            print(f"          Chunk {j}: shape={chunk.shape}, max={chunk.max():.3f}, mean={chunk.mean():.5f}")

    # Final flush
    leftover = handler.flush()
    print(f"\nFlush: {'got chunk, shape=' + str(leftover.shape) if leftover is not None else 'nothing (too short)'}")
    print("\nSelf-test complete.")
