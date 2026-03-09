"""
vad.py — Voice Activity Detection (VAD)
=========================================

VAD = "Is there speech in this audio chunk, or just silence/noise?"

WHY VAD IS CRITICAL:
  Without VAD, we'd waste compute transcribing silence, breathing,
  background noise, etc. VAD gates the pipeline — only send real
  speech to the expensive models (Whisper, ECAPA-TDNN).

TWO IMPLEMENTATIONS PROVIDED:
  1. EnergyVAD   — Our own algorithm (from scratch). Simple, fast, educational.
  2. SileroVAD   — Production-grade ML model (torch-based). More accurate.

Both expose the same interface: detect(audio) → bool
The pipeline uses SileroVAD with EnergyVAD as fallback.

STUDENT NOTE:
  EnergyVAD is the "from scratch" part — understand it fully.
  SileroVAD is an external model we're using like a library.
"""

import numpy as np
import logging
import torch
from typing import Tuple, Optional

logger = logging.getLogger("vad")


# ══════════════════════════════════════════════════════════════
# 1. ENERGY-BASED VAD (FROM SCRATCH)
# ══════════════════════════════════════════════════════════════

class EnergyVAD:
    """
    Energy-based Voice Activity Detection — implemented entirely from scratch.

    ALGORITHM:
      Speech has significantly higher energy than silence.
      We compute the Root Mean Square (RMS) energy of the audio frame
      and compare against an adaptive threshold.

    This is the classic approach used before ML-based VAD existed.
    It works surprisingly well in clean environments.

    HOW IT WORKS (step by step):
      1. Split chunk into small frames (e.g., 20ms each)
      2. Compute RMS energy of each frame
      3. Compare against threshold
      4. Count frames with speech energy
      5. If enough frames have speech → chunk is speech

    LIMITATION: Fails in noisy environments (music, traffic, etc.)
    For noisy environments, use SileroVAD instead.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 20,          # Frame size in milliseconds
        energy_threshold: float = 0.01,  # RMS threshold
        speech_ratio_threshold: float = 0.3,  # Fraction of frames needing speech
        adaptive: bool = True         # Whether to adapt threshold to background noise
    ):
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_ms / 1000)  # Samples per frame
        self.energy_threshold = energy_threshold
        self.speech_ratio_threshold = speech_ratio_threshold
        self.adaptive = adaptive

        # For adaptive thresholding: keep a rolling history of noise energy
        self._noise_floor_history = []
        self._noise_floor_window = 10   # Track last 10 chunks
        self._noise_floor = energy_threshold  # Current estimated noise floor

        logger.info(f"EnergyVAD: frame_size={self.frame_size} samples, threshold={energy_threshold}")

    # ──────────────────────────────────────────────────────────
    # Main Detection
    # ──────────────────────────────────────────────────────────

    def detect(self, audio: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if speech is present in audio chunk.

        Parameters:
        -----------
        audio : np.ndarray — float32 audio, shape (N,)

        Returns:
        --------
        (is_speech, confidence_score)
          is_speech       : bool  — True if speech detected
          confidence_score: float — 0.0 to 1.0
        """
        if len(audio) == 0:
            return False, 0.0

        # Step 1: Split into frames
        frames = self._split_into_frames(audio)

        if len(frames) == 0:
            return False, 0.0

        # Step 2: Compute RMS energy per frame
        frame_energies = np.array([self._rms_energy(f) for f in frames])

        # Step 3: Update adaptive noise floor (optional)
        if self.adaptive:
            self._update_noise_floor(frame_energies)

        # Step 4: Determine active threshold
        threshold = self._get_threshold()

        # Step 5: Count speech frames (energy above threshold)
        speech_frames = np.sum(frame_energies > threshold)
        total_frames = len(frames)
        speech_ratio = speech_frames / total_frames

        # Step 6: Decide
        is_speech = speech_ratio >= self.speech_ratio_threshold

        # Confidence = how much the ratio exceeds the threshold
        confidence = min(1.0, speech_ratio / self.speech_ratio_threshold) if is_speech else speech_ratio

        logger.debug(
            f"EnergyVAD: frames={total_frames}, speech_frames={speech_frames}, "
            f"ratio={speech_ratio:.2f}, threshold={threshold:.5f}, "
            f"is_speech={is_speech}, confidence={confidence:.2f}"
        )

        return is_speech, confidence

    # ──────────────────────────────────────────────────────────
    # Internal Algorithms (all from scratch)
    # ──────────────────────────────────────────────────────────

    def _split_into_frames(self, audio: np.ndarray) -> list:
        """
        Split audio array into equal-size frames.

        Example: 24000 samples with frame_size=320 → 75 frames

        IMPLEMENTATION:
          We use stride tricks to avoid copying data.
          Each frame is a view into the original array.
        """
        n_frames = len(audio) // self.frame_size
        if n_frames == 0:
            return [audio]  # Return whole chunk as single frame

        frames = []
        for i in range(n_frames):
            start = i * self.frame_size
            end = start + self.frame_size
            frames.append(audio[start:end])

        # Handle leftover samples (partial frame at end)
        remainder = audio[n_frames * self.frame_size:]
        if len(remainder) > self.frame_size // 2:  # Only if more than half a frame
            frames.append(remainder)

        return frames

    def _rms_energy(self, frame: np.ndarray) -> float:
        """
        Root Mean Square (RMS) energy — FROM SCRATCH.

        Formula: RMS = sqrt( mean( x[i]^2 ) )

        This is the standard measure of signal power.
        Speech has RMS > 0.01 typically; silence < 0.001.
        """
        if len(frame) == 0:
            return 0.0
        return float(np.sqrt(np.mean(frame ** 2)))

    def _zero_crossing_rate(self, frame: np.ndarray) -> float:
        """
        Zero-Crossing Rate (ZCR) — FROM SCRATCH.

        Counts how many times the signal crosses zero per second.
        Speech (voiced): ZCR ~ 50-100 Hz
        Fricatives:      ZCR ~ 1000+ Hz
        Silence:         ZCR ~  0 Hz

        Used as supplementary feature alongside energy.
        """
        if len(frame) < 2:
            return 0.0

        # Count sign changes
        sign_changes = 0
        for i in range(1, len(frame)):
            if (frame[i] >= 0) != (frame[i-1] >= 0):
                sign_changes += 1

        # Normalize to rate per second
        duration_sec = len(frame) / self.sample_rate
        return sign_changes / duration_sec

    def _update_noise_floor(self, frame_energies: np.ndarray):
        """
        Adaptive noise floor estimation — FROM SCRATCH.

        Tracks the 10th percentile of recent frame energies.
        This approximates the background noise level.
        Speech threshold is then set relative to the noise floor.
        """
        # Take the lower 20% of frame energies as noise estimate
        noise_estimate = float(np.percentile(frame_energies, 20))
        self._noise_floor_history.append(noise_estimate)

        # Keep only the last N chunks
        if len(self._noise_floor_history) > self._noise_floor_window:
            self._noise_floor_history.pop(0)

        # Smooth the noise floor (exponential moving average)
        if len(self._noise_floor_history) > 0:
            alpha = 0.1  # Smoothing factor
            new_floor = float(np.mean(self._noise_floor_history))
            self._noise_floor = (1 - alpha) * self._noise_floor + alpha * new_floor

    def _get_threshold(self) -> float:
        """
        Get the current detection threshold.

        If adaptive: threshold = noise_floor * multiplier
        If fixed:    threshold = initial energy_threshold
        """
        if self.adaptive and self._noise_floor > 0:
            # Speech is typically 3-5x louder than noise floor
            return self._noise_floor * 4.0
        return self.energy_threshold

    def get_debug_info(self, audio: np.ndarray) -> dict:
        """Return detailed frame-by-frame energy info for debugging."""
        frames = self._split_into_frames(audio)
        energies = [self._rms_energy(f) for f in frames]
        zcrs = [self._zero_crossing_rate(f) for f in frames]
        threshold = self._get_threshold()

        return {
            "n_frames": len(frames),
            "mean_energy": float(np.mean(energies)),
            "max_energy": float(np.max(energies)),
            "threshold": threshold,
            "noise_floor": self._noise_floor,
            "frame_energies": [round(e, 5) for e in energies],
            "frame_zcrs": [round(z, 1) for z in zcrs],
            "speech_frame_count": int(np.sum(np.array(energies) > threshold)),
        }


# ══════════════════════════════════════════════════════════════
# 2. SILERO VAD (ML-BASED, PRODUCTION GRADE)
# ══════════════════════════════════════════════════════════════

class SileroVAD:
    """
    Silero VAD — deep learning Voice Activity Detection.

    This uses a small, fast neural network trained specifically for VAD.
    It's much more robust than energy-based VAD in noisy environments.

    Model: silero-vad (https://github.com/snakers4/silero-vad)
    Size: ~1.8MB
    Latency: <1ms per chunk on CPU

    HOW IT DIFFERS FROM ENERGY VAD:
      - Uses LSTM neural network trained on speech/non-speech examples
      - Handles music, noise, reverb much better
      - Returns probability (0 to 1) of speech presence
      - More accurate at detecting soft speech
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        window_size_samples: int = 512  # ~32ms window
    ):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.window_size_samples = window_size_samples
        self.model = None
        self._loaded = False

        # Fallback in case Silero fails to load
        self._fallback = EnergyVAD(sample_rate=sample_rate)

    def load(self):
        """Load the Silero VAD model from torch hub."""
        try:
            logger.info("Loading Silero VAD model...")
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            self.model.eval()
            self._loaded = True
            logger.info("Silero VAD loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load Silero VAD: {e}. Using EnergyVAD fallback.")
            self._loaded = False

    def detect(self, audio: np.ndarray) -> Tuple[bool, float]:
        """
        Detect speech using Silero neural network.

        Parameters:
        -----------
        audio : np.ndarray — float32, shape (N,), 16kHz

        Returns:
        --------
        (is_speech, probability)
        """
        if not self._loaded or self.model is None:
            # Fall back to energy-based VAD
            return self._fallback.detect(audio)

        try:
            return self._silero_detect(audio)
        except Exception as e:
            logger.warning(f"Silero detect failed: {e}. Using fallback.")
            return self._fallback.detect(audio)

    def _silero_detect(self, audio: np.ndarray) -> Tuple[bool, float]:
        """
        Run Silero model on sliding windows and average the probabilities.

        We process the audio in windows (512 samples each) and
        aggregate the probabilities to decide on the full chunk.
        """
        audio_tensor = torch.FloatTensor(audio)
        n_samples = len(audio_tensor)

        probabilities = []

        # Slide a window across the chunk
        for start in range(0, n_samples, self.window_size_samples):
            end = min(start + self.window_size_samples, n_samples)
            window = audio_tensor[start:end]

            # Pad if too short
            if len(window) < self.window_size_samples:
                pad_len = self.window_size_samples - len(window)
                window = torch.cat([window, torch.zeros(pad_len)])

            with torch.no_grad():
                speech_prob = self.model(window, self.sample_rate).item()
                probabilities.append(speech_prob)

        if not probabilities:
            return False, 0.0

        # Average probability across all windows
        avg_prob = float(np.mean(probabilities))
        is_speech = avg_prob >= self.threshold

        return is_speech, avg_prob

    def reset_states(self):
        """Reset model hidden states between calls."""
        if self.model is not None:
            try:
                self.model.reset_states()
            except:
                pass


# ══════════════════════════════════════════════════════════════
# 3. COMBINED VAD (uses both for higher accuracy)
# ══════════════════════════════════════════════════════════════

class CombinedVAD:
    """
    Combined VAD that uses both Energy and Silero.

    Decision rule:
      - If Silero is loaded: use Silero (more accurate)
      - Fallback to Energy VAD if Silero unavailable
      - Optionally: require both to agree (conservative mode)

    For a college project, this shows understanding of both approaches.
    """

    def __init__(self, sample_rate: int = 16000, threshold: float = 0.45):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.energy_vad = EnergyVAD(sample_rate=sample_rate, adaptive=True)
        self.silero_vad = SileroVAD(sample_rate=sample_rate)

    def initialize(self):
        """Load models."""
        self.silero_vad.load()

    def detect(self, audio: np.ndarray) -> Tuple[bool, float]:
        """
        Combined detection with fallback logic.

        Returns:
        --------
        (is_speech, confidence)
        """
        # Always run energy VAD (fast, no model needed)
        energy_speech, energy_conf = self.energy_vad.detect(audio)

        # Run Silero if loaded
        if self.silero_vad._loaded:
            silero_speech, silero_conf = self.silero_vad.detect(audio)

            # Combine: weighted average of confidences
            # Silero is more reliable, so weight it higher
            combined_conf = 0.3 * energy_conf + 0.7 * silero_conf
            is_speech = combined_conf >= self.threshold

            logger.debug(
                f"CombinedVAD: energy=({energy_speech},{energy_conf:.2f}) "
                f"silero=({silero_speech},{silero_conf:.2f}) → "
                f"combined={combined_conf:.2f}, is_speech={is_speech}"
            )
            return is_speech, combined_conf
        else:
            # Silero not available, use energy only
            return energy_speech, energy_conf


# ──────────────────────────────────────────────────────────────
# Standalone Test
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    print("=== VAD Self-Test ===\n")

    sr = 16000
    vad = EnergyVAD(sample_rate=sr, adaptive=True)

    # Test 1: Silence
    silence = np.zeros(sr, dtype=np.float32)
    is_speech, conf = vad.detect(silence)
    print(f"Silence → is_speech={is_speech}, confidence={conf:.3f}  (expected: False)")

    # Test 2: Pure tone (not speech, but loud)
    tone = (np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr)) * 0.5).astype(np.float32)
    is_speech, conf = vad.detect(tone)
    print(f"440Hz tone → is_speech={is_speech}, confidence={conf:.3f}")

    # Test 3: Noise (simulates speech energy)
    noise = (np.random.randn(sr) * 0.3).astype(np.float32)
    is_speech, conf = vad.detect(noise)
    print(f"White noise → is_speech={is_speech}, confidence={conf:.3f}")

    # Test 4: Low energy (quiet background)
    quiet_noise = (np.random.randn(sr) * 0.002).astype(np.float32)
    is_speech, conf = vad.detect(quiet_noise)
    print(f"Quiet noise → is_speech={is_speech}, confidence={conf:.3f}  (expected: False)")

    # Debug info
    print("\nDebug frame info for noise sample:")
    debug = vad.get_debug_info(noise[:sr//4])
    for k, v in debug.items():
        if k != "frame_energies" and k != "frame_zcrs":
            print(f"  {k}: {v}")

    print("\nSelf-test complete.")
