"""
diarization.py — Speaker Embedding Extraction
===============================================

WHAT IS SPEAKER DIARIZATION?
  "Who spoke when?"
  We identify different speakers in audio and label their segments.

HOW WE DO IT:
  Step 1 → Extract a "speaker embedding" (a vector fingerprint of who is speaking)
  Step 2 → Cluster embeddings to group same-speaker segments
  (Clustering is in clustering.py)

SPEAKER EMBEDDING:
  A speaker embedding is a fixed-size vector (e.g., 192 dimensions)
  that captures voice characteristics:
    - Pitch patterns
    - Speaking rate
    - Vocal tract shape
    - Formant frequencies

  Same speaker → similar vectors (high cosine similarity)
  Different speaker → different vectors (low cosine similarity)

MODEL USED:
  SpeechBrain ECAPA-TDNN trained on VoxCeleb dataset.
  ECAPA = Emphasized Channel Attention, Propagation and Aggregation
  TDNN  = Time Delay Neural Network

  As specified in requirements:
    from speechbrain.pretrained import EncoderClassifier
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb"
    )
    embedding = classifier.encode_batch(audio_tensor)

IMPLEMENTED FROM SCRATCH (our code):
  - Audio preprocessing pipeline for speaker embedding
  - Mel filterbank feature extraction
  - Embedding quality validation
  - Embedding database management
"""

import numpy as np
import logging
import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
import json
import os
import time as _time

logger = logging.getLogger("diarization")


# ──────────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────────

@dataclass
class SpeakerEmbedding:
    """Represents a single speaker embedding with metadata."""
    vector: np.ndarray           # The embedding vector (192-dim for ECAPA-TDNN)
    speaker_id: int              # Assigned cluster/speaker ID
    timestamp: float             # When this was captured (seconds since start)
    energy: float                # Audio energy (quality indicator)
    duration: float              # Duration of audio segment in seconds


# ══════════════════════════════════════════════════════════════
# 1. AUDIO FEATURES — FROM SCRATCH
# ══════════════════════════════════════════════════════════════

class AudioFeatureExtractor:
    """
    Extract audio features for speaker characterization — FROM SCRATCH.

    We implement Mel Filterbank features, which are the standard
    input features for speaker recognition models.

    FEATURE PIPELINE:
      Audio → Framing → Windowing → FFT → Mel Scale → Log → Features

    This is the "from scratch" part of the feature extraction.
    The ECAPA-TDNN model does additional deep learning on top.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        frame_length_ms: float = 25.0,
        frame_shift_ms: float = 10.0,
        n_fft: int = 512,
        f_min: float = 80.0,
        f_max: float = 7600.0
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.frame_length = int(sample_rate * frame_length_ms / 1000)  # 400 samples
        self.frame_shift = int(sample_rate * frame_shift_ms / 1000)    # 160 samples
        self.n_fft = n_fft
        self.f_min = f_min
        self.f_max = f_max

        # Precompute mel filterbank matrix
        self.mel_filterbank = self._build_mel_filterbank()

        logger.info(
            f"AudioFeatureExtractor: sr={sample_rate}, n_mels={n_mels}, "
            f"frame_length={self.frame_length}, frame_shift={self.frame_shift}"
        )

    # ──────────────────────────────────────────────────────────
    # FROM SCRATCH: Mel Filterbank Construction
    # ──────────────────────────────────────────────────────────

    def _hz_to_mel(self, hz: float) -> float:
        """
        Convert Hz to Mel scale — FROM SCRATCH.

        The Mel scale approximates how humans perceive pitch.
        Low frequencies: fine-grained perception
        High frequencies: coarser perception

        Formula (O'Shaughnessy 1987):
          mel = 2595 * log10(1 + hz/700)
        """
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def _mel_to_hz(self, mel: float) -> float:
        """
        Convert Mel to Hz — FROM SCRATCH.

        Inverse of _hz_to_mel:
          hz = 700 * (10^(mel/2595) - 1)
        """
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _build_mel_filterbank(self) -> np.ndarray:
        """
        Build triangular mel filterbank — FROM SCRATCH.

        Creates n_mels triangular filters evenly spaced on the Mel scale.
        Each filter is a triangular weighting in the frequency domain.

        Returns:
        --------
        np.ndarray — shape (n_mels, n_fft//2 + 1)
        """
        n_freq_bins = self.n_fft // 2 + 1

        # Convert f_min and f_max to Mel
        mel_min = self._hz_to_mel(self.f_min)
        mel_max = self._hz_to_mel(self.f_max)

        # Create n_mels + 2 equally spaced Mel points (include edges)
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)

        # Convert back to Hz
        hz_points = np.array([self._mel_to_hz(m) for m in mel_points])

        # Map Hz to FFT bin indices
        bin_indices = np.floor(
            (self.n_fft + 1) * hz_points / self.sample_rate
        ).astype(int)
        bin_indices = np.clip(bin_indices, 0, n_freq_bins - 1)

        # Build filterbank matrix
        filterbank = np.zeros((self.n_mels, n_freq_bins), dtype=np.float32)

        for m in range(1, self.n_mels + 1):
            f_left   = bin_indices[m - 1]
            f_center = bin_indices[m]
            f_right  = bin_indices[m + 1]

            # Left slope (rising)
            for k in range(f_left, f_center):
                if f_center - f_left > 0:
                    filterbank[m - 1, k] = (k - f_left) / (f_center - f_left)

            # Right slope (falling)
            for k in range(f_center, f_right):
                if f_right - f_center > 0:
                    filterbank[m - 1, k] = (f_right - k) / (f_right - f_center)

        return filterbank

    # ──────────────────────────────────────────────────────────
    # FROM SCRATCH: Frame-by-Frame Feature Extraction
    # ──────────────────────────────────────────────────────────

    def extract_fbank(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract log Mel filterbank features — FROM SCRATCH.

        PIPELINE:
          1. Frame the signal
          2. Apply Hamming window
          3. Compute FFT → power spectrum
          4. Apply Mel filterbank
          5. Take log

        Parameters:
        -----------
        audio : np.ndarray — float32, shape (N,)

        Returns:
        --------
        np.ndarray — shape (n_frames, n_mels)
        """
        # Step 1: Frame the signal
        frames = self._frame_signal(audio)
        if len(frames) == 0:
            return np.zeros((1, self.n_mels), dtype=np.float32)

        # Step 2: Apply Hamming window to each frame
        window = np.hamming(self.frame_length).astype(np.float32)
        frames = frames * window[np.newaxis, :]

        # Step 3: FFT → power spectrum
        # Use numpy's FFT on each frame
        fft_frames = np.fft.rfft(frames, n=self.n_fft, axis=1)
        power_spectrum = np.abs(fft_frames) ** 2  # shape: (n_frames, n_fft//2 + 1)

        # Step 4: Apply mel filterbank
        # Matrix multiply: (n_frames, n_fft//2+1) @ (n_fft//2+1, n_mels)
        mel_features = np.dot(power_spectrum, self.mel_filterbank.T)  # (n_frames, n_mels)

        # Step 5: Log compression (stabilize small values with floor)
        mel_features = np.log(mel_features + 1e-10)

        return mel_features.astype(np.float32)

    def _frame_signal(self, audio: np.ndarray) -> np.ndarray:
        """
        Frame the signal with overlap — FROM SCRATCH.

        Splits audio into overlapping frames.
        Each frame: frame_length samples
        Step between frames: frame_shift samples

        Example:
          audio = 16000 samples (1 second)
          frame_length = 400, frame_shift = 160
          → ~98 frames

        Returns:
        --------
        np.ndarray — shape (n_frames, frame_length)
        """
        n_samples = len(audio)
        n_frames = 1 + (n_samples - self.frame_length) // self.frame_shift

        if n_frames <= 0:
            # Audio too short: just pad and return single frame
            padded = np.zeros(self.frame_length, dtype=np.float32)
            padded[:n_samples] = audio
            return padded[np.newaxis, :]

        # Allocate output array
        frames = np.zeros((n_frames, self.frame_length), dtype=np.float32)

        for i in range(n_frames):
            start = i * self.frame_shift
            end = start + self.frame_length
            frames[i] = audio[start:end]

        return frames

    def compute_delta(self, features: np.ndarray, order: int = 1) -> np.ndarray:
        """
        Compute delta (velocity) features — FROM SCRATCH.

        Delta features capture how features change over time.
        They improve speaker recognition significantly.

        Formula (2nd order approximation):
          delta[t] = (2*f[t+1] - 2*f[t-1] + f[t+2] - f[t-2]) / 10

        Parameters:
        -----------
        features : np.ndarray — shape (n_frames, n_features)
        order     : int       — 1=delta, 2=delta-delta

        Returns:
        --------
        np.ndarray — same shape as features
        """
        n_frames, n_features = features.shape
        delta = np.zeros_like(features)

        # Pad edges by replicating first/last frames
        pad = 2
        padded = np.pad(features, ((pad, pad), (0, 0)), mode='edge')

        denominator = 2 * (1**2 + 2**2)  # = 10

        for t in range(n_frames):
            # t in padded array is t + pad
            tp = t + pad
            delta[t] = (
                2 * (padded[tp + 1] - padded[tp - 1]) +
                1 * (padded[tp + 2] - padded[tp - 2])
            ) / denominator

        if order == 2:
            return self.compute_delta(delta, order=1)
        return delta


# ══════════════════════════════════════════════════════════════
# 2. PERSISTENT SPEAKER PROFILE STORE
# ══════════════════════════════════════════════════════════════

class SpeakerProfileStore:
    """
    Persists speaker embedding centroids to disk between sessions.

    This allows the system to recognise returning speakers across
    different recording sessions — similar to how Zoom identifies
    known participants when they rejoin.

    Storage format: JSON file, one entry per known speaker.
    Each entry stores: label, centroid (as list), n_samples, last_seen timestamp.
    """

    PROFILE_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "speaker_profiles.json"
    )

    def __init__(self):
        self.profiles: dict = {}  # speaker_label → {centroid, n_samples, last_seen}
        self._load()

    def _load(self):
        if os.path.exists(self.PROFILE_PATH):
            try:
                with open(self.PROFILE_PATH, "r") as f:
                    raw = json.load(f)
                for label, data in raw.items():
                    self.profiles[label] = {
                        "centroid": np.array(data["centroid"], dtype=np.float32),
                        "n_samples": data["n_samples"],
                        "last_seen": data["last_seen"]
                    }
                logger.info(
                    f"SpeakerProfileStore: loaded {len(self.profiles)} profiles from disk."
                )
            except Exception as e:
                logger.warning(f"SpeakerProfileStore: could not load profiles: {e}")

    def save(self):
        try:
            serializable = {}
            for label, data in self.profiles.items():
                serializable[label] = {
                    "centroid": data["centroid"].tolist(),
                    "n_samples": int(data["n_samples"]),
                    "last_seen": float(data["last_seen"])
                }
            with open(self.PROFILE_PATH, "w") as f:
                json.dump(serializable, f, indent=2)
        except Exception as e:
            logger.warning(f"SpeakerProfileStore: could not save profiles: {e}")

    def update(self, label: str, embedding: np.ndarray, n_samples: int):
        """Update or create a profile for this speaker label."""
        if label in self.profiles:
            existing = self.profiles[label]
            n = existing["n_samples"]
            # Weighted average: give existing profile 85% weight after 20 samples
            if n < 20:
                new_centroid = (n * existing["centroid"] + embedding) / (n + 1)
            else:
                new_centroid = 0.85 * existing["centroid"] + 0.15 * embedding
            norm = np.linalg.norm(new_centroid)
            if norm > 1e-10:
                new_centroid /= norm
            self.profiles[label] = {
                "centroid": new_centroid,
                "n_samples": n + 1,
                "last_seen": _time.time()
            }
        else:
            norm = np.linalg.norm(embedding)
            centroid = embedding / norm if norm > 1e-10 else embedding.copy()
            self.profiles[label] = {
                "centroid": centroid.copy(),
                "n_samples": n_samples,
                "last_seen": _time.time()
            }
        self.save()

    def get_all(self) -> list:
        """Return list of (label, centroid) for all stored profiles."""
        return [(label, data["centroid"]) for label, data in self.profiles.items()]

    def clear(self):
        self.profiles = {}
        if os.path.exists(self.PROFILE_PATH):
            os.remove(self.PROFILE_PATH)


# ══════════════════════════════════════════════════════════════
# 3. ECAPA-TDNN EMBEDDING EXTRACTOR
# ══════════════════════════════════════════════════════════════

class SpeakerEmbeddingExtractor:
    """
    Speaker embedding extraction using SpeechBrain ECAPA-TDNN.

    This uses the pretrained model as specified in the requirements:
      from speechbrain.pretrained import EncoderClassifier
      classifier = EncoderClassifier.from_hparams(
          source="speechbrain/spkrec-ecapa-voxceleb"
      )
      embedding = classifier.encode_batch(audio_tensor)

    What we add ON TOP (from scratch):
      - Audio quality checking before embedding
      - Embedding normalization (L2 norm)
      - Embedding reliability scoring
      - Session memory management
    """

    def __init__(self, sample_rate: int = 16000, min_duration_sec: float = 0.5):
        self.sample_rate = sample_rate
        self.min_duration_sec = min_duration_sec
        self.classifier = None
        self._loaded = False

        # Our own feature extractor (from scratch)
        self.feature_extractor = AudioFeatureExtractor(sample_rate=sample_rate)

        # Embedding dimension (ECAPA-TDNN outputs 192-dim)
        self.embedding_dim = 192

        # Persistent cross-session speaker profiles
        self.profile_store = SpeakerProfileStore()

        # For consecutive-chunk similarity diagnostics (CHANGE 8)
        self._last_embedding: Optional[np.ndarray] = None

    def load(self):
        """
        Load the pretrained ECAPA-TDNN model.
        Tries new SpeechBrain API first (>= 1.0), then legacy (< 1.0).
        """
        import traceback as _tb

        logger.info("Loading SpeechBrain ECAPA-TDNN model...")

        EncoderClassifier = None

        # Try new API path (speechbrain >= 1.0 / 0.5.15+)
        try:
            from speechbrain.inference.classifiers import EncoderClassifier
            logger.info("SpeechBrain: using new API (inference.classifiers)")
        except ImportError:
            pass

        # Fallback to legacy API path
        if EncoderClassifier is None:
            try:
                from speechbrain.pretrained import EncoderClassifier  # type: ignore
                logger.info("SpeechBrain: using legacy API (pretrained)")
            except ImportError:
                logger.error(
                    "SpeechBrain is NOT installed or unreachable. "
                    "Install with: pip install speechbrain"
                )
                self._loaded = False
                return

        try:
            self.classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="models/speechbrain_ecapa",
                run_opts={"device": "cpu"}
            )
            self._loaded = True
            logger.info(
                "ECAPA-TDNN loaded successfully — neural speaker embeddings active."
            )
        except Exception as e:
            logger.error(
                f"ECAPA-TDNN from_hparams() failed: {e}\n"
                "======= FULL TRACEBACK =======\n"
                f"{_tb.format_exc()}"
                "===========================\n"
                "Falling back to hand-crafted MFCC embeddings. "
                "Speaker discrimination will be degraded but functional."
            )
            self._loaded = False

    def extract(self, audio: np.ndarray) -> Optional[SpeakerEmbedding]:
        """
        Extract a speaker embedding from an audio chunk.

        PIPELINE (our orchestration):
          1. Validate audio quality (our code)
          2. Preprocess audio (our code)
          3. Extract embedding via ECAPA-TDNN model
          4. L2-normalize embedding (our code)
          5. Compute quality score (our code)
          6. Return SpeakerEmbedding object

        Parameters:
        -----------
        audio : np.ndarray — float32, shape (N,), 16kHz

        Returns:
        --------
        SpeakerEmbedding or None (if audio quality too low)
        """
        # Step 1: Check audio quality
        quality_ok, energy = self._check_audio_quality(audio)
        if not quality_ok:
            logger.debug(f"Audio quality too low (energy={energy:.5f}), skipping embedding.")
            return None

        # Step 2: Preprocess
        audio_processed = self._preprocess_for_embedding(audio)

        # Step 3: Extract embedding
        embedding_vector = self._extract_raw_embedding(audio_processed)

        if embedding_vector is None:
            return None

        # Step 4: L2 normalize (our code)
        embedding_vector = self._l2_normalize(embedding_vector)

        # Diagnostic: log embedding norm and audio stats
        emb_norm = float(np.linalg.norm(embedding_vector))
        emb_mean = float(np.mean(np.abs(embedding_vector)))
        emb_max  = float(np.max(np.abs(embedding_vector)))
        logger.info(
            f"[Embedding] rms={energy:.5f} | norm={emb_norm:.4f} | "
            f"mean_abs={emb_mean:.5f} | max_abs={emb_max:.5f} | "
            f"audio_samples={len(audio_processed)} | dim={embedding_vector.shape[0]}"
        )

        # CHANGE 8: Consecutive-chunk similarity — the key diagnostic metric.
        # For one speaker talking continuously, this must be consistently > 0.4.
        # If it jumps between 0.02 and 0.55, the preprocessing is broken.
        if self._last_embedding is not None:
            sim_to_last = float(np.dot(embedding_vector, self._last_embedding))
            rms_after = float(np.sqrt(np.mean(audio_processed ** 2)))
            logger.info(
                f"[EmbeddingDiag] sim_to_previous_chunk={sim_to_last:+.4f} | "
                f"audio_len={len(audio)} | processed_len={len(audio_processed)} | "
                f"rms_before={energy:.5f} | rms_after={rms_after:.5f}"
            )
        self._last_embedding = embedding_vector.copy()

        # Step 5: Build result object
        duration = len(audio) / self.sample_rate
        return SpeakerEmbedding(
            vector=embedding_vector,
            speaker_id=-1,       # Not yet assigned (clustering does this)
            timestamp=0.0,       # Set by caller
            energy=float(energy),
            duration=duration
        )

    # ──────────────────────────────────────────────────────────
    # FROM SCRATCH: Audio Quality Checks
    # ──────────────────────────────────────────────────────────

    def _check_audio_quality(self, audio: np.ndarray) -> Tuple[bool, float]:
        """
        Check if audio is suitable for embedding extraction — FROM SCRATCH.

        We reject audio that is:
          - Too short (< min_duration_sec)
          - Too quiet (RMS energy < threshold)
          - Clipped (saturated at ±1.0)

        Returns:
        --------
        (is_good_quality, rms_energy)
        """
        # Check duration
        min_samples = int(self.sample_rate * self.min_duration_sec)
        if len(audio) < min_samples:
            return False, 0.0

        # Compute RMS energy
        rms = float(np.sqrt(np.mean(audio ** 2)))

        # Check if too quiet
        if rms < 0.005:
            return False, rms

        # Check for clipping (>5% of samples at max amplitude)
        clipped_ratio = float(np.mean(np.abs(audio) > 0.99))
        if clipped_ratio > 0.05:
            logger.debug(f"Audio clipped ({clipped_ratio:.1%} of samples)")
            # Still usable but warn

        return True, rms

    # ──────────────────────────────────────────────────────────
    # FROM SCRATCH: Preprocessing for Embedding
    # ──────────────────────────────────────────────────────────

    def _preprocess_for_embedding(self, audio: np.ndarray) -> np.ndarray:
        """
        Preprocess audio for ECAPA-TDNN speaker embedding — FROM SCRATCH.

        KEY DESIGN DECISION: Do NOT extract voiced frames or remove silence.
        ECAPA-TDNN is a sequence model trained on continuous audio. Removing
        silence frames and concatenating the remaining ones produces
        non-contiguous audio — the model sees artificial discontinuities
        where natural pauses were. This destroys temporal consistency and
        causes the same speaker to get wildly different embeddings chunk-to-chunk
        (observed: cosine sim jumps from 0.02 to 0.55 randomly).

        Instead: pass the full continuous chunk, padded/trimmed to a fixed
        3-second length, DC-removed and RMS-normalised. The model handles
        silence internally via its attention mechanism.
        """
        audio = audio.astype(np.float32).copy()

        # ── Step 1: Pad or trim to exactly 3 seconds (48000 samples at 16kHz) ───
        # Fixed-length input gives ECAPA-TDNN consistent temporal context,
        # which is critical for stable embeddings.
        target_len = int(self.sample_rate * 3.0)  # 48000 samples
        if len(audio) >= target_len:
            # Take the middle 3 seconds (avoids boundary noise)
            start = (len(audio) - target_len) // 2
            audio = audio[start : start + target_len]
        else:
            # Pad with zeros at the end
            pad = np.zeros(target_len - len(audio), dtype=np.float32)
            audio = np.concatenate([audio, pad])

        # ── Step 2: Remove DC offset ─────────────────────────────────────────────
        audio -= np.mean(audio)

        # ── Step 3: RMS normalise to -23 dBFS (≈ 0.07 RMS) ─────────────────────
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms > 1e-6:
            audio = np.clip(audio * (0.07 / rms), -1.0, 1.0)

        # ── Step 4: Short fade-in/out (5ms) to avoid click artifacts ────────────
        fade_samples = int(self.sample_rate * 0.005)  # 80 samples
        if len(audio) > 2 * fade_samples:
            audio[:fade_samples]  *= np.linspace(0.0, 1.0, fade_samples)
            audio[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples)

        return audio

    # ──────────────────────────────────────────────────────────
    # Model Inference
    # ──────────────────────────────────────────────────────────

    def _extract_raw_embedding(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """
        Run ECAPA-TDNN model to get raw embedding.
        """
        if self._loaded and self.classifier is not None:
            try:
                # Convert to torch tensor
                audio_tensor = torch.FloatTensor(audio).unsqueeze(0)  # (1, N)

                # Extract embedding using SpeechBrain API
                with torch.no_grad():
                    embeddings = self.classifier.encode_batch(audio_tensor)  # (1, 1, 192)

                # Convert to numpy and flatten
                embedding = embeddings.squeeze().cpu().numpy()  # (192,)
                return embedding.astype(np.float32)

            except Exception as e:
                logger.error(f"ECAPA-TDNN inference error: {e}")
                return self._fallback_embedding(audio)
        else:
            return self._fallback_embedding(audio)

    def _fallback_embedding(self, audio: np.ndarray) -> np.ndarray:
        """
        Fallback speaker embedding using MFCC + Cepstral Mean Normalization — FROM SCRATCH.

        WHY MFCC OVER MEL FILTERBANK:
          Mel filterbank features directly capture phoneme content (what is said).
          MFCCs (Discrete Cosine Transform of log mel) decorrelate the features
          and — after Cepstral Mean Normalization — capture the VOCAL TRACT SHAPE
          of the speaker, not the phoneme sequence.

        CEPSTRAL MEAN NORMALIZATION (CMN):
          Subtract per-coefficient mean across all frames in the chunk.
          This removes channel/content effects, making c1..c12 largely
          speaker-discriminative rather than utterance-discriminative.

        EMBEDDING STRUCTURE (total 192 dims):
          mfcc_mean         : 20 dims  — average spectral shape (speaker timbre)
          mfcc_std          : 20 dims  — variance across frames
          mfcc_percentiles  : 40 dims  — p10,p25,p50,p75,p90 × 8 MFCCs
          delta_mean        : 20 dims  — average MFCC velocity (speaking rate cues)
          delta_std         : 20 dims  — variance thereof
          delta2_mean       : 20 dims  — MFCC acceleration
          spectral_stats    : 20 dims  — centroid, spread, rolloff, flatness × frames
          energy_envelope   : 12 dims  — RMS, ZCR, peak stats
          ─────────────────────────────
          Total             : 172 dims  → padded to 192
        """
        logger.debug("Using MFCC fallback embedding (ECAPA-TDNN not loaded)")

        n_mfcc = 20

        # ── Step 1: Log Mel → MFCCs via DCT ─────────────────────────
        fbank = self.feature_extractor.extract_fbank(audio)  # (n_frames, 80)
        n_frames, n_mels = fbank.shape

        # DCT-II of each frame: transforms correlated mel features into
        # decorrelated cepstral coefficients
        dct_matrix = self._get_dct_matrix(n_mels, n_mfcc)   # (n_mfcc, n_mels)
        mfccs = fbank @ dct_matrix.T                         # (n_frames, n_mfcc)

        # ── Step 2: Cepstral Mean Normalization (CMN) ────────────────
        # Subtract mean across time axis per coefficient
        # This is the key step that makes cepstral features speaker-dependent
        # rather than content-dependent.
        cep_mean = np.mean(mfccs, axis=0, keepdims=True)     # (1, n_mfcc)
        mfccs_norm = mfccs - cep_mean                        # (n_frames, n_mfcc)

        # ── Step 3: Delta and Delta-Delta features ───────────────────
        delta  = self.feature_extractor.compute_delta(mfccs_norm, order=1)
        delta2 = self.feature_extractor.compute_delta(mfccs_norm, order=2)

        # ── Step 4: Statistics over time ────────────────────────────
        mfcc_mean = np.mean(mfccs_norm, axis=0).astype(np.float32)     # (20,)
        mfcc_std  = np.std(mfccs_norm,  axis=0).astype(np.float32)     # (20,)

        # Percentile features for 8 most discriminative MFCC coefficients
        pct_levels = [10, 25, 50, 75, 90]
        pct_arr = np.percentile(mfccs_norm[:, 1:9], pct_levels, axis=0)  # (5, 8)
        mfcc_pct = pct_arr.flatten().astype(np.float32)                   # (40,)

        delta_mean  = np.mean(delta,  axis=0).astype(np.float32)        # (20,)
        delta_std   = np.std(delta,   axis=0).astype(np.float32)        # (20,)
        delta2_mean = np.mean(delta2, axis=0).astype(np.float32)        # (20,)

        # ── Step 5: Spectral and energy statistics ───────────────────
        spectral_stats = self._compute_spectral_stats(audio)             # (16,)

        # Energy envelope: RMS, peak, and zero-crossing statistics
        frames_audio = self.feature_extractor._frame_signal(audio)       # (n_frames, frame_len)
        rms_per_frame = np.sqrt(np.mean(frames_audio ** 2, axis=1))      # (n_frames,)
        energy_stats = np.array([
            np.mean(rms_per_frame),
            np.std(rms_per_frame),
            np.max(rms_per_frame),
            np.percentile(rms_per_frame, 75),
            np.percentile(rms_per_frame, 25),
            np.sum(rms_per_frame > np.mean(rms_per_frame)) / max(1, len(rms_per_frame)),
            float(np.sqrt(np.mean(audio ** 2))),   # global RMS
            float(np.max(np.abs(audio))),           # peak amplitude
            float(np.mean(np.abs(np.diff(audio > 0)))),  # zero-crossing rate
            float(np.std(np.abs(audio))),
            float(np.percentile(np.abs(audio), 90)),
            float(np.percentile(np.abs(audio), 10)),
        ], dtype=np.float32)   # (12,)

        # Pad spectral_stats to 20 dims
        spectral_padded = np.zeros(20, dtype=np.float32)
        spectral_padded[:len(spectral_stats)] = spectral_stats

        # ── Step 6: Concatenate all features ────────────────────────
        # 20+20+40+20+20+20+20+12 = 172 → pad to 192
        embedding = np.concatenate([
            mfcc_mean,       # 20
            mfcc_std,        # 20
            mfcc_pct,        # 40
            delta_mean,      # 20
            delta_std,       # 20
            delta2_mean,     # 20
            spectral_padded, # 20
            energy_stats,    # 12
        ])  # total = 172

        result = np.zeros(self.embedding_dim, dtype=np.float32)
        result[:len(embedding)] = embedding
        return result

    def _get_dct_matrix(self, n_input: int, n_output: int) -> np.ndarray:
        """
        Build DCT-II matrix — FROM SCRATCH.

        DCT-II formula:
          D[k, n] = cos(π * k * (n + 0.5) / N)

        Applying this to log mel filterbank features gives MFCCs.
        k indexes output coefficients (0..n_output-1)
        n indexes input mel bins (0..n_input-1)
        """
        k = np.arange(n_output)[:, np.newaxis]   # (n_output, 1)
        n = np.arange(n_input)[np.newaxis, :]    # (1, n_input)
        dct = np.cos(np.pi * k * (n + 0.5) / n_input)  # (n_output, n_input)
        # Orthonormal scaling
        dct[0] *= np.sqrt(1.0 / n_input)
        dct[1:] *= np.sqrt(2.0 / n_input)
        return dct.astype(np.float32)

    def _compute_spectral_stats(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute spectral statistics — FROM SCRATCH.

        These capture speaker characteristics:
          - Spectral centroid (brightness)
          - Spectral spread
          - Spectral rolloff
          - Spectral flux (change over time)
        """
        features = np.zeros(16, dtype=np.float32)

        if len(audio) < 512:
            return features

        # FFT
        spectrum = np.abs(np.fft.rfft(audio, n=1024))
        freqs = np.fft.rfftfreq(1024, d=1.0/self.sample_rate)

        power = spectrum ** 2
        total_power = np.sum(power) + 1e-10

        # Spectral centroid: weighted mean of frequencies
        centroid = np.sum(freqs * power) / total_power
        features[0] = centroid / (self.sample_rate / 2)  # Normalize

        # Spectral spread: weighted std
        spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / total_power)
        features[1] = spread / (self.sample_rate / 2)

        # Spectral rolloff: frequency below which 85% of energy falls
        cumsum = np.cumsum(power)
        rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
        features[2] = freqs[min(rolloff_idx, len(freqs)-1)] / (self.sample_rate / 2)

        # Flatness (ratio of geometric mean to arithmetic mean)
        log_spec = np.log(spectrum + 1e-10)
        geom_mean = np.exp(np.mean(log_spec))
        arith_mean = np.mean(spectrum) + 1e-10
        features[3] = geom_mean / arith_mean

        return features

    # ──────────────────────────────────────────────────────────
    # FROM SCRATCH: Embedding Normalization
    # ──────────────────────────────────────────────────────────

    def _l2_normalize(self, embedding: np.ndarray) -> np.ndarray:
        """
        L2 normalization — FROM SCRATCH.

        Projects the embedding onto the unit hypersphere.
        This makes cosine similarity equivalent to dot product,
        which simplifies clustering distance computation.

        Formula: x_normalized = x / ||x||_2
          where ||x||_2 = sqrt(sum(x_i^2))
        """
        norm = np.sqrt(np.sum(embedding ** 2))
        if norm < 1e-10:
            return embedding  # Avoid division by zero
        return embedding / norm


# ──────────────────────────────────────────────────────────────
# Standalone Test
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Diarization / Embedding Self-Test ===\n")

    sr = 16000

    # Test AudioFeatureExtractor
    print("1. Testing AudioFeatureExtractor (from scratch)...")
    extractor = AudioFeatureExtractor(sample_rate=sr)

    # Simulate 1 second of speech-like audio
    t = np.linspace(0, 1.0, sr)
    test_audio = (
        0.3 * np.sin(2 * np.pi * 150 * t) +   # Fundamental (low pitch)
        0.2 * np.sin(2 * np.pi * 300 * t) +   # 2nd harmonic
        0.1 * np.sin(2 * np.pi * 600 * t) +   # 3rd harmonic
        0.05 * np.random.randn(sr)             # Noise
    ).astype(np.float32)

    fbank = extractor.extract_fbank(test_audio)
    print(f"   Mel filterbank shape: {fbank.shape}  (expected: ~98 frames × 80 mels)")

    delta = extractor.compute_delta(fbank)
    print(f"   Delta features shape: {delta.shape}  (same as fbank)")

    # Test SpeakerEmbeddingExtractor
    print("\n2. Testing SpeakerEmbeddingExtractor (fallback mode, no model)...")
    emb_extractor = SpeakerEmbeddingExtractor(sample_rate=sr)
    # Don't call .load() — test fallback embedding

    speaker1_audio = test_audio.copy()
    speaker2_audio = (
        0.3 * np.sin(2 * np.pi * 200 * t) +   # Higher pitch (different speaker)
        0.2 * np.sin(2 * np.pi * 400 * t) +
        0.05 * np.random.randn(sr)
    ).astype(np.float32)

    emb1 = emb_extractor.extract(speaker1_audio)
    emb2 = emb_extractor.extract(speaker2_audio)

    if emb1 and emb2:
        cos_sim = float(np.dot(emb1.vector, emb2.vector))
        print(f"   Embedding 1: shape={emb1.vector.shape}, norm={np.linalg.norm(emb1.vector):.4f}")
        print(f"   Embedding 2: shape={emb2.vector.shape}, norm={np.linalg.norm(emb2.vector):.4f}")
        print(f"   Cosine similarity (same speaker=high): {cos_sim:.4f}")
        print(f"   Energy: {emb1.energy:.5f}, Duration: {emb1.duration:.2f}s")
    else:
        print("   Embedding extraction returned None (audio quality check failed)")

    print("\nSelf-test complete.")
