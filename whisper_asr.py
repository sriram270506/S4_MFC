"""
whisper_asr.py — Automatic Speech Recognition (ASR)
====================================================

This module handles speech-to-text conversion using Faster-Whisper.

WHAT IS FASTER-WHISPER?
  Faster-Whisper is an optimized version of OpenAI's Whisper model.
  It uses CTranslate2 for faster inference and lower memory usage.
  On CPU: ~3-5x faster than original Whisper.

WHY WHISPER?
  - State-of-the-art accuracy for ASR
  - Handles multiple languages automatically
  - Works well with varied microphone quality
  - Outputs word-level timestamps (via WhisperX)

MODEL SIZES:
  tiny   → fastest, lowest accuracy (~39M params)
  base   → good balance (~74M params)
  small  → better accuracy (~244M params) ← recommended for laptops
  medium → high accuracy (~769M params)  ← if you have GPU
  large  → best accuracy (~1.5B params)  ← need GPU

WE IMPLEMENT FROM SCRATCH:
  - Audio validation before sending to Whisper
  - Subtitle merging logic
  - Word timestamp alignment
  - Confidence-based filtering
"""

import numpy as np
import logging
import time
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

logger = logging.getLogger("whisper_asr")


# ──────────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────────

@dataclass
class TranscriptionResult:
    """Result from ASR for a single audio chunk."""
    text: str                          # Transcribed text
    language: str                      # Detected language (e.g., "en")
    confidence: float                  # Average token probability
    words: List[Dict]                  # Word-level timestamps
    processing_time_ms: float          # How long transcription took


@dataclass
class WordInfo:
    """A single word with timing information."""
    word: str
    start: float    # Start time relative to chunk start (seconds)
    end: float      # End time
    probability: float  # Confidence for this word


# ══════════════════════════════════════════════════════════════
# FASTER-WHISPER ASR ENGINE
# ══════════════════════════════════════════════════════════════

class WhisperASR:
    """
    ASR using Faster-Whisper model.

    Handles:
      - Model loading and initialization
      - Audio validation before sending to model
      - Transcription with word timestamps
      - Confidence filtering (reject low-quality transcriptions)
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",    # int8 = quantized (faster, smaller)
        language: Optional[str] = "en",
        beam_size: int = 5,
        min_confidence: float = 0.45   # Reject transcriptions below this confidence
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.min_confidence = min_confidence

        self.model = None
        self._loaded = False

        # Rolling context prompt for Whisper (reduces hallucinations
        # and improves word-boundary accuracy across chunks)
        self._context_prompt = ""

        # Statistics
        self._n_transcriptions = 0
        self._total_audio_sec = 0.0
        self._total_proc_ms = 0.0

        logger.info(
            f"WhisperASR: model={model_size}, device={device}, "
            f"compute_type={compute_type}, language={language}"
        )

    def load(self):
        """Load the Faster-Whisper model."""
        try:
            logger.info(f"Loading Faster-Whisper ({self.model_size})...")
            from faster_whisper import WhisperModel

            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root="models/whisper"
            )
            self._loaded = True
            logger.info("Faster-Whisper loaded successfully.")
        except ImportError:
            logger.error("faster_whisper not installed. Run: pip install faster-whisper")
            self._loaded = False
        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")
            self._loaded = False

    # ──────────────────────────────────────────────────────────
    # Main Transcription
    # ──────────────────────────────────────────────────────────

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> Optional[TranscriptionResult]:
        """
        Transcribe audio chunk to text.

        Parameters:
        -----------
        audio       : np.ndarray — float32, shape (N,), 16kHz
        sample_rate : int        — Should always be 16000 for Whisper

        Returns:
        --------
        TranscriptionResult or None (if audio too short/quiet or model error)
        """
        # Step 1: Validate audio (our code)
        valid, reason = self._validate_audio(audio, sample_rate)
        if not valid:
            logger.debug(f"Transcription skipped: {reason}")
            return None

        start_time = time.time()

        if not self._loaded or self.model is None:
            # Demo mode: return placeholder
            return self._demo_transcription(audio)

        try:
            result = self._run_whisper(audio, sample_rate)
            proc_ms = (time.time() - start_time) * 1000

            if result:
                result.processing_time_ms = proc_ms
                self._update_stats(len(audio) / sample_rate, proc_ms)

            return result

        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            return None

    # ──────────────────────────────────────────────────────────
    # FROM SCRATCH: Audio Validation
    # ──────────────────────────────────────────────────────────

    def _validate_audio(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> Tuple[bool, str]:
        """
        Validate audio before sending to Whisper — FROM SCRATCH.

        Whisper is slow and expensive. We pre-screen audio to avoid
        wasting compute on silence or garbage inputs.

        Checks:
          1. Minimum length (0.5 seconds)
          2. Minimum energy (not silence)
          3. Sample rate is correct
          4. No NaN/Inf values
        """
        # Check 1: Length
        min_samples = int(sample_rate * 0.5)
        if len(audio) < min_samples:
            return False, f"Too short: {len(audio)} samples (need {min_samples})"

        # Check 2: Energy (silence detection)
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 0.004:
            return False, f"Too quiet: RMS={rms:.5f}"

        # Check 3: Sample rate hint
        if sample_rate != 16000:
            logger.warning(f"Expected 16000 Hz, got {sample_rate} Hz. Whisper may give poor results.")

        # Check 4: NaN/Inf safety
        if not np.isfinite(audio).all():
            return False, "Audio contains NaN or Inf values"

        return True, "OK"

    # ──────────────────────────────────────────────────────────
    # Whisper Inference
    # ──────────────────────────────────────────────────────────

    def _run_whisper(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> Optional[TranscriptionResult]:
        """Run faster-whisper inference and return structured result."""

        # Apply pre-emphasis here (and ONLY here) — FROM SCRATCH.
        # Formula: y[n] = x[n] - 0.97 * x[n-1]
        # Applied once over the full continuous chunk so there are no
        # per-burst spectral discontinuities (unlike audio_stream which
        # used to apply it per incoming browser packet).
        # Pre-emphasis boosts high-frequency consonants (s, t, f, k, p)
        # which Whisper's internal Mel filterbank would otherwise underweight.
        audio_for_asr = np.append(audio[0], audio[1:] - 0.97 * audio[:-1]).astype(np.float32)

        segments, info = self.model.transcribe(
            audio_for_asr,
            language=self.language,
            beam_size=self.beam_size,
            word_timestamps=True,
            # ── ANTI-HALLUCINATION SETTINGS ─────────────────────────────────
            # temperature=0.0 forces greedy decoding.  Temperatures > 0 make
            # Whisper sample tokens randomly; on short/noisy audio this
            # produces fluent but completely wrong text (e.g. "Thank you for
            # watching").  Never use a temperature list here.
            temperature=0.0,
            # condition_on_previous_text=True feeds the rolling context prompt
            # back into each chunk.  Once hallucinated text enters the context,
            # it poisons every subsequent chunk because Whisper "continues"
            # from the bad text.  Must be False for streaming live speech.
            condition_on_previous_text=False,
            initial_prompt=None,          # No prompt — avoids context poisoning
            # Run Whisper's internal VAD as a second gate on top of our own.
            # This suppresses hallucinations on silent sections that slipped
            # through Energy/Silero VAD.
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=400,
                threshold=0.35,
            ),
            # ── QUALITY GATES ───────────────────────────────────────────────
            compression_ratio_threshold=2.0,   # Low ratio = repetitive = likely hallucination
            no_speech_threshold=0.6,            # Reject low-speech-probability segments
            log_prob_threshold=-1.0,            # Reject very low log-prob sequences
        )

        # Collect all segments
        all_text_parts = []
        all_words = []
        all_probs = []

        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue

            all_text_parts.append(text)

            # Extract word-level info
            if hasattr(segment, 'words') and segment.words:
                for word in segment.words:
                    all_words.append({
                        "word": word.word,
                        "start": round(float(word.start), 3),
                        "end": round(float(word.end), 3),
                        "probability": round(float(word.probability), 3)
                    })
                    all_probs.append(word.probability)
            else:
                # No word timestamps available
                all_words.append({
                    "word": text,
                    "start": 0.0,
                    "end": len(audio) / sample_rate,
                    "probability": 0.8
                })
                all_probs.append(0.8)

        if not all_text_parts:
            return None

        # Compute average confidence
        avg_confidence = float(np.mean(all_probs)) if all_probs else 0.0

        full_text = " ".join(all_text_parts)

        # Clean up text (our code)
        full_text = self._clean_text(full_text)

        if not full_text:
            return None

        # Filter low-confidence results
        if avg_confidence < self.min_confidence:
            logger.debug(f"Low confidence transcription ({avg_confidence:.2f}), skipping.")
            return None

        # NOTE: We intentionally do NOT maintain a rolling context prompt.
        # condition_on_previous_text=False makes the context prompt unused anyway,
        # and storing hallucinated text would poison future chunks.

        return TranscriptionResult(
            text=full_text,
            language=info.language,
            confidence=avg_confidence,
            words=all_words,
            processing_time_ms=0.0
        )

    # ──────────────────────────────────────────────────────────
    # FROM SCRATCH: Text Cleaning
    # ──────────────────────────────────────────────────────────

    def _clean_text(self, text: str) -> str:
        """
        Clean up transcription artifacts and hallucinations — FROM SCRATCH.

        Whisper is trained on internet audio and has memorised common
        patterns that it hallucinates onto unclear or silent audio:
          • YouTube sign-off phrases ("Thanks for watching", "Subscribe", ...)
          • Music/noise markers ([BLANK_AUDIO], (music), ...)
          • Filler repetition ("Hello, hello, hello, hello, ...")

        We filter all of these out before displaying to the user.
        """
        import re

        # ── 1. Remove Whisper bracket/paren tokens ───────────────────────
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\(.*?\)', '', text)

        # ── 2. Known Whisper hallucination phrases ───────────────────────
        # These are strings Whisper outputs on silence/noise; any segment
        # that contains one of these (case-insensitive) is discarded entirely.
        HALLUCINATION_BLOCKLIST = [
            "thank you for watching",
            "thanks for watching",
            "please subscribe",
            "don't forget to subscribe",
            "like and subscribe",
            "hit the subscribe",
            "smash the like button",
            "in the next video",
            "see you next time",
            "see you in the next",
            "bye bye",
            "goodbye",
            "translate by",
            "subtitles by",
            "transcribed by",
            "www.",
            ".com",
            "subtitled by",
            "amara.org",
        ]
        text_lower = text.lower()
        for phrase in HALLUCINATION_BLOCKLIST:
            if phrase in text_lower:
                logger.debug(f"Hallucination blocked: '{text[:60]}'")
                return ''

        # ── 3. Repetition detection — FROM SCRATCH ──────────────────────
        # Whisper sometimes loops: "Hello Hello Hello Hello Hello"
        # or spells acronyms as letters: "m q t t is equal to m q t t"
        words = text.split()
        if len(words) >= 4:
            from collections import Counter

            # (a) Single-word repetition
            word_counts = Counter(w.lower().strip('.,!?') for w in words)
            most_common_word, most_common_count = word_counts.most_common(1)[0]
            word_rep_ratio = most_common_count / len(words)

            # (b) Single-character token ratio (catches letter-by-letter acronym loops)
            # Common real single-char words: a, i — allow those
            single_char_tokens = [w for w in words
                                   if len(w.strip('.,!?')) == 1
                                   and w.lower().strip('.,!?') not in {'a', 'i'}]
            single_char_ratio = len(single_char_tokens) / len(words)

            # (c) Bigram repetition — catches "m q t t" repeating
            bigrams = [(words[i].lower(), words[i+1].lower()) for i in range(len(words)-1)]
            bigram_counts = Counter(bigrams)
            most_common_bigram_count = bigram_counts.most_common(1)[0][1] if bigrams else 0
            bigram_rep_ratio = most_common_bigram_count / max(1, len(bigrams))

            if word_rep_ratio > 0.4 and len(words) >= 4:
                logger.debug(
                    f"Word repetition: '{most_common_word}' "
                    f"{most_common_count}/{len(words)} ({word_rep_ratio:.0%}). Skipping."
                )
                return ''
            if single_char_ratio > 0.35 and len(words) >= 6:
                logger.debug(
                    f"Single-char token loop detected "
                    f"({single_char_ratio:.0%} single-char tokens). Skipping."
                )
                return ''
            if bigram_rep_ratio > 0.4 and len(words) >= 6:
                logger.debug(
                    f"Bigram repetition detected (ratio={bigram_rep_ratio:.0%}). Skipping."
                )
                return ''

        # ── 4. Whitespace cleanup ────────────────────────────────────────
        text = re.sub(r'\s+', ' ', text).strip()

        # ── 5. Remove standalone punctuation / very short remnants ───────
        if text in {'.', ',', '!', '?', '...', '-', '--'}:
            return ''
        if len(text) < 2:
            return ''

        return text

    # ──────────────────────────────────────────────────────────
    # FROM SCRATCH: Word Timestamp Alignment
    # ──────────────────────────────────────────────────────────

    def align_words_to_timeline(
        self,
        words: List[Dict],
        chunk_offset_sec: float
    ) -> List[Dict]:
        """
        Adjust word timestamps to absolute timeline — FROM SCRATCH.

        Whisper gives timestamps relative to the start of each chunk.
        We adjust them to be relative to the start of the session.

        Parameters:
        -----------
        words            : list of word dicts with "start", "end"
        chunk_offset_sec : float — start time of this chunk in session

        Returns:
        --------
        list of word dicts with adjusted timestamps
        """
        adjusted = []
        for word in words:
            adjusted.append({
                **word,
                "start": round(word["start"] + chunk_offset_sec, 3),
                "end": round(word["end"] + chunk_offset_sec, 3),
            })
        return adjusted

    # ──────────────────────────────────────────────────────────
    # Demo Mode
    # ──────────────────────────────────────────────────────────

    def _demo_transcription(self, audio: np.ndarray) -> TranscriptionResult:
        """
        Return fake transcription for UI testing — no model needed.
        Used when Whisper isn't loaded.
        """
        demo_phrases = [
            "Hello everyone, welcome to the meeting.",
            "Good morning, can everyone hear me?",
            "Let's get started with the agenda.",
            "I agree with that point.",
            "Could you elaborate on that?",
            "That's a great idea.",
            "Let me check the numbers.",
            "I have a question about that.",
        ]
        import random
        phrase = random.choice(demo_phrases)

        return TranscriptionResult(
            text=phrase,
            language="en",
            confidence=0.85,
            words=[{"word": w, "start": i*0.3, "end": (i+1)*0.3, "probability": 0.85}
                   for i, w in enumerate(phrase.split())],
            processing_time_ms=50.0
        )

    # ──────────────────────────────────────────────────────────
    # Statistics
    # ──────────────────────────────────────────────────────────

    def reset(self):
        """Reset rolling context — call between sessions to avoid cross-session contamination."""
        self._context_prompt = ""
        logger.info("WhisperASR context prompt reset.")

    def _update_stats(self, audio_sec: float, proc_ms: float):
        self._n_transcriptions += 1
        self._total_audio_sec += audio_sec
        self._total_proc_ms += proc_ms

    def get_stats(self) -> dict:
        """Return ASR performance statistics."""
        avg_proc = self._total_proc_ms / max(1, self._n_transcriptions)
        rtf = (self._total_proc_ms / 1000) / max(1e-6, self._total_audio_sec)
        return {
            "n_transcriptions": self._n_transcriptions,
            "total_audio_sec": round(self._total_audio_sec, 1),
            "avg_processing_ms": round(avg_proc, 1),
            "real_time_factor": round(rtf, 3),  # < 1.0 means faster than real-time
            "model_loaded": self._loaded,
            "model_size": self.model_size,
        }


# ──────────────────────────────────────────────────────────────
# Standalone Test
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== WhisperASR Self-Test ===\n")

    sr = 16000
    asr = WhisperASR(model_size="base", device="cpu")

    # Test 1: Validation checks
    print("1. Audio validation tests:")

    # Too short
    short_audio = np.zeros(1000, dtype=np.float32)
    valid, reason = asr._validate_audio(short_audio, sr)
    print(f"   Short audio → valid={valid}, reason='{reason}'")

    # Silent
    silent = np.zeros(sr, dtype=np.float32)
    valid, reason = asr._validate_audio(silent, sr)
    print(f"   Silent audio → valid={valid}, reason='{reason}'")

    # Good audio
    good = (np.random.randn(sr) * 0.3).astype(np.float32)
    valid, reason = asr._validate_audio(good, sr)
    print(f"   Good audio → valid={valid}, reason='{reason}'")

    # Test 2: Text cleaning
    print("\n2. Text cleaning tests:")
    test_texts = [
        "[BLANK_AUDIO]",
        "(music) Hello everyone",
        "  Hello   world  ",
        ".",
        "[MUSIC] What did you say? [LAUGHTER]",
    ]
    for t in test_texts:
        cleaned = asr._clean_text(t)
        print(f"   '{t}' → '{cleaned}'")

    # Test 3: Demo transcription (no model needed)
    print("\n3. Demo transcription (no model):")
    demo = asr._demo_transcription(good)
    print(f"   Text: '{demo.text}'")
    print(f"   Confidence: {demo.confidence}")
    print(f"   Words: {len(demo.words)}")

    # Test 4: Word timestamp alignment
    print("\n4. Word timestamp alignment:")
    words = [
        {"word": "Hello", "start": 0.1, "end": 0.4, "probability": 0.9},
        {"word": "world", "start": 0.5, "end": 0.8, "probability": 0.85},
    ]
    aligned = asr.align_words_to_timeline(words, chunk_offset_sec=5.0)
    for w in aligned:
        print(f"   '{w['word']}': {w['start']:.2f}s – {w['end']:.2f}s (adjusted by +5s)")

    print("\nSelf-test complete.")
