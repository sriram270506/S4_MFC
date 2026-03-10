"""
process_audio.py — Standalone Audio Processor
===============================================

Runs the FULL diarization pipeline on EITHER:
  (A) A recorded audio file (.wav, .mp3, .flac, .ogg, .m4a)
  (B) Live microphone input (no browser/WebSocket needed)

Output: Printed to terminal in real time.

Usage:
------
  # Process a recorded file:
  python process_audio.py --file meeting.wav

  # Live microphone (press Ctrl+C to stop):
  python process_audio.py --mic

  # Recorded file with options:
  python process_audio.py --file audio.wav --model small --speakers 4

  # Show all options:
  python process_audio.py --help

How it works:
-------------
  This script reuses ALL the backend modules (vad, diarization,
  clustering, whisper_asr, pipeline) but instead of a WebSocket,
  it reads audio from a file or mic and prints results directly.

  File mode:   reads file → splits into 1.5s chunks → pipeline → print
  Mic mode:    opens PyAudio stream → chunks → pipeline → print

FROM SCRATCH in this file:
  - Audio file loading + resampling to 16kHz
  - File chunking loop
  - PyAudio mic capture loop
  - Terminal color printing (ANSI codes)
  - Progress bar for file mode
  - Real-time latency measurement
"""

import os
import sys
import time
import argparse
import threading
import signal
import numpy as np

# ── Add backend/ to Python path so we can import our modules ──
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR  = os.path.join(SCRIPT_DIR, "backend")
sys.path.insert(0, BACKEND_DIR)

# ── Import our pipeline modules ──
from vad        import CombinedVAD
from diarization import SpeakerEmbeddingExtractor
from clustering  import OnlineSpeakerClusterer
from whisper_asr import WhisperASR
from pipeline    import SpeechPipeline, SubtitleBuffer, SubtitleResult

# ══════════════════════════════════════════════════════════
# TERMINAL COLOR PRINTER — FROM SCRATCH
# ══════════════════════════════════════════════════════════

# ANSI escape codes for terminal colors
ANSI_RESET  = "\033[0m"
ANSI_BOLD   = "\033[1m"
ANSI_DIM    = "\033[2m"

# 8 speaker colors mapped to ANSI terminal colors
SPEAKER_ANSI = [
    "\033[96m",   # Speaker 1 — Cyan
    "\033[92m",   # Speaker 2 — Green
    "\033[93m",   # Speaker 3 — Yellow
    "\033[95m",   # Speaker 4 — Magenta
    "\033[94m",   # Speaker 5 — Blue
    "\033[91m",   # Speaker 6 — Red
    "\033[97m",   # Speaker 7 — White
    "\033[33m",   # Speaker 8 — Dark Yellow
]

ANSI_CYAN   = "\033[96m"
ANSI_GREEN  = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_RED    = "\033[91m"
ANSI_GREY   = "\033[90m"
ANSI_WHITE  = "\033[97m"


def speaker_color(speaker_id: int) -> str:
    """Return ANSI color for a speaker index — FROM SCRATCH."""
    return SPEAKER_ANSI[speaker_id % len(SPEAKER_ANSI)]


def print_subtitle(result: dict):
    """
    Print a single subtitle line to terminal with color — FROM SCRATCH.

    Format:
      [00:03]  Speaker 1 │ Hello everyone, welcome to the meeting.
    """
    sid   = result.get("speaker_id", 0)
    color = speaker_color(sid)
    ts    = result.get("timestamp", "??:??")
    spk   = result.get("speaker",   "Speaker ?")
    text  = result.get("text",      "")
    conf  = result.get("confidence", 0.0)
    partial = result.get("is_partial", False)

    # Timestamp in grey
    ts_str  = f"{ANSI_GREY}[{ts}]{ANSI_RESET}"

    # Speaker name in speaker color + bold
    spk_str = f"{color}{ANSI_BOLD}{spk:<12}{ANSI_RESET}"

    # Text — dim if partial
    if partial:
        text_str = f"{ANSI_DIM}{text}…{ANSI_RESET}"
    else:
        text_str = f"{ANSI_WHITE}{text}{ANSI_RESET}"

    # Confidence in grey (only show if not 100%)
    conf_str = f"  {ANSI_GREY}({int(conf*100)}%){ANSI_RESET}" if conf < 0.99 else ""

    print(f"  {ts_str}  {spk_str} │ {text_str}{conf_str}")


def print_header(title: str):
    """Print a styled section header."""
    width = 60
    print()
    print(f"{ANSI_CYAN}{'─' * width}{ANSI_RESET}")
    print(f"{ANSI_CYAN}{ANSI_BOLD}  {title}{ANSI_RESET}")
    print(f"{ANSI_CYAN}{'─' * width}{ANSI_RESET}")


def print_speaker_summary(pipeline: SpeechPipeline):
    """Print end-of-session speaker statistics."""
    stats = pipeline.get_session_stats()
    speakers = stats.get("speakers", [])

    print()
    print(f"{ANSI_CYAN}{'─' * 60}{ANSI_RESET}")
    print(f"{ANSI_BOLD}  Session Summary{ANSI_RESET}")
    print(f"{ANSI_CYAN}{'─' * 60}{ANSI_RESET}")
    print(f"  Duration:       {ANSI_WHITE}{stats.get('elapsed_sec', 0):.1f}s{ANSI_RESET}")
    print(f"  Chunks:         {ANSI_WHITE}{stats.get('chunks_processed', 0)}{ANSI_RESET}")
    print(f"  Speech ratio:   {ANSI_WHITE}{(1-stats.get('silence_ratio',0))*100:.0f}%{ANSI_RESET}")
    print(f"  Speakers found: {ANSI_WHITE}{stats.get('n_speakers_found', 0)}{ANSI_RESET}")

    if speakers:
        print()
        print(f"  {'Speaker':<14} {'Segments':>8}  {'~Duration':>10}")
        print(f"  {'─'*14}  {'─'*8}  {'─'*10}")
        for s in speakers:
            color = speaker_color(s["speaker_id"])
            label = f"{color}{s['label']}{ANSI_RESET}"
            print(f"  {label:<24} {s['n_segments']:>8}  {s['total_duration_sec']:>8.1f}s")

    asr = stats.get("asr", {})
    if asr.get("n_transcriptions", 0) > 0:
        print()
        print(f"  ASR ({asr['model_size']}):  "
              f"avg {asr['avg_processing_ms']:.0f}ms/chunk  "
              f"RTF={asr['real_time_factor']:.2f}")
    print(f"{ANSI_CYAN}{'─' * 60}{ANSI_RESET}")
    print()


def progress_bar(current: int, total: int, width: int = 30) -> str:
    """
    Build a simple ASCII progress bar — FROM SCRATCH.

    Example:  [████████████░░░░░░░░]  45%
    """
    if total <= 0:
        return ""
    ratio   = min(1.0, current / total)
    filled  = int(ratio * width)
    bar     = "█" * filled + "░" * (width - filled)
    pct     = int(ratio * 100)
    return f"[{bar}] {pct:3d}%"


# ══════════════════════════════════════════════════════════
# AUDIO FILE LOADING — FROM SCRATCH (with soundfile/librosa)
# ══════════════════════════════════════════════════════════

def load_audio_file(filepath: str, target_sr: int = 16000) -> tuple:
    """
    Load any audio file and resample to target sample rate — FROM SCRATCH logic.

    Supports: .wav, .mp3, .flac, .ogg, .m4a (via soundfile + librosa)

    Steps (our code):
      1. Try soundfile first (fastest, no ffmpeg needed)
      2. Fall back to librosa (handles mp3, m4a via ffmpeg)
      3. Convert to mono by averaging channels (our code)
      4. Resample to 16kHz (our code using linear interpolation)
      5. Normalize amplitude

    Parameters:
    -----------
    filepath  : str — Path to audio file
    target_sr : int — Target sample rate (16000 for Whisper)

    Returns:
    --------
    (audio: np.ndarray, sample_rate: int, duration_sec: float)
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    print(f"  Loading: {ANSI_WHITE}{filepath}{ANSI_RESET}  ({ext})")

    audio = None
    sr    = None

    # ── Try soundfile (wav, flac, ogg) ───────────────────
    try:
        import soundfile as sf
        audio, sr = sf.read(filepath, dtype='float32', always_2d=False)
        print(f"  Loaded via soundfile: {len(audio)} samples @ {sr}Hz")
    except Exception as e:
        print(f"  soundfile failed ({e}), trying librosa...")

    # ── Fall back to librosa (mp3, m4a, etc.) ────────────
    if audio is None:
        try:
            import librosa
            audio, sr = librosa.load(filepath, sr=None, mono=False, dtype=np.float32)
            print(f"  Loaded via librosa: shape={audio.shape} @ {sr}Hz")
        except Exception as e:
            raise RuntimeError(f"Could not load audio file: {e}\n"
                               "Install: pip install soundfile librosa")

    # ── Convert to mono — FROM SCRATCH ───────────────────
    if audio.ndim == 2:
        # Shape is (channels, samples) or (samples, channels)
        if audio.shape[0] < audio.shape[1]:
            # (channels, samples) — average channels
            audio = np.mean(audio, axis=0)
        else:
            # (samples, channels)
            audio = np.mean(audio, axis=1)
        print(f"  Converted to mono")

    audio = audio.astype(np.float32)

    # ── Resample to target_sr — FROM SCRATCH ─────────────
    if sr != target_sr:
        audio = resample_audio(audio, sr, target_sr)
        print(f"  Resampled: {sr}Hz → {target_sr}Hz")
        sr = target_sr

    # ── Normalize amplitude ───────────────────────────────
    max_val = np.max(np.abs(audio))
    if max_val > 1e-6:
        audio = audio / max_val * 0.95
    audio = np.clip(audio, -1.0, 1.0)

    duration = len(audio) / sr
    print(f"  Duration: {duration:.2f}s  |  Samples: {len(audio)}  |  SR: {sr}Hz")

    return audio, sr, duration


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio using linear interpolation — FROM SCRATCH.

    This is a simple but functional resampling method.
    For production, scipy.signal.resample is better, but this shows
    the concept clearly.

    HOW IT WORKS:
      We compute the time axis for original and target, then
      interpolate the signal at the new time points.

      orig_times  = [0, 1/orig_sr, 2/orig_sr, ...]
      target_times = [0, 1/target_sr, 2/target_sr, ...]

      For each target time, find position in orig and interpolate.

    Parameters:
    -----------
    audio     : np.ndarray — Original audio
    orig_sr   : int        — Original sample rate
    target_sr : int        — Target sample rate

    Returns:
    --------
    np.ndarray — Resampled audio
    """
    # Try scipy first (better quality)
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g  = gcd(orig_sr, target_sr)
        up = target_sr // g
        dn = orig_sr   // g
        return resample_poly(audio, up, dn).astype(np.float32)
    except ImportError:
        pass

    # Fallback: linear interpolation (our own)
    duration    = len(audio) / orig_sr
    orig_times  = np.linspace(0, duration, len(audio),        endpoint=False)
    target_times = np.linspace(0, duration, int(duration * target_sr), endpoint=False)
    resampled   = np.interp(target_times, orig_times, audio)
    return resampled.astype(np.float32)


# ══════════════════════════════════════════════════════════
# MODE A: PROCESS RECORDED FILE — TWO-PASS APPROACH
# ══════════════════════════════════════════════════════════

def process_file(
    filepath:       str,
    model_size:     str | None = None,
    chunk_sec:      float = 1.5,
    max_speakers:   int   = 8,
    known_speakers: int   = 0
):
    """
    TWO-PASS file processing for accurate speaker diarization.

    WHY TWO PASSES?
      Online clustering (one-chunk-at-a-time) drifts because the SAME
      speaker sounds different at the start vs end of a recording.
      Early chunks set centroids that later chunks miss.

      With a full file we can do better:
        PASS 1 — Extract ALL embeddings + run VAD (fast, no ASR)
        PASS 2 — Batch cluster ALL embeddings at once (far more accurate)
               — Then run ASR with the correct speaker labels

    FROM SCRATCH in this function:
      - Two-pass loop structure
      - Embedding collection array
      - Batch cosine distance matrix for final clustering
      - Re-labelling loop after batch cluster
    """
    print_header(f"FILE MODE (two-pass) — {os.path.basename(filepath)}")

    # ── Step 1: Load audio ────────────────────────────────
    try:
        audio, sr, duration = load_audio_file(filepath, target_sr=16000)
    except Exception as e:
        print(f"\n{ANSI_RED}  ERROR: {e}{ANSI_RESET}\n")
        return

    chunk_samples = int(sr * chunk_sec)
    total_chunks  = (len(audio) + chunk_samples - 1) // chunk_samples
    print(f"  Chunks: {total_chunks}  ({chunk_sec}s each)\n")

    # ── Step 2: Load models ───────────────────────────────
    print(f"  {ANSI_YELLOW}Loading ML models...{ANSI_RESET}")
    import asyncio, sys
    sys.path.insert(0, BACKEND_DIR)
    from vad         import CombinedVAD
    from diarization import SpeakerEmbeddingExtractor
    from clustering  import AgglomerativeClusterer, OnlineSpeakerClusterer
    from whisper_asr import WhisperASR

    vad      = CombinedVAD(sample_rate=sr);      vad.initialize()
    embedder = SpeakerEmbeddingExtractor(sample_rate=sr); embedder.load()

    runtime_probe = SpeechPipeline.__new__(SpeechPipeline)
    auto_model_size, _auto_fallback, asr_device, asr_compute_type, _fallback_compute_type = (
        runtime_probe._resolve_asr_runtime()
    )
    selected_model_size = model_size or auto_model_size

    print(
        f"  ASR runtime: model={selected_model_size} device={asr_device} "
        f"compute_type={asr_compute_type}"
    )
    asr = WhisperASR(
        model_size=selected_model_size,
        device=asr_device,
        compute_type=asr_compute_type,
        language="en",
    ); asr.load()
    print(f"  {ANSI_GREEN}Models loaded.{ANSI_RESET}\n")

    # ── PASS 1: VAD + Embedding for every chunk ───────────
    print(f"  {ANSI_YELLOW}Pass 1/2 — VAD + speaker embeddings...{ANSI_RESET}")

    chunk_meta = []   # list of {idx, timestamp, is_speech, embedding, audio}
    all_embeddings = []  # only speech embeddings, for clustering

    for i in range(total_chunks):
        start = i * chunk_samples
        end   = min(start + chunk_samples, len(audio))
        chunk = audio[start:end]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

        ts = _format_ts(i * chunk_sec)

        bar = progress_bar(i, total_chunks)
        print(f"\r  {ANSI_GREY}{bar}  {i+1}/{total_chunks}{ANSI_RESET}",
              end="", flush=True)

        is_speech, _ = vad.detect(chunk)
        emb_result   = embedder.extract(chunk) if is_speech else None

        chunk_meta.append({
            "idx":       i,
            "timestamp": ts,
            "is_speech": is_speech,
            "embedding": emb_result.vector.copy() if emb_result else None,
            "audio":     chunk.copy(),
        })

        if emb_result is not None:
            all_embeddings.append((i, emb_result.vector.copy()))

    print("\r" + " " * 70 + "\r", end="")
    n_speech = sum(1 for c in chunk_meta if c["is_speech"])
    print(f"  Pass 1 done: {n_speech}/{total_chunks} chunks have speech\n")

    if not all_embeddings:
        print(f"  {ANSI_RED}No speech detected in file.{ANSI_RESET}")
        return

    # ── PASS 2a: Batch cluster ALL embeddings ────────────
    # This is the key improvement over online clustering:
    # we see ALL embeddings before deciding boundaries.
    print(f"  {ANSI_YELLOW}Pass 2/2 — Batch clustering + transcription...{ANSI_RESET}\n")

    emb_indices = [e[0] for e in all_embeddings]  # chunk indices
    emb_matrix  = np.array([e[1] for e in all_embeddings], dtype=np.float32)

    # Decide cluster count
    if known_speakers > 0:
        n_clusters = known_speakers
    else:
        # Auto-estimate: try values 1..max_speakers, pick best via silhouette
        n_clusters = _estimate_n_speakers(emb_matrix, max_speakers)

    batch_clusterer = AgglomerativeClusterer(
        n_clusters=n_clusters,
        linkage="average"
    )
    labels = batch_clusterer.fit_predict(emb_matrix)  # shape: (n_speech_chunks,)

    # ── Post-cluster cleanup: absorb noise clusters ──────
    # Any cluster with only 1 segment is almost certainly NOT a real speaker
    # (it's a noise spike, cough, background sound, etc.)
    # We remap those chunk labels to their nearest real cluster.
    # FROM SCRATCH:

    MIN_SEGS_FOR_REAL_SPEAKER = 2
    labels = list(labels)
    n_total = len(labels)

    # Count how many segments each raw label has
    from collections import Counter
    label_counts = Counter(labels)

    # Find which labels are "real" speakers (enough segments)
    real_labels = {lab for lab, cnt in label_counts.items()
                   if cnt >= MIN_SEGS_FOR_REAL_SPEAKER}

    if real_labels and len(real_labels) < len(label_counts):
        # Compute centroid for each label
        label_centroids = {}
        for lab in set(labels):
            indices = [i for i, l in enumerate(labels) if l == lab]
            vecs = emb_matrix[indices]
            c = vecs.mean(axis=0)
            c = c / (np.linalg.norm(c) + 1e-10)
            label_centroids[lab] = c

        # Remap noise labels → nearest real label by cosine similarity
        noise_labels = set(labels) - real_labels
        remap = {}
        for noise_lab in noise_labels:
            noise_c = label_centroids[noise_lab]
            best_real = max(real_labels,
                key=lambda r: float(np.dot(noise_c, label_centroids[r])))
            remap[noise_lab] = best_real

        labels = [remap.get(l, l) for l in labels]
        absorbed = len(noise_labels)
        print(f"  {ANSI_GREY}Absorbed {absorbed} noise cluster(s) into nearest real speaker{ANSI_RESET}")

    # Sort cluster IDs by first appearance (Speaker 1 = first to speak)
    label_order = {}
    for lab in labels:
        if lab not in label_order:
            label_order[lab] = len(label_order)
    ordered_labels = [label_order[l] for l in labels]

    chunk_speaker = {}  # chunk_idx → speaker_id
    for chunk_idx, spk_id in zip(emb_indices, ordered_labels):
        chunk_speaker[chunk_idx] = spk_id

    n_final_speakers = len(set(ordered_labels))
    print(f"  {ANSI_GREEN}Final speaker count: {n_final_speakers}{ANSI_RESET}\n")

    # Speaker colors (same palette as clustering.py)
    COLORS = ["#4FC3F7","#AED581","#FFB74D","#F48FB1",
              "#CE93D8","#80DEEA","#FFCC02","#FF8A65"]

    # ── PASS 2b: ASR with correct speaker labels ─────────
    print_header("Transcript")
    total_results = 0
    speaker_stats = {}  # speaker_id → {segments, duration}

    for i, meta in enumerate(chunk_meta):
        if not meta["is_speech"]:
            continue

        chunk      = meta["audio"]
        ts         = meta["timestamp"]
        chunk_idx  = meta["idx"]
        speaker_id = chunk_speaker.get(chunk_idx, 0)
        speaker    = f"Speaker {speaker_id + 1}"
        color      = COLORS[speaker_id % len(COLORS)]

        bar = progress_bar(i, total_chunks)
        print(f"\r  {ANSI_GREY}{bar}  {i+1}/{total_chunks}{ANSI_RESET}",
              end="", flush=True)

        asr_result = asr.transcribe(chunk, sample_rate=sr)
        if asr_result is None or not asr_result.text.strip():
            continue

        print("\r" + " " * 70 + "\r", end="")
        result = {
            "timestamp":  ts,
            "speaker":    speaker,
            "speaker_id": speaker_id,
            "text":       asr_result.text,
            "confidence": asr_result.confidence,
            "color":      color,
            "is_partial": False,
        }
        print_subtitle(result)
        total_results += 1

        if speaker_id not in speaker_stats:
            speaker_stats[speaker_id] = {"label": speaker, "segments": 0,
                                          "duration": 0.0, "color": color}
        speaker_stats[speaker_id]["segments"] += 1
        speaker_stats[speaker_id]["duration"] += chunk_sec

    print("\r" + " " * 70 + "\r", end="")

    if total_results == 0:
        print(f"  {ANSI_YELLOW}No transcriptions produced.{ANSI_RESET}")

    # ── Summary ───────────────────────────────────────────
    _print_file_summary(speaker_stats, asr, duration)


def _estimate_n_speakers(embeddings: np.ndarray, max_k: int) -> int:
    """
    Estimate number of speakers using Gap-like statistic on cosine distances.
    FROM SCRATCH — no sklearn needed.

    ALGORITHM:
      For each candidate k (1 to max_k):
        1. Cluster embeddings into k groups
        2. Compute mean INTRA-cluster cosine distance (lower = tighter = better)
        3. Compute mean INTER-cluster cosine distance (higher = more separated = better)
        4. Score = inter_dist - intra_dist  (maximise this)

      The k with the highest score is the best estimate.
      We also apply a minimum-segment filter: any cluster with fewer than
      MIN_SEGS segments is noise, so we subtract those from k.

    WHY THIS IS BETTER THAN ELBOW:
      The elbow method needs a visible "knee" in the curve, which is
      unreliable for ECAPA embeddings. This gap statistic is more robust.
    """
    from clustering import AgglomerativeClusterer

    n = len(embeddings)
    if n <= 2:
        return 1

    MIN_SEGS = 2          # ignore clusters smaller than this
    max_k    = min(max_k, n // 2, 6)   # cap at n/2 and 6
    if max_k < 1:
        return 1

    # L2-normalise
    norms  = np.sqrt(np.sum(embeddings**2, axis=1, keepdims=True))
    normed = embeddings / np.maximum(norms, 1e-10)

    # Full pairwise cosine distance matrix (1 - similarity)
    # Shape: (n, n)
    sim_matrix  = normed @ normed.T                          # cosine similarities
    dist_matrix = 1.0 - np.clip(sim_matrix, -1.0, 1.0)     # cosine distances
    np.fill_diagonal(dist_matrix, 0.0)

    best_k     = 1
    best_score = -999.0
    print(f"  {ANSI_GREY}  k  intra   inter   score   valid_clusters{ANSI_RESET}")

    for k in range(1, max_k + 1):
        clusterer = AgglomerativeClusterer(n_clusters=k, linkage="average")
        labels    = clusterer.fit_predict(normed)

        # Count segments per cluster — filter noise clusters
        counts        = np.bincount(labels, minlength=k)
        valid_clusters = int(np.sum(counts >= MIN_SEGS))

        # Intra-cluster mean distance (want LOW)
        intra_dists = []
        for c in range(k):
            idx = np.where(labels == c)[0]
            if len(idx) < 2:
                continue
            # Mean pairwise distance within cluster
            sub = dist_matrix[np.ix_(idx, idx)]
            n_pairs = len(idx) * (len(idx) - 1)
            if n_pairs > 0:
                intra_dists.append(float(sub.sum() / n_pairs))

        avg_intra = float(np.mean(intra_dists)) if intra_dists else 0.0

        # Inter-cluster mean distance (want HIGH)
        inter_dists = []
        for ci in range(k):
            for cj in range(ci + 1, k):
                idx_i = np.where(labels == ci)[0]
                idx_j = np.where(labels == cj)[0]
                if len(idx_i) == 0 or len(idx_j) == 0:
                    continue
                sub = dist_matrix[np.ix_(idx_i, idx_j)]
                inter_dists.append(float(sub.mean()))

        avg_inter = float(np.mean(inter_dists)) if inter_dists else 0.0

        # Score: separation minus compactness
        # Heavily penalise k values where most clusters are noise
        noise_penalty = 0.15 * (k - valid_clusters)
        score = avg_inter - avg_intra - noise_penalty

        print(f"  {ANSI_GREY}  k={k}  {avg_intra:.3f}   {avg_inter:.3f}   "
              f"{score:.3f}   {valid_clusters}/{k}{ANSI_RESET}")

        if score > best_score:
            best_score = score
            best_k     = valid_clusters   # use valid cluster count, not k

    best_k = max(1, best_k)
    print(f"  {ANSI_GREEN}  → Best estimate: {best_k} speaker(s){ANSI_RESET}\n")
    return best_k


def _print_file_summary(speaker_stats: dict, asr, total_duration: float):
    """Print end-of-file summary."""
    print()
    print(f"{ANSI_CYAN}{'─' * 60}{ANSI_RESET}")
    print(f"{ANSI_BOLD}  Session Summary{ANSI_RESET}")
    print(f"{ANSI_CYAN}{'─' * 60}{ANSI_RESET}")
    print(f"  File duration:  {ANSI_WHITE}{total_duration:.1f}s{ANSI_RESET}")
    print(f"  Speakers found: {ANSI_WHITE}{len(speaker_stats)}{ANSI_RESET}")

    if speaker_stats:
        print()
        print(f"  {'Speaker':<14} {'Segments':>8}  {'~Duration':>10}")
        print(f"  {'─'*14}  {'─'*8}  {'─'*10}")
        for sid in sorted(speaker_stats):
            s     = speaker_stats[sid]
            color = speaker_color(sid)
            label = f"{color}{s['label']}{ANSI_RESET}"
            print(f"  {label:<24} {s['segments']:>8}  {s['duration']:>8.1f}s")

    asr_stats = asr.get_stats()
    if asr_stats["n_transcriptions"] > 0:
        print()
        print(f"  ASR ({asr_stats['model_size']}):  "
              f"avg {asr_stats['avg_processing_ms']:.0f}ms/chunk  "
              f"RTF={asr_stats['real_time_factor']:.2f}")
    print(f"{ANSI_CYAN}{'─' * 60}{ANSI_RESET}\n")


# ══════════════════════════════════════════════════════════
# MODE B: LIVE MICROPHONE — FROM SCRATCH
# ══════════════════════════════════════════════════════════

def process_mic(
    model_size:   str = "base",
    chunk_sec:    float = 1.5,
    sample_rate:  int = 16000,
    max_speakers: int = 8
):
    """
    Process live microphone input through the full pipeline.

    Uses PyAudio to capture microphone audio in chunks,
    feeds each chunk into the pipeline, and prints subtitles.

    Press Ctrl+C to stop.

    FROM SCRATCH:
      - PyAudio stream setup
      - Chunk accumulation loop
      - Graceful stop on Ctrl+C
    """
    print_header("MICROPHONE MODE")

    # ── Check PyAudio ─────────────────────────────────────
    try:
        import pyaudio
    except ImportError:
        print(f"\n{ANSI_RED}  PyAudio not installed.{ANSI_RESET}")
        print(f"  Install with:  pip install pyaudio\n")
        print(f"  On Linux:      sudo apt-get install portaudio19-dev && pip install pyaudio")
        print(f"  On Mac:        brew install portaudio && pip install pyaudio")
        return

    # ── Initialize pipeline ───────────────────────────────
    print(f"  {ANSI_YELLOW}Loading ML models...{ANSI_RESET}")
    pipeline = SpeechPipeline(
        sample_rate=sample_rate,
        whisper_model=model_size,
        max_speakers=max_speakers
    )
    import asyncio
    asyncio.run(pipeline.initialize())
    print(f"  {ANSI_GREEN}Models loaded.{ANSI_RESET}")

    # ── Open mic stream ───────────────────────────────────
    pa          = pyaudio.PyAudio()
    chunk_size  = int(sample_rate * chunk_sec)   # samples per chunk
    # PyAudio reads int16 frames, we convert to float32
    FRAME_SIZE  = 1024   # small read size for low latency

    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=FRAME_SIZE
        )
    except OSError as e:
        print(f"\n{ANSI_RED}  Could not open microphone: {e}{ANSI_RESET}")
        print(f"  Check that a microphone is connected and not in use.")
        pa.terminate()
        return

    # ── State ─────────────────────────────────────────────
    audio_buffer   = []
    buffer_samples = 0
    session_start  = time.time()
    is_running     = True

    # Handle Ctrl+C gracefully
    def on_sigint(sig, frame):
        nonlocal is_running
        is_running = False
    signal.signal(signal.SIGINT, on_sigint)

    print_header("Transcript  (press Ctrl+C to stop)")
    print(f"  {ANSI_GREY}Listening... speak into your microphone{ANSI_RESET}\n")

    # ── Main capture loop — FROM SCRATCH ──────────────────
    try:
        while is_running:
            # Read raw bytes from mic
            try:
                raw = stream.read(FRAME_SIZE, exception_on_overflow=False)
            except OSError:
                continue

            # Convert int16 bytes → float32 array (our code)
            int16_array = np.frombuffer(raw, dtype=np.int16)
            float32_chunk = int16_to_float32(int16_array)

            # Accumulate into buffer
            audio_buffer.append(float32_chunk)
            buffer_samples += len(float32_chunk)

            # When we have a full chunk, process it
            if buffer_samples >= chunk_size:
                # Merge buffer
                full_audio = np.concatenate(audio_buffer)
                chunk      = full_audio[:chunk_size]
                leftover   = full_audio[chunk_size:]

                # Reset buffer with leftover
                audio_buffer   = [leftover] if len(leftover) > 0 else []
                buffer_samples = len(leftover)

                # Timestamp
                elapsed   = time.time() - session_start
                timestamp = _format_ts(elapsed)

                # Run pipeline
                results = pipeline.process_chunk(chunk, timestamp)

                # Print subtitles
                for r in results:
                    print_subtitle(r)

    except Exception as e:
        print(f"\n{ANSI_RED}  Stream error: {e}{ANSI_RESET}")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

    # Flush remaining buffer
    if buffer_samples > int(sample_rate * 0.5):
        final_audio = np.concatenate(audio_buffer) if audio_buffer else np.array([], dtype=np.float32)
        elapsed     = time.time() - session_start
        final_results = pipeline.process_chunk(final_audio, _format_ts(elapsed))
        flush_results = pipeline.process_flush()
        for r in final_results + flush_results:
            print_subtitle(r)

    print(f"\n  {ANSI_GREY}Recording stopped.{ANSI_RESET}")
    print_speaker_summary(pipeline)


# ══════════════════════════════════════════════════════════
# HELPERS — FROM SCRATCH
# ══════════════════════════════════════════════════════════

def int16_to_float32(int16_array: np.ndarray) -> np.ndarray:
    """
    Convert int16 PCM samples to float32 — FROM SCRATCH.

    Microphones output int16 by default (range: -32768 to 32767).
    Our pipeline needs float32 in range [-1.0, 1.0].

    Formula: float = int16 / 32768.0
    """
    return (int16_array.astype(np.float32) / 32768.0)


def _format_ts(seconds: float) -> str:
    """Format seconds → MM:SS — FROM SCRATCH."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


# ══════════════════════════════════════════════════════════
# CLI ARGUMENT PARSER
# ══════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="process_audio.py",
        description="Real-Time Speaker Diarization — File or Microphone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_audio.py --file meeting.wav
  python process_audio.py --file interview.mp3 --model small
  python process_audio.py --mic
  python process_audio.py --mic --model tiny --speakers 2
        """
    )

    # Input source (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--file", "-f",
        metavar="PATH",
        help="Path to audio file (.wav, .mp3, .flac, .ogg, .m4a)"
    )
    group.add_argument(
        "--mic", "-m",
        action="store_true",
        help="Use live microphone input"
    )

    # Options
    parser.add_argument(
        "--model",
           choices=["auto", "tiny", "base", "small", "medium", "large", "large-v3"],
           default="auto",
           help="Whisper model size (default: auto). "
               "auto picks medium on 6GB CUDA, small on CPU."
    )
    parser.add_argument(
        "--chunk",
        type=float,
        default=1.5,
        metavar="SECONDS",
        help="Audio chunk size in seconds (default: 1.5)"
    )
    parser.add_argument(
        "--speakers",
        type=int,
        default=0,
        metavar="N",
        help="Known number of speakers (e.g. 3). "
             "If set, output is forced to exactly N speakers after processing. "
             "Leave at 0 to auto-detect."
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=16000,
        metavar="HZ",
        help="Sample rate for mic mode (default: 16000)"
    )

    return parser


# ══════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════

def main():
    # ── Banner ────────────────────────────────────────────
    print()
    print(f"{ANSI_CYAN}{'═' * 60}{ANSI_RESET}")
    print(f"{ANSI_CYAN}{ANSI_BOLD}  LiveTranscribe — Speaker Diarization System{ANSI_RESET}")
    print(f"{ANSI_GREY}  VAD → ECAPA-TDNN Embeddings → Clustering → Whisper ASR{ANSI_RESET}")
    print(f"{ANSI_CYAN}{'═' * 60}{ANSI_RESET}")

    parser = build_parser()
    args   = parser.parse_args()

    print(f"\n  Model:      {ANSI_WHITE}{args.model}{ANSI_RESET}")
    print(f"  Chunk:      {ANSI_WHITE}{args.chunk}s{ANSI_RESET}")
    spk_hint = str(args.speakers) if args.speakers > 0 else "auto-detect"
    print(f"  Speakers:     {ANSI_WHITE}{spk_hint}{ANSI_RESET}")

    if args.file:
        process_file(
            filepath=args.file,
            model_size=None if args.model == "auto" else args.model,
            chunk_sec=args.chunk,
            max_speakers=args.speakers if args.speakers > 0 else 8,
            known_speakers=args.speakers
        )
    elif args.mic:
        process_mic(
            model_size="medium" if args.model == "auto" else args.model,
            chunk_sec=args.chunk,
            sample_rate=args.sr,
            max_speakers=args.speakers
        )


if __name__ == "__main__":
    main()