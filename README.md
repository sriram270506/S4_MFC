# S4_MFC / LiveTranscribe

Detailed project README and revision notes for the current codebase.

This document is written as a study guide, a project walkthrough, and a codebase map. It reflects the current repository structure and current runtime behavior, not the older project layout that the repository originally documented.

## 1. What This Project Does

This project is a real-time speech transcription and speaker diarization system.

In plain terms, it does three jobs at the same time:

1. Capture microphone audio from a browser or from a local script.
2. Convert speech to text using Whisper.
3. Decide which speaker said each chunk of text using speaker embeddings and clustering.

The system has two major usage modes:

1. Live web mode
   - Run `python main.py`
   - Open `http://127.0.0.1:8000`
   - Speak into the microphone from the browser
   - View live transcript plus speaker labels in the web UI

2. Standalone CLI mode
   - Run `python process_audio.py --file your_audio.wav`
   - Or run `python process_audio.py --mic`
   - View results in the terminal

## 2. Current Important Reality Checks

These points are important because they describe the repository exactly as it works now:

1. The repository is flat at the root level.
   - There is no active `backend/` folder and no active `frontend/` folder.
   - The main code files are all in the repository root.

2. The live web app is currently locked to Whisper `medium` on CUDA only.
   - Live mode is not meant to fall back to CPU.
   - If CUDA is unavailable, the live app should fail early instead of silently switching to a smaller model.

3. The live diarization pipeline is currently tuned for at most 3 speakers.
   - The online clusterer in the live path is capped at 3 speakers.

4. The current live server is meant to be opened at:
   - `http://127.0.0.1:8000`
   - or `http://localhost:8000`

5. `0.0.0.0` is not a browser URL.
   - It is only a bind address used by servers.
   - The browser should use `127.0.0.1` or `localhost`.

6. The current README before this rewrite was outdated.
   - It described a different folder layout.
   - It described older model defaults.
   - It described running the frontend separately, which is not how the current app serves the UI.

## 3. Repository Layout

Current top-level files and folders:

```text
S4_MFC/
|- main.py
|- pipeline.py
|- audio_stream.py
|- vad.py
|- diarization.py
|- clustering.py
|- whisper_asr.py
|- process_audio.py
|- download_models.py
|- index.html
|- script.js
|- style.css
|- requirements.txt
|- speaker_profiles.json
|- models/
|  |- whisper/
|  |- speechbrain_ecapa/
|- __pycache__/
```

## 4. High-Level Architecture

### 4.1 Live Web Mode Flow

The live web path works like this:

```text
Browser microphone
  -> Web Audio API capture in script.js
  -> Float32 PCM chunks over WebSocket
  -> main.py websocket endpoint (/ws/audio)
  -> AudioStreamHandler in audio_stream.py
  -> bounded queue in main.py
  -> SpeechPipeline.process_chunk() in pipeline.py
     -> CombinedVAD in vad.py
     -> SpeakerEmbeddingExtractor in diarization.py
     -> OnlineSpeakerClusterer in clustering.py
     -> WhisperASR in whisper_asr.py
     -> SubtitleBuffer in pipeline.py
  -> JSON messages back to browser
  -> script.js renders transcript, speaker labels, diagnostics
```

### 4.2 Standalone CLI Flow

There are actually two different CLI behaviors:

1. File mode in `process_audio.py`
   - Uses a two-pass pipeline for better speaker separation on recorded files.
   - Pass 1: run VAD and collect all embeddings.
   - Pass 2: batch-cluster all speech chunks, then run ASR with the final speaker labels.

2. Microphone mode in `process_audio.py`
   - Uses the online `SpeechPipeline` path, similar to the live web system.

That distinction matters because file mode is intentionally more accurate for diarization than the live path.

## 5. File-by-File Responsibilities

This section is the most important code map in the repository.

### 5.1 `main.py`

Purpose:

1. Starts the FastAPI application.
2. Loads all ML models once during app startup.
3. Serves the web UI files.
4. Owns the WebSocket endpoint used by the browser.

Key responsibilities:

1. Creates a single global `SpeechPipeline` instance in the FastAPI lifespan block.
2. Serves:
   - `/` -> `index.html`
   - `/style.css` -> `style.css`
   - `/script.js` -> `script.js`
   - `/health` -> health JSON
3. Exposes `/ws/audio` for browser audio streaming.
4. Uses an `asyncio.Queue(maxsize=8)` to buffer ready chunks before processing.
5. Adapts chunk duration from 1.5s up to 2.5s under queue pressure.
6. Sends periodic `ping` messages so the browser WebSocket stays alive during heavy processing.
7. Flushes remaining audio and open subtitles when the client disconnects.

Current bind settings:

1. Host: `127.0.0.1`
2. Port: `8000`

### 5.2 `pipeline.py`

Purpose:

1. This is the orchestrator of the entire live pipeline.
2. It decides what happens to each chunk of audio.

Main classes:

1. `SubtitleResult`
   - Normalized output object for subtitle lines.

2. `SubtitleBuffer`
   - Merges same-speaker chunks into a cleaner running subtitle line.

3. `SpeechPipeline`
   - Owns VAD, embeddings, clustering, ASR, buffering, revision logic, and diagnostics.

Current live defaults in `SpeechPipeline`:

1. `sample_rate = 16000`
2. `whisper_model = "medium"`
3. `speaker_threshold = 0.40`
4. `max_speakers = 3`
5. `CombinedVAD` threshold = `0.45`
6. Subtitle merge window = `3.0s`
7. Minimum turn duration = `1.0s`
8. New speaker hold = `0.9s`
9. Candidate smoothing hits = `2`
10. Delayed refinement window = `15.0s`
11. Delayed refinement interval = `10.0s`
12. Recent segment buffer length = `64`

Important current policy:

1. `_resolve_asr_runtime()` now forces live ASR to `medium` on CUDA only.
2. The small-model fallback is disabled in the live pipeline.
3. If CUDA is unavailable, initialization should fail intentionally.

What `process_chunk()` does:

1. Reads runtime pressure metadata from the queue.
2. Runs VAD and exits early on non-speech.
3. Extracts a speaker embedding.
4. Assigns a speaker ID using online clustering.
5. Computes boundary evidence from embedding drift and VAD confidence.
6. Applies turn hysteresis to avoid rapid speaker ping-pong.
7. Runs ASR if the queue policy allows it.
8. Merges diarization plus ASR into subtitle events.
9. Emits metrics events for the frontend diagnostics panel.
10. Emits revision events when delayed reclustering decides an earlier segment should be relabeled.

### 5.3 `audio_stream.py`

Purpose:

1. Converts incoming short browser audio bursts into fixed-size chunks for the ML pipeline.

Key behavior:

1. Default chunk size = `1.5s`.
2. Overlap ratio = `0.2`.
3. With 16 kHz audio, default chunk size = `24000` samples.
4. Overlap = `4800` samples.
5. Uses a deque as the internal accumulation buffer.
6. Applies slow session-level AGC after chunk extraction.
7. Does not apply pre-emphasis here anymore.

Why pre-emphasis was removed from this stage:

1. Applying it on tiny incoming browser packets caused spectral discontinuities.
2. Those discontinuities hurt speaker embeddings.
3. Pre-emphasis is now applied only inside Whisper ASR, once per complete chunk.

Current slow AGC settings:

1. `enable_slow_agc = True`
2. `agc_target_rms = 0.07`
3. `agc_smoothing = 0.05`

### 5.4 `vad.py`

Purpose:

1. Decides whether a chunk should be treated as speech or not.

Main classes:

1. `EnergyVAD`
2. `SileroVAD`
3. `CombinedVAD`

`EnergyVAD` details:

1. Frame size = 20 ms.
2. At 16 kHz, that becomes `320` samples per frame.
3. Base energy threshold = `0.01` RMS.
4. Required speech-frame ratio = `0.3`.
5. Adaptive mode = enabled by default.
6. Adaptive threshold uses recent low-energy history to estimate a noise floor.

`SileroVAD` details:

1. Default threshold = `0.5`.
2. Window size = `512` samples.
3. Loaded from `torch.hub`.
4. Falls back to `EnergyVAD` if loading fails.

`CombinedVAD` details:

1. Default decision threshold = `0.45`.
2. Always runs energy VAD.
3. Uses Silero if available.
4. Combines confidence as:
   - `0.3 * energy_conf + 0.7 * silero_conf`

### 5.5 `diarization.py`

Purpose:

1. Extracts speaker embeddings from audio.
2. Stores speaker profile centroids on disk.
3. Provides compatibility patches for library-version mismatches on Windows.

Main responsibilities:

1. Build features for speaker representation.
2. Load SpeechBrain ECAPA-TDNN.
3. Extract normalized speaker vectors.
4. Persist speaker profiles to `speaker_profiles.json`.

Current important implementation notes:

1. ECAPA-TDNN is the primary embedding model.
2. Output embeddings are 192-dimensional.
3. The module includes compatibility shims for:
   - `torchaudio >= 2.10`
   - `huggingface_hub 1.x`
4. On Windows, it forces SpeechBrain downloads to use copy-based local strategy instead of symlinks.

That Windows fix matters because symlink-based caching often fails without elevated privileges or Developer Mode.

About `speaker_profiles.json`:

1. It stores saved centroids, sample counts, and timestamps.
2. It is updated during sessions.
3. It is loaded at startup.
4. In the current code, it acts more like persisted metadata/profile memory than direct cluster bootstrapping.
5. The online clusterer is still reset for each new live session.

### 5.6 `clustering.py`

Purpose:

1. Performs online speaker assignment using cosine similarity.
2. Creates new speaker clusters when enough evidence accumulates.
3. Optionally merges clusters that are almost identical.

Main class:

1. `OnlineSpeakerClusterer`

Core algorithm:

1. Compare each new embedding against all active cluster centroids.
2. If best similarity >= threshold, assign to that speaker and update the centroid.
3. If best similarity < threshold, count it as a miss.
4. Only create a new speaker after enough consecutive misses.

Current live thresholds:

1. `similarity_threshold = 0.40`
2. `new_speaker_patience = 4`
3. `merge_threshold = 0.75`
4. `max_inactive_seconds = 300.0`

Interpretation of these values:

1. Scores around `0.40` to `0.60` are typically treated as same-speaker territory in the live path.
2. Scores in the `0.20` to `0.35` range are treated as likely speaker changes.
3. `new_speaker_patience = 4` means the system waits for a sustained mismatch before creating a new speaker, instead of reacting to a single noisy chunk.

Current live speaker cap:

1. The `SpeechPipeline` passes `max_speakers = 3` to the online clusterer.

### 5.7 `whisper_asr.py`

Purpose:

1. Performs speech-to-text conversion.
2. Validates audio before inference.
3. Applies anti-hallucination controls.
4. Filters weak or empty results.

Main class:

1. `WhisperASR`

Current live ASR configuration:

1. Model = `medium`
2. Device = `cuda`
3. Compute type = `float16`
4. Language = `en`
5. Beam size = `5`
6. Minimum confidence = `0.45`

Current inference details that matter:

1. Audio validation rejects:
   - chunks shorter than 0.5 seconds
   - chunks with RMS below `0.004`
   - NaN or Inf audio
2. Pre-emphasis is applied here, once over the full chunk.
3. `condition_on_previous_text = False`
4. `temperature = 0.0`
5. `initial_prompt = None`
6. `vad_filter = True`
7. Whisper internal VAD parameters:
   - `min_silence_duration_ms = 400` in quality mode
   - `min_silence_duration_ms = 250` in fast mode
   - `threshold = 0.35`
8. Quality gates:
   - `compression_ratio_threshold = 2.0`
   - `no_speech_threshold = 0.6`
   - `log_prob_threshold = -1.0`

### 5.8 `process_audio.py`

Purpose:

1. Standalone processor for audio files and direct microphone input.
2. Useful for testing without the browser UI.

Important distinction:

1. File mode is two-pass and more offline-oriented.
2. Mic mode is live and online-oriented.

File mode (`--file`) behavior:

1. Load and normalize the full file.
2. Resample to 16 kHz if needed.
3. Split into chunks.
4. Pass 1: detect speech and extract embeddings.
5. Estimate speaker count.
6. Run batch agglomerative clustering.
7. Pass 2: transcribe chunks using the final speaker labels.
8. Print transcript and session summary.

Mic mode (`--mic`) behavior:

1. Uses PyAudio.
2. Builds `SpeechPipeline`.
3. Reads int16 PCM frames from the microphone.
4. Converts them to float32.
5. Processes online and prints subtitles to the terminal.

Current CLI notes:

1. Parser default is `--model auto`.
2. In file mode, runtime probing now follows the same CUDA policy used by the pipeline.
3. Because the live pipeline is CUDA-only now, standalone behavior assumes the same current environment.

### 5.9 `download_models.py`

Purpose:

1. Legacy helper script to pre-download models.

What it currently downloads:

1. Faster-Whisper `base`
2. SpeechBrain ECAPA-TDNN
3. Silero VAD

Important current caveat:

1. The live app now uses Whisper `medium`, not `base`.
2. So this script is no longer perfectly aligned with the live runtime defaults.
3. Running the live app for the first time may still download the `medium` model even if `download_models.py` already downloaded `base`.

### 5.10 `index.html`

Purpose:

1. Defines the browser UI.

Main UI sections:

1. Header with connection state and VAD indicator.
2. Sidebar with controls, session stats, speaker registry, config, and diagnostics.
3. Transcript panel with partial/live line support.

Current config controls in the UI:

1. Chunk size in milliseconds
2. Sample rate selector
3. WebSocket host field
4. Browser DSP toggle

### 5.11 `script.js`

Purpose:

1. Browser-side audio capture and UI behavior.

Main jobs:

1. Request microphone access.
2. Open WebSocket connection.
3. Capture audio using `ScriptProcessorNode`.
4. Compute client-side RMS for the local VAD indicator.
5. Accumulate audio into chunks and send binary float32 audio.
6. Receive backend JSON events and render them.
7. Maintain diagnostics and speaker registry state.
8. Export transcript to a text file.

Current frontend defaults:

1. Default sample rate = `16000`
2. Default chunk size = `1500 ms`
3. WebSocket reconnect delay constant = `2000 ms`
4. Default WebSocket host field = `localhost:8000`
5. Browser DSP default = Off

Important note about the sample-rate selector:

1. The UI offers `16000` and `22050`.
2. The backend is designed around `16000`.
3. For the current system, `16000` is the safe and correct setting.

### 5.12 `style.css`

Purpose:

1. Styles the live UI.

What it controls:

1. Layout and panels
2. Button styles
3. Transcript line appearance
4. Speaker-color accent styling
5. Diagnostics and badges
6. Revised subtitle highlighting

### 5.13 `requirements.txt`

Purpose:

1. Lists Python package dependencies.

Important practical note:

1. `torch` and `torchaudio` may need to be installed with a CUDA-specific wheel separately.
2. In the current working environment, the system was fixed using a CUDA-enabled PyTorch build.

### 5.14 `speaker_profiles.json`

Purpose:

1. Stores persisted speaker profile centroids and metadata.

What it contains:

1. Label
2. Centroid vector
3. Number of samples
4. Last seen timestamp

## 6. Live WebSocket Contract

### 6.1 Frontend to Backend

The browser sends:

1. Raw binary audio over WebSocket.
2. Format: float32 PCM bytes.
3. Mono.
4. Expected by backend as 16 kHz.

Backend decoding line of thought:

1. `main.py` receives bytes.
2. Converts them with `np.frombuffer(..., dtype=np.float32)`.

### 6.2 Backend to Frontend Message Types

The backend sends several JSON shapes.

#### A. Normal finalized subtitle event

```json
{
  "timestamp": "00:03",
  "speaker": "Speaker 1",
  "speaker_id": 0,
  "text": "Hello everyone",
  "confidence": 0.87,
  "color": "#4FC3F7",
  "is_partial": false,
  "segment_id": 12
}
```

#### B. Partial subtitle event

This is used for the live partial line in the frontend.

```json
{
  "timestamp": "00:03",
  "speaker": "Speaker 1",
  "speaker_id": 0,
  "text": "Hello every...",
  "confidence": 0.81,
  "color": "#4FC3F7",
  "is_partial": true,
  "segment_id": 12
}
```

#### C. Metrics event

Used by the diagnostics panel.

```json
{
  "type": "metrics",
  "timestamp": "4.5s",
  "queue_depth": 0,
  "queue_max": 8,
  "queue_pressure": 0.0,
  "dropped_chunks": 0,
  "chunk_seconds": 1.5,
  "asr_mode": "primary",
  "vad_conf": 0.73,
  "embedding_drift": 0.4215,
  "boundary_score": 0.336,
  "stage_latency_ms": {
    "vad": 2.1,
    "embedding": 41.5,
    "assignment": 0.8,
    "refinement": 0.0,
    "asr": 932.4,
    "merge": 0.3,
    "total": 977.1
  },
  "revisions": 1
}
```

#### D. Revision event

Used when delayed reclustering decides a previous segment should be relabeled.

```json
{
  "type": "revision",
  "segment_id": 12,
  "timestamp": "00:03",
  "old_speaker_id": 0,
  "new_speaker_id": 1,
  "new_speaker": "Speaker 2",
  "color": "#AED581",
  "reason": "rolling_recluster"
}
```

#### E. Ping event

```json
{
  "type": "ping"
}
```

#### F. Error event

```json
{
  "error": "...",
  "timestamp": "00:00",
  "speaker": "System",
  "text": "Processing error: ..."
}
```

## 7. Current Live Runtime Defaults and Thresholds

### 7.1 Live server and queue

| Setting | Current value | Where it lives |
|---|---:|---|
| Host | `127.0.0.1` | `main.py` |
| Port | `8000` | `main.py` |
| WebSocket endpoint | `/ws/audio` | `main.py` |
| Queue max size | `8` | `main.py` |
| Base chunk seconds | `1.5` | `main.py` |
| Max chunk seconds under pressure | `2.5` | `main.py` |
| Ping interval | `15s` | `main.py` |

### 7.2 Audio buffering

| Setting | Current value | Where it lives |
|---|---:|---|
| Sample rate | `16000` | `audio_stream.py`, `pipeline.py`, `main.py` |
| Chunk size | `24000 samples` at 1.5s | `audio_stream.py` |
| Overlap ratio | `0.20` | `audio_stream.py` |
| Overlap size | `4800 samples` at 1.5s | `audio_stream.py` |
| AGC target RMS | `0.07` | `audio_stream.py` |
| AGC smoothing | `0.05` | `audio_stream.py` |

### 7.3 VAD

| Setting | Current value | Where it lives |
|---|---:|---|
| Energy frame size | `20 ms` | `vad.py` |
| Energy threshold | `0.01` | `vad.py` |
| Speech ratio threshold | `0.3` | `vad.py` |
| Silero threshold | `0.5` | `vad.py` |
| CombinedVAD threshold | `0.45` | `vad.py`, `pipeline.py` |

### 7.4 Diarization and speaker change

| Setting | Current value | Where it lives |
|---|---:|---|
| Speaker similarity threshold | `0.40` | `clustering.py`, `pipeline.py` |
| New speaker patience | `4` misses | `clustering.py`, `pipeline.py` |
| Merge threshold | `0.75` | `clustering.py` |
| Max speakers in live path | `3` | `pipeline.py` |
| Min turn duration | `1.0s` | `pipeline.py` |
| New speaker hold | `0.9s` | `pipeline.py` |
| Candidate hits for reassignment | `2` | `pipeline.py` |
| Delayed refine window | `15s` | `pipeline.py` |
| Delayed refine interval | `10s` | `pipeline.py` |

### 7.5 ASR

| Setting | Current value | Where it lives |
|---|---:|---|
| Live Whisper model | `medium` | `pipeline.py` |
| Device | `cuda` | `pipeline.py` |
| Compute type | `float16` | `pipeline.py` |
| Fallback model | disabled | `pipeline.py` |
| ASR min confidence | `0.45` | `whisper_asr.py` |
| Audio min duration | `0.5s` | `whisper_asr.py` |
| Audio RMS minimum | `0.004` | `whisper_asr.py` |
| Whisper no-speech threshold | `0.6` | `whisper_asr.py` |
| Whisper log-prob threshold | `-1.0` | `whisper_asr.py` |
| Compression ratio threshold | `2.0` | `whisper_asr.py` |

### 7.6 Subtitle merging

| Setting | Current value | Where it lives |
|---|---:|---|
| Merge window | `3.0s` | `pipeline.py` |
| Max chars per subtitle | `150` | `pipeline.py` |
| Flush other speakers immediately on switch | yes | `pipeline.py` |

## 8. Web GUI Behavior

### 8.1 What the browser does

When you click Start Recording:

1. `script.js` opens a WebSocket.
2. It requests microphone access.
3. It creates an `AudioContext`.
4. It captures raw audio through a `ScriptProcessorNode`.
5. It computes RMS locally for the level meter and VAD dot.
6. It accumulates chunks up to the configured chunk size.
7. It sends raw float32 audio to the backend.

### 8.2 What the browser renders

The UI shows:

1. Connection status
2. Local VAD dot
3. Audio level meter
4. Session duration
5. Total chunks sent
6. Number of detected speakers
7. Number of transcript lines
8. Speaker registry with color badges
9. Diagnostics panel with queue, latency, confidence, ASR mode, and embedding drift
10. Live transcript feed
11. Partial line while a subtitle is still open

### 8.3 Browser DSP toggle

The UI has a Browser DSP setting:

1. Off (default)
   - Echo cancellation off
   - Noise suppression off
   - Auto gain control off
   - Better for preserving speaker identity

2. On
   - Cleaner capture
   - May reduce diarization quality because browser DSP can distort speaker characteristics

## 9. Standalone Script Behavior

### 9.1 File mode

Use:

```bash
python process_audio.py --file sample.wav
```

This mode is more suitable for evaluation because it can see the whole file and cluster globally.

Pipeline summary for file mode:

1. Load audio with `soundfile` or `librosa`
2. Convert to mono if needed
3. Resample to 16 kHz if needed
4. Normalize amplitude
5. Split into chunks
6. Pass 1: detect speech and extract embeddings
7. Estimate speaker count
8. Run batch clustering
9. Pass 2: transcribe each speech chunk
10. Print transcript and session summary

### 9.2 Mic mode

Use:

```bash
python process_audio.py --mic
```

Mic mode is closer to live behavior:

1. Opens microphone via PyAudio
2. Builds `SpeechPipeline`
3. Accumulates audio into chunks
4. Runs online pipeline on each chunk
5. Prints subtitles in real time

## 10. Model and Cache Locations

### 10.1 Whisper cache

Live and standalone Whisper downloads go under:

```text
models/whisper/
```

That means once a model is downloaded, it should not be re-downloaded unless:

1. The cache folder is deleted.
2. A different model size is requested.
3. The cache root changes.

### 10.2 SpeechBrain ECAPA cache

Speaker embedding model files live under:

```text
models/speechbrain_ecapa/
```

### 10.3 Silero cache

Silero VAD is loaded through `torch.hub`, so its cache usually lives in the user profile cache, for example:

```text
C:\Users\<username>\.cache\torch\hub\
```

### 10.4 Persisted speaker profiles

Profiles are stored in:

```text
speaker_profiles.json
```

## 11. Setup and Run Commands

### 11.1 Install dependencies

From the repository root:

```bash
pip install -r requirements.txt
```

Important:

1. For this project's current live configuration, a CUDA-enabled PyTorch build is required.
2. Generic `pip install torch` may install a CPU-only build depending on environment and index configuration.

### 11.2 Start the live web app

```bash
python main.py
```

Open:

```text
http://127.0.0.1:8000
```

or:

```text
http://localhost:8000
```

### 11.3 Process a recorded file

```bash
python process_audio.py --file sample.wav
```

### 11.4 Process microphone input from terminal

```bash
python process_audio.py --mic
```

### 11.5 Optional model pre-download

```bash
python download_models.py
```

Important caveat again:

1. This helper currently downloads `base`, not `medium`.
2. The first live startup may still download `medium`.

## 12. Troubleshooting and Windows Notes

### 12.1 Browser URL not working

Use:

1. `http://127.0.0.1:8000`
2. `http://localhost:8000`

Do not use:

1. `http://0.0.0.0:8000`

### 12.2 App fails during startup with CUDA-related issues

Reason:

1. The current live pipeline is intentionally CUDA-only.

Check:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If it prints `False`, the live app is not in a valid runtime state.

### 12.3 Whisper download seems stuck

What usually happens:

1. Hugging Face logs show token and redirect requests.
2. Then a large model blob downloads quietly.
3. The server does not become reachable until startup finishes.

This is normal on first run.

### 12.4 Windows symlink warnings from Hugging Face

Meaning:

1. Hugging Face wants to use symlinks for efficient cache storage.
2. Windows may block that without Developer Mode or admin privileges.

Effect:

1. Caching still works.
2. It may use more disk space.

### 12.5 SpeechBrain / torchaudio compatibility errors

This repository already contains fixes inside `diarization.py` for:

1. `torchaudio.list_audio_backends` removal in newer torchaudio versions
2. `huggingface_hub` API changes
3. Windows copy-vs-symlink issues during model fetch

### 12.6 Why speaker separation may still be imperfect

Even with current tuning, live diarization remains harder than offline diarization because:

1. Live mode sees only short chunks.
2. Room acoustics and overlapping speech reduce separation quality.
3. Browser DSP can blur speaker identity.
4. Short turns can look acoustically similar.

### 12.7 Why the 22 kHz UI option is risky

The backend is designed around 16 kHz processing.

For the current project, use:

1. `16 kHz`

## 13. Presentation and Viva Revision Notes

If you need to explain this project clearly in a presentation or viva, these are the key ideas.

### 13.1 Core system idea

This is not just speech-to-text.

It combines:

1. Voice Activity Detection
2. Speaker embedding extraction
3. Online speaker clustering
4. ASR with Whisper
5. Subtitle merging and UI rendering

### 13.2 What is the hardest problem here?

The hardest part is speaker diarization in live audio.

Why:

1. The system does not know in advance how many speakers there are.
2. Short live chunks can make the same speaker look different over time.
3. Noise, room echo, and browser processing can distort identity features.

### 13.3 Why use ECAPA-TDNN?

Because it gives a compact vector representation of speaker identity.

That vector is then clustered with cosine similarity.

### 13.4 Why use Whisper medium on GPU?

Because:

1. It is significantly more accurate than very small Whisper models.
2. Real-time performance is much better on GPU.
3. The current machine was tuned around this model.

### 13.5 Why is file mode more accurate for speaker separation?

Because file mode uses a two-pass strategy:

1. It sees the whole recording first.
2. It clusters globally.
3. Then it transcribes with more stable speaker labels.

Live mode cannot do that because it must decide online.

### 13.6 What is subtitle merging doing?

It prevents every 1.5-second chunk from appearing as a separate transcript line.

Instead:

1. Same-speaker nearby chunks are merged.
2. Speaker changes flush the previous line.
3. The frontend can still show partial lines and later revisions.

## 14. Suggested Reading Order for Studying the Code

If you want to understand the project properly, read the files in this order:

1. `main.py`
   - Understand server entry, startup, and WebSocket handling.

2. `script.js`
   - Understand what the browser sends and receives.

3. `pipeline.py`
   - Understand the main orchestration logic.

4. `audio_stream.py`
   - Understand chunk creation and overlap.

5. `vad.py`
   - Understand how non-speech is filtered out.

6. `diarization.py`
   - Understand embeddings and compatibility fixes.

7. `clustering.py`
   - Understand online speaker assignment.

8. `whisper_asr.py`
   - Understand ASR, text filtering, and anti-hallucination settings.

9. `process_audio.py`
   - Understand the offline/two-pass path and CLI workflows.

10. `index.html` and `style.css`
   - Understand final UI structure and presentation.

## 15. Final Summary

The current repository is a real-time diarization plus transcription system with:

1. A browser UI served directly by FastAPI
2. WebSocket audio streaming
3. Adaptive chunking and bounded queue handling
4. Combined VAD
5. ECAPA-TDNN speaker embeddings
6. Online cosine-similarity clustering
7. Whisper medium on CUDA for live ASR
8. Subtitle merging, metrics, and revision events
9. A more accurate two-pass CLI path for recorded files

If you remember only one sentence for a presentation, use this:

This project captures live audio, filters speech, extracts speaker identity embeddings, clusters speakers online, transcribes speech with Whisper, and displays speaker-labeled subtitles in real time through a browser UI.