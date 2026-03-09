# LiveTranscribe — Real-Time Speaker Diarization System

A complete real-time speech recognition and speaker diarization system
built for a college project. Multiple speakers are automatically identified
and labeled in real time using a full ML pipeline.

---

## Project Structure

```
speech_diarization/
├── backend/
│   ├── main.py          ← FastAPI server + WebSocket endpoint
│   ├── audio_stream.py  ← Audio chunking & buffering (FROM SCRATCH)
│   ├── vad.py           ← Voice Activity Detection (FROM SCRATCH + Silero)
│   ├── diarization.py   ← Speaker embeddings: ECAPA-TDNN + feature extraction (FROM SCRATCH)
│   ├── clustering.py    ← Online agglomerative clustering (FROM SCRATCH)
│   ├── whisper_asr.py   ← Speech-to-text with Faster-Whisper
│   ├── pipeline.py      ← Full orchestration + subtitle merging (FROM SCRATCH)
│   └── requirements.txt
├── frontend/
│   ├── index.html       ← UI
│   ├── style.css        ← Dark theme
│   └── script.js        ← Web Audio API capture (FROM SCRATCH)
└── models/
    └── download_models.py  ← Pre-download all ML models
```

---

## Pipeline Overview

```
Microphone Audio
     │
     ▼ (Web Audio API, 16kHz float32)
Audio Chunking (1.5s chunks)
     │
     ▼ (audio_stream.py — FROM SCRATCH)
VAD: Is there speech?
     │ YES
     ▼ (vad.py — EnergyVAD FROM SCRATCH + Silero)
Speaker Embedding Extraction
     │
     ▼ (diarization.py — ECAPA-TDNN model)
Online Clustering → Assign "Speaker 1", "Speaker 2"...
     │
     ▼ (clustering.py — cosine similarity FROM SCRATCH)
ASR: Transcribe to text
     │
     ▼ (whisper_asr.py — Faster-Whisper)
Subtitle Merging
     │
     ▼ (pipeline.py — FROM SCRATCH)
WebSocket → Frontend → Display
```

---

## From-Scratch Implementations

| Module            | What We Built From Scratch |
|-------------------|----------------------------|
| `audio_stream.py` | Pre-emphasis filter, amplitude normalization, overlap chunking |
| `vad.py`          | Frame energy computation, adaptive noise floor, zero-crossing rate |
| `diarization.py`  | Mel filterbank construction, FFT framing, delta features, spectral stats |
| `clustering.py`   | Cosine similarity, online centroid update, agglomerative clustering loop, cluster merging |
| `pipeline.py`     | SubtitleBuffer merge algorithm, session management |
| `script.js`       | RMS computation, Float32→bytes conversion, audio level meter, speaker registry UI |

---

## Installation

### 1. Install Python dependencies

```bash
cd speech_diarization/backend
pip install -r requirements.txt
```

For CPU-only PyTorch (smaller download):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2. Download ML models (do this once)

```bash
python models/download_models.py
```

This downloads:
- Faster-Whisper "base" (~150MB)
- SpeechBrain ECAPA-TDNN (~100MB)
- Silero VAD (~2MB)

---

## Running the System

### Step 1: Start backend server

```bash
cd speech_diarization/backend
python main.py
```

You should see:
```
INFO:     Loading ML models — this may take 30–60 seconds...
INFO:     All models loaded. Server ready.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 2: Open frontend

Open `frontend/index.html` directly in your browser:
- Chrome: `File → Open File → index.html`
- Or serve locally: `python -m http.server 3000` then visit `http://localhost:3000`

### Step 3: Record

1. Click **Start Recording**
2. Allow microphone access when prompted
3. Speak — transcriptions appear within ~2 seconds
4. Multiple speakers are automatically labeled: Speaker 1, Speaker 2, etc.
5. Click **Stop** when done

---

## Output Format

Each subtitle line sent from backend to frontend:
```json
{
  "timestamp": "00:03",
  "speaker": "Speaker 1",
  "speaker_id": 0,
  "text": "Hello everyone, welcome to the meeting.",
  "confidence": 0.89,
  "color": "#4FC3F7",
  "is_partial": false
}
```

---

## Testing Individual Modules

Each backend file has a self-test at the bottom:

```bash
python backend/audio_stream.py   # Test chunking + pre-emphasis
python backend/vad.py            # Test energy VAD on test signals
python backend/diarization.py    # Test Mel filterbank + embedding
python backend/clustering.py     # Test online clustering with fake embeddings
python backend/whisper_asr.py    # Test text cleaning + validation
python backend/pipeline.py       # Test full pipeline with synthetic audio
```

---

## Keyboard Shortcuts

| Shortcut     | Action           |
|--------------|------------------|
| Ctrl+Space   | Start/Stop       |
| Ctrl+L       | Clear transcript |

---

## Configuration

In the UI sidebar, you can adjust:
- **Chunk (ms)**: Audio chunk size sent to backend (500–3000ms)
- **Sample Rate**: 16kHz recommended for Whisper
- **WS Host**: Backend address (default: localhost:8000)

In `backend/pipeline.py`:
- `whisper_model`: "tiny", "base", "small", "medium"
- `speaker_threshold`: Cosine similarity threshold (0.0–1.0, default 0.75)
- `max_speakers`: Maximum concurrent speakers

---

## Troubleshooting

**Microphone not working?**
- Use Chrome or Firefox (Safari has limited Web Audio API support)
- Allow microphone permission when prompted
- Check that no other app is using the mic

**Backend not connecting?**
- Make sure `python main.py` is running first
- Check that the WS Host in the UI matches (default: `localhost:8000`)
- CORS is enabled for all origins in development

**Models not loading?**
- Run `python models/download_models.py` first
- The system has fallbacks: EnergyVAD if Silero fails, random embeddings if ECAPA fails
- Whisper in demo mode returns placeholder text if model isn't loaded

**Transcription is slow?**
- Use `whisper_model="tiny"` for faster (less accurate) results
- Ensure you're using `compute_type="int8"` (quantized)
- For real-time on CPU, "tiny" or "base" are recommended

---

## Architecture Notes (for college report)

### Why Agglomerative Clustering?
- Does not require knowing the number of speakers in advance
- Works well with cosine distance in embedding space
- Online version (one embedding at a time) enables real-time use

### Why ECAPA-TDNN?
- Trained on 1.2M speaker segments from VoxCeleb
- 192-dimensional embeddings capture pitch, formants, speaking style
- Fast inference: ~10ms per 1.5s chunk on CPU

### Why Faster-Whisper?
- CTranslate2 backend = 3-5x faster than original Whisper
- INT8 quantization reduces memory ~4x with minimal accuracy loss
- Word-level timestamps enable precise subtitle alignment

### Latency Analysis
| Stage              | Time (CPU) |
|--------------------|------------|
| VAD                | < 5ms      |
| ECAPA-TDNN         | ~10ms      |
| Clustering         | < 1ms      |
| Whisper (base)     | ~300–500ms |
| Network (WebSocket)| < 5ms      |
| **Total**          | **~400ms** |
