"""
main.py — FastAPI WebSocket Server
Real-Time Speech Recognition & Speaker Diarization System

This is the entry point for the backend.
It:
  - Starts a FastAPI app
  - Opens a WebSocket endpoint /ws/audio
  - Receives raw PCM audio bytes from the browser
  - Feeds them into the full pipeline (VAD → Embedding → Clustering → ASR)
  - Returns JSON subtitles back to the frontend in real time
"""

import asyncio
import logging
import json
import time
import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np

# --- Our own modules ---
from audio_stream import AudioStreamHandler
from pipeline import SpeechPipeline

# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("main")

# ──────────────────────────────────────────────────────────────
# Global pipeline instance
# ──────────────────────────────────────────────────────────────
pipeline: SpeechPipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all ML models at startup; log shutdown."""
    global pipeline
    logger.info("Loading ML models — this may take 30–60 seconds...")
    pipeline = SpeechPipeline()
    await pipeline.initialize()
    logger.info("All models loaded. Server ready.")
    yield
    logger.info("Server shutting down.")


# ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Real-Time Speaker Diarization API",
    description="WebSocket-based speech recognition and speaker identification",
    version="2.0.0",
    lifespan=lifespan
)

# Allow browser to connect (CORS for local development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

@app.get("/")
async def serve_index():
    return FileResponse(BASE_DIR / "index.html")

@app.get("/style.css")
async def serve_css():
    return FileResponse(BASE_DIR / "style.css", media_type="text/css")

@app.get("/script.js")
async def serve_js():
    return FileResponse(BASE_DIR / "script.js", media_type="application/javascript")

# ──────────────────────────────────────────────────────────────
# Health Check
# ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "pipeline_ready": pipeline is not None}

# ──────────────────────────────────────────────────────────────
# WebSocket Endpoint
# ──────────────────────────────────────────────────────────────
@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    """
    Main WebSocket endpoint.

    Protocol:
      Frontend → Backend : raw PCM float32 bytes (16kHz, mono)
      Backend  → Frontend : JSON string with subtitle data

    JSON format returned:
    {
      "timestamp": "00:03",
      "speaker": "Speaker 1",
      "text": "Hello everyone",
      "speaker_id": 0
    }
    """
    await websocket.accept()
    client_id = id(websocket)
    logger.info(f"Client connected: {client_id}")

    # Each WebSocket connection gets its own stream handler
    # Reuse the global pipeline (models already loaded at startup) but reset its state
    stream_handler = AudioStreamHandler(sample_rate=16000, chunk_seconds=2.0)

    # Reset cluster/subtitle state for this new session (models stay loaded)
    pipeline.reset()

    session_start_time = time.time()
    processing = False  # Guard: True while a chunk is being processed

    async def send_ping():
        """Send periodic pings to keep WebSocket alive during long processing."""
        try:
            while True:
                await asyncio.sleep(15)
                await websocket.send_text(json.dumps({"type": "ping"}))
        except Exception:
            pass

    ping_task = asyncio.create_task(send_ping())

    try:
        while True:
            # Receive raw bytes from browser
            raw_bytes = await websocket.receive_bytes()

            # Convert bytes → numpy float32 array
            audio_chunk = np.frombuffer(raw_bytes, dtype=np.float32).copy()

            # Feed into stream handler (accumulates audio into fixed-size chunks)
            complete_chunks = stream_handler.add_audio(audio_chunk)

            if not complete_chunks:
                continue

            # If processing is slow, only process the LATEST chunk (drop stale ones)
            # This prevents the server from falling behind real-time
            if len(complete_chunks) > 1:
                logger.info(f"Dropping {len(complete_chunks) - 1} stale chunk(s) to stay real-time")
            chunk = complete_chunks[-1]  # Only process the freshest chunk

            # Calculate timestamp relative to session start
            elapsed = time.time() - session_start_time
            timestamp = format_timestamp(elapsed)

            # Run the full pipeline on this chunk
            results = await asyncio.get_event_loop().run_in_executor(
                None,  # uses default ThreadPoolExecutor
                pipeline.process_chunk,
                chunk,
                timestamp
            )

            # Send each subtitle result back to frontend
            for result in results:
                await websocket.send_text(json.dumps(result))
                logger.info(f"Sent: [{result['timestamp']}] {result['speaker']}: {result['text']}")

    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {client_id}")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}", exc_info=True)
        try:
            await websocket.send_text(json.dumps({
                "error": str(e),
                "timestamp": "00:00",
                "speaker": "System",
                "text": f"Processing error: {str(e)}"
            }))
        except:
            pass
    finally:
        ping_task.cancel()


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


# ──────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
