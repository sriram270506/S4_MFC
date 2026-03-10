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

    base_chunk_seconds = 1.5
    max_chunk_seconds = 2.5
    current_chunk_seconds = base_chunk_seconds
    queue_max = 8
    dropped_chunks = 0

    # Each WebSocket connection gets its own stream handler.
    stream_handler = AudioStreamHandler(sample_rate=16000, chunk_seconds=base_chunk_seconds)

    # Reset cluster/subtitle state for this new session (models stay loaded).
    pipeline.reset()
    session_start_time = time.time()
    chunk_queue: asyncio.Queue = asyncio.Queue(maxsize=queue_max)

    async def send_result(result: dict):
        await websocket.send_text(json.dumps(result))

    async def send_ping():
        """Send periodic pings to keep WebSocket alive during long processing."""
        try:
            while True:
                await asyncio.sleep(15)
                await websocket.send_text(json.dumps({"type": "ping"}))
        except Exception:
            pass

    async def queue_worker():
        """Stage worker: dequeue chunk -> pipeline -> websocket send."""
        nonlocal dropped_chunks
        while True:
            item = await chunk_queue.get()
            if item is None:
                chunk_queue.task_done()
                break

            try:
                meta = {
                    "queue_depth": chunk_queue.qsize(),
                    "queue_max": queue_max,
                    "dropped_chunks": dropped_chunks,
                    "chunk_seconds": current_chunk_seconds,
                }
                results = await asyncio.get_event_loop().run_in_executor(
                    None,
                    pipeline.process_chunk,
                    item["chunk"],
                    item["timestamp"],
                    meta,
                )
                for result in results:
                    await send_result(result)
            except Exception as e:
                logger.debug(f"Queue worker processing warning: {e}")
            finally:
                chunk_queue.task_done()

    async def enqueue_chunk(chunk: np.ndarray, timestamp: str):
        """Queue stage input with bounded-queue adaptive handling."""
        nonlocal current_chunk_seconds, dropped_chunks
        pressure = chunk_queue.qsize() / queue_max

        # Adapt chunk duration before dropping anything.
        if pressure >= 0.75 and current_chunk_seconds < max_chunk_seconds:
            current_chunk_seconds = min(max_chunk_seconds, current_chunk_seconds + 0.25)
            stream_handler.set_chunk_seconds(current_chunk_seconds)
            logger.info(
                "Queue pressure high (%.2f). Increased chunk size to %.2fs",
                pressure,
                current_chunk_seconds,
            )
        elif pressure <= 0.25 and current_chunk_seconds > base_chunk_seconds:
            current_chunk_seconds = max(base_chunk_seconds, current_chunk_seconds - 0.25)
            stream_handler.set_chunk_seconds(current_chunk_seconds)

        # Bounded queue: drop oldest only as a last resort.
        if chunk_queue.full():
            try:
                _ = chunk_queue.get_nowait()
                chunk_queue.task_done()
                dropped_chunks += 1
                logger.warning(
                    "Queue full. Dropped oldest queued chunk (total dropped=%d)",
                    dropped_chunks,
                )
            except asyncio.QueueEmpty:
                pass

        await chunk_queue.put({"chunk": chunk, "timestamp": timestamp})

    ping_task = asyncio.create_task(send_ping())
    worker_task = asyncio.create_task(queue_worker())

    try:
        while True:
            raw_bytes = await websocket.receive_bytes()
            audio_chunk = np.frombuffer(raw_bytes, dtype=np.float32).copy()
            complete_chunks = stream_handler.add_audio(audio_chunk)

            if not complete_chunks:
                continue

            for chunk in complete_chunks:
                elapsed = time.time() - session_start_time
                timestamp = format_timestamp(elapsed)
                await enqueue_chunk(chunk, timestamp)

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
        except Exception:
            pass
    finally:
        # Flush remaining buffered audio into queue.
        try:
            leftover = stream_handler.flush()
            if leftover is not None:
                elapsed = time.time() - session_start_time
                await enqueue_chunk(leftover, format_timestamp(elapsed))
        except Exception as e:
            logger.debug(f"Stream flush skipped: {e}")

        try:
            await chunk_queue.join()
            await chunk_queue.put(None)
            await worker_task
        except Exception as e:
            logger.debug(f"Queue shutdown warning: {e}")

        # Explicit subtitle flush on stop/disconnect.
        try:
            for result in pipeline.process_flush():
                await send_result(result)
        except Exception as e:
            logger.debug(f"Pipeline flush warning: {e}")

        ping_task.cancel()
        worker_task.cancel()


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
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info"
    )
