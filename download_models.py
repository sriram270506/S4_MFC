"""
download_models.py — Pre-download all ML models
================================================

Run this ONCE before starting the server to pre-download all models.
This avoids downloading during the first WebSocket connection (which
can cause timeouts).

Usage:
  python models/download_models.py

Models downloaded:
  1. Faster-Whisper "base" (ASR)       — ~150MB
  2. SpeechBrain ECAPA-TDNN            — ~100MB
  3. Silero VAD                        — ~2MB
"""

import os
import sys
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_models")

# ── Directories ───────────────────────────────────────────
MODELS_DIR  = os.path.dirname(os.path.abspath(__file__))
WHISPER_DIR = os.path.join(MODELS_DIR, "whisper")
ECAPA_DIR   = os.path.join(MODELS_DIR, "speechbrain_ecapa")

os.makedirs(WHISPER_DIR, exist_ok=True)
os.makedirs(ECAPA_DIR, exist_ok=True)


def download_whisper(model_size: str = "base"):
    """Download Faster-Whisper model."""
    logger.info(f"=== Downloading Faster-Whisper ({model_size}) ===")
    try:
        from faster_whisper import WhisperModel
        start = time.time()
        model = WhisperModel(model_size, device="cpu", compute_type="int8", download_root=WHISPER_DIR)
        elapsed = time.time() - start
        logger.info(f"✓ Faster-Whisper '{model_size}' downloaded in {elapsed:.1f}s → {WHISPER_DIR}")
        del model
        return True
    except ImportError:
        logger.error("faster-whisper not installed. Run: pip install faster-whisper")
        return False
    except Exception as e:
        logger.error(f"Failed to download Whisper: {e}")
        return False


def download_ecapa_tdnn():
    """Download SpeechBrain ECAPA-TDNN speaker embedding model."""
    logger.info("=== Downloading SpeechBrain ECAPA-TDNN ===")
    try:
        from speechbrain.pretrained import EncoderClassifier
        import torch

        start = time.time()
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=ECAPA_DIR,
            run_opts={"device": "cpu"}
        )
        elapsed = time.time() - start
        logger.info(f"✓ ECAPA-TDNN downloaded in {elapsed:.1f}s → {ECAPA_DIR}")

        # Quick inference test
        logger.info("  Running inference test...")
        import numpy as np
        dummy_audio = torch.FloatTensor(np.random.randn(16000).astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            emb = classifier.encode_batch(dummy_audio)
        logger.info(f"  Embedding shape: {emb.shape} ✓")
        del classifier
        return True
    except ImportError:
        logger.error("speechbrain not installed. Run: pip install speechbrain")
        return False
    except Exception as e:
        logger.error(f"Failed to download ECAPA-TDNN: {e}")
        return False


def download_silero_vad():
    """Download Silero VAD model via torch.hub."""
    logger.info("=== Downloading Silero VAD ===")
    try:
        import torch
        import numpy as np

        start = time.time()
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        model.eval()
        elapsed = time.time() - start
        logger.info(f"✓ Silero VAD downloaded in {elapsed:.1f}s")

        # Quick test
        dummy = torch.FloatTensor(np.zeros(512, dtype=np.float32))
        with torch.no_grad():
            prob = model(dummy, 16000).item()
        logger.info(f"  VAD test probability: {prob:.3f} ✓")
        del model
        return True
    except Exception as e:
        logger.warning(f"Failed to download Silero VAD: {e} (will use EnergyVAD fallback)")
        return False


def verify_all():
    """Verify all required packages are installed."""
    logger.info("\n=== Verifying Installations ===")

    packages = {
        "fastapi":        "FastAPI",
        "uvicorn":        "Uvicorn",
        "numpy":          "NumPy",
        "torch":          "PyTorch",
        "faster_whisper": "Faster-Whisper",
        "speechbrain":    "SpeechBrain",
        "sklearn":        "Scikit-learn",
        "soundfile":      "SoundFile",
    }

    all_ok = True
    for pkg, name in packages.items():
        try:
            __import__(pkg)
            logger.info(f"  ✓ {name}")
        except ImportError:
            logger.error(f"  ✗ {name} — run: pip install {pkg}")
            all_ok = False

    return all_ok


if __name__ == "__main__":
    print("=" * 60)
    print("  LiveTranscribe — Model Downloader")
    print("=" * 60)
    print()

    # Step 1: Verify packages
    packages_ok = verify_all()
    if not packages_ok:
        print("\n⚠️  Some packages are missing. Install them first:")
        print("   pip install -r backend/requirements.txt")
        sys.exit(1)

    print()

    # Step 2: Download models
    results = {}
    results["whisper"]    = download_whisper("base")
    results["ecapa_tdnn"] = download_ecapa_tdnn()
    results["silero_vad"] = download_silero_vad()

    print()
    print("=" * 60)
    print("  Download Summary")
    print("=" * 60)
    for name, ok in results.items():
        status = "✓ OK" if ok else "✗ FAILED (check logs above)"
        print(f"  {name:<20} {status}")
    print()

    if all(results.values()):
        print("✅  All models ready! You can now start the server:")
        print("   python backend/main.py")
    else:
        print("⚠️  Some models failed to download.")
        print("   The system will use fallbacks where available.")
        print("   You can still start the server: python backend/main.py")
    print()
