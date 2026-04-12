"""Standalone FunASR server for Chinese/English/Japanese speech-to-text.

Provides:
- POST /transcribe — batch transcription (file upload)
- WS /ws/transcribe — streaming 2-pass transcription
- GET /health — readiness check
"""

import io
import json
import logging
import os
import struct
import sys
import tempfile
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

logger = logging.getLogger("funasr-server")

asr_model: Optional[object] = None
vad_model: Optional[object] = None

# Fun-ASR repo must be on sys.path for model.py's sibling imports (ctc, tools.utils)
_FUN_ASR_DIR = str(Path(__file__).resolve().parent / "Fun-ASR")
if _FUN_ASR_DIR not in sys.path:
    sys.path.insert(0, _FUN_ASR_DIR)

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # 16-bit
# Silence detection for streaming: peak < threshold for this duration triggers sentence boundary
SILENCE_THRESHOLD = 500  # int16 peak amplitude
SILENCE_DURATION_MS = 600  # ms of silence to trigger sentence end
SILENCE_DURATION_SAMPLES = int(SILENCE_DURATION_MS * SAMPLE_RATE / 1000)
# Minimum speech duration to bother transcribing (avoid tiny fragments)
MIN_SPEECH_SAMPLES = int(0.3 * SAMPLE_RATE)  # 300ms


def _wrap_pcm_in_wav(pcm_bytes: bytes, sample_rate: int = 16000, channels: int = 1, sample_width: int = 2) -> bytes:
    """Wrap raw PCM data in a WAV header."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def _read_wav_as_int16(wav_bytes: bytes) -> np.ndarray:
    """Read WAV bytes and return mono int16 numpy array."""
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        assert wf.getnchannels() == 1, "Expected mono audio"
        assert wf.getsampwidth() == 2, "Expected 16-bit audio"
        frames = wf.readframes(wf.getnframes())
    return np.frombuffer(frames, dtype=np.int16)


def _extract_segment(samples: np.ndarray, start_ms: int, end_ms: int, sample_rate: int) -> np.ndarray:
    """Extract a segment from audio samples given start/end in milliseconds."""
    start_sample = int(start_ms * sample_rate / 1000)
    end_sample = int(end_ms * sample_rate / 1000)
    return samples[start_sample:end_sample]


def _transcribe_segment(samples: np.ndarray) -> str:
    """Transcribe a numpy int16 audio segment. Returns text or empty string."""
    if len(samples) < MIN_SPEECH_SAMPLES:
        return ""
    wav_bytes = _wrap_pcm_in_wav(samples.tobytes())
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_bytes)
        tmp_path = tmp.name
    try:
        result = asr_model.generate(input=tmp_path, cache={}, batch_size=1, itn=True)
        if result and isinstance(result, list) and len(result) > 0:
            entry = result[0]
            t = entry.get("text", "") if isinstance(entry, dict) else str(entry)
            return t.strip()
    finally:
        os.unlink(tmp_path)
    return ""


def _transcribe_full_with_vad(all_samples: np.ndarray) -> str:
    """Run VAD + per-segment ASR on full audio (pass 2). Returns joined text."""
    if len(all_samples) < MIN_SPEECH_SAMPLES:
        return ""

    wav_bytes = _wrap_pcm_in_wav(all_samples.tobytes())
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_bytes)
        tmp_path = tmp.name

    try:
        vad_result = vad_model.generate(input=tmp_path)
    except Exception:
        os.unlink(tmp_path)
        raise

    segments = []
    if vad_result and isinstance(vad_result, list) and len(vad_result) > 0:
        segments = vad_result[0].get("value", [])

    if not segments:
        os.unlink(tmp_path)
        return ""

    texts = []
    for start_ms, end_ms in segments:
        seg = _extract_segment(all_samples, start_ms, end_ms, SAMPLE_RATE)
        if len(seg) == 0:
            continue
        t = _transcribe_segment(seg)
        if t:
            texts.append(t)

    os.unlink(tmp_path)
    return "".join(texts)


class StreamingSilenceDetector:
    """Detects sentence boundaries by tracking silence duration in streaming audio."""

    def __init__(self):
        self.speech_started = False
        self.silence_count = 0  # consecutive silent samples
        self.speech_start_idx = 0  # sample index where current speech started
        self.total_samples = 0

    def feed(self, samples: np.ndarray) -> list[tuple[int, int]]:
        """Feed new samples, return list of (start_idx, end_idx) for completed sentences."""
        boundaries = []
        # Process in small windows for efficiency
        window_size = int(SAMPLE_RATE * 0.03)  # 30ms windows

        for i in range(0, len(samples), window_size):
            window = samples[i:i + window_size]
            if len(window) == 0:
                continue

            peak = int(np.max(np.abs(window)))
            is_silent = peak < SILENCE_THRESHOLD

            if not self.speech_started:
                if not is_silent:
                    self.speech_started = True
                    self.speech_start_idx = self.total_samples + i
                    self.silence_count = 0
            else:
                if is_silent:
                    self.silence_count += len(window)
                    if self.silence_count >= SILENCE_DURATION_SAMPLES:
                        # Sentence boundary detected
                        end_idx = self.total_samples + i - self.silence_count + len(window)
                        boundaries.append((self.speech_start_idx, end_idx))
                        self.speech_started = False
                        self.silence_count = 0
                else:
                    self.silence_count = 0

        self.total_samples += len(samples)
        return boundaries


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    global asr_model, vad_model
    from funasr import AutoModel

    logger.info("Loading VAD model...")
    vad_model = AutoModel(model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch", disable_update=True)

    logger.info("Loading Fun-ASR-Nano-2512 model...")
    asr_model = AutoModel(
        model="FunAudioLLM/Fun-ASR-Nano-2512",
        trust_remote_code=True,
        remote_code=os.path.join(_FUN_ASR_DIR, "model.py"),
        disable_update=True,
    )
    logger.info("Models loaded successfully")
    yield
    asr_model = None
    vad_model = None


app = FastAPI(title="FunASR Server", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": asr_model is not None and vad_model is not None}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """Batch transcription endpoint (kept for non-streaming use)."""
    if asr_model is None or vad_model is None:
        return JSONResponse(status_code=503, content={"error": "Models not loaded"})

    audio_bytes = await file.read()

    is_pcm = False
    if file.content_type and "pcm" in file.content_type:
        is_pcm = True
    if file.filename and file.filename.endswith(".pcm"):
        is_pcm = True

    if is_pcm:
        audio_bytes = _wrap_pcm_in_wav(audio_bytes)

    samples = _read_wav_as_int16(audio_bytes)
    text = _transcribe_full_with_vad(samples)
    return {"text": text}


@app.websocket("/ws/transcribe")
async def ws_transcribe(ws: WebSocket):
    """Streaming 2-pass transcription.

    Client sends:
      - Binary frames: PCM16 16kHz mono audio chunks
      - Text frame: {"type": "finish"} when recording ends

    Server sends:
      - {"type": "partial", "text": "..."} per detected sentence (pass 1)
      - {"type": "final", "text": "..."}  full re-transcription (pass 2)
    """
    await ws.accept()

    if asr_model is None or vad_model is None:
        await ws.send_json({"type": "error", "error": "Models not loaded"})
        await ws.close()
        return

    detector = StreamingSilenceDetector()
    all_chunks: list[bytes] = []  # raw PCM bytes for full buffer
    all_samples_list: list[np.ndarray] = []  # numpy arrays for indexing
    partial_texts: list[str] = []

    try:
        while True:
            message = await ws.receive()

            if message["type"] == "websocket.disconnect":
                break

            if "bytes" in message and message["bytes"]:
                pcm_bytes = message["bytes"]
                all_chunks.append(pcm_bytes)
                samples = np.frombuffer(pcm_bytes, dtype=np.int16)
                all_samples_list.append(samples)

                # Check for sentence boundaries
                boundaries = detector.feed(samples)
                if boundaries and len(all_samples_list) > 0:
                    full_so_far = np.concatenate(all_samples_list)
                    for start_idx, end_idx in boundaries:
                        seg = full_so_far[start_idx:end_idx]
                        text = _transcribe_segment(seg)
                        if text:
                            partial_texts.append(text)
                            combined = "".join(partial_texts)
                            await ws.send_json({"type": "partial", "text": combined})

            elif "text" in message and message["text"]:
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue

                if data.get("type") == "finish":
                    # Pass 2: full re-transcription with VAD model
                    if all_samples_list:
                        full_audio = np.concatenate(all_samples_list)
                        final_text = _transcribe_full_with_vad(full_audio)
                    else:
                        final_text = ""

                    await ws.send_json({"type": "final", "text": final_text})
                    break

    except WebSocketDisconnect:
        logger.debug("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await ws.send_json({"type": "error", "error": str(e)})
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="127.0.0.1", port=10095)
