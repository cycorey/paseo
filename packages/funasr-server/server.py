"""Standalone FunASR server for Chinese/English/Japanese speech-to-text."""

import io
import os
import sys
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

model: Optional[object] = None

# Fun-ASR repo must be on sys.path for model.py's sibling imports (ctc, tools.utils)
_FUN_ASR_DIR = str(Path(__file__).resolve().parent / "Fun-ASR")
if _FUN_ASR_DIR not in sys.path:
    sys.path.insert(0, _FUN_ASR_DIR)


def _wrap_pcm_in_wav(pcm_bytes: bytes, sample_rate: int = 16000, channels: int = 1, sample_width: int = 2) -> bytes:
    """Wrap raw PCM data (16kHz mono 16-bit) in a WAV header."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the FunASR model on startup."""
    global model
    from funasr import AutoModel

    model = AutoModel(
        model="FunAudioLLM/Fun-ASR-Nano-2512",
        trust_remote_code=True,
        remote_code=os.path.join(_FUN_ASR_DIR, "model.py"),
    )
    yield
    model = None


app = FastAPI(title="FunASR Server", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(status_code=503, content={"error": "Model not loaded"})

    audio_bytes = await file.read()

    # Detect raw PCM: content_type contains "pcm" or filename is "audio.pcm"
    is_pcm = False
    if file.content_type and "pcm" in file.content_type:
        is_pcm = True
    if file.filename and file.filename.endswith(".pcm"):
        is_pcm = True

    if is_pcm:
        audio_bytes = _wrap_pcm_in_wav(audio_bytes)

    # Write to temp file — Fun-ASR-Nano requires file path input, not raw bytes
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        result = model.generate(input=tmp_path, cache={}, batch_size=1, itn=True)
    finally:
        os.unlink(tmp_path)

    # FunASR returns a list of dicts; extract the text from the first result.
    text = ""
    if result and isinstance(result, list) and len(result) > 0:
        first = result[0]
        if isinstance(first, dict):
            text = first.get("text", "")
        else:
            text = str(first)

    return {"text": text}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=10095)
