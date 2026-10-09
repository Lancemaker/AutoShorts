"""Whisper transcription and its reusable cache."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from .config import VOICE_TRACK, WHISPER_MODEL, WORK, configure_cuda_dlls
except ImportError:  # pragma: no cover - used when run directly from this folder.
    from config import VOICE_TRACK, WHISPER_MODEL, WORK, configure_cuda_dlls  # type: ignore


configure_cuda_dlls()


def transcribe(
    video: Path,
    audio: Path,
    voice_track: int,
    force: bool,
) -> list[dict]:
    """Transcribe one microphone track with CUDA Whisper, caching the result."""
    cache = WORK / f"{video.stem}-transcript-track{voice_track + 1}.json"
    if cache.exists() and not force:
        print(f"Using saved transcript: {cache.name}")
        return json.loads(cache.read_text(encoding="utf-8"))

    # Retain the cache created by the prior fixed-track version.
    legacy_cache = WORK / f"{video.stem}-transcript.json"
    if voice_track == VOICE_TRACK and legacy_cache.exists() and not force:
        print(f"Using saved transcript: {legacy_cache.name}")
        return json.loads(legacy_cache.read_text(encoding="utf-8"))

    try:
        from faster_whisper import WhisperModel
    except ImportError as error:
        raise RuntimeError(
            "Whisper transcription requires faster-whisper. Install it with: "
            "python -m pip install faster-whisper"
        ) from error

    print("Loading Whisper on CUDA …")
    whisper = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
    print("Transcribing microphone track …")
    segments, _ = whisper.transcribe(str(audio), vad_filter=True, language="en")
    result = [
        {
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text.strip(),
        }
        for segment in segments
        if segment.text.strip()
    ]
    if not result:
        raise RuntimeError(
            f"No speech was recognized from zero-based audio index {voice_track}."
        )

    cache.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result
