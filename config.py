"""Configuration shared by the Shorts automation modules.

Keep model choices, video crop settings, and default OBS audio tracks here.
Track indexes are zero-based, matching FFmpeg's ``0:a:N`` notation.
"""

from __future__ import annotations

import os
import site
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VIDEOS = ROOT / "videos"
INPUT = VIDEOS / "input"
OUTPUT = VIDEOS / "output"
WORK = ROOT / "work"

SUPPORTED_VIDEO_EXTENSIONS = {".mkv", ".mp4", ".mov", ".webm", ".avi"}

OLLAMA_URL = "http://localhost:11434/api/chat"
VISION_MODEL = "gemma3:4b"
WHISPER_MODEL = "small.en"

# OBS stream indexes are zero-based.
# The defaults preserve the original two-track workflow:
#   0 = game audio, 1 = game + microphone.
GAME_TRACK = 0
VOICE_TRACK = 1
RENDER_TRACK = VOICE_TRACK

COARSE_EVERY_SECONDS = 45
COARSE_BATCH_SIZE = 4
MAX_AUDIO_CANDIDATES = 12
EVENT_CONTEXT_BEFORE_SECONDS = 12
EVENT_CONTEXT_AFTER_SECONDS = 24
MAX_SHORT_SECONDS = 59.0
MERGE_MAX_GAP_SECONDS = 8.0

# The visual cache can survive scoring/selection-rule changes.
CACHE_VERSION = "v3-text"
REVIEW_QUEUE_VERSION = "v5-merged-numbered"
MIN_DUPLICATE_OVERLAP_SECONDS = 8.0

# Fixed, centered 9:16 crop. It intentionally does not use background blur
# or a crop-selection model.
SHORTS_VIDEO_FILTER = (
    "[0:v]"
    "crop=trunc(ih*9/16/2)*2:ih:trunc((iw-trunc(ih*9/16/2)*2)/2):0,"
    "scale=1080:1920:flags=lanczos,format=yuv420p[v]"
)


def configure_cuda_dlls() -> None:
    """Make pip-installed CUDA 12 DLLs visible to CTranslate2 on Windows."""
    if os.name != "nt":
        return

    folders: list[str] = []
    for base in map(Path, site.getsitepackages()):
        for name in ("cublas", "cudnn", "cuda_runtime"):
            folder = base / "nvidia" / name / "bin"
            if folder.is_dir():
                folders.append(str(folder))

    if folders:
        os.environ["PATH"] = os.pathsep.join(folders + [os.environ.get("PATH", "")])
        for folder in folders:
            os.add_dll_directory(folder)
