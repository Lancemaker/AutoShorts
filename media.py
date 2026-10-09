"""FFmpeg/FFprobe operations and filesystem helpers."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

try:  # Supports direct execution and module execution.
    from .config import (
        INPUT,
        MAX_SHORT_SECONDS,
        OUTPUT,
        SHORTS_VIDEO_FILTER,
        SUPPORTED_VIDEO_EXTENSIONS,
        VIDEOS,
        WORK,
    )
except ImportError:  # pragma: no cover - used when run directly from this folder.
    from config import (  # type: ignore
        INPUT,
        MAX_SHORT_SECONDS,
        OUTPUT,
        SHORTS_VIDEO_FILTER,
        SUPPORTED_VIDEO_EXTENSIONS,
        VIDEOS,
        WORK,
    )


def run(command: list[str]) -> None:
    """Run an FFmpeg-family command and show it before execution."""
    print("\n> " + " ".join(command))
    subprocess.run(command, check=True)


def timestamp(seconds: float) -> str:
    """Render seconds in the compact MM:SS form used in cut labels."""
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def locate_input_dir() -> Path:
    """Prefer videos/input but retain compatibility with videos."""
    return INPUT if INPUT.exists() else VIDEOS


def find_videos() -> list[Path]:
    """Find supported source recordings without recursing into output folders."""
    folder = locate_input_dir()
    if not folder.is_dir():
        return []
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    )


def nonnegative_track(value: str) -> int:
    """Argparse validator for zero-based FFmpeg audio-track indexes."""
    track = int(value)
    if track < 0:
        raise argparse.ArgumentTypeError("Audio-track indexes start at 0.")
    return track


def validate_audio_tracks(video: Path, *tracks: int) -> None:
    """Fail before FFmpeg runs when any requested audio stream is absent."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    count = len(json.loads(result.stdout).get("streams", []))
    invalid = [track for track in sorted(set(tracks)) if track >= count]
    if invalid:
        available = f"0 to {count - 1}" if count else "none"
        requested = ", ".join(map(str, invalid))
        raise ValueError(
            f"{video.name} has {count} audio stream(s), with zero-based indexes "
            f"{available}. Requested invalid index(es): {requested}."
        )


def extract_audio(video: Path, track: int) -> Path:
    """Extract one whole OBS audio track as a mono 16 kHz WAV for analysis.

    This does not split left/right channels. The WAV is only an analysis input;
    the final MP4 keeps the original selected audio stream.
    """
    output = WORK / f"{video.stem}-track{track + 1}.wav"
    if output.exists() and output.stat().st_size:
        print(f"Using existing extracted audio: {output.name}")
        return output

    print(f"Extracting OBS audio track {track + 1} …")
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-map",
            f"0:a:{track}",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output),
        ]
    )
    return output


def video_duration(video: Path) -> float:
    """Read duration in seconds from the source container."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(video),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return float(result.stdout.strip())


def render_short(video: Path, report: dict, number: int, render_track: int) -> Path:
    """Render one fixed-center 9:16 MP4 using the selected original track."""
    start = float(report["start"])
    duration = min(float(report["end"]) - start, MAX_SHORT_SECONDS)
    output = OUTPUT / f"{report['title']}.mp4"
    print(
        f"Rendering Short {number}: {timestamp(start)}–"
        f"{timestamp(start + duration)} — {report['title']}"
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(start),
            "-i",
            str(video),
            "-t",
            str(duration),
            "-filter_complex",
            SHORTS_VIDEO_FILTER,
            "-map",
            "[v]",
            "-map",
            f"0:a:{render_track}",
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p5",
            "-cq",
            "20",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    return output
