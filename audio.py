"""Game-audio peak analysis."""

from __future__ import annotations

import audioop
import math
import wave
from pathlib import Path

try:
    from .config import (
        EVENT_CONTEXT_AFTER_SECONDS,
        EVENT_CONTEXT_BEFORE_SECONDS,
        MAX_AUDIO_CANDIDATES,
    )
except ImportError:  # pragma: no cover - used when run directly from this folder.
    from config import (  # type: ignore
        EVENT_CONTEXT_AFTER_SECONDS,
        EVENT_CONTEXT_BEFORE_SECONDS,
        MAX_AUDIO_CANDIDATES,
    )


def percentile(values: list[float], fraction: float) -> float:
    """Return a deterministic percentile from a non-empty numeric sequence."""
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def game_audio_peaks(audio: Path) -> list[dict]:
    """Find isolated high-energy moments in game-only audio, once per second."""
    with wave.open(str(audio), "rb") as wav:
        rate = wav.getframerate()
        width = wav.getsampwidth()
        channels = wav.getnchannels()
        energies: list[float] = []
        while data := wav.readframes(rate):
            if channels > 1:
                data = audioop.tomono(data, width, 0.5, 0.5)
            energies.append(math.log1p(audioop.rms(data, width)))

    if not energies:
        return []

    threshold = percentile(energies, 0.92)
    local_peaks: list[tuple[int, float]] = []
    for index, value in enumerate(energies):
        left = energies[max(0, index - 4) : index]
        right = energies[index + 1 : index + 5]
        if value >= threshold and (not left or value >= max(left)) and (
            not right or value >= max(right)
        ):
            local_peaks.append((index, value))

    selected: list[tuple[int, float]] = []
    for second, value in sorted(local_peaks, key=lambda item: item[1], reverse=True):
        if all(abs(second - chosen_second) >= 18 for chosen_second, _ in selected):
            selected.append((second, value))
        if len(selected) == MAX_AUDIO_CANDIDATES:
            break

    # Continuous engine, wind, or gunfire can lack a single local maximum.
    # In that case, pass the relatively loudest separated moments to vision.
    if not selected:
        print("No isolated audio peaks found; using relatively loud moments instead.")
        for second, value in sorted(
            enumerate(energies), key=lambda item: item[1], reverse=True
        ):
            if all(abs(second - chosen_second) >= 18 for chosen_second, _ in selected):
                selected.append((second, value))
            if len(selected) == MAX_AUDIO_CANDIDATES:
                break

    floor = min(energies)
    ceiling = max(energies)
    return sorted(
        [
            {
                "event_time": second,
                "energy": round((value - floor) / max(ceiling - floor, 0.001), 3),
                "start": max(0, second - EVENT_CONTEXT_BEFORE_SECONDS),
                "end": second + EVENT_CONTEXT_AFTER_SECONDS,
            }
            for second, value in selected
        ],
        key=lambda item: item["start"],
    )
