"""Frame extraction, Gemma prompts, visual cache, and candidate triage."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

try:
    import requests
except ImportError:  # Allow --help and static checks before dependencies are installed.
    requests = None  # type: ignore

try:
    from .config import (
        CACHE_VERSION,
        COARSE_BATCH_SIZE,
        GAME_TRACK,
        OLLAMA_URL,
        VISION_MODEL,
        VOICE_TRACK,
        WORK,
    )
    from .media import run, timestamp
except ImportError:  # pragma: no cover - used when run directly from this folder.
    from config import (  # type: ignore
        CACHE_VERSION,
        COARSE_BATCH_SIZE,
        GAME_TRACK,
        OLLAMA_URL,
        VISION_MODEL,
        VOICE_TRACK,
        WORK,
    )
    from media import run, timestamp  # type: ignore


def ollama_text(
    prompt: str,
    model: str,
    images: list[Path] | None = None,
    *,
    num_predict: int | None = None,
    temperature: float | None = None,
) -> str:
    """Ask Ollama for plain text; Python owns all structured records."""
    if requests is None:
        raise RuntimeError(
            "Visual analysis requires requests. Install it with: "
            "python -m pip install requests"
        )

    message: dict = {"role": "user", "content": prompt}
    if images:
        message["images"] = [
            base64.b64encode(path.read_bytes()).decode("ascii") for path in images
        ]

    vision_request = model == VISION_MODEL
    payload: dict = {
        "model": model,
        "stream": False,
        "messages": [message],
        "options": {
            "temperature": 0.1 if temperature is None else temperature,
            "num_ctx": 8192 if vision_request else 4096,
            "num_predict": num_predict if num_predict is not None else (
                220 if vision_request else 180
            ),
            "repeat_penalty": 1.1 if vision_request else 1.0,
        },
    }

    # A token-repeat abort is an Ollama runtime issue. Retrying a fresh request
    # is safe because this module has not mutated any analysis state yet.
    for attempt in range(1, 3):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=600)
        except requests.RequestException as error:
            raise RuntimeError(f"Could not reach Ollama at {OLLAMA_URL}: {error}") from error
        if response.ok:
            content = str(response.json().get("message", {}).get("content", "")).strip()
            if content:
                return content
            error = "Ollama returned empty content"
        else:
            detail = response.text.strip().replace("\n", " ")[:1200]
            error = f"Ollama HTTP {response.status_code}: {detail}"

        if attempt == 1 and "token repeat limit reached" in error:
            print("  Ollama repeated tokens; retrying the same local request once …")
            continue
        raise RuntimeError(error)

    raise RuntimeError("Ollama failed without an error message")


def extract_frame(video: Path, second: float, label: str) -> Path:
    """Extract and cache a reduced-size JPEG frame from a source recording."""
    folder = WORK / "frames" / video.stem
    folder.mkdir(parents=True, exist_ok=True)
    safe_label = re.sub(r"[^a-zA-Z0-9_-]", "_", label)
    image = folder / f"{safe_label}-{second:08.2f}.jpg"
    if not image.exists():
        run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{second:.2f}",
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-vf",
                "scale=960:-2",
                "-q:v",
                "3",
                "-update",
                "1",
                str(image),
            ]
        )
    return image


def visual_map_path(video: Path) -> Path:
    return WORK / f"{video.stem}-visual-map-{CACHE_VERSION}.json"


def coarse_visual_map(
    video: Path,
    duration: float,
    every: int,
    force: bool,
) -> list[dict]:
    """Build coarse, plain-language evidence from frames across the recording."""
    cache = visual_map_path(video)
    if cache.exists() and not force:
        print(f"Using saved visual map: {cache.name}")
        return json.loads(cache.read_text(encoding="utf-8"))

    times = list(range(0, int(duration), every))
    records: list[dict] = []
    print(f"Building visual map: {len(times)} frames, one every {every} seconds …")
    for batch_start in range(0, len(times), COARSE_BATCH_SIZE):
        seconds = times[batch_start : batch_start + COARSE_BATCH_SIZE]
        images = [
            extract_frame(video, second, f"coarse-{second}") for second in seconds
        ]
        labels = ", ".join(
            f"Frame {index + 1}: {timestamp(second)}"
            for index, second in enumerate(seconds)
        )
        prompt = f"""You are reviewing {len(images)} independent sampled frames from a first-person gameplay recording.

{labels}

The frames are separated in time. Do not assume that one frame directly causes the next.

Describe each frame in chronological order. For each one, state what the player is visibly doing, the main activity (quiet gameplay, travel, setup, combat, aftermath, or menu), and any clearly visible event or result. Say "no clear event visible" when appropriate.

Focus on visible gameplay and UI evidence. Do not infer kills, explosions, objectives, weapons, or outcomes unless they are clearly visible. End with one short sentence describing the overall context of these sampled moments.

Use plain text only. Do not use JSON, Markdown tables, or code blocks."""
        try:
            description = ollama_text(prompt, VISION_MODEL, images)
            records.append(
                {
                    "frame_seconds": seconds,
                    "frame_times": [timestamp(second) for second in seconds],
                    "description": description,
                }
            )
            print(
                f"  Mapped {records[-1]['frame_times'][0]}–"
                f"{records[-1]['frame_times'][-1]}"
            )
        except RuntimeError as error:
            print(f"Visual-map batch at {timestamp(seconds[0])} failed: {error}")

    cache.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return records


def speech_near(segments: list[dict], start: float, end: float) -> str:
    """Return the transcript that overlaps a candidate window."""
    return " ".join(
        segment["text"]
        for segment in segments
        if segment["end"] >= start and segment["start"] <= end
    )[:1400]


def relevant_map(records: list[dict], start: float, end: float) -> str:
    """Return up to two nearby coarse evidence descriptions."""
    nearby: list[dict] = []
    for record in records:
        frame_seconds = record.get("frame_seconds", [])
        if any(start - 45 <= second <= end + 45 for second in frame_seconds):
            nearby.append(record)
    if not nearby:
        return "[No nearby coarse-frame description.]"

    return "\n\n".join(
        f"Sampled frames {', '.join(record['frame_times'])}:\n"
        f"{record['description'][:900]}"
        for record in nearby[:2]
    )


def candidate_path(video: Path, game_track: int, voice_track: int) -> Path:
    """Return the cache name that incorporates the analysis-track indexes."""
    return WORK / (
        f"{video.stem}-visual-candidates-{CACHE_VERSION}-"
        f"game{game_track + 1}-voice{voice_track + 1}.json"
    )


def verdict_from_text(text: str) -> str:
    """Read the fixed trailing verdict from an otherwise free-text response."""
    match = re.search(
        r"(?im)^\s*VERDICT\s*:\s*(PAYOFF|NO_PAYOFF|UNCERTAIN)\s*$",
        text,
    )
    return match.group(1).upper() if match else "UNCERTAIN"


def local_candidate_score(verdict: str, energy: float) -> int:
    """Combine visual triage and audio intensity into a sorting score."""
    baseline = {"PAYOFF": 70, "UNCERTAIN": 25, "NO_PAYOFF": 0}[verdict]
    return min(100, baseline + round(25 * energy))


def visual_candidates(
    video: Path,
    duration: float,
    audio_events: list[dict],
    speech: list[dict],
    visual_map: list[dict],
    game_track: int,
    voice_track: int,
    force: bool,
) -> list[dict]:
    """Validate loud game moments with event frames, transcript, and context."""
    cache = candidate_path(video, game_track, voice_track)
    if cache.exists() and not force:
        print(f"Using saved visual candidate analysis: {cache.name}")
        return json.loads(cache.read_text(encoding="utf-8"))

    # The legacy cache was made with the original fixed indexes: game 0,
    # game+voice 1. Reuse it only for that exact compatibility case.
    legacy_cache = WORK / f"{video.stem}-visual-candidates-{CACHE_VERSION}.json"
    if (
        game_track == GAME_TRACK
        and voice_track == VOICE_TRACK
        and legacy_cache.exists()
        and not force
    ):
        print(f"Using saved visual candidate analysis: {legacy_cache.name}")
        return json.loads(legacy_cache.read_text(encoding="utf-8"))

    reports: list[dict] = []
    print(f"Visually validating {len(audio_events)} loud game-audio events …")
    for event in audio_events:
        start = float(event["start"])
        end = min(duration, float(event["end"]))
        middle = float(event["event_time"])
        frame_seconds = [start + 2, middle, min(end - 1, middle + 8)]
        images = [
            extract_frame(video, second, f"candidate-{middle}-{index}")
            for index, second in enumerate(frame_seconds)
        ]
        voice = speech_near(speech, start, end) or "[no clear speech]"
        map_context = relevant_map(visual_map, start, end)
        prompt = f"""You are the visual quality gate for an FPS YouTube Short.

The three images are close together and chronological: before ({timestamp(frame_seconds[0])}), event ({timestamp(frame_seconds[1])}), after ({timestamp(frame_seconds[2])}). A peak in game-only audio occurred at {timestamp(middle)}.

Nearby microphone transcript:
{voice}

Nearby coarse visual observations:
{map_context}

Describe only what the three event frames visibly prove: setup, player action, and visible result. Reject menu screens, waiting/lobby screens, travel, preparation, and unresolved action. Do not call an enemy dead, an explosion successful, or an objective complete unless the frames visibly support it.

Use plain text. Finish with exactly one final line:
VERDICT: PAYOFF
or
VERDICT: NO_PAYOFF
or
VERDICT: UNCERTAIN"""
        try:
            description = ollama_text(prompt, VISION_MODEL, images)
            verdict = verdict_from_text(description)
            report = {
                "start": round(start, 1),
                "end": round(end, 1),
                "event_time": middle,
                "audio_energy": event["energy"],
                "voice": voice,
                "coarse_context": map_context,
                "visual_description": description,
                "verdict": verdict,
                "local_score": local_candidate_score(verdict, event["energy"]),
            }
            reports.append(report)
            print(
                f"  {timestamp(middle)}: {verdict} — "
                f"score {report['local_score']}"
            )
        except RuntimeError as error:
            print(f"Candidate {timestamp(middle)} failed: {error}")

    cache.write_text(
        json.dumps(reports, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return reports
