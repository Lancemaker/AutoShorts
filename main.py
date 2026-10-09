"""Create a local human-review queue of FPS Shorts from OBS recordings.

The pipeline combines:

* game-only audio peaks;
* a Whisper transcript from the microphone track;
* Gemma frame descriptions returned as plain text;
* Python-owned JSON caches and review decisions; and
* a final Excel workbook for the human reviewer.

Clear payoffs and uncertain-but-promising moments are rendered. Only clear
non-events are skipped. Nothing is uploaded or published.
"""

from __future__ import annotations

import argparse

try:  # Supports direct execution and module execution.
    from .audio import game_audio_peaks
    from .config import (
        COARSE_EVERY_SECONDS,
        GAME_TRACK,
        OUTPUT,
        RENDER_TRACK,
        VOICE_TRACK,
        WORK,
    )
    from .media import (
        extract_audio,
        find_videos,
        locate_input_dir,
        nonnegative_track,
        render_short,
        validate_audio_tracks,
        video_duration,
    )
    from .review import human_review_queue, write_review_workbook
    from .speech import transcribe
    from .vision import coarse_visual_map, visual_candidates
except ImportError:  # pragma: no cover - used when run directly from this folder.
    from audio import game_audio_peaks  # type: ignore
    from config import (  # type: ignore
        COARSE_EVERY_SECONDS,
        GAME_TRACK,
        OUTPUT,
        RENDER_TRACK,
        VOICE_TRACK,
        WORK,
    )
    from media import (  # type: ignore
        extract_audio,
        find_videos,
        locate_input_dir,
        nonnegative_track,
        render_short,
        validate_audio_tracks,
        video_duration,
    )
    from review import human_review_queue, write_review_workbook  # type: ignore
    from speech import transcribe  # type: ignore
    from vision import coarse_visual_map, visual_candidates  # type: ignore


def parse_args() -> argparse.Namespace:
    """Define and parse the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-transcribe",
        action="store_true",
        help="Ignore the saved Whisper transcript.",
    )
    parser.add_argument(
        "--refresh-visual",
        action="store_true",
        help="Rebuild plain-text visual evidence caches.",
    )
    parser.add_argument(
        "--refresh-candidates",
        action="store_true",
        help="Rebuild event candidates using the saved visual map.",
    )
    parser.add_argument(
        "--coarse-every",
        type=int,
        default=COARSE_EVERY_SECONDS,
        help="Seconds between storyboard frames (default: 45).",
    )
    parser.add_argument(
        "--game-track",
        type=nonnegative_track,
        default=GAME_TRACK,
        help="Zero-based game-only track used for loud-event analysis (default: 0).",
    )
    parser.add_argument(
        "--voice-track",
        type=nonnegative_track,
        default=VOICE_TRACK,
        help="Zero-based microphone track used by Whisper (default: 1).",
    )
    parser.add_argument(
        "--render-track",
        "--output-track",
        dest="render_track",
        type=nonnegative_track,
        default=RENDER_TRACK,
        help=(
            "Zero-based original track copied into each Short "
            "(default: same as the voice track)."
        ),
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Analyze and build the review queue, but do not create MP4 files or Excel.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full local analysis, human-review queue, and render flow."""
    args = parse_args()

    if args.coarse_every <= 0:
        raise ValueError("--coarse-every must be greater than zero.")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    videos = find_videos()
    if not videos:
        raise FileNotFoundError(f"No video found in {locate_input_dir()}")

    for video in videos:
        print(f"\nProcessing {video.name}")
        validate_audio_tracks(
            video,
            args.game_track,
            args.voice_track,
            args.render_track,
        )
        print(
            "Audio indexes (zero-based): "
            f"game={args.game_track}, voice={args.voice_track}, "
            f"render={args.render_track}"
        )

        voice_audio = extract_audio(video, args.voice_track)
        game_audio = extract_audio(video, args.game_track)
        speech = transcribe(
            video,
            voice_audio,
            args.voice_track,
            args.force_transcribe,
        )
        duration = video_duration(video)
        visual_map = coarse_visual_map(
            video,
            duration,
            args.coarse_every,
            args.refresh_visual,
        )
        reports = visual_candidates(
            video,
            duration,
            game_audio_peaks(game_audio),
            speech,
            visual_map,
            args.game_track,
            args.voice_track,
            args.refresh_visual or args.refresh_candidates,
        )
        selected = human_review_queue(
            video,
            reports,
            args.game_track,
            args.voice_track,
        )
        high_count = sum(
            report["priority"] == "ALTO POTENCIAL" for report in selected
        )
        review_count = len(selected) - high_count
        skipped_count = sum(
            report["verdict"] == "NO_PAYOFF" for report in reports
        )
        print(
            f"Human-review queue: {high_count} high-potential + "
            f"{review_count} uncertain. Skipped {skipped_count} "
            "no-interest candidates."
        )

        if args.no_render:
            continue

        rendered: list[dict] = []
        for number, report in enumerate(selected, start=1):
            output = render_short(
                video,
                report,
                number,
                args.render_track,
            )
            rendered.append(report | {"rendered_file": str(output)})

        if rendered:
            workbook = write_review_workbook(
                video,
                rendered,
                skipped_count,
                visual_map,
            )
            print(f"Excel review workbook: {workbook}")


if __name__ == "__main__":
    main()
