"""Create a local human-review queue of FPS Shorts from OBS recordings.

This version deliberately keeps language models away from JSON generation:

* OBS recording 1 (game only) finds loud/action events.
* OBS recording 2 (game + microphone) is transcribed by Whisper.
* Gemma sees frames and returns plain-language visual descriptions only.
* Python owns timestamps, cache files, the review queue, and every JSON record.
* You are the final judge: clear payoffs and uncertain-but-promising moments
  are rendered for review; only clear non-events are skipped.
* After rendering, an Excel workbook organizes every cut and its evidence.

Nothing is uploaded or published. Output always uses a fixed central 9:16 crop.
"""

from __future__ import annotations

import argparse
import audioop
import base64
import json
import math
import os
import re
import site
import subprocess
import wave
from datetime import datetime, timedelta
from pathlib import Path

import requests
from faster_whisper import WhisperModel


ROOT = Path(__file__).resolve().parent
VIDEOS = ROOT / "videos"
INPUT = VIDEOS / "input"
OUTPUT = VIDEOS / "output"
WORK = ROOT / "work"

OLLAMA_URL = "http://localhost:11434/api/chat"
VISION_MODEL = "gemma3:4b"
WHISPER_MODEL = "small.en"

# OBS stream indexes are zero-based. recording1 = game only; recording2 = game + voice.
GAME_TRACK = 0
VOICE_TRACK = 1

COARSE_EVERY_SECONDS = 45
COARSE_BATCH_SIZE = 4
MAX_AUDIO_CANDIDATES = 12
# Keep the proven visual evidence cache. Selection rules can evolve without
# making Gemma inspect all 65 storyboard frames again.
CACHE_VERSION = "v3-text"
REVIEW_QUEUE_VERSION = "v4-human-review"
MIN_DUPLICATE_OVERLAP_SECONDS = 8.0


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


configure_cuda_dlls()


def run(command: list[str]) -> None:
    print("\n> " + " ".join(command))
    subprocess.run(command, check=True)


def timestamp(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def locate_input_dir() -> Path:
    return INPUT if INPUT.exists() else VIDEOS


def find_videos() -> list[Path]:
    extensions = {".mkv", ".mp4", ".mov", ".webm", ".avi"}
    return sorted(path for path in locate_input_dir().iterdir() if path.suffix.lower() in extensions)


def extract_audio(video: Path, track: int) -> Path:
    output = WORK / f"{video.stem}-track{track + 1}.wav"
    if output.exists() and output.stat().st_size:
        print(f"Using existing extracted audio: {output.name}")
        return output
    print(f"Extracting OBS audio track {track + 1} …")
    run([
        "ffmpeg", "-y", "-i", str(video), "-map", f"0:a:{track}",
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(output),
    ])
    return output


def transcribe(video: Path, audio: Path, force: bool) -> list[dict]:
    cache = WORK / f"{video.stem}-transcript.json"
    if cache.exists() and not force:
        print(f"Using saved transcript: {cache.name}")
        return json.loads(cache.read_text(encoding="utf-8"))
    print("Loading Whisper on CUDA …")
    whisper = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
    print("Transcribing microphone track …")
    segments, _ = whisper.transcribe(str(audio), vad_filter=True, language="en")
    result = [
        {"start": round(segment.start, 2), "end": round(segment.end, 2), "text": segment.text.strip()}
        for segment in segments
        if segment.text.strip()
    ]
    if not result:
        raise RuntimeError("No speech was recognized from OBS recording 2.")
    cache.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def ollama_text(
    prompt: str,
    model: str,
    images: list[Path] | None = None,
    *,
    num_predict: int | None = None,
    temperature: float | None = None,
) -> str:
    """Return plain text. No model response is used as program JSON."""
    message: dict = {"role": "user", "content": prompt}
    if images:
        message["images"] = [base64.b64encode(path.read_bytes()).decode("ascii") for path in images]

    vision_request = model == VISION_MODEL
    payload: dict = {
        "model": model,
        "stream": False,
        "messages": [message],
        "options": {
            "temperature": 0.1 if temperature is None else temperature,
            "num_ctx": 8192 if vision_request else 4096,
            "num_predict": num_predict if num_predict is not None else (220 if vision_request else 180),
            "repeat_penalty": 1.1 if vision_request else 1.0,
        },
    }
    # A token-repeat abort is an Ollama runtime issue seen with some local
    # models. Retrying a fresh request is safe because no state is mutated.
    for attempt in range(1, 3):
        response = requests.post(OLLAMA_URL, json=payload, timeout=600)
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


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))]


def game_audio_peaks(audio: Path) -> list[dict]:
    """Find isolated high-energy moments in the game-only audio, once per second."""
    with wave.open(str(audio), "rb") as wav:
        rate, width, channels = wav.getframerate(), wav.getsampwidth(), wav.getnchannels()
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
        left = energies[max(0, index - 4):index]
        right = energies[index + 1:index + 5]
        if value >= threshold and (not left or value >= max(left)) and (not right or value >= max(right)):
            local_peaks.append((index, value))

    selected: list[tuple[int, float]] = []
    for second, value in sorted(local_peaks, key=lambda item: item[1], reverse=True):
        if all(abs(second - chosen_second) >= 18 for chosen_second, _ in selected):
            selected.append((second, value))
        if len(selected) == MAX_AUDIO_CANDIDATES:
            break

    # Continuous engine, wind, or gunfire can lack a single local maximum.
    # Use the relatively loudest moments instead of returning no visual work.
    if not selected:
        print("No isolated audio peaks found; using relatively loud moments instead.")
        for second, value in sorted(enumerate(energies), key=lambda item: item[1], reverse=True):
            if all(abs(second - chosen_second) >= 18 for chosen_second, _ in selected):
                selected.append((second, value))
            if len(selected) == MAX_AUDIO_CANDIDATES:
                break

    floor, ceiling = min(energies), max(energies)
    return sorted([
        {
            "event_time": second,
            "energy": round((value - floor) / max(ceiling - floor, 0.001), 3),
            "start": max(0, second - 7),
            "end": second + 26,
        }
        for second, value in selected
    ], key=lambda item: item["start"])


def video_duration(video: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(video)],
        check=True,
        text=True,
        capture_output=True,
    )
    return float(result.stdout.strip())


def extract_frame(video: Path, second: float, label: str) -> Path:
    folder = WORK / "frames" / video.stem
    folder.mkdir(parents=True, exist_ok=True)
    safe_label = re.sub(r"[^a-zA-Z0-9_-]", "_", label)
    image = folder / f"{safe_label}-{second:08.2f}.jpg"
    if not image.exists():
        run([
            "ffmpeg", "-y", "-ss", f"{second:.2f}", "-i", str(video),
            "-frames:v", "1", "-vf", "scale=960:-2", "-q:v", "3", "-update", "1", str(image),
        ])
    return image


def visual_map_path(video: Path) -> Path:
    return WORK / f"{video.stem}-visual-map-{CACHE_VERSION}.json"


def coarse_visual_map(video: Path, duration: float, every: int, force: bool) -> list[dict]:
    cache = visual_map_path(video)
    if cache.exists() and not force:
        print(f"Using saved visual map: {cache.name}")
        return json.loads(cache.read_text(encoding="utf-8"))

    times = list(range(0, int(duration), every))
    records: list[dict] = []
    print(f"Building visual map: {len(times)} frames, one every {every} seconds …")
    for batch_start in range(0, len(times), COARSE_BATCH_SIZE):
        seconds = times[batch_start:batch_start + COARSE_BATCH_SIZE]
        images = [extract_frame(video, second, f"coarse-{second}") for second in seconds]
        labels = ", ".join(f"Frame {index + 1}: {timestamp(second)}" for index, second in enumerate(seconds))
        prompt = f"""You are reviewing {len(images)} independent sampled frames from a first-person gameplay recording.

{labels}

The frames are separated in time. Do not assume that one frame directly causes the next.

Describe each frame in chronological order. For each one, state what the player is visibly doing, the main activity (quiet gameplay, travel, setup, combat, aftermath, or menu), and any clearly visible event or result. Say "no clear event visible" when appropriate.

Focus on visible gameplay and UI evidence. Do not infer kills, explosions, objectives, weapons, or outcomes unless they are clearly visible. End with one short sentence describing the overall context of these sampled moments.

Use plain text only. Do not use JSON, Markdown tables, or code blocks."""
        try:
            description = ollama_text(prompt, VISION_MODEL, images)
            records.append({
                "frame_seconds": seconds,
                "frame_times": [timestamp(second) for second in seconds],
                "description": description,
            })
            print(f"  Mapped {records[-1]['frame_times'][0]}–{records[-1]['frame_times'][-1]}")
        except (requests.RequestException, RuntimeError) as error:
            print(f"Visual-map batch at {timestamp(seconds[0])} failed: {error}")
    cache.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return records


def speech_near(segments: list[dict], start: float, end: float) -> str:
    return " ".join(segment["text"] for segment in segments if segment["end"] >= start and segment["start"] <= end)[:1400]


def relevant_map(records: list[dict], start: float, end: float) -> str:
    nearby: list[dict] = []
    for record in records:
        frame_seconds = record.get("frame_seconds", [])
        if any(start - 45 <= second <= end + 45 for second in frame_seconds):
            nearby.append(record)
    if not nearby:
        return "[No nearby coarse-frame description.]"
    return "\n\n".join(
        f"Sampled frames {', '.join(record['frame_times'])}:\n{record['description'][:900]}"
        for record in nearby[:2]
    )


def candidate_path(video: Path) -> Path:
    return WORK / f"{video.stem}-visual-candidates-{CACHE_VERSION}.json"


def verdict_from_text(text: str) -> str:
    match = re.search(r"(?im)^\s*VERDICT\s*:\s*(PAYOFF|NO_PAYOFF|UNCERTAIN)\s*$", text)
    return match.group(1).upper() if match else "UNCERTAIN"


def local_candidate_score(verdict: str, energy: float) -> int:
    baseline = {"PAYOFF": 70, "UNCERTAIN": 25, "NO_PAYOFF": 0}[verdict]
    return min(100, baseline + round(25 * energy))


def visual_candidates(video: Path, duration: float, audio_events: list[dict], speech: list[dict], visual_map: list[dict], force: bool) -> list[dict]:
    cache = candidate_path(video)
    if cache.exists() and not force:
        print(f"Using saved visual candidate analysis: {cache.name}")
        return json.loads(cache.read_text(encoding="utf-8"))

    reports: list[dict] = []
    print(f"Visually validating {len(audio_events)} loud game-audio events …")
    for event in audio_events:
        start, end, middle = event["start"], min(duration, event["end"]), event["event_time"]
        frame_seconds = [start + 2, middle, min(end - 1, middle + 8)]
        images = [extract_frame(video, second, f"candidate-{middle}-{index}") for index, second in enumerate(frame_seconds)]
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
            print(f"  {timestamp(middle)}: {verdict} — score {report['local_score']}")
        except (requests.RequestException, RuntimeError) as error:
            print(f"Candidate {timestamp(middle)} failed: {error}")
    cache.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    return reports


def review_queue_path(video: Path) -> Path:
    return WORK / f"{video.stem}-review-queue-{REVIEW_QUEUE_VERSION}.json"


def review_priority(verdict: str) -> str:
    """Translate model triage into a human-review bucket."""
    if verdict == "PAYOFF":
        return "ALTO POTENCIAL"
    if verdict == "UNCERTAIN":
        return "EM REVISÃO"
    return "SEM INTERESSE"


def overlap_seconds(first: dict, second: dict) -> float:
    return max(0.0, min(float(first["end"]), float(second["end"])) - max(float(first["start"]), float(second["start"])))


def human_review_queue(video: Path, reports: list[dict]) -> list[dict]:
    """Render promising and uncertain candidates; leave final judgment to the user."""
    queue = [
        report | {
            "priority": review_priority(report["verdict"]),
            "review_status": "Pendente",
            "title": f"Wardogs — {timestamp(report['event_time'])}",
        }
        for report in reports
        if report["verdict"] != "NO_PAYOFF"
    ]
    queue.sort(key=lambda report: (
        0 if report["priority"] == "ALTO POTENCIAL" else 1,
        -report["local_score"],
        report["start"],
    ))

    for index, report in enumerate(queue, start=1):
        overlaps: list[str] = []
        for other_index, other in enumerate(queue, start=1):
            if index == other_index:
                continue
            shared = overlap_seconds(report, other)
            if shared >= MIN_DUPLICATE_OVERLAP_SECONDS:
                overlaps.append(f"Corte #{other_index} ({round(shared)} s)")
        report["order"] = index
        report["possible_overlap"] = ", ".join(overlaps) if overlaps else ""

    skipped = [report for report in reports if report["verdict"] == "NO_PAYOFF"]
    payload = {
        "selection_mode": "human-review",
        "render_rule": "Render PAYOFF and UNCERTAIN. Skip only NO_PAYOFF.",
        "queue": queue,
        "skipped_no_interest": skipped,
    }
    review_queue_path(video).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return queue


def review_workbook_path(video: Path) -> Path:
    return OUTPUT / f"{video.stem}-shorts-review.xlsx"


def write_review_workbook(video: Path, rendered: list[dict], skipped_count: int) -> Path:
    """Create a human-friendly Excel review queue after the MP4 files exist."""
    try:
        from openpyxl import Workbook
        from openpyxl.formatting.rule import CellIsRule
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.worksheet.datavalidation import DataValidation
        from openpyxl.worksheet.table import Table, TableStyleInfo
    except ImportError as error:
        raise RuntimeError(
            "Excel report requires openpyxl. Install it with: "
            ".\\.venv\\Scripts\\python.exe -m pip install openpyxl"
        ) from error

    workbook = Workbook()
    cuts = workbook.active
    cuts.title = "Cortes"
    cuts.sheet_view.showGridLines = False

    title_fill = PatternFill("solid", fgColor="1F4E78")
    header_fill = PatternFill("solid", fgColor="244062")
    high_fill = PatternFill("solid", fgColor="E2F0D9")
    review_fill = PatternFill("solid", fgColor="FFF2CC")
    alert_fill = PatternFill("solid", fgColor="FCE4D6")
    white_bold = Font(name="Arial", size=10, bold=True, color="FFFFFF")
    body_font = Font(name="Arial", size=10)
    title_font = Font(name="Arial", size=14, bold=True, color="1F1F1F")
    center = Alignment(horizontal="center", vertical="center")
    left = Alignment(horizontal="left", vertical="center")
    wrapped = Alignment(horizontal="left", vertical="top", wrap_text=True)

    cuts["A2"] = "Revisão de Shorts"
    cuts["A2"].font = title_font
    cuts["A3"] = "Vídeo fonte"
    cuts["B3"] = video.name
    cuts["D3"] = "Cortes renderizados"
    cuts["E3"] = len(rendered)
    cuts["G3"] = "Não renderizados"
    cuts["H3"] = skipped_count
    cuts["A4"] = "Gerado em"
    cuts["B4"] = datetime.now()
    cuts["B4"].number_format = "yyyy-mm-dd hh:mm"
    cuts["A5"] = "Regra"
    cuts["B5"] = "Alto potencial e Em revisão são renderizados. A decisão final é humana."
    cuts["B5"].alignment = left

    headers = [
        "#", "Prioridade", "Revisão humana", "Início", "Fim", "Duração",
        "Evento", "Áudio", "Título", "Arquivo MP4", "Possível sobreposição",
    ]
    header_row = 7
    for column, value in enumerate(headers, start=1):
        cell = cuts.cell(header_row, column, value)
        cell.fill = header_fill
        cell.font = white_bold
        cell.alignment = center

    for row_index, report in enumerate(rendered, start=header_row + 1):
        values = [
            report["order"],
            report["priority"],
            report["review_status"],
            timedelta(seconds=round(float(report["start"]))),
            timedelta(seconds=round(float(report["end"]))),
            timedelta(seconds=round(float(report["end"]) - float(report["start"]))),
            timedelta(seconds=round(float(report["event_time"]))),
            float(report["audio_energy"]),
            report["title"],
            Path(report["rendered_file"]).name,
            report["possible_overlap"] or "",
        ]
        for column, value in enumerate(values, start=1):
            cell = cuts.cell(row_index, column, value)
            cell.font = body_font
            cell.alignment = center if column in {1, 2, 3, 4, 5, 6, 7, 8} else left
        for column in (4, 5, 6, 7):
            cuts.cell(row_index, column).number_format = "[m]:ss"
        cuts.cell(row_index, 8).number_format = "0.0%"
        mp4_cell = cuts.cell(row_index, 10)
        try:
            mp4_cell.hyperlink = Path(report["rendered_file"]).resolve().as_uri()
            mp4_cell.style = "Hyperlink"
        except ValueError:
            pass
        priority_cell = cuts.cell(row_index, 2)
        priority_cell.fill = high_fill if report["priority"] == "ALTO POTENCIAL" else review_fill
        if report["possible_overlap"]:
            cuts.cell(row_index, 11).fill = alert_fill

    last_cut_row = header_row + len(rendered)
    if rendered:
        table = Table(displayName="CortesRenderizados", ref=f"A{header_row}:K{last_cut_row}")
        table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True, showColumnStripes=False)
        cuts.add_table(table)
        validation = DataValidation(type="list", formula1='"Pendente,Aprovado,Rejeitado,Ajustar início/fim"', allow_blank=False)
        cuts.add_data_validation(validation)
        validation.add(f"C{header_row + 1}:C{last_cut_row}")
        cuts.conditional_formatting.add(
            f"C{header_row + 1}:C{last_cut_row}",
            CellIsRule(operator="equal", formula=['"Aprovado"'], fill=high_fill),
        )
        cuts.conditional_formatting.add(
            f"C{header_row + 1}:C{last_cut_row}",
            CellIsRule(operator="equal", formula=['"Rejeitado"'], fill=alert_fill),
        )
    cuts.freeze_panes = f"A{header_row + 1}"
    widths = {"A": 6, "B": 18, "C": 20, "D": 12, "E": 12, "F": 12, "G": 12, "H": 10, "I": 28, "J": 42, "K": 30}
    for column, width in widths.items():
        cuts.column_dimensions[column].width = width

    evidence = workbook.create_sheet("Evidências")
    evidence.sheet_view.showGridLines = False
    evidence_headers = [
        "#", "Prioridade", "Título", "Fala próxima", "Descrição visual",
        "Contexto visual amostrado", "Veredito", "Score", "Possível sobreposição",
    ]
    for column, value in enumerate(evidence_headers, start=1):
        cell = evidence.cell(1, column, value)
        cell.fill = header_fill
        cell.font = white_bold
        cell.alignment = center
    for row_index, report in enumerate(rendered, start=2):
        values = [
            report["order"], report["priority"], report["title"], report["voice"],
            report["visual_description"], report["coarse_context"], report["verdict"],
            int(report["local_score"]), report["possible_overlap"] or "",
        ]
        for column, value in enumerate(values, start=1):
            cell = evidence.cell(row_index, column, value)
            cell.font = body_font
            cell.alignment = wrapped if column in {4, 5, 6, 9} else center if column in {1, 2, 7, 8} else left
        evidence.cell(row_index, 2).fill = high_fill if report["priority"] == "ALTO POTENCIAL" else review_fill
        evidence.row_dimensions[row_index].height = 110
    last_evidence_row = 1 + len(rendered)
    if rendered:
        table = Table(displayName="EvidenciasCortes", ref=f"A1:I{last_evidence_row}")
        table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True, showColumnStripes=False)
        evidence.add_table(table)
    evidence.freeze_panes = "A2"
    evidence_widths = {"A": 6, "B": 18, "C": 28, "D": 42, "E": 58, "F": 68, "G": 15, "H": 10, "I": 30}
    for column, width in evidence_widths.items():
        evidence.column_dimensions[column].width = width

    output = review_workbook_path(video)
    try:
        workbook.save(output)
    except PermissionError as error:
        raise RuntimeError(f"Close {output.name} in Excel and run again.") from error
    return output


def render_short(video: Path, report: dict, number: int) -> Path:
    start = report["start"]
    duration = min(report["end"] - start, 59.0)
    output = OUTPUT / f"{video.stem}-{number:02d}-{timestamp(start).replace(':', 'm')}s.mp4"
    # 2560x1440 source -> centered 810x1440 crop -> 1080x1920 Short.
    filters = "[0:v]crop=trunc(ih*9/16/2)*2:ih:trunc((iw-trunc(ih*9/16/2)*2)/2):0,scale=1080:1920:flags=lanczos,format=yuv420p[v]"
    print(f"Rendering Short {number}: {timestamp(start)}–{timestamp(start + duration)} — {report['title']}")
    run([
        "ffmpeg", "-y", "-ss", str(start), "-i", str(video), "-t", str(duration),
        "-filter_complex", filters, "-map", "[v]", "-map", f"0:a:{VOICE_TRACK}",
        "-c:v", "h264_nvenc", "-preset", "p5", "-cq", "20",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output),
    ])
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-transcribe", action="store_true", help="Ignore the saved Whisper transcript.")
    parser.add_argument("--refresh-visual", action="store_true", help="Rebuild plain-text visual evidence caches.")
    parser.add_argument("--coarse-every", type=int, default=COARSE_EVERY_SECONDS, help="Seconds between storyboard frames (default: 45).")
    parser.add_argument("--no-render", action="store_true", help="Analyze and build the human-review queue, but do not create MP4 files or Excel.")
    args = parser.parse_args()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    videos = find_videos()
    if not videos:
        raise FileNotFoundError(f"No video found in {locate_input_dir()}")

    for video in videos:
        print(f"\nProcessing {video.name}")
        voice_audio = extract_audio(video, VOICE_TRACK)
        game_audio = extract_audio(video, GAME_TRACK)
        speech = transcribe(video, voice_audio, args.force_transcribe)
        duration = video_duration(video)
        visual_map = coarse_visual_map(video, duration, args.coarse_every, args.refresh_visual)
        reports = visual_candidates(video, duration, game_audio_peaks(game_audio), speech, visual_map, args.refresh_visual)
        selected = human_review_queue(video, reports)
        high_count = sum(report["priority"] == "ALTO POTENCIAL" for report in selected)
        review_count = len(selected) - high_count
        skipped_count = sum(report["verdict"] == "NO_PAYOFF" for report in reports)
        print(f"Human-review queue: {high_count} high-potential + {review_count} uncertain. Skipped {skipped_count} no-interest candidates.")
        if not args.no_render:
            rendered: list[dict] = []
            for number, report in enumerate(selected, start=1):
                output = render_short(video, report, number)
                rendered.append(report | {"rendered_file": str(output)})
            if rendered:
                workbook = write_review_workbook(video, rendered, skipped_count)
                print(f"Excel review workbook: {workbook}")


if __name__ == "__main__":
    main()
