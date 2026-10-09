"""Human-review queue, duplicate warnings, and Excel workbook output."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

try:
    from .config import (
        MIN_DUPLICATE_OVERLAP_SECONDS,
        MAX_SHORT_SECONDS,
        MERGE_MAX_GAP_SECONDS,
        OUTPUT,
        REVIEW_QUEUE_VERSION,
        WORK,
    )
    from .media import timestamp
except ImportError:  # pragma: no cover - used when run directly from this folder.
    from config import (  # type: ignore
        MIN_DUPLICATE_OVERLAP_SECONDS,
        MAX_SHORT_SECONDS,
        MERGE_MAX_GAP_SECONDS,
        OUTPUT,
        REVIEW_QUEUE_VERSION,
        WORK,
    )
    from media import timestamp  # type: ignore


def review_queue_path(video: Path, game_track: int, voice_track: int) -> Path:
    """Return the JSON review-queue cache for a pair of analysis tracks."""
    return WORK / (
        f"{video.stem}-review-queue-{REVIEW_QUEUE_VERSION}-"
        f"game{game_track + 1}-voice{voice_track + 1}.json"
    )


def review_priority(verdict: str) -> str:
    """Translate model triage into a human-review bucket."""
    if verdict == "PAYOFF":
        return "ALTO POTENCIAL"
    if verdict == "UNCERTAIN":
        return "EM REVISÃO"
    return "SEM INTERESSE"


def overlap_seconds(first: dict, second: dict) -> float:
    """Return the number of seconds shared by two candidate windows."""
    return max(
        0.0,
        min(float(first["end"]), float(second["end"]))
        - max(float(first["start"]), float(second["start"])),
    )


def merge_nearby_candidates(reports: list[dict]) -> list[dict]:
    """Join usable candidate windows that overlap or are only briefly apart."""
    candidates = sorted(
        (report.copy() for report in reports if report["verdict"] != "NO_PAYOFF"),
        key=lambda report: (float(report["start"]), float(report["end"])),
    )
    merged: list[dict] = []
    for report in candidates:
        if not merged:
            merged.append(report)
            continue

        previous = merged[-1]
        combined_end = max(float(previous["end"]), float(report["end"]))
        gap = float(report["start"]) - float(previous["end"])
        if (
            gap <= MERGE_MAX_GAP_SECONDS
            and combined_end - float(previous["start"]) <= MAX_SHORT_SECONDS
        ):
            previous_events = previous.get("merged_events", [previous.copy()])
            report_events = report.get("merged_events", [report.copy()])
            previous["merged_events"] = previous_events + report_events
            previous["start"] = min(float(previous["start"]), float(report["start"]))
            previous["end"] = combined_end
            previous["event_times"] = [
                float(event["event_time"]) for event in previous["merged_events"]
            ]
            previous["event_time"] = previous["event_times"][0]
            previous["verdict"] = (
                "PAYOFF"
                if any(event["verdict"] == "PAYOFF" for event in previous["merged_events"])
                else "UNCERTAIN"
            )
            previous["local_score"] = max(
                int(event["local_score"]) for event in previous["merged_events"]
            )
            for field in ("voice", "visual_description", "coarse_context"):
                previous[field] = "\n\n".join(
                    f"[{timestamp(event['event_time'])}] {event[field]}"
                    for event in previous["merged_events"]
                    if event.get(field)
                )
        else:
            merged.append(report)
    return merged


def human_review_queue(
    video: Path,
    reports: list[dict],
    game_track: int,
    voice_track: int,
) -> list[dict]:
    """Keep all promising/uncertain cuts; only clear non-events are skipped."""
    queue = [
        report
        | {
            "priority": review_priority(report["verdict"]),
            "review_status": "Pendente",
        }
        for report in merge_nearby_candidates(reports)
    ]
    queue.sort(
        key=lambda report: (
            0 if report["priority"] == "ALTO POTENCIAL" else 1,
            -report["local_score"],
            report["start"],
        )
    )

    for index, report in enumerate(queue, start=1):
        overlaps: list[str] = []
        for other_index, other in enumerate(queue, start=1):
            if index == other_index:
                continue
            shared = overlap_seconds(report, other)
            if shared >= MIN_DUPLICATE_OVERLAP_SECONDS:
                overlaps.append(f"Corte #{other_index} ({round(shared)} s)")
        report["order"] = index
        start_label = timestamp(report["start"]).replace(":", "m")
        report["title"] = f"{video.stem} - Corte {index:02d} - {start_label}s"
        report["possible_overlap"] = ", ".join(overlaps) if overlaps else ""

    skipped = [report for report in reports if report["verdict"] == "NO_PAYOFF"]
    payload = {
        "selection_mode": "human-review",
        "render_rule": "Render PAYOFF and UNCERTAIN. Skip only NO_PAYOFF.",
        "queue": queue,
        "skipped_no_interest": skipped,
    }
    review_queue_path(video, game_track, voice_track).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return queue


def review_workbook_path(video: Path) -> Path:
    """Return the Excel review workbook path for a source recording."""
    return OUTPUT / f"{video.stem}-shorts-review.xlsx"


def write_review_workbook(
    video: Path,
    rendered: list[dict],
    skipped_count: int,
    visual_map: list[dict],
) -> Path:
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

    summary = workbook.create_sheet("Resumo do vídeo", 0)
    summary.sheet_view.showGridLines = False
    summary["A1"] = "Resumo visual do vídeo"
    summary["A1"].font = title_font
    summary["A2"] = "Vídeo fonte"
    summary["B2"] = video.name
    summary["A3"] = "Nota"
    summary["B3"] = (
        "Descrições baseadas em quadros amostrados; não representam cada segundo."
    )
    summary["B3"].alignment = left
    summary_headers = ["Intervalo amostrado", "Resumo visual"]
    for column, value in enumerate(summary_headers, start=1):
        cell = summary.cell(5, column, value)
        cell.fill = header_fill
        cell.font = white_bold
        cell.alignment = center

    for row_index, record in enumerate(visual_map, start=6):
        frame_times = record.get("frame_times", [])
        interval = (
            f"{frame_times[0]} - {frame_times[-1]}"
            if frame_times
            else ""
        )
        summary.cell(row_index, 1, interval)
        summary.cell(row_index, 2, record.get("description", ""))
        for column in (1, 2):
            cell = summary.cell(row_index, column)
            cell.font = body_font
            cell.alignment = wrapped
        summary.row_dimensions[row_index].height = 110

    last_summary_row = 5 + len(visual_map)
    if visual_map:
        table = Table(
            displayName="ResumoVisualVideo",
            ref=f"A5:B{last_summary_row}",
        )
        table.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium2",
            showRowStripes=True,
            showColumnStripes=False,
        )
        summary.add_table(table)
    summary.freeze_panes = "A6"
    summary.column_dimensions["A"].width = 24
    summary.column_dimensions["B"].width = 110
    workbook.active = 0

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
    cuts["B5"] = (
        "Alto potencial e Em revisão são renderizados. "
        "A decisão final é humana."
    )
    cuts["B5"].alignment = left

    headers = [
        "#",
        "Prioridade",
        "Revisão humana",
        "Início",
        "Fim",
        "Duração",
        "Evento",
        "Áudio",
        "Título",
        "Arquivo MP4",
        "Possível sobreposição",
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
            ", ".join(
                timestamp(event_time)
                for event_time in report.get("event_times", [report["event_time"]])
            ),
            float(report["audio_energy"]),
            report["title"],
            Path(report["rendered_file"]).name,
            report["possible_overlap"] or "",
        ]
        for column, value in enumerate(values, start=1):
            cell = cuts.cell(row_index, column, value)
            cell.font = body_font
            cell.alignment = (
                center if column in {1, 2, 3, 4, 5, 6, 8} else left
            )
        for column in (4, 5, 6):
            cuts.cell(row_index, column).number_format = "[m]:ss"
        cuts.cell(row_index, 8).number_format = "0.0%"

        mp4_cell = cuts.cell(row_index, 10)
        try:
            mp4_cell.hyperlink = Path(report["rendered_file"]).resolve().as_uri()
            mp4_cell.style = "Hyperlink"
        except ValueError:
            pass

        priority_cell = cuts.cell(row_index, 2)
        priority_cell.fill = (
            high_fill if report["priority"] == "ALTO POTENCIAL" else review_fill
        )
        if report["possible_overlap"]:
            cuts.cell(row_index, 11).fill = alert_fill

    last_cut_row = header_row + len(rendered)
    if rendered:
        table = Table(
            displayName="CortesRenderizados",
            ref=f"A{header_row}:K{last_cut_row}",
        )
        table.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium2",
            showRowStripes=True,
            showColumnStripes=False,
        )
        cuts.add_table(table)
        validation = DataValidation(
            type="list",
            formula1='"Pendente,Aprovado,Rejeitado,Ajustar início/fim"',
            allow_blank=False,
        )
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
    widths = {
        "A": 6,
        "B": 18,
        "C": 20,
        "D": 12,
        "E": 12,
        "F": 12,
        "G": 12,
        "H": 10,
        "I": 28,
        "J": 42,
        "K": 30,
    }
    for column, width in widths.items():
        cuts.column_dimensions[column].width = width

    evidence = workbook.create_sheet("Evidências")
    evidence.sheet_view.showGridLines = False
    evidence_headers = [
        "#",
        "Prioridade",
        "Título",
        "Fala próxima",
        "Descrição visual",
        "Contexto visual amostrado",
        "Veredito",
        "Score",
        "Possível sobreposição",
    ]
    for column, value in enumerate(evidence_headers, start=1):
        cell = evidence.cell(1, column, value)
        cell.fill = header_fill
        cell.font = white_bold
        cell.alignment = center

    for row_index, report in enumerate(rendered, start=2):
        values = [
            report["order"],
            report["priority"],
            report["title"],
            report["voice"],
            report["visual_description"],
            report["coarse_context"],
            report["verdict"],
            int(report["local_score"]),
            report["possible_overlap"] or "",
        ]
        for column, value in enumerate(values, start=1):
            cell = evidence.cell(row_index, column, value)
            cell.font = body_font
            cell.alignment = (
                wrapped
                if column in {4, 5, 6, 9}
                else center
                if column in {1, 2, 7, 8}
                else left
            )
        evidence.cell(row_index, 2).fill = (
            high_fill if report["priority"] == "ALTO POTENCIAL" else review_fill
        )
        evidence.row_dimensions[row_index].height = 110

    last_evidence_row = 1 + len(rendered)
    if rendered:
        table = Table(
            displayName="EvidenciasCortes",
            ref=f"A1:I{last_evidence_row}",
        )
        table.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium2",
            showRowStripes=True,
            showColumnStripes=False,
        )
        evidence.add_table(table)
    evidence.freeze_panes = "A2"
    evidence_widths = {
        "A": 6,
        "B": 18,
        "C": 28,
        "D": 42,
        "E": 58,
        "F": 68,
        "G": 15,
        "H": 10,
        "I": 30,
    }
    for column, width in evidence_widths.items():
        evidence.column_dimensions[column].width = width

    output = review_workbook_path(video)
    try:
        workbook.save(output)
    except PermissionError as error:
        raise RuntimeError(f"Close {output.name} in Excel and run again.") from error
    return output
