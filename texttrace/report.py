from __future__ import annotations

from datetime import datetime
from pathlib import Path
import html
import re
import io

import fitz  # PyMuPDF
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
)


def _esc(value):
    return html.escape(str(value)).replace("\n", "<br/>")


def _pct(value):
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return "Not run"


def _page_decor(canvas, document):
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(colors.HexColor("#172b4d"))
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(18 * mm, height - 12 * mm, "TextTrace")
    canvas.setFillColor(colors.HexColor("#69758b"))
    canvas.setFont("Helvetica", 7)
    canvas.drawRightString(width - 18 * mm, height - 12 * mm, f"TEXTTRACE ANALYSIS  |  PAGE {document.page}")
    canvas.setStrokeColor(colors.HexColor("#d8dde5"))
    canvas.line(18 * mm, height - 15 * mm, width - 18 * mm, height - 15 * mm)
    canvas.setStrokeColor(colors.HexColor("#d8dde5"))
    canvas.line(18 * mm, 14 * mm, width - 18 * mm, 14 * mm)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#69758b"))
    canvas.drawString(18 * mm, 8.5 * mm, "AI-likelihood estimates can be wrong; human review is required.")
    canvas.drawRightString(width - 18 * mm, 8.5 * mm, datetime.now().strftime("%d %b %Y"))
    canvas.restoreState()


def create_report(folder, filename, doc, ai, sim, original_bytes=None, highlighted_bytes=None):
    """Create one polished overview PDF.

    If highlighted_bytes is supplied, it is appended as original-document pages so the final
    PDF includes both the TextTrace overview and the annotated document in a single file.
    For a PDF source, pass the PDF bytes with annotations. For DOCX sources, convert the
    annotated DOCX to PDF first (LibreOffice or Word) and pass those PDF bytes.
    """
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", Path(filename).stem)[:100] or "document"
    path = folder / f"TextTrace_Report_{safe}_{datetime.now():%Y%m%d_%H%M%S}.pdf"

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="TTTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=26, leading=31, alignment=0, textColor=colors.HexColor("#172b4d"),
        spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="TTSub", parent=styles["BodyText"], fontSize=10, leading=14,
        textColor=colors.HexColor("#52627a"), spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="TTHead", parent=styles["Heading1"], fontName="Helvetica-Bold",
        fontSize=15, leading=19, textColor=colors.HexColor("#172b4d"),
        spaceBefore=7, spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="TTBody", parent=styles["BodyText"], fontSize=9, leading=13,
        textColor=colors.HexColor("#263449"), spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="TTScore", parent=styles["BodyText"], fontName="Helvetica-Bold",
        fontSize=30, leading=34, textColor=colors.HexColor("#172b4d"),
    ))
    styles.add(ParagraphStyle(
        name="TTSmall", parent=styles["BodyText"], fontSize=7.5, leading=10,
        textColor=colors.HexColor("#52627a"),
    ))

    text = doc.get("text", "") if isinstance(doc, dict) else ""
    word_count = len(re.findall(r"\b[\w'-]+\b", text))
    char_count = len(text)
    ai_percent = ai.get("ai_percent", ai.get("percent")) if ai else None
    sim_percent = sim.get("percent") if sim else None
    segments = ai.get("segments", []) if ai else []
    threshold = ai.get("threshold", None) if ai else None
    ai_classified = sum(
        1 for s in segments
        if _score(s) is not None and threshold is not None and _score(s) >= float(threshold)
    )
    models = ", ".join(ai.get("models", [])) if ai else "Not run"
    method = ai.get("method", "Not run") if ai else "Not run"
    report_date = datetime.now().strftime("%d %b %Y, %H:%M")

    story = [
        Spacer(1, 10 * mm),
        Paragraph("TextTrace", styles["TTTitle"]),
        Paragraph("AI Writing & Similarity Analysis", styles["TTSub"]),
        Spacer(1, 3 * mm),
        Paragraph("DOCUMENT OVERVIEW", styles["TTHead"]),
    ]

    # Main AI metric card
    score = _pct(ai_percent)
    ai_label = "AI-likelihood estimate"
    score_cell = [
        Paragraph(score, styles["TTScore"]),
        Paragraph(ai_label, styles["TTSmall"]),
    ]
    right_cell = [
        Paragraph("<b>Document</b><br/>" + _esc(Path(filename).name), styles["TTBody"]),
        Paragraph(f"<b>Report date</b><br/>{_esc(report_date)}", styles["TTBody"]),
        Paragraph(f"<b>Word count</b><br/>{word_count:,}", styles["TTBody"]),
    ]
    score_table = Table([[score_cell, right_cell]], colWidths=[53 * mm, 112 * mm])
    score_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#e8efff")),
        ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#f5f7fb")),
        ("BOX", (0, 0), (-1, -1), .7, colors.HexColor("#cbd5e4")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    story += [score_table, Spacer(1, 5 * mm)]

    detail_rows = [
        [Paragraph("<b>Report measure</b>", styles["TTSmall"]), Paragraph("<b>Result</b>", styles["TTSmall"])],
        [Paragraph("Segments analysed", styles["TTBody"]), Paragraph(f"{len(segments):,}" if ai else "Not run", styles["TTBody"])],
        [Paragraph("Segments above threshold", styles["TTBody"]), Paragraph(f"{ai_classified:,}" if ai else "Not run", styles["TTBody"])],
        [Paragraph("Decision threshold", styles["TTBody"]), Paragraph(f"{float(threshold):.2f}" if threshold is not None else "Not available", styles["TTBody"])],
        [Paragraph("Local-library overlap estimate", styles["TTBody"]), Paragraph(_pct(sim_percent), styles["TTBody"])],
        [Paragraph("Document characters", styles["TTBody"]), Paragraph(f"{char_count:,}", styles["TTBody"])],
    ]
    table = Table(detail_rows, colWidths=[105 * mm, 60 * mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#d5dce7")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story += [table, Spacer(1, 5 * mm)]

    caution = Table([[
        Paragraph(
            "<b>Important interpretation</b><br/>"
            "The percentage is a model estimate, not a verified authorship finding. "
            "AI detectors can produce false positives and false negatives. A similarity match "
            "does not by itself establish plagiarism. Human review is required.",
            styles["TTBody"],
        )
    ]], colWidths=[165 * mm])
    caution.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff5e8")),
        ("BOX", (0, 0), (-1, -1), .6, colors.HexColor("#e7c89f")),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story += [caution, Spacer(1, 5 * mm), Paragraph("METHOD & COVERAGE", styles["TTHead"])]
    story += [
        Paragraph("<b>Models used:</b> " + _esc(models), styles["TTBody"]),
        Paragraph("<b>Detection method:</b> " + _esc(method), styles["TTBody"]),
        Paragraph(
            "<b>Similarity coverage:</b> " +
            _esc(sim.get("method", "Similarity analysis was not run.") if sim else "Similarity analysis was not run."),
            styles["TTBody"],
        ),
    ]

    sources = sim.get("sources", []) if sim else []
    if sources:
        story.append(Paragraph("Potential local reference matches", styles["Heading2"]))
        source_rows = [[
            Paragraph("<b>Source</b>", styles["TTSmall"]),
            Paragraph("<b>Match estimate</b>", styles["TTSmall"]),
            Paragraph("<b>Location</b>", styles["TTSmall"]),
        ]]
        for source in sources[:10]:
            source_rows.append([
                Paragraph(_esc(source.get("name", "")), styles["TTSmall"]),
                Paragraph(_pct(source.get("score")), styles["TTSmall"]),
                Paragraph(_esc(source.get("path", "")), styles["TTSmall"]),
            ])
        src_table = Table(source_rows, colWidths=[48 * mm, 35 * mm, 82 * mm], repeatRows=1)
        src_table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#d5dce7")),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f8")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(src_table)
    else:
        story.append(Paragraph("No potential local reference matches were returned, or similarity analysis was not run.", styles["TTBody"]))

    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph(
        "This report compares against the sources available to TextTrace. It is not a Turnitin report "
        "and does not have access to Turnitin's private repository or a live internet-wide index.",
        styles["TTSmall"],
    ))

    SimpleDocTemplate(
        str(path), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=22 * mm, bottomMargin=19 * mm,
        title=f"TextTrace Report - {Path(filename).name}",
        author="TextTrace",
    ).build(story, onFirstPage=_page_decor, onLaterPages=_page_decor)

    # One PDF output: overview plus annotated original document pages, when provided.
    if highlighted_bytes:
        try:
            overview_pdf = fitz.open(path)
            original_pdf = fitz.open(stream=highlighted_bytes, filetype="pdf")
            overview_pdf.insert_pdf(original_pdf)
            temp_path = path.with_suffix(".combined.tmp.pdf")
            overview_pdf.save(temp_path, garbage=4, deflate=True)
            overview_pdf.close()
            original_pdf.close()
            temp_path.replace(path)
        except Exception as exc:
            # Keep a valid overview report if source-page appending is not possible.
            try:
                if 'overview_pdf' in locals():
                    overview_pdf.close()
            except Exception:
                pass
            raise RuntimeError(
                "Overview PDF was created, but the annotated document could not be appended. "
                "For DOCX, convert the annotated DOCX to PDF first; for PDF, pass annotated PDF bytes. "
                f"Details: {exc}"
            ) from exc

    return str(path)


def _score(segment):
    """Read both the current engine's 'probability' key and older 'score' key."""
    try:
        return float(segment.get("probability", segment.get("score")))
    except (TypeError, ValueError):
        return None
