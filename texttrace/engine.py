
from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG_PATH = APP_ROOT / "models" / "ensemble_config.json"

SUPPORTED_TEXT = {
    ".txt", ".md", ".py", ".js", ".java", ".cpp", ".c",
    ".h", ".cs", ".html", ".css", ".ts", ".jsx", ".sql"
}


def load_uploaded_document(filename: str, raw: bytes):
    ext = Path(filename).suffix.lower()
    meta = {"filename": filename, "extension": ext, "bytes": len(raw)}

    if ext == ".docx":
        from docx import Document

        doc = Document(io.BytesIO(raw))
        chunks = []

        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                chunks.append(paragraph.text)

        for table in doc.tables:
            for row in table.rows:
                chunks.append(" | ".join(cell.text for cell in row.cells))

        for section in doc.sections:
            for paragraph in section.header.paragraphs:
                if paragraph.text.strip():
                    chunks.append(paragraph.text)

            for paragraph in section.footer.paragraphs:
                if paragraph.text.strip():
                    chunks.append(paragraph.text)

        return "\n\n".join(chunks), meta

    if ext == ".pdf":
        import fitz

        doc = fitz.open(stream=raw, filetype="pdf")
        text = "\n\n".join(page.get_text("text") for page in doc)
        return text, {**meta, "pages": len(doc)}

    if ext in SUPPORTED_TEXT:
        return raw.decode("utf-8-sig", errors="replace"), meta

    raise ValueError(f"Unsupported extension: {ext}")


def _segments(text: str, max_chars: int = 950):
    """Split text into paragraph-local segments."""
    blocks = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]

    output = []

    for block in blocks:
        if len(block) <= max_chars:
            output.append(block)
            continue

        sentences = re.split(r"(?<=[.!?])\s+", block)
        chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            if chunk and len(chunk) + len(sentence) + 1 > max_chars:
                output.append(chunk)
                chunk = sentence

            elif len(sentence) > max_chars:
                if chunk:
                    output.append(chunk)
                    chunk = ""

                output.extend(
                    sentence[i:i + max_chars]
                    for i in range(0, len(sentence), max_chars)
                )

            else:
                chunk = (chunk + " " + sentence).strip()

        if chunk:
            output.append(chunk)

    return output or ([text.strip()] if text.strip() else [])


def _load_models():
    """Load configured local models without substituting a heuristic."""
    if not MODEL_CONFIG_PATH.is_file():
        return None, f"Model config not found: {MODEL_CONFIG_PATH}"

    try:
        import joblib

        config = json.loads(
            MODEL_CONFIG_PATH.read_text(encoding="utf-8")
        )

        models_dir = APP_ROOT / "models"

        filenames = {
            "LR": "tfidf_logistic_regression.pkl",
            "SVM": "tfidf_calibrated_linear_svm.pkl",
            "XGBoost": "xgboost_linguistic.pkl",
            "Adaptive LR": "adaptive_logistic_regression.pkl",
            "Adaptive Logistic Regression": "adaptive_logistic_regression.pkl",
        }

        models = {}
        problems = []

        for name in config.get("model_names", ["LR", "SVM"]):

            if name == "DistilRoBERTa":
                candidates = [
                    models_dir / "distilroberta_texttrace",
                    models_dir / "distilroberta",
                ]

                model_dir = next(
                    (
                        path for path in candidates
                        if path.is_dir()
                        and (path / "config.json").is_file()
                        and (path / "model.safetensors").is_file()
                    ),
                    None,
                )

                if model_dir is None:
                    problems.append(
                        "DistilRoBERTa checkpoint not found."
                    )
                    continue

                try:
                    import torch
                    from transformers import (
                        AutoTokenizer,
                        AutoModelForSequenceClassification,
                    )

                    tokenizer = AutoTokenizer.from_pretrained(
                        str(model_dir),
                        local_files_only=True,
                    )

                    transformer = (
                        AutoModelForSequenceClassification.from_pretrained(
                            str(model_dir),
                            local_files_only=True,
                        )
                    )

                    device = torch.device(
                        "cuda" if torch.cuda.is_available() else "cpu"
                    )

                    transformer.to(device)
                    transformer.eval()

                    models[name] = {
                        "kind": "transformer",
                        "tokenizer": tokenizer,
                        "model": transformer,
                        "device": device,
                    }

                except Exception as exc:
                    problems.append(
                        f"DistilRoBERTa load failed: "
                        f"{type(exc).__name__}: {exc}"
                    )

                continue

            filename = filenames.get(name)

            if filename is None:
                problems.append(
                    f"Unknown configured model name: {name}"
                )
                continue

            model_path = models_dir / filename

            if not model_path.is_file():
                problems.append(
                    f"Missing model file: {model_path.name}"
                )
                continue

            try:
                models[name] = joblib.load(model_path)

            except Exception as exc:
                problems.append(
                    f"{name} load failed: "
                    f"{type(exc).__name__}: {exc}"
                )

        if not models:
            return None, (
                "No configured model loaded. " + "; ".join(problems)
            )

        status = "Loaded: " + ", ".join(models.keys())

        if problems:
            status += " | Partial ensemble: " + "; ".join(problems)

        return (config, models), status

    except Exception as exc:
        return None, (
            f"Model loading failed: {type(exc).__name__}: {exc}"
        )


def _linguistic_features(text: str):
    """Extract the linguistic features currently supported by this engine."""
    words = re.findall(r"\b\w+\b", text)
    sentences = [
        sentence
        for sentence in re.split(r"[.!?]+", text)
        if sentence.strip()
    ]

    word_count = len(words)
    char_count = len(text)
    sentence_count = max(1, len(sentences))
    word_lengths = [len(word) for word in words]

    unique_ratio = (
        len({word.lower() for word in words}) / word_count
        if word_count else 0.0
    )

    return {
        "char_count": char_count,
        "word_count": word_count,
        "sentence_count": len(sentences),
        "avg_word_length": (
            sum(word_lengths) / len(word_lengths)
            if word_lengths else 0.0
        ),
        "avg_sentence_length": word_count / sentence_count,
        "unique_word_ratio": unique_ratio,
        "digit_ratio": (
            sum(char.isdigit() for char in text) / char_count
            if char_count else 0.0
        ),
        "uppercase_ratio": (
            sum(char.isupper() for char in text) / char_count
            if char_count else 0.0
        ),
        "punctuation_ratio": (
            sum(
                not char.isalnum() and not char.isspace()
                for char in text
            ) / char_count
            if char_count else 0.0
        ),
        "question_count": text.count("?"),
        "exclamation_count": text.count("!"),
        "comma_count": text.count(","),
        "semicolon_count": text.count(";"),
        "colon_count": text.count(":"),
        "newline_count": text.count("\n"),
        "transition_word_count": len(
            re.findall(
                r"\b(however|moreover|therefore|furthermore|"
                r"in conclusion|overall)\b",
                text.lower(),
            )
        ),
    }


def _positive_class_index(model):
    classes = list(getattr(model, "classes_", [0, 1]))

    if 1 in classes:
        return classes.index(1)

    raise ValueError(
        f"Expected AI class label 1; found classes {classes!r}"
    )


def _model_probability(model, text):
    if not hasattr(model, "predict_proba"):
        raise TypeError(
            f"{type(model).__name__} does not implement predict_proba"
        )

    prediction = model.predict_proba([text])

    return float(
        prediction[0][_positive_class_index(model)]
    )


def _xgboost_probability(model, text):
    """Score XGBoost only when its saved feature schema is available."""
    import pandas as pd

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "XGBoost model does not implement predict_proba"
        )

    names = getattr(model, "feature_names_in_", None)

    if names is None:
        booster = getattr(model, "get_booster", lambda: None)()
        names = (
            getattr(booster, "feature_names", None)
            if booster is not None else None
        )

    # FIX: names may be a NumPy array. Do not use `if not names`.
    if names is None or len(names) == 0:
        raise ValueError(
            "XGBoost has no saved feature names; "
            "refusing to guess feature order"
        )

    names = list(names)
    features = _linguistic_features(text)

    unknown = [
        name for name in names
        if name not in features
    ]

    if unknown:
        raise ValueError(
            "XGBoost training feature schema differs from the "
            "available extractor; unsupported features: "
            f"{unknown}"
        )

    row = pd.DataFrame(
        [{name: features[name] for name in names}],
        columns=names,
    )

    prediction = model.predict_proba(row)

    return float(
        prediction[0][_positive_class_index(model)]
    )


def _transformer_probability(bundle, text):
    import torch

    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    device = bundle["device"]

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0]

    if probabilities.numel() < 2:
        raise ValueError(
            "Transformer checkpoint is not a two-class classifier"
        )

    # Assumes the trained checkpoint uses class 1 for AI.
    # Verify this mapping against the checkpoint's training labels.
    return float(
        probabilities[1].detach().cpu().item()
    )


def analyze_document(text: str):
    """Analyze text using configured trained models."""
    chunks = _segments(text)
    loaded, status = _load_models()

    threshold = 0.56
    probabilities = [None] * len(chunks)
    model_names = []
    method = "No trained model produced a score."
    model_score_means = {}

    if loaded:
        config, models = loaded

        try:
            threshold = float(
                config.get("threshold", 0.56)
            )
        except (TypeError, ValueError):
            threshold = 0.56

        weights_config = config.get("weights", {})
        per_model_scores = {
            name: [] for name in models
        }
        used_models = set()

        for index, chunk in enumerate(chunks):
            available = []

            for name, model in models.items():
                try:
                    if name == "DistilRoBERTa":
                        probability = _transformer_probability(
                            model, chunk
                        )
                    elif name == "XGBoost":
                        probability = _xgboost_probability(
                            model, chunk
                        )
                    else:
                        probability = _model_probability(
                            model, chunk
                        )

                    weight = float(
                        weights_config.get(name, 1.0)
                    )

                    if weight > 0:
                        available.append(
                            (name, probability, weight)
                        )
                        per_model_scores[name].append(probability)

                except Exception as exc:
                    message = (
                        f"{name} inference skipped: "
                        f"{type(exc).__name__}: {exc}"
                    )

                    if message not in status:
                        status += " | " + message

            total_weight = sum(
                weight for _, _, weight in available
            )

            if total_weight > 0:
                probabilities[index] = sum(
                    probability * weight
                    for _, probability, weight in available
                ) / total_weight

                used_models.update(
                    name for name, _, _ in available
                )

        model_names = [
            name for name in models
            if name in used_models
        ]

        model_score_means = {
            name: (
                sum(scores) / len(scores) * 100
                if scores else None
            )
            for name, scores in per_model_scores.items()
        }

        if model_names:
            method = (
                "Weighted ensemble of successfully scoring "
                "configured local models; weights and threshold "
                "read from ensemble_config.json."
            )
        else:
            method = (
                "Configured models loaded, but none produced "
                "a valid score."
            )

    segments = [
        {
            "index": index + 1,
            "text": chunk,
            "probability": probabilities[index],
        }
        for index, chunk in enumerate(chunks)
    ]

    valid_probabilities = [
        probability for probability in probabilities
        if probability is not None
    ]

    ai_count = sum(
        probability >= threshold
        for probability in valid_probabilities
    )

    # This is the percentage of scored segments above threshold,
    # NOT the ensemble's average AI probability.
    ai_percent = (
        100.0 * ai_count / len(valid_probabilities)
        if valid_probabilities else None
    )

    return {
        "segments": segments,
        "threshold": threshold,
        "ai_percent": ai_percent,
        "models": model_names or ["No trained model score"],
        "model_scores": model_score_means,
        "method": method,
        "model_status": status,
        "word_count": len(re.findall(r"\b\w+\b", text)),
        "trained_model_available": bool(valid_probabilities),
    }


def analyze_similarity(text: str, repository: Path):
    """Compare against local reference files only."""
    repository = Path(repository)
    references = []

    if repository.exists():
        for path in repository.rglob("*"):
            if (
                not path.is_file()
                or path.suffix.lower()
                not in SUPPORTED_TEXT | {".docx", ".pdf"}
            ):
                continue

            try:
                raw = path.read_bytes()
                source, _ = load_uploaded_document(
                    path.name, raw
                )

                if (
                    source.strip()
                    and source.strip() != text.strip()
                ):
                    references.append({
                        "name": path.name,
                        "path": str(path),
                        "text": source,
                    })

            except Exception:
                continue

    if not references:
        return {
            "percent": 0.0,
            "sources": [],
            "method": (
                "TF-IDF cosine similarity against local reference "
                "files; no readable sources found. This is not "
                "web-wide search."
            ),
        }

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    chunks = _segments(text)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=50000,
    )

    matrix = vectorizer.fit_transform(
        chunks + [reference["text"] for reference in references]
    )

    similarities = cosine_similarity(
        matrix[:len(chunks)],
        matrix[len(chunks):],
    )

    overlap = float(
        (similarities.max(axis=1) >= 0.25).mean() * 100
    )

    sources = []

    for index, reference in enumerate(references):
        best_chunk = int(
            similarities[:, index].argmax()
        )
        score = float(
            similarities[best_chunk, index] * 100
        )

        if score >= 20:
            sources.append({
                "name": reference["name"],
                "path": reference["path"],
                "score": score,
                "excerpt": chunks[best_chunk][:300],
            })

    sources.sort(
        key=lambda source: source["score"],
        reverse=True,
    )

    return {
        "percent": overlap,
        "sources": sources[:15],
        "method": (
            "TF-IDF cosine similarity against local reference "
            "files only; not the live internet or private "
            "Turnitin database."
        ),
    }


def _text_norm(text):
    return re.sub(r"\s+", " ", text).strip()


def annotate_docx(raw: bytes, result: dict):
    """Highlight above-threshold spans in a copy of a DOCX."""
    from lxml import etree

    namespace = {
        "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    }
    w = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

    targets = [
        re.sub(r"\s+", " ", segment["text"]).strip()
        for segment in result["segments"]
        if segment.get("probability") is not None
        and float(segment["probability"])
        >= float(result["threshold"])
        and segment.get("text", "").strip()
    ]

    targets = [
        target for target in targets
        if len(target) >= 12
    ]

    output = io.BytesIO()

    def split_run(run, text_node, start, end):
        original = text_node.text or ""

        pieces = [
            (original[:start], False),
            (original[start:end], True),
            (original[end:], False),
        ]

        parent = run.getparent()

        if parent is None:
            return

        insert_at = parent.index(run)

        for offset, (part, highlighted) in enumerate(pieces):
            if not part:
                continue

            clone = etree.fromstring(
                etree.tostring(run)
            )

            clone_text_nodes = clone.xpath(
                ".//w:t", namespaces=namespace
            )

            if not clone_text_nodes:
                continue

            clone_text_nodes[0].text = part

            if part[:1].isspace() or part[-1:].isspace():
                clone_text_nodes[0].set(
                    "{http://www.w3.org/XML/1998/namespace}space",
                    "preserve",
                )

            properties = clone.find(w + "rPr")

            if properties is None:
                properties = etree.Element(w + "rPr")
                clone.insert(0, properties)

            old_highlight = properties.find(w + "highlight")

            if old_highlight is not None:
                properties.remove(old_highlight)

            if highlighted:
                highlight = etree.SubElement(
                    properties, w + "highlight"
                )
                highlight.set(w + "val", "yellow")

            parent.insert(insert_at + offset, clone)

        parent.remove(run)

    with (
        zipfile.ZipFile(io.BytesIO(raw), "r") as source,
        zipfile.ZipFile(
            output, "w", zipfile.ZIP_DEFLATED
        ) as destination,
    ):
        for item in source.infolist():
            data = source.read(item.filename)

            editable_part = (
                item.filename == "word/document.xml"
                or re.fullmatch(
                    r"word/(header|footer|footnotes|endnotes)\d*\.xml",
                    item.filename,
                )
            )

            if (
                item.filename.startswith("word/")
                and item.filename.endswith(".xml")
                and editable_part
            ):
                try:
                    root = etree.fromstring(data)

                    for paragraph in root.xpath(
                        ".//w:p", namespaces=namespace
                    ):
                        nodes = paragraph.xpath(
                            ".//w:t", namespaces=namespace
                        )

                        if not nodes:
                            continue

                        characters = []
                        mapping = []

                        for node in nodes:
                            value = node.text or ""
                            run = node.getparent()

                            while (
                                run is not None
                                and run.tag != w + "r"
                            ):
                                run = run.getparent()

                            for local_index, char in enumerate(value):
                                characters.append(char)
                                mapping.append(
                                    (run, node, local_index)
                                )

                        paragraph_text = "".join(characters)
                        normalized = re.sub(
                            r"\s+", " ", paragraph_text
                        ).strip()

                        if not normalized:
                            continue

                        matched_spans = []

                        for target in targets:
                            position = normalized.find(target)

                            while position >= 0:
                                matched_spans.append(
                                    (position, position + len(target))
                                )
                                position = normalized.find(
                                    target,
                                    position + max(1, len(target)),
                                )

                        if not matched_spans:
                            continue

                        norm_to_raw = []
                        previous_space = False

                        for raw_index, char in enumerate(paragraph_text):
                            if char.isspace():
                                if not previous_space:
                                    norm_to_raw.append(raw_index)
                                previous_space = True
                            else:
                                norm_to_raw.append(raw_index)
                                previous_space = False

                        for start, end in sorted(
                            set(matched_spans), reverse=True
                        ):
                            if (
                                start >= len(norm_to_raw)
                                or end - 1 >= len(norm_to_raw)
                            ):
                                continue

                            raw_start = norm_to_raw[start]
                            raw_end = norm_to_raw[end - 1] + 1
                            fragments = {}

                            for raw_index in range(
                                raw_start,
                                min(raw_end, len(mapping)),
                            ):
                                run, node, local_index = mapping[raw_index]

                                if run is None or node is None:
                                    continue

                                key = (id(run), id(node))

                                if key not in fragments:
                                    fragments[key] = [
                                        run,
                                        node,
                                        local_index,
                                        local_index + 1,
                                    ]
                                else:
                                    fragments[key][2] = min(
                                        fragments[key][2],
                                        local_index,
                                    )
                                    fragments[key][3] = max(
                                        fragments[key][3],
                                        local_index + 1,
                                    )

                            for run, node, lo, hi in sorted(
                                fragments.values(),
                                key=lambda fragment: fragment[2],
                                reverse=True,
                            ):
                                run_text_nodes = run.xpath(
                                    "./w:t", namespaces=namespace
                                )

                                complex_children = [
                                    child for child in run
                                    if child.tag not in (
                                        w + "rPr", w + "t"
                                    )
                                ]

                                if (
                                    len(run_text_nodes) == 1
                                    and not complex_children
                                    and lo < hi
                                ):
                                    split_run(
                                        run, node, lo, hi
                                    )

                    data = etree.tostring(
                        root,
                        xml_declaration=True,
                        encoding="UTF-8",
                        standalone=True,
                    )

                except Exception:
                    # Keep original XML if safe editing fails.
                    pass

            destination.writestr(item, data)

    return output.getvalue()


def annotate_pdf(raw: bytes, result: dict):
    """Highlight above-threshold AI-likely snippets in a PDF."""
    import fitz

    document = fitz.open(
        stream=raw,
        filetype="pdf",
    )

    for segment in result["segments"]:
        if segment.get("probability") is None:
            continue

        if (
            float(segment["probability"])
            < float(result["threshold"])
        ):
            continue

        text = re.sub(
            r"\s+", " ", segment.get("text", "")
        ).strip()

        if len(text) < 12:
            continue

        if len(text) <= 90:
            snippets = [text]
        else:
            snippets = [
                text[:75],
                text[-75:],
            ]

            middle = max(0, len(text) // 2 - 35)
            snippets.append(text[middle:middle + 70])

        for page in document:
            for snippet in snippets:
                if len(snippet.strip()) < 15:
                    continue

                try:
                    rectangles = page.search_for(snippet)

                    for rectangle in rectangles:
                        annotation = page.add_highlight_annot(
                            rectangle
                        )

                        annotation.set_info(
                            content=(
                                "TextTrace AI-likely segment; "
                                "model estimate, not proof of authorship."
                            )
                        )

                        annotation.update()

                except Exception:
                    continue

    return document.tobytes(
        garbage=4,
        deflate=True,
    )


def make_pdf_report(
    filename: str,
    result: dict,
    similarity=None,
):
    """Generate the TextTrace analysis report as PDF bytes."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import (
        getSampleStyleSheet,
        ParagraphStyle,
    )
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        PageBreak,
    )

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="TextTrace Analysis Report",
    )

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="TTTitle",
            parent=styles["Title"],
            textColor=colors.HexColor("#3c55d8"),
            fontSize=22,
            spaceAfter=10,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SmallText",
            parent=styles["BodyText"],
            fontSize=8,
            leading=11,
        )
    )

    story = [
        Paragraph("TextTrace", styles["TTTitle"]),
        Paragraph(
            "AI Analysis Studio · Document Review Report",
            styles["Heading2"],
        ),
        Spacer(1, 8),
    ]

    def escape_text(value):
        return (
            str(value)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    story.extend([
        Paragraph(
            "<b>Document:</b> " + escape_text(filename),
            styles["BodyText"],
        ),
        Paragraph(
            f"<b>Word count:</b> "
            f"{result.get('word_count', 0):,}",
            styles["BodyText"],
        ),
        Spacer(1, 8),
    ])

    ai_percent = result.get("ai_percent")

    rows = [
        ["Metric", "Result"],
        [
            "Segments above threshold",
            (
                f"{ai_percent:.1f}%"
                if ai_percent is not None
                else "Unavailable"
            ),
        ],
        ["Segments analysed", str(len(result["segments"]))],
        ["Decision threshold", f"{result['threshold']:.2f}"],
        ["Models", ", ".join(result["models"])],
        ["Method", result["method"]],
        [
            "Model status",
            result.get("model_status", "Not available"),
        ],
    ]

    table = Table(
        [
            [
                Paragraph(escape_text(cell), styles["SmallText"])
                for cell in row
            ]
            for row in rows
        ],
        colWidths=[48 * mm, 112 * mm],
        repeatRows=1,
    )

    table.setStyle(
        TableStyle([
            (
                "BACKGROUND", (0, 0), (-1, 0),
                colors.HexColor("#dce6ff"),
            ),
            (
                "GRID", (0, 0), (-1, -1),
                0.4, colors.HexColor("#b7c4df"),
            ),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("PADDING", (0, 0), (-1, -1), 7),
        ])
    )

    story.extend([
        table,
        Spacer(1, 12),
        Paragraph("Important limitation", styles["Heading2"]),
        Paragraph(
            "AI-likelihood scores are uncertain model signals, "
            "not proof of authorship. False positives and false "
            "negatives are possible. Review the original passages "
            "in context.",
            styles["BodyText"],
        ),
        Spacer(1, 10),
        Paragraph("Segment review", styles["Heading2"]),
    ])

    for segment in result["segments"]:
        score = segment.get("probability")

        if score is None:
            label = "Score unavailable"
            score_text = ""
        else:
            label = (
                "Above threshold"
                if score >= result["threshold"]
                else "Below threshold"
            )
            score_text = f" · {score * 100:.1f}%"

        story.append(
            Paragraph(
                f"<b>Segment {segment['index']:03d} · "
                f"{label}{score_text}</b>",
                styles["BodyText"],
            )
        )

        story.append(
            Paragraph(
                escape_text(segment["text"]),
                styles["SmallText"],
            )
        )

        story.append(Spacer(1, 5))

    if similarity:
        story.extend([
            PageBreak(),
            Paragraph("Similarity review", styles["Heading2"]),
            Paragraph(
                escape_text(similarity.get("method", "")),
                styles["BodyText"],
            ),
            Paragraph(
                f"Local overlap estimate: "
                f"{similarity.get('percent', 0):.1f}%",
                styles["BodyText"],
            ),
        ])

        for source in similarity.get("sources", []):
            story.append(
                Paragraph(
                    f"<b>{escape_text(source['name'])} · "
                    f"{source['score']:.1f}%</b>",
                    styles["BodyText"],
                )
            )

            story.append(
                Paragraph(
                    escape_text(source["excerpt"]),
                    styles["SmallText"],
                )
            )

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#66718b"))

        canvas.drawString(
            18 * mm,
            10 * mm,
            "TextTrace · AI scores are not authorship verdicts",
        )

        canvas.drawRightString(
            A4[0] - 18 * mm,
            10 * mm,
            f"Page {doc.page}",
        )

        canvas.restoreState()

    document.build(
        story,
        onFirstPage=footer,
        onLaterPages=footer,
    )

    return buffer.getvalue()


def make_html_review(result: dict):
    """Generate an HTML review with above-threshold segments highlighted."""
    import html

    rows = []

    for segment in result["segments"]:
        score = segment.get("probability")

        if score is None:
            css_class = "unknown"
            score_label = "Score unavailable"
        else:
            css_class = (
                "ai"
                if score >= result["threshold"]
                else "human"
            )
            score_label = f"{score * 100:.1f}%"

        rows.append(
            f'<p class="{css_class}">'
            f'<b>Segment {segment["index"]:03d} · '
            f'{score_label}</b><br>'
            f'{html.escape(segment["text"])}</p>'
        )

    return (
        "<!doctype html><meta charset='utf-8'>"
        "<title>TextTrace Highlighted Review</title>"
        "<style>"
        "body{font:16px/1.6 Arial;max-width:900px;margin:40px auto}"
        ".ai{background:#fff176;padding:8px}"
        ".human{padding:8px}"
        ".unknown{background:#eee;padding:8px}"
        "h1{color:#405eea}"
        "</style>"
        "<h1>TextTrace · AI-highlighted review</h1>"
        "<p>Yellow means above the model threshold; "
        "this is not proof of AI authorship.</p>"
        + "".join(rows)
    )
