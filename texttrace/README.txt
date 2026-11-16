# TextTrace Report Update

This package contains an updated `report.py` that creates a polished TextTrace overview PDF.
It can append the annotated original PDF pages to make one combined PDF.

## Integration

For a PDF upload:
```python
annotated_pdf = annotate_pdf(uploaded_bytes, ai_result)
report_path = create_report(
    reports_folder, uploaded_filename, doc_metadata, ai_result, similarity_result,
    original_bytes=uploaded_bytes,
    highlighted_bytes=annotated_pdf,
)
```

For DOCX, convert the annotated DOCX to PDF before passing `highlighted_bytes`.
Use LibreOffice headless or a configured Microsoft Word conversion service; the report module
does not silently flatten/rebuild DOCX layout itself.

This file expects `ai_result` segments to contain `probability` and the result threshold to be
in `ai_result["threshold"]`. It also supports older `score` segment keys for counting only.
