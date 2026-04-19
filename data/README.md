# Data layout

- `**raw/**` — full PDFs and other inputs you do not want mixed with code (defaults point here).
- `**processed/**` — page-range slices from `split_pdf_pages` (default OCR input lives here).

PDFs are gitignored (`*.pdf`); 