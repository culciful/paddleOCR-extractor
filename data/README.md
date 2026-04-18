# Data layout

- **`raw/`** — full PDFs and other inputs you do not want mixed with code (defaults point here).
- **`processed/`** — page-range slices from `split_pdf_pages` (default OCR input lives here).

PDFs are gitignored (`*.pdf`); keep large binaries out of Git and only commit this README / `.gitkeep` if you want empty dirs tracked.

Structured extraction outputs live under **`extractions/<book>/`** at the repo root (see `extractions/liuxuesheng/README.md`).
