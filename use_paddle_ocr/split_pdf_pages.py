from __future__ import annotations

import argparse
from pathlib import Path

from pypdf import PdfReader, PdfWriter


def split_pdf_pages(input_pdf: Path, output_pdf: Path, start_page: int, end_page: int) -> None:
    if start_page < 1 or end_page < 1:
        raise ValueError("Pages are 1-indexed; please check the start/end arguments.")
    if end_page < start_page:
        raise ValueError("end_page must be >= start_page.")
    if not input_pdf.exists():
        raise FileNotFoundError(f"Input file not found: {input_pdf.resolve()}")

    reader = PdfReader(str(input_pdf))
    total = len(reader.pages)

    if start_page > total:
        raise ValueError(f"start_page={start_page} exceeds total pages {total}.")

    end_page = min(end_page, total)

    writer = PdfWriter()
    for idx in range(start_page - 1, end_page):
        writer.add_page(reader.pages[idx])

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with output_pdf.open("wb") as f:
        writer.write(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a PDF by page range (1-indexed, inclusive).")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/liuxuesheng.pdf"),
        help="Source PDF (default: data/raw/liuxuesheng.pdf).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/liuxuesheng_42_492.pdf"),
        help="Output PDF slice (default: data/processed/liuxuesheng_42_492.pdf).",
    )
    parser.add_argument("--start", type=int, default=42, help="Start page (1-indexed). Default: 42")
    parser.add_argument("--end", type=int, default=492, help="End page (1-indexed, inclusive). Default: 492")
    args = parser.parse_args()

    split_pdf_pages(args.input, args.output, args.start, args.end)
    print(f"Exported: {args.output.resolve()}")


if __name__ == "__main__":
    main()
