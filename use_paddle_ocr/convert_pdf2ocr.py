from __future__ import annotations

import argparse
import os
from pathlib import Path

import paddle
from paddleocr import PaddleOCR


def build_ocr(target_device: str) -> PaddleOCR:
    return PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device=target_device,
    )


def run_ocr(input_file: Path, out_root: Path) -> None:
    print("Working directory:", os.getcwd())
    print("Input exists:", input_file.exists())

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file.resolve()}")

    json_dir = out_root / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    device = "gpu" if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0 else "cpu"
    print("OCR device (initial):", device)

    try:
        ocr = build_ocr(device)
    except RuntimeError as err:
        err_msg = str(err)
        # RTX 50 series GPUs may trigger "Unsupported GPU architecture" on older Paddle builds.
        if device == "gpu" and "Unsupported GPU architecture" in err_msg:
            print("Detected unsupported GPU architecture for this Paddle build; falling back to CPU.")
            device = "cpu"
            ocr = build_ocr(device)
        else:
            raise

    print("OCR device (actual):", device)

    results = ocr.predict(str(input_file))

    for page_idx, res in enumerate(results, start=1):
        # PaddleOCR writes one JSON file per page; store under a dedicated folder and rename
        # to a stable page-based filename.
        res.save_to_json(str(json_dir))

        # Default filename is usually like:
        # - {stem}_{page_index}_res.json
        # where page_index is 0-based.
        default_json = json_dir / f"{input_file.stem}_{page_idx - 1}_res.json"
        target_json = json_dir / f"page_{page_idx:04}.json"

        if default_json.exists() and not target_json.exists():
            default_json.rename(target_json)

        if page_idx % 10 == 0:
            print(f"Pages processed: {page_idx}")

    print(f"Done. JSON directory: {json_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PaddleOCR on each PDF page and save per-page JSON.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/liuxuesheng_42_492.pdf"),
        help="Input PDF path (default: sliced PDF under data/processed/).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/liuxuesheng"),
        help="Output directory root. Default: output/liuxuesheng",
    )
    args = parser.parse_args()

    run_ocr(args.input, args.out)


if __name__ == "__main__":
    main()
