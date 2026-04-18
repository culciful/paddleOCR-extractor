import argparse
import json
import re
from pathlib import Path
from typing import Dict, Final, List, Optional, Tuple

# --- Person-entry line patterns (OCR-tolerant year span in parentheses) ---

# Same rules as PERSON_START years: 4-digit era, OCR dash variants, optional tail / unknown-year markers.
YEARS_CORE: Final[str] = r"\d{3,4}[—\-一][\d\?？]{0,4}|\d{3,4}[—\-一]?"

PERSON_START_PATTERN = re.compile(
    rf"^(?P<name>[\u4e00-\u9fa5]{{1,4}})[（(](?P<years>{YEARS_CORE})[）)]?"
)

# STROKE_TITLE_PATTERN: stroke-count index lines, e.g. "二画" / "二画丁", "十七画以上",
# "戴 十七画以上" / "戴 檀 廢 魏 十七画以上" (OCR may insert spaces between surnames).
STROKE_TITLE_PATTERN = re.compile(
    r"^(?:[\u4e00-\u9fa5]{1,12}(?:\s+[\u4e00-\u9fa5]{1,12})*\s+|[\u4e00-\u9fa5]{1,24}\s*)?"
    r"[一二三四五六七八九十]+画(?:以上)?(?:[\u4e00-\u9fa5]+)?$"
)
PURE_NUMBER_PATTERN = re.compile(r"^[0-9０-９]{1,3}$")

def normalize_years(raw: str) -> str:
    """Unify common OCR variants in the parenthetical year span for downstream use."""
    if not raw:
        return raw
    s = raw.replace("？", "?")
    s = re.sub(r"[一—]", "-", s)
    return s

def load_lines(json_path: Path) -> List[Dict]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rec_texts = data["rec_texts"]
    rec_boxes = data["rec_boxes"]
    rec_scores = data.get("rec_scores", [1.0] * len(rec_texts))
    page_index = data.get("page_index", 0)

    lines = []
    for idx, (text, box, score) in enumerate(zip(rec_texts, rec_boxes, rec_scores)):
        x1, y1, x2, y2 = box
        lines.append(
            {
                "idx": idx,
                "page_index": page_index,
                "text": str(text).strip(),
                "score": float(score),
                "box": [x1, y1, x2, y2],
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "w": x2 - x1,
                "h": y2 - y1,
            }
        )
    return lines


def assign_columns(lines: List[Dict]) -> List[Dict]:
    if not lines:
        return lines

    # For this dictionary-like layout, text is mostly left-aligned in each column.
    # Using x1 is usually more stable than x-center.
    xs = [line["x1"] for line in lines]
    x_span = max(xs) - min(xs)
    n = len(xs)

    if x_span < 80 or n < 8:
        for line in lines:
            line["col"] = 0
        return lines

    def kmeans_1d(values: List[float], k: int, max_iter: int = 40):
        v_min, v_max = min(values), max(values)
        if v_min == v_max:
            return [v_min] * k, [0] * len(values), 0.0

        # Evenly-spaced initialization across the x-range.
        if k == 1:
            centers = [(v_min + v_max) / 2.0]
        else:
            step = (v_max - v_min) / (k - 1)
            centers = [v_min + i * step for i in range(k)]
        assignments = [0] * len(values)

        for _ in range(max_iter):
            changed = False
            for i, x in enumerate(values):
                new_cluster = min(range(k), key=lambda c: abs(x - centers[c]))
                if assignments[i] != new_cluster:
                    assignments[i] = new_cluster
                    changed = True

            groups = {i: [] for i in range(k)}
            for x, c in zip(values, assignments):
                groups[c].append(x)

            for c in range(k):
                if groups[c]:
                    centers[c] = sum(groups[c]) / len(groups[c])

            if not changed:
                break

        inertia = sum((x - centers[c]) ** 2 for x, c in zip(values, assignments))
        return centers, assignments, inertia

    # Try k=1,2,3 and pick smallest k that is good enough.
    candidates = {}
    for k in (1, 2, 3):
        centers, assignments, inertia = kmeans_1d(xs, k)
        counts = [0] * k
        for c in assignments:
            counts[c] += 1

        candidates[k] = {
            "centers": centers,
            "assignments": assignments,
            "inertia": inertia,
        }

    # Elbow-style selection with conservative upgrade.
    best_k = 1
    if True:
        improve_1_to_2 = (
            (candidates[1]["inertia"] - candidates[2]["inertia"]) / max(candidates[1]["inertia"], 1e-6)
        )
        if improve_1_to_2 > 0.45:
            best_k = 2
    if best_k == 2 and 3 in candidates:
        improve_2_to_3 = (
            (candidates[2]["inertia"] - candidates[3]["inertia"]) / max(candidates[2]["inertia"], 1e-6)
        )
        if improve_2_to_3 > 0.35:
            best_k = 3

    centers = candidates[best_k]["centers"]
    assignments = candidates[best_k]["assignments"]
    cluster_order = sorted(range(best_k), key=lambda c: centers[c])
    cluster_to_col = {cluster: col for col, cluster in enumerate(cluster_order)}

    for line, cluster in zip(lines, assignments):
        line["col"] = cluster_to_col[cluster]

    return lines


def is_noise_line(line: Dict, page_y_min: int, page_y_max: int) -> bool:
    text = line["text"]
    x1, y1, x2, y2 = line["box"]
    page_span = max(1, page_y_max - page_y_min)
    top_band_limit = page_y_min + max(80, int(page_span * 0.08))
    bottom_band_start = page_y_min + int(page_span * 0.9)

    if not text:
        return True

    # Ignore section index like "二画", "二画丁" (STROKE_TITLE_PATTERN).
    if STROKE_TITLE_PATTERN.fullmatch(text):
        return True

    # Ignore tiny page-number-like digits near page bottom area.
    if PURE_NUMBER_PATTERN.fullmatch(text):
        if y1 > bottom_band_start and line["w"] <= 20:
            return True
        if line["w"] <= 20 and line["h"] <= 25:
            return True

    # Ignore top index marker block area.
    if y2 < top_band_limit and "画" in text:
        return True

    return False


def finalize_person(person: Dict) -> Dict:
    person["content"] = "".join(person.pop("lines"))
    return person


def extract_people(lines: List[Dict], current: Optional[Dict], page_seq: Dict[str, int]) -> Tuple[List[Dict], Optional[Dict]]:
    if not lines:
        return [], current

    page_y_min = min(line["y1"] for line in lines)
    page_y_max = max(line["y2"] for line in lines)
    cleaned = []
    for line in lines:
        if not is_noise_line(line, page_y_min, page_y_max):
            cleaned.append(line)

    cleaned = assign_columns(cleaned)
    ordered = sorted(cleaned, key=lambda x: (x["col"], x["y1"], x["x1"]))

    people: List[Dict] = []
    page_no = str(lines[0].get("page_index", 0) + 1)

    for line in ordered:
        text = line["text"]
        matched = PERSON_START_PATTERN.match(text)

        if matched:
            if current:
                people.append(finalize_person(current))

            page_seq[page_no] = page_seq.get(page_no, 0) + 1

            current = {
                "name": matched.group("name"),
                "years": normalize_years(matched.group("years")),
                "start_idx": f"{page_no}-{page_seq[page_no]}",
                "lines": [text],
            }
            continue

        if current:
            current["lines"].append(text)

    return people, current


def list_page_json_files(input_dir: Path, limit: int) -> List[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir.resolve()}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir.resolve()}")
    files = sorted(input_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found under: {input_dir.resolve()}")
    if limit <= 0:
        return files
    return files[:limit]


def sanitize_filename(raw: str) -> str:
    cleaned = re.sub(r"[^\w\-]+", "_", raw, flags=re.UNICODE).strip("_")
    return cleaned or "entry"


def write_llm_artifacts(people: List[Dict], output_dir: Path) -> Dict:
    entries_dir = output_dir / "entries"
    entries_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    n = len(people)
    id_width = max(4, len(str(n))) if n else 4

    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for idx, person in enumerate(people, start=1):
            start_idx = str(person.get("start_idx", f"unknown-{idx}"))
            name = str(person.get("name", "unknown"))
            years = str(person.get("years", ""))
            content = str(person.get("content", "")).strip()
            # Sortable filename: global id (same as manifest "id"), not page-local start_idx.
            file_stem = sanitize_filename(f"{idx:0{id_width}d}_{name}")
            text_path = entries_dir / f"{file_stem}.txt"
            text_path.write_text(content + "\n", encoding="utf-8")

            record = {
                "id": idx,
                "start_idx": start_idx,
                "name": name,
                "years": years,
                "content_path": str(text_path.relative_to(output_dir)),
                "content_chars": len(content),
            }
            manifest_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "total_people": len(people),
        "manifest": str(manifest_path),
        "entries_dir": str(entries_dir),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract biography entries from page-level PaddleOCR JSON files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("output/liuxuesheng/json"),
        help="Directory containing page-level JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/liuxuesheng/people_all_pages_aggregated.json"),
        help="Path to save aggregated extraction result.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/liuxuesheng/people_llm"),
        help="Directory for LLM-friendly artifacts (entries + manifest).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="How many page JSON files to process; <=0 means all files.",
    )
    args = parser.parse_args()

    input_files = list_page_json_files(args.input_dir, args.limit)
    all_people: List[Dict] = []
    pages: List[Dict] = []
    current_person: Optional[Dict] = None
    page_seq: Dict[str, int] = {}

    for idx, json_path in enumerate(input_files, start=1):
        lines = load_lines(json_path)
        people, current_person = extract_people(lines, current_person, page_seq)
        page_index = int(lines[0].get("page_index", idx - 1)) if lines else idx - 1
        page_no = page_index + 1

        pages.append(
            {
                "source_file": json_path.name,
                "page_index": page_index,
                "page_no": page_no,
                "people_count": len(people),
                "people": people,
            }
        )
        all_people.extend(people)
        print(f"[{idx}/{len(input_files)}] {json_path.name}: {len(people)} closed entries")

    if current_person:
        finalized = finalize_person(current_person)
        all_people.append(finalized)
        if pages:
            pages[-1]["people"].append(finalized)
            pages[-1]["people_count"] = len(pages[-1]["people"])
        print(f"Finalized trailing cross-page entry: {finalized['start_idx']} {finalized['name']}")

    result = {
        "input_dir": str(args.input_dir),
        "processed_files": len(input_files),
        "total_people": len(all_people),
        "pages": pages,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved aggregated result to: {args.output}")

    llm_summary = write_llm_artifacts(all_people, args.output_dir)
    print(f"Saved LLM artifacts to: {args.output_dir}")
    print(f"Manifest: {llm_summary['manifest']}")


if __name__ == "__main__":
    main()
