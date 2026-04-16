import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional


PERSON_START_PATTERN = re.compile(
    r"^(?P<name>[\u4e00-\u9fa5]{1,4})（(?P<years>\d{3,4}[—\-][\d\?？]{0,4}|\d{3,4}[—\-]?)）"
)
STROKE_TITLE_PATTERN = re.compile(r"^[一二三四五六七八九十]+画(?:[\u4e00-\u9fa5]+)?$")
PURE_NUMBER_PATTERN = re.compile(r"^[0-9０-９]{1,3}$")


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

    # Ignore the section index like "二画", "二画丁".
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


def is_page_number_candidate(line: Dict, page_y_min: int, page_y_max: int) -> bool:
    text = line["text"].strip()
    page_span = max(1, page_y_max - page_y_min)
    bottom_band_start = page_y_min + int(page_span * 0.9)
    if not PURE_NUMBER_PATTERN.fullmatch(text):
        return False
    if line["y1"] < bottom_band_start:
        return False
    # Keep small numeric marks only.
    if line["w"] > 60 or line["h"] > 40:
        return False
    return True


def resolve_pdf_page_number(candidates: List[Dict]) -> Optional[str]:
    if not candidates:
        return None
    # Pick the lowest candidate first; if multiple are on same baseline,
    # concatenate from left to right to support split digits.
    candidates = sorted(candidates, key=lambda x: (-x["y1"], x["x1"]))
    baseline_y = candidates[0]["y1"]
    same_line = [c for c in candidates if abs(c["y1"] - baseline_y) <= 12]
    same_line = sorted(same_line, key=lambda x: x["x1"])
    page_no = "".join(c["text"] for c in same_line)
    return page_no if page_no else None


def extract_people(lines: List[Dict]) -> List[Dict]:
    if not lines:
        return []

    page_y_min = min(line["y1"] for line in lines)
    page_y_max = max(line["y2"] for line in lines)
    cleaned = []
    page_num_candidates = []
    for line in lines:
        if not is_noise_line(line, page_y_min, page_y_max):
            cleaned.append(line)
        elif is_page_number_candidate(line, page_y_min, page_y_max):
            page_num_candidates.append(line)

    cleaned = assign_columns(cleaned)
    ordered = sorted(cleaned, key=lambda x: (x["col"], x["y1"], x["x1"]))

    people: List[Dict] = []
    current: Optional[Dict] = None
    pdf_page_no = resolve_pdf_page_number(page_num_candidates)
    fallback_page_no = str(lines[0].get("page_index", 0) + 1)
    page_no = pdf_page_no or fallback_page_no

    for line in ordered:
        text = line["text"]
        matched = PERSON_START_PATTERN.match(text)

        if matched:
            if current:
                current["content"] = "".join(current.pop("lines"))
                people.append(current)

            current = {
                "name": matched.group("name"),
                "years": matched.group("years"),
                "start_idx": "",
                "lines": [text],
            }
            continue

        if current:
            current["lines"].append(text)

    if current:
        current["content"] = "".join(current.pop("lines"))
        people.append(current)

    for i, person in enumerate(people, start=1):
        person["start_idx"] = f"{page_no}-{i}"

    return people


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract biography entries from PaddleOCR JSON output.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/output_42_0_res.json"),
        help="Path to PaddleOCR JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save extracted entries as JSON.",
    )
    args = parser.parse_args()

    lines = load_lines(args.input)
    people = extract_people(lines)

    if not people:
        print("未提取到人物条目，请检查正则或过滤规则。")
        return

    for i, person in enumerate(people, start=1):
        print(f"{i}. {person['name']}（{person['years']}）")
        print(person["content"])
        print("-" * 60)

    if args.output:
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(people, f, ensure_ascii=False, indent=2)
        print(f"已保存到: {args.output}")


if __name__ == "__main__":
    main()
