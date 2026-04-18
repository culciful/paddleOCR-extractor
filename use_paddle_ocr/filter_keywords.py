#!/usr/bin/env python3
"""
Scan manifest + entry bodies; keep rows whose text contains any of the given substrings.

Use for country / topic recall (e.g. --keywords Germany or --keywords France,Belgium; pass Chinese literals as needed).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set


def load_manifest_rows(manifest_path: Path, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def resolve_keywords(keywords_csv: str | None) -> List[str]:
    out: List[str] = []
    if keywords_csv:
        out.extend(p.strip() for p in keywords_csv.split(",") if p.strip())
    # dedupe preserving order
    seen: Set[str] = set()
    uniq: List[str] = []
    for k in out:
        if k not in seen:
            seen.add(k)
            uniq.append(k)
    return uniq


def keywords_hit(text: str, keywords: List[str]) -> List[str]:
    return [k for k in keywords if k in text]


def process_row(row: Dict[str, Any], base_dir: Path, keywords: List[str]) -> Dict[str, Any] | None:
    rel = row.get("content_path") or ""
    path = base_dir / rel
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    hits = keywords_hit(text, keywords)
    if not hits:
        return None
    return {
        "id": row.get("id"),
        "start_idx": row.get("start_idx"),
        "name": row.get("name"),
        "years": row.get("years"),
        "content_path": rel,
        "content_chars": row.get("content_chars"),
        "keywords_matched": hits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read manifest rows; keep entries whose body text contains any given substring.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("output/liuxuesheng/people_llm/manifest.jsonl"),
        help="Path to manifest.jsonl",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("output/liuxuesheng/people_llm"),
        help="Directory that content_path is relative to",
    )
    parser.add_argument(
        "--keywords",
        type=str,
        default="德国",
        help="Comma-separated substrings to search for in entry body",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N manifest rows; 0 means all",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/liuxuesheng/people_llm/filtered_by_keywords.json"),
        help="Write JSON with summary + matches",
    )
    args = parser.parse_args()

    keywords = resolve_keywords(args.keywords or None)
    if not keywords:
        parser.error("Provide at least one substring via --keywords a,b,c")

    rows = load_manifest_rows(args.manifest, args.limit)
    matches: List[Dict[str, Any]] = []
    for row in rows:
        rec = process_row(row, args.base_dir, keywords)
        if rec:
            matches.append(rec)

    summary = {
        "manifest": str(args.manifest),
        "base_dir": str(args.base_dir),
        "keywords": keywords,
        "match_mode": "any_keyword_in_body",
        "limit": args.limit,
        "processed": len(rows),
        "matched_count": len(matches),
        "output": str(args.output),
    }

    out_obj = {"summary": summary, "matches": matches}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
