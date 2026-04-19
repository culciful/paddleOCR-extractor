#!/usr/bin/env python3
"""
Batch biography extraction via local Ollama (e.g. qwen2.5:7b).

Reads output/.../filtered_by_keywords.json, loads each entry body from base_dir,
calls Ollama /api/chat with format=json, appends one JSON object per line to a JSONL file.

Example:
  cd /path/to/usePaddle
  python -m use_paddle_ocr.extraction.extract_bio_ollama --resume --limit 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


SYSTEM_PROMPT = """你是工具书人物小传的信息抽取助手。用户文本来自 OCR，可能有错字、标点混乱；可能夹有背面透印导致的无关片段。

你必须只输出一个 JSON 对象（不要 markdown、不要解释），字段与类型如下：
{
  "person_name": string | null,
  "lifespan": string | null,
  "highest_education": string | null,
  "education": [
    {
      "time_period": string,
      "location": string,
      "institution": string,
      "program_or_degree": string
    }
  ],
  "employment": [
    {
      "time_period": string,
      "location": string,
      "position": string
    }
  ]
}

规则：
- 仅抽取与「本条主人公」生平相关的内容；明显属于他人、另一段无关传记或印刷噪声的句子不要写入 education/employment。
- 时间若原文含糊，time_period 可填空字符串 ""，不要猜测具体年份。
- 地点与学校分工：学校、学院、大学、中学、研究机构等名称填 institution；省/市/国等地理信息填 location。若原文只有校名没有地点，location 用 ""。不要把校名只写在 location 而留空 institution（除非原文完全无法区分机构类型，此时可退化为 institution 填原文机构称呼）。
- 若原文是「国名+校名」连写且无法拆分（如「瑞士楚里西大学」「德国柏林大学」），location 填国名（二字），institution 填校名剩余部分。
- education、employment 数组中每一条对象必须包含该结构下的全部键，禁止省略键；无信息一律用空字符串 ""，不要用 null 作为列表项里的字段值。
- employment 尽量「一用人单位一条」：同一句话里并列多个单位且各自有不同职务时，拆成多条；每条 position 以单一职务或紧密相关的一两个头衔为主，避免把十几个机关用顿号堆进同一个 location 或 position；若原文确为一句笼统列举且无法合理拆分，再保留一条并简短概括。
- 转学、多次就读：拆成多条 education。
- 学业上的「毕业」「考入」「入读」写入 education，不要写入 employment；employment 只保留任职、教职、行政职务等。
- person_name、lifespan 以正文为主；若与用户给出的索引名/年生年不一致，仍如实填正文理解结果，不要强行改成索引。
- 顶层 person_name、lifespan、highest_education 无把握时可用 null；education / employment 数组内字段只用 ""，不要用 null。
- highest_education 只写学位层级词即可（如 博士、硕士、学士），不要写「留学某国」等前缀。
"""


def load_done_ids(jsonl_path: Path) -> Set[int]:
    """Ids that already have a successful extraction (failed rows are not skipped)."""
    done: Set[int] = set()
    if not jsonl_path.is_file():
        return done
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                ext = obj.get("extracted")
                if isinstance(obj.get("id"), int) and isinstance(ext, dict):
                    done.add(obj["id"])
            except json.JSONDecodeError:
                continue
    return done


def strip_code_fence(text: str) -> str:
    t = text.strip()
    m = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", t, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return t


def parse_model_json(content: str) -> Dict[str, Any]:
    s = strip_code_fence(content)
    return json.loads(s)


EDU_KEYS = ("time_period", "location", "institution", "program_or_degree")
EMP_KEYS = ("time_period", "location", "position")


def _optional_str_scalar(v: Any) -> Any:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _row_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return str(v).strip()


def normalize_extracted(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Fill missing keys, coerce list items to uniform string fields (downstream-safe)."""
    out: Dict[str, Any] = {
        "person_name": _optional_str_scalar(raw.get("person_name")),
        "lifespan": _optional_str_scalar(raw.get("lifespan")),
        "highest_education": _optional_str_scalar(raw.get("highest_education")),
        "education": [],
        "employment": [],
    }

    edu_in = raw.get("education")
    if isinstance(edu_in, list):
        for item in edu_in:
            if isinstance(item, dict):
                out["education"].append({k: _row_str(item.get(k)) for k in EDU_KEYS})

    emp_in = raw.get("employment")
    if isinstance(emp_in, list):
        for item in emp_in:
            if isinstance(item, dict):
                out["employment"].append({k: _row_str(item.get(k)) for k in EMP_KEYS})

    return out


# 常见二字国名/地区前缀，用于「瑞士楚里西大学」→ location + institution 拆分
_COUNTRY_PREFIX_2: Tuple[str, ...] = (
    "瑞士",
    "德国",
    "法国",
    "英国",
    "美国",
    "日本",
    "韩国",
    "瑞典",
    "荷兰",
    "俄国",
    "苏联",
    "越南",
    "印度",
    "加拿大",
    "澳洲",
    "澳大利",
    "意大利",
    "比利时",
    "奥地利",
    "西班牙",
    "葡萄牙",
    "土耳其",
    "墨西哥",
    "新加坡",
    "菲律宾",
    "泰国",
    "朝鲜",
    "丹麦",
    "挪威",
    "芬兰",
    "波兰",
    "捷克",
    "希腊",
    "埃及",
    "南非",
)


def infer_highest_education_from_body(body: str) -> Optional[str]:
    """从开篇「留学…，学位」句兜底；只返回学位词（博士/硕士/学士…），不含「留学」前缀。"""
    head = body.strip()[:220]
    if not head or "留学" not in head:
        return None
    deg = r"(?:名誉博士|副博士|博士后|博士|硕士|学士|研究生)"
    m = re.search(rf"留学[^。\n]{{1,52}}?[，,、]\s*{deg}", head)
    if m:
        inner = re.search(deg, m.group(0))
        if inner:
            return inner.group(0)
    m2 = re.search(r"留学[^。\n]{1,52}?。", head)
    if m2:
        inner = re.search(deg, m2.group(0))
        if inner:
            return inner.group(0)
    return None


_DEGREE_WORD = re.compile(r"(?:名誉博士|副博士|博士后|博士|硕士|学士|研究生)")


def shorten_highest_education_field(raw: Optional[str]) -> Optional[str]:
    """去掉「留学…，」类前缀，只保留学位词；无留学前缀且非纯学位短语则保持原样。"""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if _DEGREE_WORD.fullmatch(s):
        return s
    if "留学" in s:
        m = re.search(r"留学[^，。\n]{0,48}?[，,、]\s*((?:名誉博士|副博士|博士后|博士|硕士|学士|研究生))", s)
        if m:
            return m.group(1)
        m2 = _DEGREE_WORD.search(s)
        if m2:
            return m2.group(0)
    return s


def _fix_edu_country_school_split(row: Dict[str, str]) -> None:
    loc = (row.get("location") or "").strip()
    inst = (row.get("institution") or "").strip()
    if inst or not loc:
        return
    if not any(k in loc for k in ("大学", "学院", "中学", "专门学校")):
        return
    for p in _COUNTRY_PREFIX_2:
        if loc.startswith(p) and len(loc) > len(p):
            rest = loc[len(p) :].lstrip("、，, ")
            if rest:
                row["location"] = p
                row["institution"] = rest
                return
    row["institution"] = loc
    row["location"] = ""


def _employment_is_graduation_only(row: Dict[str, str]) -> bool:
    pos = (row.get("position") or "").strip()
    if "毕业" not in pos:
        return False
    if any(k in pos for k in ("任", "聘为", "兼任", "出任", "担任")):
        return False
    if pos.endswith("毕业") or pos in ("毕业", "卒业"):
        return True
    return "毕业" in pos and len(pos) <= 28


def _patch_education_times_from_removed_graduation(
    education: List[Dict[str, str]], removed: List[Dict[str, str]]
) -> None:
    for emp in removed:
        tp = (emp.get("time_period") or "").strip()
        pos = (emp.get("position") or "").strip()
        if "毕业" not in pos:
            continue
        school = pos.replace("毕业", "").replace("。", "").strip()
        if not school:
            continue
        merged = False
        for edu in education:
            blob = (edu.get("institution") or "") + (edu.get("location") or "")
            if school and school in blob:
                if not (edu.get("time_period") or "").strip() and tp:
                    edu["time_period"] = tp
                merged = True
                break
        if not merged:
            education.append(
                {
                    "time_period": tp,
                    "location": "",
                    "institution": school,
                    "program_or_degree": "",
                }
            )


def refine_extracted(extracted: Dict[str, Any], body: str) -> Dict[str, Any]:
    """规则后处理：学历兜底、国名+校名拆分、去掉误放入 employment 的毕业行。"""
    edu = extracted.get("education")
    if isinstance(edu, list):
        for row in edu:
            if isinstance(row, dict):
                _fix_edu_country_school_split(row)

    emp = extracted.get("employment")
    removed: List[Dict[str, str]] = []
    if isinstance(emp, list):
        kept: List[Dict[str, str]] = []
        for row in emp:
            if not isinstance(row, dict):
                continue
            if _employment_is_graduation_only(row):
                removed.append(dict(row))
            else:
                kept.append(row)
        extracted["employment"] = kept
        if removed and isinstance(edu, list):
            _patch_education_times_from_removed_graduation(edu, removed)

    if extracted.get("highest_education") is None:
        inferred = infer_highest_education_from_body(body)
        if inferred:
            extracted["highest_education"] = inferred

    he = extracted.get("highest_education")
    if he is not None:
        extracted["highest_education"] = shorten_highest_education_field(he)

    return extracted


def ollama_chat(
    base_url: str,
    model: str,
    system: str,
    user: str,
    timeout_sec: float,
    temperature: float,
) -> str:
    url = base_url.rstrip("/") + "/api/chat"
    body: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": temperature},
    }
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read().decode("utf-8")
    payload = json.loads(raw)
    msg = payload.get("message") or {}
    content = msg.get("content")
    if not isinstance(content, str):
        raise ValueError(f"Unexpected Ollama response: {repr(payload)[:500]}")
    return content


def build_user_message(index_name: str, index_years: str, start_idx: str, body: str) -> str:
    return (
        "【索引参考（可能含 OCR 错误，仅供对照）】\n"
        f"start_idx（本书页码/栏位，仅追溯，输出 JSON 中勿重复该字段）: {start_idx}\n"
        f"姓名: {index_name}\n"
        f"年生年: {index_years}\n\n"
        "【正文】\n"
        f"{body.strip()}\n"
    )


def extract_with_retries(
    base_url: str,
    model: str,
    user_msg: str,
    retries: int,
    timeout_sec: float,
    temperature: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], str]:
    last_content = ""
    last_err: Optional[str] = None
    user_round = user_msg
    for attempt in range(retries):
        try:
            last_content = ollama_chat(
                base_url, model, SYSTEM_PROMPT, user_round, timeout_sec, temperature
            )
            parsed = parse_model_json(last_content)
            return parsed, None, last_content
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as e:
            last_err = f"{type(e).__name__}: {e}"
        if attempt < retries - 1:
            user_round = (
                user_msg
                + "\n\n【重要】上一回复不是合法 JSON 或解析失败。请严格只输出一个 JSON 对象，"
                "键名与 SYSTEM 说明一致；education 与 employment 的每个数组元素必须包含全部子键且值为字符串（可 \"\"），"
                "不要 markdown 代码块，不要其它文字。"
            )
            time.sleep(1.0)
    return None, last_err, last_content


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract structured biography fields via Ollama (local).",
    )
    parser.add_argument(
        "--filtered-json",
        type=Path,
        default=Path("output/liuxuesheng/people_llm/filtered_by_keywords.json"),
        help="Path to filtered_by_keywords.json",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Override summary.base_dir (default: read from filtered JSON)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("extractions/liuxuesheng/bio_ollama.jsonl"),
        help="Append-only JSONL output path",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://127.0.0.1:11434",
        help="Ollama server base URL",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5:7b",
        help="Ollama model name (must be pulled locally)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N matches (0 = all)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip entry ids already present in output JSONL",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between successful requests",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="HTTP timeout per request (seconds)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Ollama temperature (lower = stabler JSON)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Attempts per entry on parse/network failure",
    )
    args = parser.parse_args()

    filtered_path: Path = args.filtered_json
    if not filtered_path.is_file():
        print(f"Missing file: {filtered_path}", file=sys.stderr)
        sys.exit(1)

    with filtered_path.open("r", encoding="utf-8") as f:
        bundle = json.load(f)
    matches: List[Dict[str, Any]] = bundle.get("matches") or []
    summary: Dict[str, Any] = bundle.get("summary") or {}
    base_dir: Path = args.base_dir or Path(str(summary.get("base_dir") or "output/liuxuesheng/people_llm"))

    out_path: Path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = load_done_ids(out_path) if args.resume else set()

    mode = "a" if out_path.exists() else "w"
    processed = 0
    skipped = 0
    failed = 0

    with out_path.open(mode, encoding="utf-8") as out_f:
        for row in matches:
            eid = row.get("id")
            if not isinstance(eid, int):
                continue
            if eid in done_ids:
                skipped += 1
                continue
            rel = row.get("content_path") or ""
            path = base_dir / rel
            body = path.read_text(encoding="utf-8") if path.is_file() else ""
            index_name = str(row.get("name") or "")
            index_years = str(row.get("years") or "")
            start_idx = str(row.get("start_idx") or "")
            user_msg = build_user_message(index_name, index_years, start_idx, body)

            extracted, err, raw_content = extract_with_retries(
                args.ollama_url,
                args.model,
                user_msg,
                retries=max(1, args.retries),
                timeout_sec=args.timeout,
                temperature=args.temperature,
            )

            if isinstance(extracted, dict):
                extracted = normalize_extracted(extracted)
                extracted = refine_extracted(extracted, body)

            record: Dict[str, Any] = {
                "id": eid,
                "start_idx": row.get("start_idx"),
                "index_name": index_name,
                "index_years": index_years,
                "content_path": rel,
                "ollama_model": args.model,
                "extracted": extracted,
                "error": err,
            }
            if extracted is None:
                record["raw_content"] = raw_content[:8000] if raw_content else ""
                failed += 1
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            processed += 1
            if isinstance(extracted, dict):
                done_ids.add(eid)
            if args.sleep > 0:
                time.sleep(args.sleep)
            if args.limit > 0 and processed >= args.limit:
                break

    print(
        json.dumps(
            {
                "output": str(out_path),
                "processed_this_run": processed,
                "skipped_resume": skipped,
                "failed_this_run": failed,
                "model": args.model,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
