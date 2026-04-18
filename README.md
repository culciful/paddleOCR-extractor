# use-paddle-ocr — PaddleOCR biography pipeline

End goal: run **PaddleOCR** on PDF pages, then **split text into one JSON + one text file per person** for downstream search or LLM use.

**Install:** `pip install -r requirements.txt` plus Paddle (`paddlepaddle` / GPU) from the [official install guide](https://www.paddlepaddle.org.cn/install/quick). 

---

## Layout

```text
use_paddle_ocr/           # Python package (import use_paddle_ocr)
  extraction/             # Place structured field-extraction code here
  split_pdf_pages.py
  convert_pdf2ocr.py
  extract_people.py
  filter_keywords.py
data/raw/                 # full PDFs
data/processed/           # page-range slices
output/liuxuesheng/       # OCR + people (gitignored)
  json/
  people_llm/             # entries + manifest (upstream for extraction)
extractions/liuxuesheng/  # tracked: structured extraction outputs (see README there)
```

Run CLIs **from the repository root**:

```bash
cd /path/to/usePaddle
python -m use_paddle_ocr.split_pdf_pages --help
```

---

## Pipeline

```text
python -m use_paddle_ocr.split_pdf_pages   → data/processed/*.pdf
        ↓
python -m use_paddle_ocr.convert_pdf2ocr   → output/<name>/json/page_*.json
        ↓
python -m use_paddle_ocr.extract_people   → people_llm/ + aggregated JSON
        ↓
python -m use_paddle_ocr.filter_keywords   → optional keyword JSON
        ↓
(use_paddle_ocr.extraction — add your field extraction scripts)
```

---

## Modules


| Module                              | Role                                                        |
| ----------------------------------- | ----------------------------------------------------------- |
| `use_paddle_ocr.split_pdf_pages`    | 1-based inclusive page range → PDF slice.                   |
| `use_paddle_ocr.convert_pdf2ocr`    | PaddleOCR; per-page JSON under `<out>/json/`.               |
| `use_paddle_ocr.extract_people`     | Page JSON → people + `entries/` + `manifest.jsonl` (incl. person-start regex + `normalize_years`). |
| `use_paddle_ocr.filter_keywords`    | Substring filter over manifest + bodies.                    |
| `use_paddle_ocr.extraction`         | Package for **structured data extraction** (add your code). |


---

## 1. `split_pdf_pages`

```bash
python -m use_paddle_ocr.split_pdf_pages \
  --input data/raw/liuxuesheng.pdf \
  --output data/processed/liuxuesheng_42_492.pdf \
  --start 42 --end 492
```


| Argument            | Default                                 |
| ------------------- | --------------------------------------- |
| `--input`           | `data/raw/liuxuesheng.pdf`              |
| `--output`          | `data/processed/liuxuesheng_42_492.pdf` |
| `--start` / `--end` | `42` / `492` (1-based, inclusive)       |


---

## 2. `convert_pdf2ocr`

```bash
python -m use_paddle_ocr.convert_pdf2ocr --input data/processed/your_slice.pdf --out output/liuxuesheng
```


| Argument  | Default                                 |
| --------- | --------------------------------------- |
| `--input` | `data/processed/liuxuesheng_42_492.pdf` |
| `--out`   | `output/liuxuesheng`                    |


---

## 3. `extract_people`

```bash
python -m use_paddle_ocr.extract_people \
  --input-dir output/liuxuesheng/json \
  --output output/liuxuesheng/people_all_pages_aggregated.json \
  --output-dir output/liuxuesheng/people_llm \
  --limit 0
```


| Argument  | Default | Notes                         |
| --------- | ------- | ----------------------------- |
| `--limit` | `0`     | `<= 0` = all page JSON files. |


**Entry files:** `entries/<id>_<name>.txt` with zero-padded global `id` (min width 4). `manifest.jsonl` keeps `start_idx` (e.g. `3-5`) for page traceability.

---

## 4. `filter_keywords`

```bash
python -m use_paddle_ocr.filter_keywords --keyword 德国
python -m use_paddle_ocr.filter_keywords --keywords 法国,比利时
```


| Argument     | Default                   |
| ------------ | ------------------------- |
| `--limit`    | `0` (= all manifest rows) |
| `--keywords` | `德国`                      |


---

## 5. Extractions

- **Data (tracked):** `extractions/liuxuesheng/` — write JSON/CSV/LLM outputs here so they are not under gitignored `output/`.
- **Code:** `use_paddle_ocr/extraction/` — add field-extraction modules / CLIs here.

---

## Typical flow

```bash
python -m use_paddle_ocr.split_pdf_pages
python -m use_paddle_ocr.convert_pdf2ocr
python -m use_paddle_ocr.extract_people
python -m use_paddle_ocr.filter_keywords
```

