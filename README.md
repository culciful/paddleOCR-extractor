# use-paddle-ocr — PaddleOCR biography pipeline

End goal: run **PaddleOCR** on PDF pages, then **split text into one JSON + one text file per person** for downstream search or LLM use.

---

## Install (environment)

1. **Python** 3.10 or newer (`requires-python` in `pyproject.toml`).
2. **Conda environment (recommended):**
  ```bash
   conda create -n usepaddle python=3.11 -y
   conda activate usepaddle
  ```
3. **Python packages for OCR:**
  ```bash
   pip install -U pip
   pip install -r requirements.txt
  ```
4. **PaddlePaddle:** the `paddlepaddle` wheel is **not** listed in `requirements.txt` (CPU vs GPU and CUDA version vary by machine). Install the build that matches your OS and GPU from the [official quick install](https://www.paddlepaddle.org.cn/install/quick), for example a GPU wheel or `paddlepaddle` for CPU-only.

### Local LLM extraction (optional)

The CLI `[extract_bio_ollama](use_paddle_ocr/extraction/extract_bio_ollama.py)` sends each biography text to a **local** model through **[Ollama](https://ollama.com/download)**’s HTTP API. The script uses only the Python standard library (`urllib`).

**Setup (once per machine)**

Commands below are for **Linux** (see [Ollama download](https://ollama.com/download) for macOS / Windows installers or other install options).

1. **Install Ollama** (official script; requires `sudo` when prompted):
  ```bash
   curl -fsSL https://ollama.com/install.sh | sh
  ```
2. **Start the Ollama daemon** (pick one):
  ```bash
   # Recommended if the installer registered a systemd service:
   sudo systemctl enable --now ollama
  ```
3. **Pull the default model** (download weights once; size is several GB):
  ```bash
   ollama pull qwen2.5:7b
  ```
4. **Sanity check** (optional):
  ```bash
   ollama -v
   ollama list
  ```

**Why `qwen2.5:7b` is the default**

- The corpus is **Chinese** biographical prose with OCR noise; **Qwen** models are a practical default for Chinese understanding at modest size.
- **7B** (often served quantized by Ollama) is a balance between **quality** and **VRAM** on a typical single consumer GPU (e.g. ~12 GB): large enough for structured JSON extraction on short entries, small enough to run comfortably next to other workloads.
- It is a **default, not a requirement**: pass `--model` to any tag you have pulled (e.g. a larger Qwen instruct variant if it fits your GPU).

**How to improve results later**

- **Model:** try a larger instruct model (e.g. 14B quantized) if VRAM allows, or switch the same script’s URL to a **cloud** OpenAI-compatible endpoint and a stronger hosted model for difficult rows only.
- **Prompt and schema:** tighten the system prompt or add a **second pass** (small repair prompt) for rows that fail JSON parse or fail manual QA.
- **Rules:** extend the existing **post-processing** in `extract_bio_ollama.py` (field normalization, heuristics) so the model does less brittle work.
- **Evaluation:** keep a small **gold-checked** set of entries; compare runs when you change model or prompt.
- **Grounding:** optional **RAG** or gazetteers later to normalize school names and fix OCR entities without bloating the main prompt.

---

## Layout

```text
use_paddle_ocr/           # Python package (import use_paddle_ocr)
  extraction/             # Structured field extraction
    extract_bio_ollama.py # Local Ollama → JSONL (education / employment, etc.)
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
python -m use_paddle_ocr.extraction.extract_bio_ollama   → optional JSONL (local Ollama)
```

---

## Modules


| Module                                         | Role                                                                                                |
| ---------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `use_paddle_ocr.split_pdf_pages`               | 1-based inclusive page range → PDF slice.                                                           |
| `use_paddle_ocr.convert_pdf2ocr`               | PaddleOCR; per-page JSON under `<out>/json/`.                                                       |
| `use_paddle_ocr.extract_people`                | Page JSON → people + `entries/` + `manifest.jsonl` (incl. person-start regex + `normalize_years`).  |
| `use_paddle_ocr.filter_keywords`               | Substring filter over manifest + bodies.                                                            |
| `use_paddle_ocr.extraction.extract_bio_ollama` | Local **Ollama** HTTP API: structured biography JSON per person → `extractions/…/bio_ollama.jsonl`. |
| `use_paddle_ocr.extraction`                    | Package namespace for extraction CLIs / helpers.                                                    |


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
python -m use_paddle_ocr.filter_keywords --keywords 德国
python -m use_paddle_ocr.filter_keywords --keywords 法国,比利时
```


| Argument     | Default                   |
| ------------ | ------------------------- |
| `--limit`    | `0` (= all manifest rows) |
| `--keywords` | `德国`                      |


---

## 5. `extract_bio_ollama` (local Ollama)

Reads `output/…/filtered_by_keywords.json` (or `--filtered-json`), loads each `content_path` under `--base-dir`, calls Ollama with `format: json`, and **appends** one JSON object per line to `--output` (default `extractions/liuxuesheng/bio_ollama.jsonl`). Each line includes outer fields such as `id`, `start_idx`, `index_name`, and `extracted` (person fields + `education` / `employment` lists).

```bash
# Ollama must be running; model must be pulled, e.g. ollama pull qwen2.5:7b
python -m use_paddle_ocr.extraction.extract_bio_ollama --resume --limit 5
```


| Argument          | Default                                                   | Notes                                                                    |
| ----------------- | --------------------------------------------------------- | ------------------------------------------------------------------------ |
| `--filtered-json` | `output/liuxuesheng/people_llm/filtered_by_keywords.json` | Input bundle from `filter_keywords`.                                     |
| `--base-dir`      | from JSON `summary.base_dir`                              | Root for `content_path` text files.                                      |
| `--output`        | `extractions/liuxuesheng/bio_ollama.jsonl`                | Append-only JSONL.                                                       |
| `--ollama-url`    | `http://127.0.0.1:11434`                                  | Ollama base URL.                                                         |
| `--model`         | `qwen2.5:7b`                                              | Must exist locally (`ollama list`).                                      |
| `--resume`        | off                                                       | Skip `id`s that already have a successful `extracted` dict in the JSONL. |
| `--limit`         | `0`                                                       | `0` = process all matches in the filtered file.                          |


---

## 6. Extractions

- **Data (tracked):** `extractions/liuxuesheng/` — write JSON/CSV/LLM outputs here so they are not under gitignored `output/`.
- **Code:** `use_paddle_ocr/extraction/` — e.g. `extract_bio_ollama.py`; add more modules as needed.

---

## Typical flow

```bash
python -m use_paddle_ocr.split_pdf_pages
python -m use_paddle_ocr.convert_pdf2ocr
python -m use_paddle_ocr.extract_people
python -m use_paddle_ocr.filter_keywords
# optional: after Ollama is installed and running
python -m use_paddle_ocr.extraction.extract_bio_ollama --resume
```

