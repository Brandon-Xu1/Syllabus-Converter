# MySyllabus (Syllabus Converter)

[![CI](https://github.com/Brandon-Xu1/Syllabus-Converter/actions/workflows/ci.yml/badge.svg)](https://github.com/Brandon-Xu1/Syllabus-Converter/actions/workflows/ci.yml)

**Live demo: [syllabus-converter.onrender.com](https://syllabus-converter.onrender.com/)** — hosted on a free tier, so if the first load takes a few seconds it's just waking up. Try it with [`samples/sample_syllabus.txt`](samples/sample_syllabus.txt).

A Flask web app that turns a course syllabus (PDF, DOCX, or TXT) into calendar deadlines. Upload a syllabus, review the extracted due dates grouped by month, add individual events to Google Calendar, or download everything as an `.ics` file. Results can be saved to a per-browser history — no account required.

## How it works

1. **Text extraction** — `pypdf` pulls raw text from PDFs (with an OCR fallback for scanned PDFs when tesseract is installed), `python-docx` reads DOCX paragraphs and tables, and TXT files are read directly.
2. **Candidate selection** — a recall-first pass (`build_candidate_text`) keeps only lines containing deadline keywords or date-like tokens, plus ±3 lines of context, merging overlapping ranges. This shrinks the text sent to the LLM. If the result is under 1,500 characters, the full text is sent instead so nothing is missed.
3. **LLM extraction** — the candidate text is sent to OpenAI (`gpt-4o-mini` by default) using **structured outputs**: a strict JSON Schema constrains the model's response shape, with a graceful fallback to free-form JSON parsing for models that don't support it. Texts over 24,000 characters are chunked and results merged.
4. **Validation** — the model's output is treated as untrusted: dates must match `YYYY-MM-DD` exactly, are normalized to the user-selected academic year (wrapping terms like Aug–Jan resolve January dates to the following year), invalid dates (e.g. Feb 29 in a non-leap year) are dropped, dates outside the selected term window are filtered out, and duplicates are removed.
5. **Output** — dated events get Google Calendar links and RFC 5545-style `.ics` export; recurring items ("quiz every Friday") are shown separately and never become fake dated events.

Extraction results are cached in SQLite keyed by a SHA-256 hash of the text and parameters, so re-uploading the same syllabus is instant and costs no API calls.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your OPENAI_API_KEY
```

Optional OCR support for scanned PDFs (needs the tesseract and poppler system packages — see `requirements-ocr.txt`):

```bash
pip install -r requirements-ocr.txt
```

## Run

```bash
python app.py                  # development
gunicorn app:app               # production-style
```

Then open http://127.0.0.1:5000 (or gunicorn's port).

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | (required) | OpenAI API key used for extraction |
| `OPENAI_DEADLINE_MODEL` | `gpt-4o-mini` | Model used for extraction |
| `SYLLABUS_MAX_CHARS` | `24000` | Chunk size for long syllabi |
| `FLASK_SECRET_KEY` | random per process | Session signing key; set for stable sessions across restarts |
| `MAX_UPLOAD_MB` | `10` | Maximum upload size (larger uploads get a 413) |
| `EXTRACT_RATE_LIMIT` | `20 per hour` | Per-IP rate limit on extraction requests |
| `RATELIMIT_STORAGE_URI` | `memory://` | Rate-limit counter storage (use Redis for multi-process) |
| `SYLLABUS_DB_PATH` | `./syllabus.db` | SQLite database location (point at a mounted disk in production) |
| `TRUST_PROXY` | off | Set to `1` behind a reverse proxy so rate limiting sees real client IPs (auto-enabled on Render/Railway) |

## Deployment

The app ships deploy-ready three ways:

- **Render** — `render.yaml` is a Blueprint: in the Render dashboard choose *New → Blueprint*, point it at this repo, and set `OPENAI_API_KEY` when prompted. It builds the `Dockerfile` (which includes tesseract + poppler, so OCR works) and serves via gunicorn.
- **Railway / Heroku-style** — the `Procfile` runs gunicorn; set `OPENAI_API_KEY`, `FLASK_SECRET_KEY`, and `TRUST_PROXY=1`.
- **Any Docker host** — `docker build -t syllabus-converter . && docker run -p 8000:8000 --env-file .env syllabus-converter`.

Notes: the container runs a single gunicorn worker because SQLite and the in-memory rate limiter assume one process (scale with threads, or move to Postgres/Redis). On free tiers without a persistent disk, the SQLite file is ephemeral — saved history resets on redeploy; set `SYLLABUS_DB_PATH` to a mounted disk to persist it.

## Tests

```bash
python -m pytest
```

Tests cover the term-window date logic (including wrapping Aug–Jan terms), the extraction pipeline with a mocked LLM, LLM-output parsing (structured, fenced, and prose-embedded JSON), candidate-text selection, DOCX extraction, ICS escaping/formatting, upload size limits, rate limiting, and the web routes. No API key is needed to run them. CI runs the suite on Python 3.12 and 3.14 for every push.

## Benchmark

```bash
python scripts/benchmark.py samples/sample_syllabus.txt
```

Reports how much the candidate-text selector shrinks the LLM input, and (when `OPENAI_API_KEY` is set) cold vs. cached extraction latency per file. Pass your own `.pdf`/`.docx`/`.txt` files and optional `--year/--start-month/--end-month`.

## Project structure

```
app.py                  # Flask app: routes, extraction pipeline, SQLite persistence, ICS generation
templates/              # index (upload + results), history, history detail
static/style.css        # styling
tests/                  # pytest suite
scripts/benchmark.py    # token-reduction and latency measurement
samples/                # sample syllabus for benchmarking
Dockerfile, render.yaml, Procfile   # deployment
syllabus.db             # local SQLite database (created automatically, not tracked)
```

## Notes and limitations

- History is tied to a browser cookie (`visitor_id`), not a user account, so it doesn't follow you across devices.
- Scanned PDFs require the optional OCR dependencies (included in the Docker image).
- Single-process deployment model; see the deployment notes before scaling out.
