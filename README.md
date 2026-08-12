# MySyllabus (Syllabus Converter)

A Flask web app that turns a course syllabus (PDF or TXT) into calendar deadlines. Upload a syllabus, review the extracted due dates grouped by month, add individual events to Google Calendar, or download everything as an `.ics` file. Results can be saved to a per-browser history — no account required.

## How it works

1. **Text extraction** — `pypdf` pulls raw text from the uploaded PDF (or the TXT file is read directly).
2. **Candidate selection** — a recall-first pass (`build_candidate_text`) keeps only lines containing deadline keywords or date-like tokens, plus ±3 lines of context, merging overlapping ranges. This shrinks the text sent to the LLM. If the result is under 1,500 characters, the full text is sent instead so nothing is missed.
3. **LLM extraction** — the candidate text is sent to OpenAI (`gpt-4o-mini` by default) with a strict JSON schema and a prompt that forbids inferring dates that aren't explicitly in the text. Texts over 24,000 characters are chunked and results merged.
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

## Run

```bash
python app.py
# or: flask --app app run
```

Then open http://127.0.0.1:5000.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | (required) | OpenAI API key used for extraction |
| `OPENAI_DEADLINE_MODEL` | `gpt-4o-mini` | Model used for extraction |
| `SYLLABUS_MAX_CHARS` | `24000` | Chunk size for long syllabi |
| `FLASK_SECRET_KEY` | random per process | Session signing key; set for stable sessions across restarts |

## Tests

```bash
python -m pytest
```

Tests cover the term-window date logic (including wrapping Aug–Jan terms), the extraction pipeline with a mocked LLM, candidate-text selection, ICS escaping/formatting, and the web routes. No API key is needed to run them.

## Project structure

```
app.py                  # Flask app: routes, extraction pipeline, SQLite persistence, ICS generation
templates/              # index (upload + results), history, history detail
static/style.css        # styling
tests/                  # pytest suite
syllabus.db             # local SQLite database (created automatically, not tracked)
```

## Notes and limitations

- History is tied to a browser cookie (`visitor_id`), not a user account, so it doesn't follow you across devices.
- Scanned/image-only PDFs won't work — the PDF must contain extractable text.
- This is a personal project; it is not hardened for public deployment.
