from __future__ import annotations

import re
from datetime import datetime, date, timedelta
from typing import List, Dict, Any
import json
import os
import sqlite3
import uuid
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env or your environment.")


from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    make_response,
)

try:
    from pypdf import PdfReader  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    PdfReader = None

app = Flask(__name__)
app.secret_key = "replace-me"  # simple default; override in production

# --- Persistence (SQLite) ---
DB_PATH = os.path.join(os.path.dirname(__file__), "syllabus.db")

def _db_conn():
    return sqlite3.connect(DB_PATH)

def _init_db():
    with _db_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS saved_entry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                visitor_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                term_year INTEGER NULL,
                start_month INTEGER NULL,
                end_month INTEGER NULL,
                events_json TEXT NOT NULL,
                recurring_json TEXT NOT NULL,
                label TEXT NULL
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_saved_entry_visitor_created ON saved_entry(visitor_id, created_at DESC);")

_init_db()


def _get_or_create_visitor_id() -> str:
    vid = request.cookies.get("visitor_id")
    if vid and isinstance(vid, str) and len(vid) <= 64:
        return vid
    return uuid.uuid4().hex

@app.after_request
def _ensure_visitor_cookie(resp):
    if not request.cookies.get("visitor_id"):
        resp.set_cookie("visitor_id", _get_or_create_visitor_id(), httponly=True, samesite="Lax", max_age=60*60*24*365*2)
    return resp

# Using ChatGPT for extraction; no heuristic diagnostics needed


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    events: List[Dict[str, Any]] = []
    recurring_items: List[Dict[str, Any]] = []

    if request.method == "POST":
        uploaded = request.files.get("syllabus")
        # Persist AI mode preference from the form
        session["use_ai"] = bool(request.form.get("use_ai"))
        if not uploaded or uploaded.filename == "":
            error = "Please choose a syllabus file to upload."
        else:
            try:
                semester = (request.form.get("semester") or "").strip() or None
                year_str = (request.form.get("year") or "").strip()
                year = int(year_str) if year_str.isdigit() else None
                sm_str = (request.form.get("start_month") or "").strip()
                em_str = (request.form.get("end_month") or "").strip()
                start_month = int(sm_str) if sm_str.isdigit() else None
                end_month = int(em_str) if em_str.isdigit() else None

                text = extract_text(uploaded)
                print("PDF text chars:", len(text))
                print("PDF preview:", text[:300].replace("\n"," "))
                due_dates = extract_due_dates(text, year)
                session["ai_status"] = "active"
                # Remember user parameters for Save Entry
                session["term_year"] = year
                session["start_month"] = start_month
                session["end_month"] = end_month
                # Split dated and recurring items
                dated_events, recurring_items = split_events(due_dates)
                # Store only dated events for ICS usage
                session["events"] = [
                    {"date": event["date"].isoformat(), "description": event["description"]}
                    for event in dated_events
                ]
                # Store recurring items separately for display-only
                session["recurring"] = [
                    {"date": None, "description": item["description"]}
                    for item in recurring_items
                ]
                # Only build calendar links for dated events
                events = add_calendar_links(dated_events)
                if not events:
                    flash("No due dates were detected. Try cleaning up the PDF or uploading a text export.", "info")
            except RuntimeError as exc:
                error = str(exc)
            except Exception:  # pragma: no cover - generic failure handler
                error = "We couldn't read that file. Make sure it is a standard, text-based PDF."

    else:
        # Reset UI to original state on refresh/open by clearing transient session state
        for key in ("events", "recurring", "use_ai", "ai_status"):
            session.pop(key, None)

    # Only group dated events for monthly view
    grouped = group_events_by_month(events)

    return render_template(
        "index.html",
        grouped_events=grouped,
        recurring_items=session.get("recurring", []),
        error=error,
        use_ai=session.get("use_ai", False),
        ai_status=session.get("ai_status", "off"),
    )


@app.route("/save-entry", methods=["POST"])
def save_entry():
    stored = session.get("events")
    recurring = session.get("recurring")
    if not stored and not recurring:
        flash("Nothing to save yet. Upload and extract first.", "warning")
        return redirect(url_for("index"))

    visitor_id = _get_or_create_visitor_id()
    label = (request.form.get("label") or "").strip() or None
    term_year = session.get("term_year")
    start_month = session.get("start_month")
    end_month = session.get("end_month")

    events_json = json.dumps(stored or [])
    recurring_json = json.dumps(recurring or [])

    with _db_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO saved_entry (visitor_id, created_at, term_year, start_month, end_month, events_json, recurring_json, label)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                visitor_id,
                datetime.utcnow().isoformat(timespec="seconds") + "Z",
                term_year,
                start_month,
                end_month,
                events_json,
                recurring_json,
                label,
            ),
        )
        entry_id = cur.lastrowid

    flash("Entry saved.", "info")
    resp = redirect(url_for("history"))
    # Ensure cookie is set on redirect
    if not request.cookies.get("visitor_id"):
        resp.set_cookie("visitor_id", visitor_id, httponly=True, samesite="Lax", max_age=60*60*24*365*2)
    return resp


@app.route("/history")
def history():
    visitor_id = _get_or_create_visitor_id()
    with _db_conn() as conn:
        rows = conn.execute(
            "SELECT id, created_at, term_year, start_month, end_month, events_json, label FROM saved_entry WHERE visitor_id=? ORDER BY created_at DESC",
            (visitor_id,),
        ).fetchall()

    entries = []
    for (eid, created_at, term_year, sm, em, events_json, label) in rows:
        try:
            count = len(json.loads(events_json or "[]"))
        except Exception:
            count = 0
        entries.append({
            "id": eid,
            "created_at": created_at,
            "term_year": term_year,
            "start_month": sm,
            "end_month": em,
            "label": label or "",
            "count": count,
        })

    return render_template("history.html", entries=entries)


@app.route("/history/<int:entry_id>")
def history_detail(entry_id: int):
    visitor_id = _get_or_create_visitor_id()
    with _db_conn() as conn:
        row = conn.execute(
            "SELECT id, visitor_id, created_at, term_year, start_month, end_month, events_json, recurring_json, label FROM saved_entry WHERE id=?",
            (entry_id,),
        ).fetchone()
    if not row or row[1] != visitor_id:
        flash("Entry not found.", "error")
        return redirect(url_for("history"))

    _, _, created_at, term_year, sm, em, events_json, recurring_json, label = row
    try:
        raw_events = json.loads(events_json or "[]")
    except Exception:
        raw_events = []
    try:
        recurring_items = json.loads(recurring_json or "[]")
    except Exception:
        recurring_items = []

    # Rehydrate dated events to datetime and add calendar links
    dated_events = []
    for item in raw_events:
        try:
            dt = datetime.fromisoformat(item["date"]).replace(tzinfo=None)
            dated_events.append({"date": dt, "description": item["description"]})
        except Exception:
            continue

    events_with_links = add_calendar_links(dated_events)
    grouped = group_events_by_month(events_with_links)

    return render_template(
        "history_detail.html",
        label=label or "",
        created_at=created_at,
        grouped_events=grouped,
        recurring_items=recurring_items,
    )


@app.route("/delete-entry/<int:entry_id>", methods=["POST"])
def delete_entry(entry_id: int):
    visitor_id = _get_or_create_visitor_id()
    with _db_conn() as conn:
        row = conn.execute("SELECT visitor_id FROM saved_entry WHERE id=?", (entry_id,)).fetchone()
        if not row or row[0] != visitor_id:
            flash("Entry not found.", "error")
            return redirect(url_for("history"))
        conn.execute("DELETE FROM saved_entry WHERE id=?", (entry_id,))
    flash("Entry deleted.", "info")
    return redirect(url_for("history"))


@app.route("/download-ics", methods=["POST"])
def download_ics():
    stored = session.get("events")
    if not stored:
        flash("Upload a syllabus first to generate an ICS file.", "warning")
        return redirect(url_for("index"))

    ics = build_ics(
        [
            {"date": datetime.fromisoformat(item["date"]), "description": item["description"]}
            for item in stored
        ]
    )
    response = make_response(ics)
    response.headers["Content-Disposition"] = "attachment; filename=syllabus_due_dates.ics"
    response.headers["Content-Type"] = "text/calendar; charset=utf-8"
    return response


def extract_text(uploaded_file) -> str:
    """Extract raw text from an uploaded PDF or text file."""
    filename = uploaded_file.filename.lower()
    if filename.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")

    if not filename.endswith(".pdf"):
        raise RuntimeError("Please upload a PDF or plain text file.")

    if PdfReader is None:
        raise RuntimeError("Install the 'pypdf' package to enable PDF parsing: pip install pypdf")

    uploaded_file.stream.seek(0)
    reader = PdfReader(uploaded_file.stream)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)

_URL_RE = re.compile(r'https?://', re.I)

def _has_url(text: str) -> bool:
    return bool(_URL_RE.search(text))

def _in_plausible_academic_window(d: datetime, base_year: int) -> bool:
    """Accept only dates near the inferred term (default: same year +/- 1)."""
    return (base_year - 1) <= d.year <= (base_year + 1)

def _looks_like_historical_year(d: datetime) -> bool:
    """Hard reject obviously historical years (tune as needed)."""
    return d.year < 1990



def extract_due_dates(text: str, user_year: int | None = None) -> List[Dict[str, Any]]:
    """Use ChatGPT to extract due dates and recurring items from the syllabus text.

    Returns a list of dicts with keys:
      - "date": datetime | None
      - "description": str
    """
    # Debug: ensure PDF gave us real text
    print("PDF text chars:", len(text))

    raw_items = _extract_with_chatgpt(text, user_year=user_year)
    events: List[Dict[str, Any]] = []

    for item in raw_items:
        title = str(item.get("title", "")).strip() or "Due"
        # Allow due_date to be None
        due_date_val = item.get("due_date")
        recurrence_val = item.get("recurrence")
        description = str(item.get("description", title)).strip()

        # If explicit due_date provided, parse it into datetime
        if isinstance(due_date_val, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", due_date_val.strip()):
            try:
                dt = datetime.strptime(due_date_val.strip(), "%Y-%m-%d")
                # Enforce authoritative user_year if provided
                if user_year is not None:
                    try:
                        dt = dt.replace(year=user_year)
                    except ValueError:
                        # Invalid date in target year (e.g., Feb 29 on non-leap year)
                        continue
                events.append({"date": dt, "description": f"{title}: {description}"})
            except Exception:
                # Skip invalid date strings
                continue
        else:
            # No explicit date. If recurrence exists, keep as undated recurring item.
            rec_str = (str(recurrence_val).strip() if recurrence_val is not None else "")
            if rec_str:
                events.append({"date": None, "description": f"{title}: {rec_str}"})
            # If neither due_date nor recurrence, ignore

    # Deduplicate and sort
    seen = set()
    unique: List[Dict[str, Any]] = []
    # Sort with None dates last, keep deterministic order
    def _sort_key(e: Dict[str, Any]):
        return (e["date"] is None, e["date"] or datetime.max)
    for e in sorted(events, key=_sort_key):
        if e["date"] is None:
            sig = (None, e["description"].lower())
        else:
            sig = (e["date"].date().isoformat(), e["description"].lower())
        if sig in seen:
            continue
        seen.add(sig)
        unique.append(e)

    print("Events kept:", len(unique))
    return unique








# (legacy date token finder removed)


# (legacy date token parser removed)


# (legacy token stripping removed)


# (legacy helper removed)


# ----- Heuristic and optional-AI classification to filter out non-deadlines -----

_POSITIVE_KEYWORDS = [
    "due",
    "deadline",
    "submit",
    "submission",
    "turn in",
    "upload",
    "deliver",
    "deliverable",
    "exam",
    "test",
    "quiz",
    "midterm",
    "final",
    "assignment",
    "homework",
    "hw",
    "project",
    "paper",
    "essay",
    "lab",
    "report",
    "proposal",
    "draft",
    "milestone",
    "checkpoint",
    "presentation",
    "gradescope",
    "canvas",
    "turnitin",
]

_NEGATIVE_KEYWORDS = [
    "published",
    "accessed",
    "retrieved",
    "copyright",
    "isbn",
    "doi",
    "pp.",
    "vol.",
    "no.",
    "edition",
    "ed.",
    "eds.",
    "press",
    "journal",
    "proceedings",
    "conference",
    "arxiv",
    "url",
    "http",
    "https",
    "university press",
    "oxford",
    "cambridge",
    "springer",
    "wiley",
    "sage",
    "financial times",
    "new york times",
    "washington post",
    "wall street journal",
]

_POSITIVE_KEYWORDS = []  # legacy stub to avoid straggler references


def add_calendar_links(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from urllib.parse import quote

    enhanced = []
    for event in events:
        date = event["date"]
        description = event["description"].strip()
        start = date.strftime("%Y%m%d")
        end = (date + timedelta(days=1)).strftime("%Y%m%d")
        base_url = "https://calendar.google.com/calendar/render?action=TEMPLATE"
        link = f"{base_url}&text={quote(description)}&dates={start}/{end}&details={quote('Imported from syllabus calendar')}"
        enhanced.append({"date": date, "description": description, "google_link": link})
    return enhanced


def group_events_by_month(events: List[Dict[str, Any]]):
    grouped = {}
    for event in events:
        key = event["date"].strftime("%B %Y")
        grouped.setdefault(key, []).append(event)
    for event_list in grouped.values():
        event_list.sort(key=lambda e: e["date"])
    return dict(sorted(grouped.items(), key=lambda item: datetime.strptime(item[0], "%B %Y")))


def build_ics(events: List[Dict[str, Any]]) -> str:
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Syllabus Calendar//EN",
    ]
    for event in events:
        date = event["date"].strftime("%Y%m%d")
        end_date = (event["date"] + timedelta(days=1)).strftime("%Y%m%d")
        description = _escape_ics_text(event["description"].replace("\n", " "))
        uid = f"{date}-{abs(hash(description))}@syllabus-calendar"
        lines.extend(
            [
                "BEGIN:VEVENT",
                f"UID:{uid}",
                f"DTSTAMP:{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}",
                f"DTSTART;VALUE=DATE:{date}",
                f"DTEND;VALUE=DATE:{end_date}",
                f"SUMMARY:{description}",
                "END:VEVENT",
            ]
        )
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines) + "\r\n"


def _escape_ics_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,")


if __name__ == "__main__":
    pass


def _extract_with_chatgpt(text: str, user_year: int | None = None) -> List[Dict[str, Any]]:
    """Call the ChatGPT API to extract JSON events from the syllabus text."""
    try:
        # New SDK style
        from openai import OpenAI  # type: ignore
    except Exception:
        print("openai SDK not installed")
        return []

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("No OPENAI_API_KEY in env")
        return []

    client = OpenAI(api_key=api_key)
    model = os.environ.get("OPENAI_DEADLINE_MODEL", "gpt-4o-mini")

    # Bound token usage
    max_chars = int(os.environ.get("SYLLABUS_MAX_CHARS", "24000"))
    snippet = text if len(text) <= max_chars else text[:max_chars]

    system = (
        "You extract graded deadlines and due dates. Output ONLY valid JSON (no prose). "
        "The user-provided academic year is authoritative and must override the syllabus text. "
        "Never infer or guess calendar dates from month headers, semester ranges, or context. "
        "Only include a due_date if there is an explicit date string present in the text."
    )
    target_year = str(user_year) if user_year is not None else "<MUST_FILL_YEAR>"
    user = (
        "You are extracting graded deadlines from a syllabus.\n\n"
        f"The user has specified the academic year as {target_year}. This year is authoritative and MUST override the syllabus text.\n\n"
        "Rules:\n"
        f"- Output ONLY valid JSON (no prose).\n"
        f"- Normalize all dates to the user-specified year {target_year}, even if the syllabus explicitly lists a different year.\n"
        f"- If a date’s month/day appears but the year in the syllabus conflicts, replace the year with {target_year}.\n"
        f"- Never output dates outside {target_year}.\n"
        "- Never infer or guess missing month/day values.\n"
        "- If an item is recurring (e.g., 'every Wednesday') and no explicit calendar date is given, set due_date to null and include the recurrence text.\n\n"
        "Only include graded items (assignments, exams, quizzes, projects, papers).\n\n"
        "Schema per item:\n"
        '{ "title": string, "due_date": "YYYY-MM-DD" | null, "recurrence": string | null, "description": string }\n\n'
        f"Syllabus text:\n```\n{snippet}\n```"
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
        )
        content = (resp.choices[0].message.content or "").strip()
        print("RAW MODEL LEN:", len(content))
    except Exception as e:
        print("OpenAI call failed:", repr(e))
        return []

    # Strip code fences if present
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?", "", content).strip()
        if content.endswith("```"):
            content = content[:-3].strip()

    # Parse JSON; if it fails, try to salvage the first JSON array
    try:
        data = json.loads(content)
    except Exception:
        m = re.search(r"\[.*\]", content, flags=re.S)
        if not m:
            print("No JSON array found in model output.")
            return []
        try:
            data = json.loads(m.group(0))
        except Exception as e:
            print("JSON parse error:", repr(e))
            return []

    if not isinstance(data, list):
        print("Model output not a list.")
        return []

    return [d for d in data if isinstance(d, dict)]


def split_events(events: List[Dict[str, Any]]):
    """Split events into dated and recurring (undated) lists.

    Returns (dated_events, recurring_items)
    """
    dated: List[Dict[str, Any]] = []
    recurring: List[Dict[str, Any]] = []
    for e in events:
        if e.get("date") is None:
            recurring.append(e)
        else:
            dated.append(e)
    return dated, recurring
