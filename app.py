from __future__ import annotations

from dateutil import parser as dparser
import re
from datetime import datetime, date, timedelta
from typing import List, Dict, Any
import json
import os
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

# Using ChatGPT for extraction; no heuristic diagnostics needed


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    events: List[Dict[str, Any]] = []

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
                due_dates = extract_due_dates(text)
                session["ai_status"] = "active"
                session["events"] = [
                    {"date": event["date"].isoformat(), "description": event["description"]}
                    for event in due_dates
                ]
                events = add_calendar_links(due_dates)
                if not events:
                    flash("No due dates were detected. Try cleaning up the PDF or uploading a text export.", "info")
            except RuntimeError as exc:
                error = str(exc)
            except Exception:  # pragma: no cover - generic failure handler
                error = "We couldn't read that file. Make sure it is a standard, text-based PDF."

    else:
        # Reset UI to original state on refresh/open by clearing transient session state
        for key in ("events", "use_ai", "ai_status"):
            session.pop(key, None)

    grouped = group_events_by_month(events)

    return render_template(
        "index.html",
        grouped_events=grouped,
        error=error,
        use_ai=session.get("use_ai", False),
        ai_status=session.get("ai_status", "off"),
    )


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



def extract_due_dates(text: str) -> List[Dict[str, Any]]:
    """Use ChatGPT to extract due dates from the syllabus text.

    Returns a list of dicts: {"date": datetime, "description": str}
    """
    # Debug: ensure PDF gave us real text
    print("PDF text chars:", len(text))

    raw_items = _extract_with_chatgpt(text)
    events: List[Dict[str, Any]] = []

    for item in raw_items:
        title = str(item.get("title", "")).strip() or "Due"
        due_date = str(item.get("due_date", "")).strip()
        description = str(item.get("description", title)).strip()

        dt = None
        try:
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", due_date):
                dt = datetime.strptime(due_date, "%Y-%m-%d")
            else:
                # Accept natural formats and then normalize
                dt_tmp = dparser.parse(due_date, fuzzy=True, dayfirst=False)
                year = dt_tmp.year if dt_tmp.year and dt_tmp.year > 1900 else datetime.now().year
                dt = datetime(year, dt_tmp.month, dt_tmp.day)
        except Exception:
            continue

        events.append({"date": dt, "description": f"{title}: {description}"})

    # Deduplicate and sort
    seen = set()
    unique = []
    for e in sorted(events, key=lambda e: e["date"]):
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


def _extract_with_chatgpt(text: str) -> List[Dict[str, Any]]:
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

    system = "You extract syllabus deadlines and due dates. Output ONLY valid JSON (no prose)."
    user = (
        "Extract all concrete graded deadlines (quizzes, exams, outlines, term papers, projects, reflections) "
        "from this syllabus. Normalize dates to YYYY-MM-DD. Fields per item:\n"
        '{ "title": "<short name>", "due_date": "YYYY-MM-DD", "description": "<details if available>" }\n\n'
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

