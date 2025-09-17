from __future__ import annotations

import re
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Iterable

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


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    events: List[Dict[str, Any]] = []

    if request.method == "POST":
        uploaded = request.files.get("syllabus")
        if not uploaded or uploaded.filename == "":
            error = "Please choose a syllabus file to upload."
        else:
            try:
                text = extract_text(uploaded)
                due_dates = extract_due_dates(text)
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
        stored = session.get("events", [])
        if stored:
            events = add_calendar_links(
                [
                    {"date": datetime.fromisoformat(item["date"]), "description": item["description"]}
                    for item in stored
                ]
            )

    grouped = group_events_by_month(events)

    return render_template("index.html", grouped_events=grouped, error=error)


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


def extract_due_dates(text: str) -> List[Dict[str, Any]]:
    """Parse the syllabus text and return detected due dates with descriptions."""

    lines = [" ".join(line.split()) for line in text.splitlines()]
    today = datetime.today().date()
    base_year = _infer_year(text, default_year=today.year)
    events: List[Dict[str, Any]] = []

    for idx, line in enumerate(lines):
        if not line:
            continue

        matches = list(_find_date_matches(line))
        if not matches:
            continue

        context = line
        # If the current line is short, append the following line for more detail.
        if len(context) < 25 and idx + 1 < len(lines):
            context = f"{context} {lines[idx + 1]}".strip()

        for token in matches:
            parsed = _parse_date_token(token, today, base_year)
            if not parsed:
                continue

            cleaned_description = _strip_token_from_context(token, context)
            events.append({"date": parsed, "description": cleaned_description})

    # Deduplicate by date and description pair while preserving order
    seen = set()
    unique_events = []
    for event in sorted(events, key=lambda e: e["date"]):
        signature = (event["date"].date().isoformat(), event["description"].lower())
        if signature in seen:
            continue
        seen.add(signature)
        unique_events.append(event)

    return unique_events


_MONTHS_PATTERN = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_DATE_PATTERNS = (
    rf"\b{_MONTHS_PATTERN}\s+\d{{1,2}}(?:,\s*\d{{4}})?\b",
    r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b",
    r"\b\d{4}-\d{1,2}-\d{1,2}\b",
)
_DATE_REGEXES = [re.compile(pattern, re.IGNORECASE) for pattern in _DATE_PATTERNS]


def _find_date_matches(line: str) -> Iterable[str]:
    for regex in _DATE_REGEXES:
        for match in regex.findall(line):
            if isinstance(match, tuple):
                yield match[0]
            else:
                yield match


def _parse_date_token(token: str, today: date, base_year: int) -> datetime | None:
    cleaned = token.strip()
    cleaned = re.sub(r"(\d)(st|nd|rd|th)", r"\1", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)

    has_alpha = any(ch.isalpha() for ch in cleaned)
    if has_alpha:
        cleaned = cleaned.lower().title()

    date_formats = (
        "%B %d, %Y",
        "%b %d, %Y",
        "%B %d %Y",
        "%b %d %Y",
        "%B %d",
        "%b %d",
        "%m/%d/%Y",
        "%m/%d/%y",
        "%m/%d",
        "%m-%d-%Y",
        "%m-%d-%y",
        "%m-%d",
        "%Y-%m-%d",
    )

    for fmt in date_formats:
        try:
            parsed = datetime.strptime(cleaned, fmt)
            if "%Y" not in fmt:
                # No year present in the token; use the inferred syllabus year.
                parsed = parsed.replace(year=base_year)
            elif parsed.year < 100:
                parsed = parsed.replace(year=2000 + parsed.year)
            return parsed
        except ValueError:
            continue

    return None


def _strip_token_from_context(token: str, context: str) -> str:
    pattern = re.compile(re.escape(token), re.IGNORECASE)
    cleaned = pattern.sub("", context).strip(" -:\u2013\u2014")
    if not cleaned:
        cleaned = "Due"
    return cleaned


def _infer_year(text: str, default_year: int) -> int:
    """Infer the intended syllabus year from the document text.

    Strategy:
    - Look for explicit 4-digit years (2000–2100) and pick the most frequent.
    - On ties or no matches, fall back to the provided default_year.
    """
    years = [int(y) for y in re.findall(r"\b(20\d{2}|19\d{2})\b", text)]
    if not years:
        return default_year

    # Tally occurrences
    counts: dict[int, int] = {}
    for y in years:
        if 2000 <= y <= 2100:
            counts[y] = counts.get(y, 0) + 1

    if not counts:
        return default_year

    # Pick the most common; on tie choose closest to default_year, then smallest.
    max_count = max(counts.values())
    candidates = [y for y, c in counts.items() if c == max_count]
    candidates.sort(key=lambda y: (abs(y - default_year), y))
    return candidates[0]


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
    app.run(debug=True)
