from __future__ import annotations

import re
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Iterable
import os

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

# Track last AI classification diagnostics for the request/session
_LAST_AI_DIAG: Dict[str, int] = {"attempts": 0, "used": 0, "failures": 0}


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
                use_ai = bool(session.get("use_ai", False))
                base_year = _infer_year(text, default_year=datetime.today().year)
                if semester and start_month and end_month:
                    sem_window = (start_month, end_month, (year or base_year))
                else:
                    sem_window = semester_window_from_choice(semester, year or base_year)

                due_dates = extract_due_dates(
                    text,
                    use_ai=use_ai,
                    semester_window=sem_window,
                    base_year_override=base_year,
                )
                # Save AI status for display
                ai_diag = dict(_LAST_AI_DIAG)
                if use_ai:
                    if ai_diag.get("used", 0) > 0:
                        session["ai_status"] = "active"
                    else:
                        session["ai_status"] = "heuristic"
                else:
                    session["ai_status"] = "off"
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



def extract_due_dates(text: str, use_ai: bool | None = None,
                      semester_window=None, base_year_override: int | None = None) -> List[Dict[str, Any]]:
    """Parse the syllabus text and return detected due dates with descriptions."""

    lines = [" ".join(line.split()) for line in text.splitlines()]
    today = datetime.today().date()
    base_year = base_year_override or _infer_year(text, default_year=today.year)
    events: List[Dict[str, Any]] = []

    # reset diagnostics
    global _LAST_AI_DIAG
    _LAST_AI_DIAG = {"attempts": 0, "used": 0, "failures": 0}

    for idx, line in enumerate(lines):
        if not line:
            continue

        matches = list(_find_date_matches(line))
        if not matches:
            continue

        context = line
        # If the current line is short, append the following line for more detail,
        # but avoid pulling in citation/metadata-looking lines.
        if len(context) < 25 and idx + 1 < len(lines):
            nxt = lines[idx + 1]
            if nxt and not _has_negative_citation_cues(nxt):
                # Avoid merging if the next line itself contains a date token (likely a separate event)
                has_date_in_next = any(True for _ in _find_date_matches(nxt))
                if not has_date_in_next:
                    context = f"{context} {nxt}".strip()
        
        for token in matches:
            parsed = _parse_date_token(token, today, base_year)
            if not parsed:
                continue

            # 1) reject obviously historical dates
            if _looks_like_historical_year(parsed):
                continue

            has_positive = _has_positive_deadline_cues(context)

            # If a semester window is provided, enforce it unless there is a strong positive cue
            if semester_window and not in_semester_window(parsed, semester_window) and not has_positive:
                continue

            # 2) if date is outside the term window (base_year ± 1) and no positive cue, skip
            if not _in_plausible_academic_window(parsed, base_year) and not has_positive:
                continue

            # 3) if the line looks like a citation or has a URL and no positive cue, skip
            if (_has_negative_citation_cues(context) or _has_url(context)) and not has_positive:
                continue

            if not _looks_like_deadline(context, use_ai=use_ai, ai_diag=_LAST_AI_DIAG):
                continue

            cleaned_description = _strip_token_from_context(token, context)
            # Very short/empty descriptions without any deadline cues are likely not real
            if len(cleaned_description.split()) <= 2 and not _has_positive_deadline_cues(context):
                continue

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


def semester_window_from_choice(semester: str | None, year: int | None):
    if not semester or not year:
        return None
    s = semester.lower()
    if s == "fall":
        return (8, 12, year)
    if s == "spring":
        return (1, 5, year)
    if s == "summer":
        return (5, 8, year)  # adjust if your school differs
    return None


def in_semester_window(dt: datetime, window) -> bool:
    if not window:
        return True
    m1, m2, y = window
    return (dt.year == y) and (m1 <= dt.month <= m2)


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

_TIME_HINTS = [
    r"\b11:?59\b",
    r"\b(1[0-2]|0?[1-9]):[0-5][0-9]\s*(am|pm)\b",
    r"\bmidnight\b",
    r"\bnoon\b",
]
_BY_TIME_RE = re.compile(
    r"\bby\s+(?:11:?59|midnight|noon|(?:1[0-2]|0?[1-9]):[0-5][0-9]\s*(?:am|pm))\b", 
    re.I
)

_BY_DATE_RE = re.compile(
    rf"\bby\s+(?:{_MONTHS_PATTERN}\s+\d{{1,2}}(?:,\s*\d{{4}})?|\d{{1,2}}[/-]\d{{1,2}}(?:[/-]\d{{2,4}})?)\b", 
    re.I
)

_DUE_BY_RE = re.compile(r"\bdue\s+by\b", re.I)


_MEETING_ONLY_HINTS = [
    "lecture",
    "class",
    "topic",
    "week ",
    "session",
]


def _has_positive_deadline_cues(text: str) -> bool:
    t = text.lower()
    if any(k in t for k in _POSITIVE_KEYWORDS):
        return True
    # “due by …” or “by <time/date>”
    if _DUE_BY_RE.search(t) or _BY_TIME_RE.search(t) or _BY_DATE_RE.search(t):
        return True
    # explicit time hints also count
    for pat in _TIME_HINTS:
        if re.search(pat, t, flags=re.IGNORECASE):
            return True
    return False


def _has_negative_citation_cues(text: str) -> bool:
    t = text.lower()
    if any(k in t for k in _NEGATIVE_KEYWORDS):
        return True
    # Common citation patterns: "Lastname (2015)", "(2015)" near author-like tokens
    if re.search(r"\b[A-Z][A-Za-z\-]+\s*\(\d{4}\)", text):
        return True
    if re.search(r"\(\d{4}\)\s*[.;]?$", text):
        return True
    # Year alongside journal-like markers
    if re.search(r"\b\d{4}\b.*\b(pp\.|doi|vol\.|no\.)", t):
        return True
    return False


def _meeting_only_context(text: str) -> bool:
    t = text.lower()
    return any(h in t for h in _MEETING_ONLY_HINTS) and not _has_positive_deadline_cues(t)


def _looks_like_deadline(context: str, use_ai: bool | None = None, ai_diag: Dict[str, int] | None = None) -> bool:
    """Decide if a line/context with a date likely represents a deadline.

    Strategy:
    - If strong negative citation cues and no positive cues -> reject.
    - If meeting-only phrasing without positive cues -> reject.
    - Otherwise require some positive cues OR sufficiently descriptive text.
    """
    # Optional AI hook: enable via use_ai flag (preferred) or env var USE_AI_CLASSIFIER
    if use_ai is None:
        use_ai = bool(os.environ.get("USE_AI_CLASSIFIER"))
    if use_ai:
        try:
            if ai_diag is not None:
                ai_diag["attempts"] = ai_diag.get("attempts", 0) + 1
            ans = _ai_says_deadline(context)
            if isinstance(ans, bool):
                if ai_diag is not None:
                    ai_diag["used"] = ai_diag.get("used", 0) + 1
                if ans is False:
                    return False
            # If ans is None, fall back to heuristics
        except Exception:
            if ai_diag is not None:
                ai_diag["failures"] = ai_diag.get("failures", 0) + 1
            # Fall back to heuristics if AI is unavailable/fails
            pass

    if _has_negative_citation_cues(context) and not _has_positive_deadline_cues(context):
        return False
    if _meeting_only_context(context):
        return False
    # Reject citation-tail dates like ", Month Day, Year" when there are no positive cues
    if re.search(rf",\s*{_MONTHS_PATTERN}\s+\d{{1,2}},\s*\d{{4}}\b", context, flags=re.IGNORECASE) \
            and not _has_positive_deadline_cues(context):
        return False

    # Prefer lines that explicitly look like deliverables/exams or have time hints
    if _has_positive_deadline_cues(context):
        return True

    # Otherwise, be conservative to avoid false positives from citations.
    # Keep only if the non-date text is reasonably descriptive (>= 4 words)
    words = [w for w in re.split(r"\s+", context.strip()) if w]
    return len(words) >= 4


def _ai_says_deadline(context: str) -> bool | None:
    """Optional AI classifier. Returns True/False, or None on failure.

    Requires: pip install openai, OPENAI_API_KEY, and USE_AI_CLASSIFIER=1.
    """
    try:
        import openai  # type: ignore
    except Exception:
        return None

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    openai.api_key = "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"
    try:
        prompt = (
            "You classify a syllabus line as deadline or not. "
            "Return only 'yes' or 'no'.\n\n"
            f"Line: {context!r}\n"
        )
        # Minimal tokens; compatible with legacy clients. Adjust as needed.
        resp = openai.Completion.create(
            model=os.environ.get("OPENAI_DEADLINE_MODEL", "gpt-3.5-turbo-instruct"),
            prompt=prompt,
            max_tokens=1,
            temperature=0,
        )
        text = resp.choices[0].text.strip().lower()
        if text.startswith("y"):
            return True
        if text.startswith("n"):
            return False
    except Exception:
        return None
    return None


def _infer_year(text: str, default_year: int) -> int:
    """Infer the intended syllabus year from the document text.

    Strategy:
    - Look for explicit 4-digit years (2000–2100) and pick the most frequent.
    - On ties or no matches, fall back to the provided default_year.
    """
    # Prefer years found on non-citation lines to avoid bias from references
    lines = text.splitlines()
    candidate_years: list[int] = []
    for ln in lines:
        if _has_negative_citation_cues(ln):
            continue
        candidate_years.extend(int(y) for y in re.findall(r"\b(20\d{2}|19\d{2})\b", ln))

    years = candidate_years or [int(y) for y in re.findall(r"\b(20\d{2}|19\d{2})\b", text)]
    if not years:
        return default_year

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
    pass
