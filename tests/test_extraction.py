from datetime import datetime

import app as app_module
from app import (
    build_candidate_text,
    build_ics,
    extract_due_dates,
    split_events,
)


def _fake_llm(items):
    def fake(text, user_year=None, cache_params=None):
        return items
    return fake


def test_extract_filters_window_and_dedupes(monkeypatch):
    monkeypatch.setattr(app_module, "_extract_with_chatgpt", _fake_llm([
        {"title": "HW 1", "due_date": "2025-09-10", "recurrence": None, "description": "Homework 1"},
        {"title": "HW 1", "due_date": "2025-09-10", "recurrence": None, "description": "Homework 1"},
        {"title": "Old date", "due_date": "2025-03-01", "recurrence": None, "description": "outside window"},
        {"title": "Quiz", "due_date": None, "recurrence": "every Friday", "description": "Weekly quiz"},
        {"title": "Bad", "due_date": "not-a-date", "recurrence": None, "description": "junk"},
        {"title": "No info", "due_date": None, "recurrence": None, "description": "ignored"},
    ]))
    events = extract_due_dates("syllabus text", user_year=2025, start_month=8, end_month=12)
    dated, recurring = split_events(events)
    assert [e["date"] for e in dated] == [datetime(2025, 9, 10)]
    assert len(recurring) == 1
    assert "every Friday" in recurring[0]["description"]


def test_wrapped_window_resolves_january_to_next_year(monkeypatch):
    monkeypatch.setattr(app_module, "_extract_with_chatgpt", _fake_llm([
        {"title": "Final", "due_date": "2025-01-06", "recurrence": None, "description": "Final exam"},
        {"title": "Midterm", "due_date": "2025-10-15", "recurrence": None, "description": "Midterm"},
    ]))
    events = extract_due_dates("syllabus text", user_year=2025, start_month=8, end_month=1)
    dates = [e["date"] for e in events]
    assert datetime(2026, 1, 6) in dates
    assert datetime(2025, 10, 15) in dates


def test_year_enforced_without_month_window(monkeypatch):
    monkeypatch.setattr(app_module, "_extract_with_chatgpt", _fake_llm([
        {"title": "HW", "due_date": "2019-09-10", "recurrence": None, "description": "wrong year in text"},
    ]))
    events = extract_due_dates("syllabus text", user_year=2025)
    assert events[0]["date"] == datetime(2025, 9, 10)


def test_candidate_text_falls_back_to_full_text_when_small():
    text = "HW 1 due Sep 10\n" + "xxxx yyyy zzzz\n" * 5
    assert build_candidate_text(text) == text


def test_candidate_text_selects_deadline_lines_from_large_text():
    filler_before = [f"xxxx yyyy zzzz {i}" for i in range(100)]
    deadline_lines = [
        f"Assignment {i} due September 10 on Gradescope submission portal" for i in range(40)
    ]
    filler_after = [f"xxxx yyyy zzzz {i}" for i in range(200, 300)]
    text = "\n".join(filler_before + deadline_lines + filler_after)

    candidate = build_candidate_text(text)
    assert candidate != text
    assert "Assignment 3 due" in candidate
    assert "xxxx yyyy zzzz 50" not in candidate


def test_build_ics_escapes_and_formats():
    ics = build_ics([{"date": datetime(2025, 9, 10), "description": "Essay; draft, final"}])
    assert "SUMMARY:Essay\\; draft\\, final" in ics
    assert "DTSTART;VALUE=DATE:20250910" in ics
    assert "DTEND;VALUE=DATE:20250911" in ics
    assert ics.startswith("BEGIN:VCALENDAR\r\n")
    assert ics.endswith("\r\n")
