#!/usr/bin/env python3
"""Benchmark candidate-text reduction and cold vs cached extraction latency.

Usage:
    python scripts/benchmark.py samples/sample_syllabus.txt [more files ...]
    python scripts/benchmark.py --year 2025 --start-month 8 --end-month 12 my_syllabus.pdf

Accepts the same formats as the app (.pdf, .docx, .txt). Latency measurement
calls the OpenAI API and requires OPENAI_API_KEY (from .env or the
environment); without a key, only text-reduction stats are reported.

The cold run bypasses the cache read (but still writes it), so the warm run
that follows measures a true cache hit.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module  # noqa: E402


class _FileUpload:
    """Minimal stand-in for a Flask/Werkzeug uploaded file."""

    def __init__(self, path: str):
        self.filename = os.path.basename(path)
        self.stream = open(path, "rb")

    def read(self) -> bytes:
        self.stream.seek(0)
        return self.stream.read()

    def close(self):
        self.stream.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="Syllabus files (.pdf, .docx, .txt)")
    parser.add_argument("--year", type=int, default=None, help="Academic year (authoritative)")
    parser.add_argument("--start-month", type=int, default=None, help="Term start month (1-12)")
    parser.add_argument("--end-month", type=int, default=None, help="Term end month (1-12)")
    args = parser.parse_args()

    have_key = bool(os.environ.get("OPENAI_API_KEY"))
    if not have_key:
        print("Note: OPENAI_API_KEY not set — reporting text-reduction stats only.\n")

    rows = []
    for path in args.files:
        upload = _FileUpload(path)
        try:
            text = app_module.extract_text(upload)
        finally:
            upload.close()

        candidate = app_module.build_candidate_text(text)
        snippet = text if len(candidate) < 1500 else candidate
        reduction = 100.0 * (1 - len(snippet) / len(text)) if text else 0.0
        row = {
            "file": os.path.basename(path),
            "full_chars": len(text),
            "sent_chars": len(snippet),
            "reduction_pct": reduction,
        }

        if have_key:
            t0 = time.perf_counter()
            events = app_module.extract_due_dates(
                text, args.year, args.start_month, args.end_month, refresh_cache=True
            )
            row["cold_s"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            app_module.extract_due_dates(text, args.year, args.start_month, args.end_month)
            row["warm_s"] = time.perf_counter() - t0
            row["events"] = len(events)

        rows.append(row)

    header = f"{'file':<32} {'full':>8} {'sent':>8} {'reduc':>7}"
    if have_key:
        header += f" {'cold':>8} {'warm':>9} {'events':>7}"
    print(header)
    print("-" * len(header))
    for r in rows:
        line = f"{r['file']:<32} {r['full_chars']:>8} {r['sent_chars']:>8} {r['reduction_pct']:>6.1f}%"
        if have_key:
            line += f" {r['cold_s']:>7.2f}s {r['warm_s']*1000:>7.1f}ms {r['events']:>7}"
        print(line)

    if len(rows) > 1:
        avg_red = sum(r["reduction_pct"] for r in rows) / len(rows)
        print(f"\nAverage input reduction: {avg_red:.1f}%")
        if have_key:
            avg_cold = sum(r["cold_s"] for r in rows) / len(rows)
            avg_warm = sum(r["warm_s"] for r in rows) / len(rows)
            print(f"Average cold extraction: {avg_cold:.2f}s")
            print(f"Average cached extraction: {avg_warm*1000:.1f}ms ({avg_cold/avg_warm:.0f}x faster)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
