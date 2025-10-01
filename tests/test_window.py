from datetime import datetime

from app import window_bounds, resolve_date_str, in_window


def test_non_wrap_jan_may():
    start, end = window_bounds(1, 5, 2025)
    assert start == datetime(2025, 1, 1)
    assert end >= datetime(2025, 5, 31)
    d = resolve_date_str("March 5", 1, 5, 2025)
    assert d == datetime(2025, 3, 5)
    assert in_window(d, start, end)
    d2 = resolve_date_str("Nov 6", 1, 5, 2025)
    assert d2 is None or not in_window(d2, start, end)


def test_wrap_aug_jan():
    start, end = window_bounds(8, 1, 2025)
    assert start == datetime(2025, 8, 1)
    # end should be sometime in Jan 2026 end
    assert end.year in (2025, 2026)
    d_fall = resolve_date_str("Oct 15", 8, 1, 2025)
    assert d_fall == datetime(2025, 10, 15)
    assert in_window(d_fall, start, end)
    d_jan = resolve_date_str("Jan 6", 8, 1, 2025)
    assert d_jan == datetime(2026, 1, 6)
    assert in_window(d_jan, start, end)


def test_invalid_day_feb_30():
    d = resolve_date_str("Feb 30", 1, 5, 2025)
    assert d is None

