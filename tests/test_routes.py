import app as app_module


def _client():
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_index_renders():
    resp = _client().get("/")
    assert resp.status_code == 200
    assert b"MySyllabus" in resp.data


def test_index_sets_visitor_cookie():
    resp = _client().get("/")
    cookies = resp.headers.getlist("Set-Cookie")
    assert any(c.startswith("visitor_id=") for c in cookies)


def test_upload_without_file_shows_error():
    resp = _client().post("/", data={})
    assert resp.status_code == 200
    assert b"Please choose a syllabus file to upload." in resp.data


def test_history_renders_empty_for_new_visitor():
    resp = _client().get("/history")
    assert resp.status_code == 200
    assert b"No saved entries yet." in resp.data


def test_download_ics_without_events_redirects():
    resp = _client().post("/download-ics")
    assert resp.status_code == 302


def test_save_entry_without_events_redirects():
    resp = _client().post("/save-entry")
    assert resp.status_code == 302
