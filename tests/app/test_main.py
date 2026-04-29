from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from ki_geodaten.app.main import create_app


def test_app_starts_and_serves_index(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = create_app(
        executor_factory=lambda: ThreadPoolExecutor(max_workers=1),
        token_counter=lambda value: len(value.split()),
    )
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "Text-to-Polygon" in response.text
    assert "score-filter" in response.text
    assert "segment-opacity" in response.text
    assert "reject-below-score" in response.text
    assert "compare-panel" in response.text
    assert "job-search" in response.text
    assert "show-failed-jobs" in response.text


def test_static_app_includes_basemap_switcher():
    app_js = (
        __import__("pathlib")
        .Path("ki_geodaten/app/static/app.js")
        .read_text(encoding="utf-8")
    )
    assert "L.control.layers" in app_js
    assert "by_dop20c" in app_js


def test_static_app_includes_review_filters():
    app_js = (
        __import__("pathlib")
        .Path("ki_geodaten/app/static/app.js")
        .read_text(encoding="utf-8")
    )
    assert "passesReviewFilter" in app_js
    assert "rejectAcceptedBelowScore" in app_js
    assert "reviewStatsEl" in app_js


def test_static_app_includes_job_comparison_tools():
    app_js = (
        __import__("pathlib")
        .Path("ki_geodaten/app/static/app.js")
        .read_text(encoding="utf-8")
    )
    assert "renderComparison" in app_js
    assert "missed_marked" in app_js
    assert "compare-table" in app_js
    assert "/summary" in app_js


def test_static_app_includes_job_list_controls():
    app_js = (
        __import__("pathlib")
        .Path("ki_geodaten/app/static/app.js")
        .read_text(encoding="utf-8")
    )
    assert "renameJob" in app_js
    assert "jobMatchesFilters" in app_js
    assert "/label" in app_js
    assert "segmentOpacityEl" in app_js


def test_app_lifespan_initializes_db(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = create_app(
        executor_factory=lambda: ThreadPoolExecutor(max_workers=1),
        token_counter=lambda value: len(value.split()),
    )
    with TestClient(app):
        pass
    assert (tmp_path / "data" / "jobs.db").exists()


def test_executor_factory_override_used(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sentinel = ThreadPoolExecutor(max_workers=1)
    app = create_app(
        executor_factory=lambda: sentinel,
        token_counter=lambda value: len(value.split()),
    )
    with TestClient(app):
        assert app.state.geojson_executor is sentinel


def test_token_counter_override_used(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = create_app(
        executor_factory=lambda: ThreadPoolExecutor(max_workers=1),
        token_counter=lambda value: 42,
    )
    with TestClient(app):
        assert app.state.token_counter("anything") == 42
