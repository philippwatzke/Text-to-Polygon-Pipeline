from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ki_geodaten.config import Settings
from ki_geodaten.jobs.store import init_schema

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).parent

TokenCounter = Callable[[str], int]
ExecutorFactory = Callable[[], Executor]
GEOJSON_CACHE_MAX_ENTRIES = 64


class LRUCache(OrderedDict):
    """Bounded LRU cache so a long-running server doesn't grow unbounded."""

    def __init__(self, maxsize: int) -> None:
        super().__init__()
        self._maxsize = maxsize

    def __setitem__(self, key, value) -> None:
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        while len(self) > self._maxsize:
            self.popitem(last=False)

    def get(self, key, default=None):
        if key in self:
            self.move_to_end(key)
            return self[key]
        return default


def _default_executor_factory() -> Executor:
    try:
        return ProcessPoolExecutor(max_workers=2)
    except PermissionError as exc:
        if os.environ.get("KI_GEODATEN_ALLOW_THREAD_EXECUTOR_FALLBACK") != "1":
            raise
        logger.warning(
            "ProcessPoolExecutor unavailable (%s); using dedicated ThreadPoolExecutor",
            exc,
        )
        return ThreadPoolExecutor(max_workers=2)


def _data_paths(root: Path) -> tuple[Path, Path, Path]:
    data_root = root / "data"
    dop_dir = data_root / "dop"
    results_dir = data_root / "results"
    dop_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return data_root, data_root / "jobs.db", results_dir


def _default_token_counter(settings: Settings) -> TokenCounter:
    try:
        from ki_geodaten.pipeline.segmenter import Sam3TextTokenCounter

        counter = Sam3TextTokenCounter(
            settings.SAM3_MODEL_ID,
            local_files_only=settings.SAM3_LOCAL_FILES_ONLY,
        )
        counter.load()
        return counter
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Falling back to whitespace token counter; real SAM encoder unavailable: %s",
            exc,
        )
        return lambda text: len(text.split())


def lifespan_factory(executor_factory: ExecutorFactory, token_counter: TokenCounter | None):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings: Settings = app.state.settings
        data_root, db_path, results_dir = _data_paths(Path.cwd())
        init_schema(db_path)
        app.state.data_root = data_root
        app.state.db_path = db_path
        app.state.results_dir = results_dir
        app.state.export_lock = threading.Lock()
        app.state.geojson_cache = LRUCache(GEOJSON_CACHE_MAX_ENTRIES)
        app.state.geojson_executor = executor_factory()
        app.state.token_counter = token_counter or _default_token_counter(settings)
        try:
            yield
        finally:
            app.state.geojson_executor.shutdown(wait=False, cancel_futures=True)

    return lifespan


def create_app(
    settings: Settings | None = None,
    *,
    executor_factory: ExecutorFactory = _default_executor_factory,
    token_counter: TokenCounter | None = None,
) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(
        title="Text-to-Polygon Pipeline",
        lifespan=lifespan_factory(executor_factory, token_counter),
    )
    app.state.settings = settings

    templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
    app.state.templates = templates
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return templates.TemplateResponse(request, "index.html")

    from ki_geodaten.app.routes.geojson import router as geojson_router
    from ki_geodaten.app.routes.jobs import router as jobs_router
    from ki_geodaten.app.routes.system import router as system_router

    app.include_router(jobs_router)
    app.include_router(geojson_router)
    app.include_router(system_router)
    return app


app = create_app()
