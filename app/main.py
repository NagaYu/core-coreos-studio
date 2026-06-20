"""
Core-CoreOS Studio · main.py
============================================================================
A deliberately tiny FastAPI server whose ONLY job is to serve the static
single-page studio to your local browser.

Why so small? Because Core-CoreOS Studio is a *zero-backend* application.
All inference — image recognition and speech-to-text — happens inside the
browser on Apple Silicon via WebGPU / WASM. The server never sees, stores,
or forwards a single byte of your images or audio. It is purely a static
file host you run on `localhost`.

Run it:
    uvicorn app.main:app --reload          # from the project root
    # or simply:
    python -m app.main
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Core-CoreOS Studio",
    description="100% on-device multimodal AI studio for Apple Silicon.",
    version="1.0.0",
    docs_url=None,       # no public API surface — this is a static host
    redoc_url=None,
)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.middleware("http")
async def security_headers(request: Request, call_next):
    """Add hardening headers.

    Note on cross-origin isolation: enabling COOP/COEP would unlock
    multi-threaded WASM (SharedArrayBuffer), but `require-corp` also blocks
    the model-weight fetch from the public CDN unless every asset ships CORP
    headers. WebGPU — our primary, fastest path on Apple Silicon — needs none
    of that, so we keep COEP OFF by default for a frictionless first run.
    Set ``COOP/COEP`` yourself if you mirror the weights locally and want the
    threaded WASM fallback to go full speed.
    """
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    # Lock down powerful features except the ones the studio actually uses
    # on-device (camera/mic are user-initiated file pickers, not live capture).
    response.headers["Permissions-Policy"] = "geolocation=(), interest-cohort=()"
    return response


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Serve the studio shell."""
    # Starlette >=0.29 signature: (request, name, context).
    return templates.TemplateResponse(request, "index.html")


@app.get("/healthz", response_class=JSONResponse)
async def healthz() -> JSONResponse:
    """Liveness probe — handy when running under a process manager."""
    return JSONResponse({"status": "ok", "service": "core-coreos-studio"})


def main() -> None:
    """Entry point for ``python -m app.main``."""
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
