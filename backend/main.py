"""FastAPI app exposing the pipeline as HTTP endpoints and serving the frontend."""
from __future__ import annotations

import logging
import os
import re
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from backend.agents.llm import LLM
from backend.agents.orchestrator import run_pipeline
from backend.config import get_settings
from backend.models import HealthResponse, TailorResponse, CoverLetterResponse
from backend.parsers.pdf_parser import ResumeFormatError
from backend import quota as quota_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("reframe")

# Scrub any string that looks like a user-supplied API key from log records.
# Defense in depth: the route handler never logs the key directly, but a
# library upstream might include it in a RuntimeError or trace.
_KEY_PREFIX_RE = re.compile(
    r"(sk-ant-[A-Za-z0-9_\-]+|sk-[A-Za-z0-9_\-]{8,}|gsk_[A-Za-z0-9_\-]+|"
    r"xai-[A-Za-z0-9_\-]+|AIza[A-Za-z0-9_\-]+)"
)


class _KeyScrubFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        if _KEY_PREFIX_RE.search(msg):
            record.msg = _KEY_PREFIX_RE.sub("[REDACTED_KEY]", msg)
            record.args = ()
        return True


for _name in ("", "reframe", "uvicorn", "uvicorn.access", "uvicorn.error"):
    logging.getLogger(_name).addFilter(_KeyScrubFilter())

settings = get_settings()


def _resolve_cors_origins() -> list[str]:
    """Tighten CORS when a DOMAIN is set but CORS_ORIGINS is still '*'."""
    if settings.domain and settings.cors_origins == ["*"]:
        logger.warning(
            "DOMAIN=%s is set but CORS_ORIGINS is '*'. Defaulting CORS to https://%s only.",
            settings.domain, settings.domain,
        )
        return [f"https://{settings.domain}"]
    return settings.cors_origins


app = FastAPI(title="Reframe", version="0.3.0")

# ── Rate limiting ─────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter


def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Return a clean JSON 429 with Retry-After."""
    response = JSONResponse(
        status_code=429,
        content={"detail": f"Rate limit exceeded: {exc.detail}"},
    )
    # SlowAPI sets Retry-After on the request state; mirror onto the response.
    retry_after = getattr(exc, "retry_after", None) or "60"
    response.headers["Retry-After"] = str(retry_after)
    return response


app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)

# ── Trusted hosts (Host header sanity) ────────────────────────────────
app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)

# ── CORS ──────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=_resolve_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=[
        "*",
        "X-User-Provider",
        "X-User-API-Key",
    ],
    expose_headers=["X-Shared-Quota-Remaining"],
)


# ── Security headers for direct API access (Caddy will override in prod) ─
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    return response


ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = ROOT / "uploads"
OUTPUT_DIR = ROOT / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

FRONTEND_DIR = ROOT / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

MEDIA_BY_EXT = {
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pdf": "application/pdf",
}

# Allowed content types for resume upload.
ALLOWED_UPLOAD_CT = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/octet-stream",  # browsers sometimes send this for unknown types
    "",  # some clients omit content-type entirely
}

# Filename sanitization: keep word chars, dot, hyphen.
_BAD_NAME_CHARS = re.compile(r"[^\w.\-]")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>Reframe</h1><p>Frontend not bundled.</p>")
    return HTMLResponse(
        index_path.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(model=settings.anthropic_model)


def _safe_filename(name: str | None) -> str:
    """Sanitize an uploaded filename. Strip path components and unsafe chars."""
    base = os.path.basename(name or "resume")
    stem, dot, ext = base.partition(".")
    stem = _BAD_NAME_CHARS.sub("_", stem) or "resume"
    ext = _BAD_NAME_CHARS.sub("", ext)
    return f"{stem}.{ext}" if ext else stem


def _save_upload(upload: UploadFile, dest_dir: Path) -> Path:
    # Content-type check.
    ct = (upload.content_type or "").lower()
    if ct not in ALLOWED_UPLOAD_CT:
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {upload.content_type}")

    safe_name = _safe_filename(upload.filename)
    suffix = Path(safe_name).suffix.lower() or ".bin"
    if suffix not in (".docx", ".pdf"):
        raise HTTPException(status_code=400, detail="Only .docx and .pdf resumes are supported.")
    dest = dest_dir / f"{uuid.uuid4().hex}{suffix}"
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    if dest.stat().st_size > settings.max_upload_bytes:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_mb} MB limit.")
    return dest


def _have_any_key() -> bool:
    return bool(
        settings.anthropic_api_key
        or settings.xai_api_key
        or settings.groq_api_key
        or settings.gemini_api_key
        or settings.openai_api_key
    )


def _build_byok_llm(provider: str | None, key: str | None) -> LLM | None:
    """If both headers are present and non-trivial, return a one-off LLM
    bound to the user key. None otherwise. Errors here are user-facing and
    deliberately scrub the key from any details."""
    if not provider or not key:
        return None
    provider = provider.strip().lower()
    key = key.strip()
    if not provider or not key:
        return None
    try:
        return LLM.from_user_key(provider, key)
    except ValueError as e:
        # ValueError messages are crafted to never include the key.
        raise HTTPException(status_code=400, detail=f"BYOK rejected: {e}") from e


@app.get("/api/quota")
async def get_quota(request: Request) -> JSONResponse:
    ip = get_remote_address(request)
    has_user_key = bool(
        request.headers.get("X-User-Provider") and request.headers.get("X-User-API-Key")
    )
    return JSONResponse({
        "remaining": quota_store.remaining(ip, settings.shared_daily_quota),
        "limit": settings.shared_daily_quota,
        "using_user_key": has_user_key,
    })


@app.post("/tailor", response_model=TailorResponse)
@limiter.limit("10/minute;60/hour")
async def tailor(
    request: Request,
    resume: UploadFile = File(..., description="Resume file (.docx or .pdf)"),
    jd: str = Form(..., description="Job description text"),
    cover_letter: bool = Form(False, description="Also generate a cover letter"),
) -> JSONResponse:
    if not jd or not jd.strip():
        raise HTTPException(status_code=400, detail="Job description is required.")

    # Resolve BYOK first. NEVER log the value of X-User-API-Key.
    byok_provider = request.headers.get("X-User-Provider")
    byok_key = request.headers.get("X-User-API-Key")
    user_llm = _build_byok_llm(byok_provider, byok_key)
    using_user_key = user_llm is not None
    client_ip = get_remote_address(request)

    if using_user_key:
        logger.info("Tailor using BYOK provider=%s ip=%s", (byok_provider or "").lower(), client_ip)
    else:
        # Shared keys path: must have at least one server key + quota left.
        if not _have_any_key():
            raise HTTPException(
                status_code=503,
                detail="Server has no shared API key configured. Add your own key in the BYOK panel.",
            )
        if quota_store.is_exhausted(client_ip, settings.shared_daily_quota):
            payload = {
                "error": "shared_quota_exhausted",
                "message": "Daily shared quota reached. Add your own API key to keep going.",
                "remaining": 0,
            }
            resp = JSONResponse(status_code=429, content=payload)
            resp.headers["X-Shared-Quota-Remaining"] = "0"
            return resp

    saved = _save_upload(resume, UPLOAD_DIR)
    suffix = saved.suffix.lower()  # .docx or .pdf

    out_name = f"tailored-{uuid.uuid4().hex}{suffix}"
    out_path = OUTPUT_DIR / out_name

    cover_pdf_path = OUTPUT_DIR / f"cover-{uuid.uuid4().hex}.pdf" if cover_letter else None

    try:
        result = run_pipeline(
            resume_docx_path=saved,
            jd_text=jd,
            output_path=out_path,
            want_cover_letter=cover_letter,
            cover_letter_pdf_path=cover_pdf_path,
            llm_override=user_llm,
        )
    except ResumeFormatError as e:
        logger.warning("Resume format rejected: %s", e)
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Pipeline failed at step")
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {type(e).__name__}: {e}") from e

    # Count shared usage only on success.
    if not using_user_key:
        quota_store.record_usage(client_ip)
    remaining_after = (
        quota_store.remaining(client_ip, settings.shared_daily_quota)
        if not using_user_key
        else settings.shared_daily_quota
    )

    cover_url = (
        f"/download/{result.cover_letter_pdf_path.name}"
        if result.cover_letter_pdf_path
        else None
    )

    body = TailorResponse(
        download_url=f"/download/{result.output_path.name}",
        output_format=result.output_format,
        match_score=result.match_report.overall_score,
        jd_analysis=result.jd,
        match_report=result.match_report,
        ats_score=result.ats_score,
        cover_letter=result.cover_letter_text,
        cover_letter_url=cover_url,
        warnings=result.warnings,
    )
    resp = JSONResponse(content=body.model_dump())
    resp.headers["X-Shared-Quota-Remaining"] = str(remaining_after)
    return resp


@app.post("/cover-letter", response_model=CoverLetterResponse)
@limiter.limit("20/minute")
async def cover_letter_only(
    request: Request,
    resume: UploadFile = File(...),
    jd: str = Form(...),
) -> CoverLetterResponse:
    """Skip resume rewriting and just generate a cover letter (returns text +
    a downloadable PDF URL)."""
    if not _have_any_key():
        raise HTTPException(status_code=503, detail="Server has no API key set for the chosen LLM_PROVIDER. Add it to .env.")
    saved = _save_upload(resume, UPLOAD_DIR)

    # We only need the candidate's structured data for the cover letter.
    from backend.agents import jd_analyzer, resume_analyzer, cover_letter as cl_agent

    if saved.suffix.lower() == ".pdf":
        from backend.parsers.pdf_parser import parse_pdf, parsed_pdf_to_analysis
        try:
            parsed = parse_pdf(saved)
        except ResumeFormatError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        heuristic = parsed_pdf_to_analysis(parsed)
        full_text = parsed.full_text
        try:
            parsed.document.close()
        except Exception:
            pass
    else:
        from backend.parsers.docx_parser import parse_docx, parsed_to_analysis
        parsed = parse_docx(saved)
        heuristic = parsed_to_analysis(parsed)
        full_text = parsed.full_text

    resume_an = resume_analyzer.analyze(full_text, heuristic=heuristic)
    jd_an = jd_analyzer.analyze(jd)
    text = cl_agent.generate(jd_an, resume_an)

    cover_pdf_path = OUTPUT_DIR / f"cover-{uuid.uuid4().hex}.pdf"
    try:
        from backend.writers.cover_letter_pdf import render_cover_letter_pdf
        render_cover_letter_pdf(
            text,
            cover_pdf_path,
            candidate_name=resume_an.candidate_name or None,
            contact=resume_an.contact or None,
        )
        download_url = f"/download/{cover_pdf_path.name}"
    except Exception:
        logger.exception("Cover letter PDF render failed")
        download_url = None

    return CoverLetterResponse(text=text, download_url=download_url)


@app.get("/download/{name}")
@limiter.limit("60/minute")
async def download(request: Request, name: str) -> FileResponse:
    if "/" in name or ".." in name or "\\" in name:
        raise HTTPException(status_code=400, detail="Bad filename")
    path = OUTPUT_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Not found")
    ext = path.suffix.lower()
    media = MEDIA_BY_EXT.get(ext, "application/octet-stream")
    friendly = "tailored-resume" + ext if name.startswith("tailored-") else (
        "cover-letter" + ext if name.startswith("cover-") else name
    )
    return FileResponse(str(path), media_type=media, filename=friendly)
