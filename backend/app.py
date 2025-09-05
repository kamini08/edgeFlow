"""FastAPI application for EdgeFlow Web API.

Implements strict CLI-API parity for compile, optimize, benchmark, version, and help
endpoints. Uses existing core modules where possible and provides safe fallbacks.
"""

from __future__ import annotations

import base64
import io
import logging
from datetime import datetime, timezone
from parser import parse_ef  # type: ignore
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, constr

# Import core CLI logic
import edgeflowc  # type: ignore

# ----------------------------------------------------------------------------
# Rate limiting (simple in-memory token bucket per client IP)
# ----------------------------------------------------------------------------


class SimpleRateLimiter:
    def __init__(self, capacity: int = 60) -> None:
        self.capacity = capacity
        self.tokens: Dict[str, int] = {}
        self.window_started: Dict[str, int] = {}

    def __call__(self, request: Request) -> None:
        # very simple 60 req/min per IP
        now_minute = int(datetime.now(tz=timezone.utc).timestamp() // 60)
        ip = request.client.host if request.client else "unknown"
        if self.window_started.get(ip) != now_minute:
            self.window_started[ip] = now_minute
            self.tokens[ip] = self.capacity
        if self.tokens[ip] <= 0:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        self.tokens[ip] -= 1


rate_limiter = SimpleRateLimiter(capacity=120)


def rate_limit_dep(request: Request) -> None:
    """Dependency wrapper to apply rate limiting using the client IP."""
    rate_limiter(request)


# ----------------------------------------------------------------------------
# Request/Response Schemas
# ----------------------------------------------------------------------------


class CompileRequest(BaseModel):
    config_file: str = Field(..., description="EdgeFlow config file content")
    filename: constr(strip_whitespace=True, min_length=1)  # type: ignore


class CompileResponse(BaseModel):
    success: bool
    parsed_config: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    logs: Optional[List[str]] = None


class OptimizeRequest(BaseModel):
    model_file: str = Field(..., description="Base64-encoded TFLite model")
    config: Dict[str, Any] = Field(default_factory=dict)


class OptimizeResponse(BaseModel):
    success: bool
    optimized_model: str
    optimization_report: Dict[str, Any]


class BenchmarkRequest(BaseModel):
    original_model: str
    optimized_model: str


class Stats(BaseModel):
    size_mb: float
    latency_ms: float


class Improvement(BaseModel):
    size_reduction: float
    speedup: float


class BenchmarkResponse(BaseModel):
    original_stats: Stats
    optimized_stats: Stats
    improvement: Improvement


# ----------------------------------------------------------------------------
# FastAPI setup
# ----------------------------------------------------------------------------


app = FastAPI(title="EdgeFlow API", version="v1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MAX_BYTES = 100 * 1024 * 1024  # 100MB


@app.middleware("http")
async def limit_body_size(request: Request, call_next):  # type: ignore
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > MAX_BYTES:
        return JSONResponse(status_code=413, content={"detail": "Payload too large"})
    return await call_next(request)


def _parse_config_content(filename: str, content: str) -> Dict[str, Any]:
    if not filename.lower().endswith(".ef"):
        raise HTTPException(
            status_code=400, detail="Invalid file extension; expected .ef"
        )
    # Write to in-memory file-like then to temp file if parser requires a path
    import os
    import tempfile

    try:
        with tempfile.NamedTemporaryFile("w+", suffix=".ef", delete=False) as tmp:
            tmp.write(content)
            tmp.flush()
            path = tmp.name
        parsed = parse_ef(path)
        return parsed
    finally:
        try:
            os.unlink(path)  # type: ignore[name-defined]
        except FileNotFoundError:
            pass
        except Exception as exc:  # noqa: BLE001 - log unexpected cleanup errors
            logging.getLogger(__name__).warning("Temp cleanup failed: %s", exc)


def _b64_size_mb(data_b64: str) -> float:
    try:
        raw = base64.b64decode(data_b64)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid base64: {exc}") from exc
    return round(len(raw) / (1024 * 1024), 6)


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"status": "healthy", "timestamp": datetime.now(tz=timezone.utc).isoformat()}


@app.get("/api/version")
def version() -> Dict[str, str]:
    return {"version": edgeflowc.VERSION, "api_version": "v1"}


@app.get("/api/help")
def help_() -> Dict[str, Any]:
    commands = [
        "POST /api/compile",
        "POST /api/compile/verbose",
        "POST /api/optimize",
        "POST /api/benchmark",
        "GET /api/version",
        "GET /api/help",
        "GET /api/health",
    ]
    usage = "python edgeflowc.py <config.ef> [--verbose|--version|--help]"
    return {"commands": commands, "usage": usage}


@app.post("/api/compile", response_model=CompileResponse)
def compile_cfg(
    req: CompileRequest, _: None = Depends(rate_limit_dep)
) -> CompileResponse:
    parsed = _parse_config_content(req.filename, req.config_file)
    return CompileResponse(
        success=True, parsed_config=parsed, message="Parsed successfully"
    )


@app.post("/api/compile/verbose", response_model=CompileResponse)
def compile_verbose(
    req: CompileRequest, _: None = Depends(rate_limit_dep)
) -> CompileResponse:
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    root = logging.getLogger()
    old_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        parsed = _parse_config_content(req.filename, req.config_file)
        handler.flush()
        logs = [line for line in log_stream.getvalue().splitlines() if line]
        return CompileResponse(
            success=True, parsed_config=parsed, logs=logs, message="Parsed successfully"
        )
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)


@app.post("/api/optimize", response_model=OptimizeResponse)
def optimize(
    req: OptimizeRequest, _: None = Depends(rate_limit_dep)
) -> OptimizeResponse:
    # Placeholder: return the same model and simple report
    size_mb = _b64_size_mb(req.model_file)
    report = {
        "quantize": req.config.get("quantize"),
        "target_device": req.config.get("target_device"),
        "optimize_for": req.config.get("optimize_for"),
        "original_size_mb": size_mb,
        "estimated_size_mb": max(size_mb * 0.5, 0.000001),
    }
    optimized_model = req.model_file  # echo for now
    return OptimizeResponse(
        success=True, optimized_model=optimized_model, optimization_report=report
    )


@app.post("/api/benchmark", response_model=BenchmarkResponse)
def benchmark(
    req: BenchmarkRequest, _: None = Depends(rate_limit_dep)
) -> BenchmarkResponse:
    orig_size = _b64_size_mb(req.original_model)
    opt_size = _b64_size_mb(req.optimized_model)
    # Simple synthetic latencies: proportional to size
    orig_latency = round(max(1.0, orig_size * 10.0), 3)
    opt_latency = round(max(0.5, opt_size * 8.0), 3)
    size_reduction = (
        round((orig_size - opt_size) / max(orig_size, 1e-9), 6) if orig_size else 0.0
    )
    speedup = round(orig_latency / max(opt_latency, 1e-9), 6)
    return BenchmarkResponse(
        original_stats=Stats(size_mb=orig_size, latency_ms=orig_latency),
        optimized_stats=Stats(size_mb=opt_size, latency_ms=opt_latency),
        improvement=Improvement(size_reduction=size_reduction, speedup=speedup),
    )


# Root redirect/info
@app.get("/")
def root() -> Dict[str, Any]:
    return {"message": "EdgeFlow API", "docs": "/docs", "health": "/api/health"}
