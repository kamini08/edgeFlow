"""FastAPI application for EdgeFlow Web API.

Implements strict CLI-API parity for compile, optimize, benchmark, version, and help
endpoints. Uses existing core modules where possible and provides safe fallbacks.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from datetime import datetime, timezone
from parser import parse_ef  # type: ignore
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, constr

# Import core CLI logic
import edgeflowc  # type: ignore
from backend.api.services.parser_service import ParserService
from code_generator import CodeGenerator

# Import EdgeFlow pipeline components
from edgeflow_ast import create_program_from_dict
from edgeflow_ir import FusionPass, IRBuilder, QuantizationPass, SchedulingPass
from explainability_reporter import generate_explainability_report
from fast_compile import fast_compile_config
from initial_check import InitialChecker, perform_initial_check
from reporter import generate_json_report  # type: ignore

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


class PipelineResponse(BaseModel):
    success: bool
    ast: Optional[Dict[str, Any]] = None
    ir_graph: Optional[Dict[str, Any]] = None
    optimization_passes: Optional[List[Dict[str, Any]]] = None
    generated_code: Optional[Dict[str, str]] = None
    optimization_report: Optional[Dict[str, Any]] = None
    explainability_report: Optional[str] = None
    message: Optional[str] = None
    errors: Optional[List[str]] = None


class FastCompileResponse(BaseModel):
    success: bool
    estimated_impact: Optional[Dict[str, Any]] = None
    validation_results: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    warnings: Optional[List[str]] = None


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


class CheckRequest(BaseModel):
    model_path: str
    config: Dict[str, Any] = Field(default_factory=dict)
    device_spec_file: Optional[str] = None


class CheckResponse(BaseModel):
    compatible: bool
    requires_optimization: bool
    issues: List[str] = []
    recommendations: List[str] = []
    fit_score: float


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
        # Prefer new parser service; fall back to day-1 helper if needed
        success, cfg, err = ParserService.parse_config_content(content)
        if success:
            return cfg
        # Fallback to existing parse_ef via temp file so older flows still work
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
        "POST /api/compile/dry-run",
        "POST /api/check",
        "POST /api/optimize",
        "POST /api/benchmark",
        "GET /api/version",
        "GET /api/help",
        "GET /api/health",
    ]
    usage = "python edgeflowc.py <config.ef> [--verbose|--dry-run|--version|--help]"
    return {"commands": commands, "usage": usage}


@app.post("/api/compile", response_model=CompileResponse)
def compile_cfg(
    req: CompileRequest, _: None = Depends(rate_limit_dep)
) -> CompileResponse:
    if not req.filename.lower().endswith(".ef"):
        raise HTTPException(
            status_code=400, detail="Invalid file extension; expected .ef"
        )
    success, cfg, err = ParserService.parse_config_content(req.config_file)
    if not success:
        raise HTTPException(status_code=400, detail=err)
    return CompileResponse(
        success=True, parsed_config=cfg, message="Parsed successfully"
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
        if not req.filename.lower().endswith(".ef"):
            raise HTTPException(
                status_code=400, detail="Invalid file extension; expected .ef"
            )
        success, cfg, err = ParserService.parse_config_content(req.config_file)
        if not success:
            raise HTTPException(status_code=400, detail=err)
        handler.flush()
        logs = [line for line in log_stream.getvalue().splitlines() if line]
        return CompileResponse(
            success=True, parsed_config=cfg, logs=logs, message="Parsed successfully"
        )
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)


@app.post("/api/optimize", response_model=OptimizeResponse)
def optimize(
    req: OptimizeRequest, _: None = Depends(rate_limit_dep)
) -> OptimizeResponse:
    # Calculate sizes
    size_mb = _b64_size_mb(req.model_file)
    optimized_size_mb = max(size_mb * 0.5, 0.000001)  # Simulated 50% reduction

    # Create stats for reporter
    unoptimized_stats = {
        "size_mb": size_mb,
        "latency_ms": size_mb * 10.0,  # Simulated latency
        "model_path": "uploaded_model.tflite",
    }
    optimized_stats = {
        "size_mb": optimized_size_mb,
        "latency_ms": optimized_size_mb * 8.0,  # Simulated improved latency
        "model_path": "optimized_model.tflite",
    }

    # Generate JSON report using reporter module
    json_report_str = generate_json_report(
        unoptimized_stats, optimized_stats, req.config
    )
    json_report_dict = json.loads(json_report_str)

    # Add quantization and target device info
    report = {
        "quantize": req.config.get("quantize"),
        "target_device": req.config.get("target_device"),
        "optimize_for": req.config.get("optimize_for"),
        "original_size_mb": size_mb,
        "estimated_size_mb": optimized_size_mb,
        **json_report_dict,  # Include full reporter metrics
    }

    optimized_model = req.model_file  # echo for now
    return OptimizeResponse(
        success=True, optimized_model=optimized_model, optimization_report=report
    )


@app.post("/api/compile/dry-run", response_model=CompileResponse)
def compile_dry_run(
    req: CompileRequest, _: None = Depends(rate_limit_dep)
) -> CompileResponse:
    """Parse-only endpoint to satisfy CLI-API parity for --dry-run."""
    if not req.filename.lower().endswith(".ef"):
        raise HTTPException(
            status_code=400, detail="Invalid file extension; expected .ef"
        )
    success, cfg, err = ParserService.parse_config_content(req.config_file)
    if not success:
        raise HTTPException(status_code=400, detail=err)
    return CompileResponse(
        success=True,
        parsed_config=cfg,
        message="Configuration parsed successfully (dry-run)",
    )


@app.post("/api/check", response_model=CheckResponse)
def check_compatibility_api(
    req: CheckRequest, _: None = Depends(rate_limit_dep)
) -> CheckResponse:
    try:
        should_opt, report = perform_initial_check(
            req.model_path, req.config, req.device_spec_file
        )
        return CheckResponse(
            compatible=report.compatible,
            requires_optimization=report.requires_optimization,
            issues=report.issues,
            recommendations=report.recommendations,
            fit_score=report.estimated_fit_score,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/check/upload", response_model=CheckResponse)
async def check_uploaded_model_api(
    model: UploadFile = File(...),
    config: str = Form("{}"),
    device_spec_file: Optional[str] = Form(None),
    _: None = Depends(rate_limit_dep),
) -> CheckResponse:
    import json as _json
    import tempfile

    try:
        cfg = _json.loads(config) if isinstance(config, str) else dict(config)
    except Exception:  # noqa: BLE001
        cfg = {}
    # Persist the uploaded file temporarily for profiling by path
    suffix = Path(model.filename or "model").suffix
    with tempfile.NamedTemporaryFile(
        "wb", suffix=suffix, delete=True
    ) as tmp:  # type: ignore[attr-defined]
        content = await model.read()
        tmp.write(content)
        tmp.flush()
        checker = InitialChecker(device_spec_file)
        target = str(cfg.get("target_device") or cfg.get("device") or "generic")
        report = checker.check_compatibility(tmp.name, target, cfg)
        return CheckResponse(
            compatible=report.compatible,
            requires_optimization=report.requires_optimization,
            issues=report.issues,
            recommendations=report.recommendations,
            fit_score=report.estimated_fit_score,
        )


@app.post("/api/pipeline", response_model=PipelineResponse)
def run_full_pipeline(
    req: CompileRequest, _: None = Depends(rate_limit_dep)
) -> PipelineResponse:
    """Run full EdgeFlow pipeline.

    Steps: parse -> AST -> IR -> optimization -> code generation.
    """
    try:
        if not req.filename.lower().endswith(".ef"):
            raise HTTPException(
                status_code=400, detail="Invalid file extension; expected .ef"
            )

        # Step 1: Parse configuration
        success, cfg, err = ParserService.parse_config_content(req.config_file)
        if not success:
            return PipelineResponse(success=False, errors=[err])

        # Step 2: Build AST
        ast = create_program_from_dict(cfg)

        # Step 3: Build IR Graph
        ir_builder = IRBuilder()
        ir_graph = ir_builder.build_from_config(cfg)

        # Step 4: Apply optimization passes
        optimization_passes = []

        # Quantization Pass
        if "quantize" in cfg and cfg["quantize"] != "none":
            quantize_pass = QuantizationPass()
            ir_graph = quantize_pass.transform(ir_graph)
            optimization_passes.append(
                {
                    "name": "QuantizationPass",
                    "description": f"Applied {cfg['quantize']} quantization",
                    "nodes_added": 1,
                }
            )

        # Fusion Pass
        if cfg.get("enable_fusion", False):
            fusion_pass = FusionPass()
            ir_graph = fusion_pass.transform(ir_graph)
            optimization_passes.append(
                {
                    "name": "FusionPass",
                    "description": "Fused operations for efficiency",
                    "nodes_added": 1,
                }
            )

        # Scheduling Pass
        schedule_pass = SchedulingPass()
        ir_graph = schedule_pass.transform(ir_graph)
        optimization_passes.append(
            {
                "name": "SchedulingPass",
                "description": "Optimized execution schedule",
                "nodes_added": 1,
            }
        )

        # Step 5: Generate code
        code_generator = CodeGenerator(ast, ir_graph)
        generated_code = {
            "python": code_generator.generate_ir_based_code("python"),
            "cpp": code_generator.generate_ir_based_code("cpp"),
            "onnx": code_generator.generate_ir_based_code("onnx"),
            "tensorrt": code_generator.generate_ir_based_code("tensorrt"),
        }

        # Step 6: Generate reports
        optimization_report = {
            "quantize": cfg.get("quantize"),
            "target_device": cfg.get("target_device"),
            "optimize_for": cfg.get("optimize_for"),
            "optimization_passes_applied": len(optimization_passes),
            "generated_backends": list(generated_code.keys()),
        }

        # Prepare IR data for explainability report
        ir_dict = ir_graph.to_dict()
        ir_transformations = {
            "passes_applied": len(optimization_passes),
            "transformations": [pass_info["name"] for pass_info in optimization_passes],
            "nodes": len(ir_dict["nodes"]),
            "edges": len(ir_dict["edges"]),
            "is_valid": True,
            "execution_order": ir_dict["execution_order"],
            "node_types": {},
        }

        # Count node types
        for node in ir_dict["nodes"]:
            node_type = node["node_type"]
            ir_transformations["node_types"][node_type] = (
                ir_transformations["node_types"].get(node_type, 0) + 1
            )

        # Prepare optimization results for explainability report
        optimization_results = {
            "optimizations_applied": [
                str(pass_info["name"]).lower().replace("pass", "")
                for pass_info in optimization_passes
            ],
            "optimization_passes": optimization_passes,
        }

        explainability_report = generate_explainability_report(
            cfg,
            optimization_results,
            ir_transformations,
            {"estimated_impact": {"size_reduction": 0.3, "speedup": 1.5}},
        )

        return PipelineResponse(
            success=True,
            ast=ast.to_dict(),
            ir_graph=ir_graph.to_dict(),
            optimization_passes=optimization_passes,
            generated_code=generated_code,
            optimization_report=optimization_report,
            explainability_report=explainability_report,
            message="Pipeline executed successfully",
        )

    except Exception as e:
        return PipelineResponse(
            success=False, errors=[f"Pipeline execution failed: {str(e)}"]
        )


@app.post("/api/fast-compile", response_model=FastCompileResponse)
def fast_compile(
    req: CompileRequest, _: None = Depends(rate_limit_dep)
) -> FastCompileResponse:
    """Fast compile with estimated impact analysis."""
    try:
        if not req.filename.lower().endswith(".ef"):
            raise HTTPException(
                status_code=400, detail="Invalid file extension; expected .ef"
            )

        # Parse configuration
        success, cfg, err = ParserService.parse_config_content(req.config_file)
        if not success:
            return FastCompileResponse(success=False, message=err)

        # Run fast compile
        result = fast_compile_config(cfg)

        return FastCompileResponse(
            success=True,
            estimated_impact=result.estimated_impact,
            validation_results={"errors": result.errors, "warnings": result.warnings},
            message="Fast compile completed successfully",
            warnings=result.warnings,
        )

    except Exception as e:
        return FastCompileResponse(
            success=False, message=f"Fast compile failed: {str(e)}"
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
