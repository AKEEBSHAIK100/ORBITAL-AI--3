import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import building_analysis, fusion, analyze
from .routers.classify import router as classify_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[SatQuery AI] Initializing services…")
    # Pre-warm building detector (non-fatal)
    try:
        from .services.building_detector import BuildingDetector
        detector = BuildingDetector.get_instance()
        if detector.is_available:
            print(f"[SatQuery AI] Building detector ready on {detector.device.upper()}.")
        else:
            print(f"[SatQuery AI] Building detector unavailable (will degrade gracefully): {detector.load_error[:120]}")
    except Exception as e:
        print(f"[SatQuery AI] Warning: Building detector init error: {e}")

    # Pre-warm BigEarthNet v2.0 classifier (non-fatal)
    try:
        from .services.ben_classifier import BENClassifier
        classifier = BENClassifier.get_instance()
        if classifier.is_available:
            print(f"[SatQuery AI] BigEarthNet v2.0 classifier ready on {classifier.device.upper()} ({classifier.model_id}).")
        else:
            print(f"[SatQuery AI] BigEarthNet classifier in fallback mode: {classifier.load_error[:120]}")
    except Exception as e:
        print(f"[SatQuery AI] Warning: BEN classifier init error: {e}")

    yield
    print("[SatQuery AI] Shutting down.")


app = FastAPI(
    title="SatQuery AI — Agentic Remote-Sensing VLM Backend",
    description=(
        "Multi-specialist deep-learning backend: building instance segmentation (YOLO), "
        "optical–SAR classical-CV fusion (OpenCV), and BigEarthNet v2.0 19-class land-cover classification (ResNet-50)."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Configure allowed origins via CORS_ORIGINS env var (comma-separated).
# Falls back to wildcard '*' for local dev / open deployments.
_raw_origins = os.getenv("CORS_ORIGINS", "*")
_allow_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()] if _raw_origins != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=_raw_origins != "*",
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router)
app.include_router(building_analysis.router)
app.include_router(fusion.router)
app.include_router(classify_router)


@app.get("/")
@app.get("/health")
@app.get("/api/health")
async def health():
    """Safe health endpoint — never raises 500 even when models are unavailable."""
    model_status: dict = {}

    try:
        from .services.building_detector import BuildingDetector
        detector = BuildingDetector.get_instance()
        model_status["building_detector"] = {
            "available": detector.is_available,
            "device": detector.device,
            "model": os.path.basename(detector.model_path) if hasattr(detector, "model_path") else "unknown",
            "error": detector.load_error if not detector.is_available else None,
        }
    except Exception as e:
        model_status["building_detector"] = {"available": False, "error": str(e)}

    try:
        from .services.ben_classifier import BENClassifier
        classifier = BENClassifier.get_instance()
        model_status["ben_classifier"] = {
            "available": classifier.is_available,
            "device": classifier.device,
            "model_id": classifier.model_id,
            "classes": 19,
            "error": classifier.load_error if not classifier.is_available else None,
        }
    except Exception as e:
        model_status["ben_classifier"] = {"available": False, "error": str(e)}

    overall = "healthy" if all(v.get("available") for v in model_status.values()) else "degraded"
    return {
        "status": overall,
        "service": "SatQuery AI Backend",
        "version": "3.0.0",
        "models": model_status,
    }


@app.get("/models")
async def models_info():
    """Expose registered analysis tools."""
    try:
        from .tools.registry import list_all_tools
        return {"tools": list_all_tools()}
    except Exception as e:
        return {"tools": {}, "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=True)
