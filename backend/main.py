import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import building_analysis, fusion, analyze
from .routers.classify import router as classify_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    lazy_warmup = os.getenv("LAZY_MODEL_WARMUP", "false").lower() in ("1", "true", "yes")
    if not lazy_warmup:
        print("[ORBITAL-AI] Initializing services (eager warmup)…")
        # Pre-warm building detector (non-fatal)
        try:
            from .services.building_detector import BuildingDetector
            detector = BuildingDetector.get_instance()
            if detector.is_available:
                print(f"[ORBITAL-AI] Building detector ready on {detector.device.upper()}.")
            else:
                print(f"[ORBITAL-AI] Building detector unavailable (will degrade gracefully): {detector.load_error[:120]}")
        except Exception as e:
            print(f"[ORBITAL-AI] Warning: Building detector init error: {e}")

        # Pre-warm BigEarthNet v2.0 classifier (non-fatal)
        try:
            from .services.ben_classifier import BENClassifier
            classifier = BENClassifier.get_instance()
            if classifier.is_available:
                print(f"[ORBITAL-AI] BigEarthNet v2.0 classifier ready on {classifier.device.upper()} ({classifier.model_id}).")
            else:
                print(f"[ORBITAL-AI] BigEarthNet classifier in fallback mode: {classifier.load_error[:120]}")
        except Exception as e:
            print(f"[ORBITAL-AI] Warning: BEN classifier init error: {e}")
    else:
        print("[ORBITAL-AI] Services initialized in lazy warmup mode (on-demand loading enabled).")

    yield
    print("[ORBITAL-AI] Shutting down — releasing model resources…")
    try:
        from .services.building_detector import BuildingDetector
        if BuildingDetector._instance is not None:
            BuildingDetector.get_instance().unload_model()
    except Exception:
        pass
    try:
        from .services.ben_classifier import BENClassifier
        if BENClassifier._instance is not None:
            BENClassifier.get_instance().unload_model()
    except Exception:
        pass
    try:
        from .services.rs_adapters import RSAdapterRuntime
        if RSAdapterRuntime._instance is not None:
            RSAdapterRuntime.get_instance().unload_models()
    except Exception:
        pass
    try:
        from .tools.registry import get_tool
        gen_tool = get_tool("rs_generalist")
        if gen_tool is not None and hasattr(gen_tool, "unload_model"):
            gen_tool.unload_model()
    except Exception:
        pass


app = FastAPI(
    title="ORBITAL-AI — Agentic Remote-Sensing VLM Backend",
    description=(
        "Remote-sensing specialist backend with executable model availability reported at runtime. "
        "Optional building detection, optical–SAR analysis, and BigEarthNet land-cover classification components are exposed only when their runtimes are available."
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
        lazy_warmup = os.getenv("LAZY_MODEL_WARMUP", "false").lower() in ("1", "true", "yes")
        if lazy_warmup and BuildingDetector._instance is None:
            model_status["building_detector"] = {
                "available": False,
                "device": "cpu",
                "model": "building_model.pt",
                "state": "standby",
                "error": "Model not loaded yet; availability will be verified on first use.",
            }
        else:
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
        lazy_warmup = os.getenv("LAZY_MODEL_WARMUP", "false").lower() in ("1", "true", "yes")
        if lazy_warmup and BENClassifier._instance is None:
            model_status["ben_classifier"] = {
                "available": False,
                "device": "cpu",
                "model_id": "BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0",
                "classes": 19,
                "state": "standby",
                "error": "Model not loaded yet; availability will be verified on first use.",
            }
        else:
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

    try:
        from .services.rs_adapters import RSAdapterRuntime, SpecialistState
        runtime = RSAdapterRuntime.get_instance()
        model_status["caption_specialist"] = {
            "available": runtime.caption_state == SpecialistState.AVAILABLE,
            "base_model": runtime.caption_base_id,
            "adapter": "BigEarthNet-derived LoRA pilot adapter",
            "device": runtime.device,
            "state": runtime.caption_state.value,
            "error": runtime.caption_unavailable_reason if runtime.caption_state != SpecialistState.AVAILABLE else None,
        }
        model_status["vqa_specialist"] = {
            "available": runtime.vqa_state == SpecialistState.AVAILABLE,
            "base_model": runtime.vqa_base_id,
            "adapter": "BigEarthNet-derived VQA LoRA pilot adapter",
            "device": runtime.device,
            "state": runtime.vqa_state.value,
            "error": runtime.vqa_unavailable_reason if runtime.vqa_state != SpecialistState.AVAILABLE else None,
        }
    except Exception as e:
        model_status["caption_specialist"] = {"available": False, "error": str(e)}
        model_status["vqa_specialist"] = {"available": False, "error": str(e)}

    overall = "healthy" if all(v.get("available") for v in model_status.values()) else "degraded"
    return {
        "status": overall,
        "service": "ORBITAL-AI Backend",
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
