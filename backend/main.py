import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import building_analysis, fusion, analyze
from .routers.classify import router as classify_router
from .services.building_detector import BuildingDetector
from .services.ben_classifier import BENClassifier

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[SatQuery AI] Initializing services…")
    # Pre-warm building detector
    try:
        detector = BuildingDetector.get_instance()
        print(f"[SatQuery AI] Building detector ready on {detector.device.upper()}.")
    except Exception as e:
        print(f"[SatQuery AI] Warning: Building detector unavailable: {e}")
    # Pre-warm BigEarthNet v2.0 classifier
    try:
        classifier = BENClassifier.get_instance()
        if classifier.is_available:
            print(f"[SatQuery AI] BigEarthNet v2.0 classifier ready on {classifier.device.upper()} ({classifier.model_id}).")
        else:
            print(f"[SatQuery AI] BigEarthNet classifier in fallback mode: {classifier.load_error[:120]}")
    except Exception as e:
        print(f"[SatQuery AI] Warning: BEN classifier unavailable: {e}")
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

# Enable CORS for Vite dev + Vercel + Express proxy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router)
app.include_router(building_analysis.router)
app.include_router(fusion.router)
app.include_router(classify_router)

@app.get("/")
@app.get("/health")
async def health():
    detector = BuildingDetector.get_instance()
    classifier = BENClassifier.get_instance()
    return {
        "status": "healthy",
        "service": "SatQuery AI Backend",
        "version": "3.0.0",
        "models": {
            "building_detector": {
                "device": detector.device,
                "model": os.path.basename(detector.model_path),
            },
            "ben_classifier": {
                "available": classifier.is_available,
                "device": classifier.device,
                "model_id": classifier.model_id,
                "classes": 19,
            },
        },
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
