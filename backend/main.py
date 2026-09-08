import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import building_analysis
from .services.building_detector import BuildingDetector

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[SatQuery AI] Initializing Building Detection Service...")
    try:
        detector = BuildingDetector.get_instance()
        print(f"[SatQuery AI] Detection model ready on {detector.device.upper()}.")
    except Exception as e:
        print(f"[SatQuery AI] Warning: Failed to pre-warm model: {e}")
    yield
    print("[SatQuery AI] Shutting down Building Detection Service.")

app = FastAPI(
    title="SatQuery AI / Orbital-AI Building Detection Service",
    description="Deep learning instance segmentation & counting pipeline for aerial and satellite imagery",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for local Vite development and Express proxies
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(building_analysis.router)

@app.get("/")
@app.get("/health")
async def health():
    detector = BuildingDetector.get_instance()
    return {
        "status": "healthy",
        "service": "Orbital-AI Building Detection Service",
        "device": detector.device,
        "model": os.path.basename(detector.model_path),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
