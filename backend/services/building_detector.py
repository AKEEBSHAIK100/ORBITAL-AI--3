import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

class BuildingDetector:
    _instance: Optional["BuildingDetector"] = None
    _model = None
    _device: str = "cpu"
    is_available: bool = False
    load_error: str = ""

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = os.getenv(
                "BUILDING_MODEL_PATH",
                str(Path(__file__).parent.parent / "models" / "building_model.pt")
            )
        self.model_path = str(Path(model_path).resolve())
        self._load_model()

    @classmethod
    def get_instance(cls) -> "BuildingDetector":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def device(self) -> str:
        return self._device

    def _load_model(self):
        print(f"[BuildingDetector] Attempting to load model from: {self.model_path}")
        if not os.path.exists(self.model_path):
            self.is_available = False
            self.load_error = f"Model file not found at: {self.model_path}"
            print(f"[BuildingDetector] WARNING: {self.load_error}")
            return

        try:
            import torch
            from ultralytics import YOLO

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            if self._device == "cpu":
                num_cores = os.cpu_count() or 4
                torch.set_num_threads(num_cores)

            self._model = YOLO(self.model_path)
            # Warmup pass
            try:
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                self._model.predict(dummy, device=self._device, verbose=False)
            except Exception as warm_err:
                print(f"[BuildingDetector] Warmup warning (non-fatal): {warm_err}")

            self.is_available = True
            print(f"[BuildingDetector] Model loaded successfully on {self._device.upper()}.")
        except Exception as e:
            self.is_available = False
            self.load_error = str(e)
            print(f"[BuildingDetector] ERROR loading model: {e}")
    def ensure_model_loaded(self) -> bool:
        """Ensures the model is loaded in memory. If unloaded, attempts to re-load."""
        if self._model is None:
            self._load_model()
        return self.is_available

    def unload_model(self) -> None:
        """Release loaded YOLO model weights from memory and clear GPU cache."""
        import gc
        self._model = None
        self.is_available = False
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        print("[BuildingDetector] Model unloaded successfully.")

    def predict_batch(
        self,
        tiles: List[np.ndarray],
        conf_threshold: float = 0.20
    ) -> List[List[Dict[str, Any]]]:
        """Run batch inference for higher throughput."""
        if not self.is_available or self._model is None:
            self.ensure_model_loaded()
        if not self.is_available or self._model is None:
            raise RuntimeError(
                f"Building detection model unavailable: {self.load_error or 'model not loaded'}"
            )
        if not tiles:
            return []

        results = self._model.predict(
            tiles,
            conf=conf_threshold,
            device=self._device,
            verbose=False,
            imgsz=640
        )

        batch_detections = []
        for result in results:
            detections = []
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            masks = result.masks.xy if result.masks is not None else []

            for i in range(len(boxes)):
                box = boxes[i].tolist()
                poly = masks[i].tolist() if i < len(masks) else [
                    [box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]
                ]
                detections.append({
                    "bbox": box,
                    "confidence": float(confs[i]),
                    "polygon": poly,
                    "bbox_area": (box[2] - box[0]) * (box[3] - box[1])
                })
            batch_detections.append(detections)
        return batch_detections

    def predict_tile(self, tile_bgr: np.ndarray, conf_threshold: float = 0.20) -> List[Dict[str, Any]]:
        return self.predict_batch([tile_bgr], conf_threshold)[0]
