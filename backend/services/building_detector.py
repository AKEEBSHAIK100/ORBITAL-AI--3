import os
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from ultralytics import YOLO

class BuildingDetector:
    _instance: Optional["BuildingDetector"] = None
    _model: Optional[YOLO] = None
    _device: str = "cpu"

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = os.getenv(
                "BUILDING_MODEL_PATH",
                os.path.join(os.path.dirname(__file__), "..", "models", "building_model.pt")
            )
        self.model_path = os.path.abspath(model_path)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._device == "cpu":
            num_cores = os.cpu_count() or 4
            torch.set_num_threads(num_cores)  # use all available cores
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
        print(f"[BuildingDetector] Loading building segmentation model from: {self.model_path}")
        print(f"[BuildingDetector] Target inference device: {self._device.upper()}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")
        
        self._model = YOLO(self.model_path)
        # Warmup
        try:
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self._model.predict(dummy, device=self._device, verbose=False)
            print("[BuildingDetector] Model loaded and warmed up successfully.")
        except Exception as e:
            print(f"[BuildingDetector] Warmup warning: {e}")

    def predict_batch(
        self,
        tiles: List[np.ndarray],
        conf_threshold: float = 0.20
    ) -> List[List[Dict[str, Any]]]:
        """Run batch inference for higher throughput."""
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
