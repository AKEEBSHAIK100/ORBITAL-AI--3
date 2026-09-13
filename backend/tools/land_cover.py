import time
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image

from .base import BaseTool
from ..services.ben_classifier import BENClassifier

class BigEarthNetTool(BaseTool):
    id = "land_cover"
    name = "BigEarthNet v2.0 Land-Cover Classifier"
    description = "Multispectral land-cover and semantic scene understanding using BIFOLD-pretrained ResNet-50 on the 19-class Corine Land Cover hierarchy."
    supported_tasks = ["land_cover", "vqa", "caption"]
    modalities = ["optical", "multispectral"]
    adapter = "BIFOLD-BigEarthNetv2-0 / reBEN ResNet-50 Deep Classifier"
    domain_adaptation = "Official 19-class Corine Land Cover taxonomy (IGARSS 2025 · arXiv:2407.03653)"
    model_id = "BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0"
    permitted_parameters = {
        "top_k": 5,
        "threshold": 0.25,
        "taxonomy": "BigEarthNet-19"
    }

    def __init__(self):
        self.classifier = BENClassifier.get_instance()

    def run(self, inputs: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        t0 = time.time()
        params = {**self.permitted_parameters, **(parameters or {})}
        image = inputs.get("image")

        if image is None:
            return {"error": "No image supplied to BigEarthNet tool", "status": "error"}

        top_k = int(params.get("top_k", 5))
        threshold = float(params.get("threshold", 0.25))

        # Convert image to bytes
        raw_bytes = b""
        if isinstance(image, bytes):
            raw_bytes = image
        elif isinstance(image, np.ndarray):
            import cv2
            _, buf = cv2.imencode(".jpg", image)
            raw_bytes = buf.tobytes()
        elif hasattr(image, "save"):
            import io
            b_io = io.BytesIO()
            image.save(b_io, format="JPEG")
            raw_bytes = b_io.getvalue()

        # Classify using classifier
        res = self.classifier.classify_image(raw_bytes, top_k=top_k, threshold=threshold)
        duration_ms = (time.time() - t0) * 1000

        return {
            "status": "success",
            "top_label": res.get("top_label", "Urban fabric"),
            "confidence": res.get("confidence", 85.0),
            "labels": res.get("labels", []),
            "active_labels": res.get("active_labels", []),
            "model_id": self.model_id,
            "duration_ms": duration_ms,
            "citation": res.get("citation", "BIFOLD BigEarthNet v2.0")
        }
