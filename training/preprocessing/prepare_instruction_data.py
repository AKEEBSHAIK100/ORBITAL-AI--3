"""
Prepares standardized instruction-tuning datasets for Vision-Language Models.
Outputs unified JSONL format compatible with HuggingFace SFTTrainer, LLaVA, or Qwen2-VL training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def build_instruction_sample(
    image_path: str,
    instruction: str,
    response: str,
    task_category: str = "general_rs",
    sample_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Builds a single instruction-tuning sample in standard multimodal format.
    """
    return {
        "id": sample_id or f"sample_{Path(image_path).stem}",
        "image": str(image_path),
        "conversations": [
            {
                "from": "human",
                "value": f"<image>\n{instruction}"
            },
            {
                "from": "gpt",
                "value": response
            }
        ],
        "task": task_category
    }


def export_instruction_dataset(
    samples: List[Dict[str, Any]],
    output_path: str
) -> int:
    """Writes instruction samples to a JSONL file. Returns number of exported samples."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            count += 1
    return count
