"""
Tool registry with lazy initialization.

All heavy tool instances (YOLO, BigEarthNet ResNet-50) are created on first
access rather than at module import time, which previously caused 15–20 s
blocking delays during container startup and health checks.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .base import BaseTool

# ── Lazy singletons ────────────────────────────────────────────────────────────
_instances: Dict[str, BaseTool] = {}


def _build_instances() -> Dict[str, BaseTool]:
    """Construct all tool instances exactly once (lazy, thread-safe enough for a single worker)."""
    from .vqa import VQATool
    from .caption import CaptionTool
    from .grounding import GroundingTool
    from .building_detection import BuildingDetectionTool
    from .land_cover import BigEarthNetTool
    from .optical_sar import OpticalSARTool
    from .change_detection import ChangeDetectionTool

    ben = BigEarthNetTool()
    building = BuildingDetectionTool()
    optical_sar = OpticalSARTool()
    change = ChangeDetectionTool()
    grounding = GroundingTool()
    caption = CaptionTool(ben_tool=ben)
    vqa = VQATool(ben_tool=ben, building_tool=building)

    return {
        "rs_vqa_adapted": vqa,
        "rs_caption_adapted": caption,
        "vqa": vqa,
        "caption": caption,
        "grounding": grounding,
        "building_detection": building,
        "land_cover": ben,
        "optical_sar": optical_sar,
        "change_detection": change,
        "change_vqa": change,
    }


def _get_all() -> Dict[str, BaseTool]:
    global _instances
    if not _instances:
        _instances = _build_instances()
    return _instances


def get_tool(tool_id: str) -> Optional[BaseTool]:
    return _get_all().get(tool_id)


def list_all_tools() -> Dict[str, Any]:
    return {tid: tool.to_spec() for tid, tool in _get_all().items()}


# Backward-compatibility: TOOLS[key] still works but initializes lazily on first access
class _LazyToolsMapping:
    """Proxy that defers tool initialization until first dict access."""
    def __getitem__(self, key: str) -> BaseTool:
        return _get_all()[key]

    def __contains__(self, key: object) -> bool:
        return key in _get_all()

    def keys(self):  # type: ignore[override]
        return _get_all().keys()

    def values(self):  # type: ignore[override]
        return _get_all().values()

    def items(self):  # type: ignore[override]
        return _get_all().items()

    def get(self, key: str, default: Optional[BaseTool] = None) -> Optional[BaseTool]:
        return _get_all().get(key, default)


TOOLS: Any = _LazyToolsMapping()
