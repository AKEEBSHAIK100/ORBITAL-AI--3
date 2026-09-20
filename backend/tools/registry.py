"""
Tool registry with per-tool on-demand lazy initialization.

All heavy tool instances (YOLO, BigEarthNet ResNet-50) and specialist adapters
are created individually on first access rather than in bulk at module import time
or on the first lookup.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import BaseTool

# ── Lazy singletons ────────────────────────────────────────────────────────────
_instances: Dict[str, BaseTool] = {}

_ALIASES: Dict[str, str] = {
    "rs_vqa_adapted": "vqa",
    "rs_caption_adapted": "caption",
    "visual_grounding": "grounding",
    "optical_sar_fusion": "optical_sar",
    "change_vqa": "change_detection",
}

CANONICAL_TOOL_IDS: List[str] = [
    "land_cover",
    "building_detection",
    "optical_sar",
    "change_detection",
    "grounding",
    "caption",
    "vqa",
    "rs_generalist",
]

ALL_TOOL_IDS: List[str] = [
    "rs_vqa_adapted",
    "rs_caption_adapted",
    "vqa",
    "caption",
    "grounding",
    "visual_grounding",
    "building_detection",
    "land_cover",
    "optical_sar",
    "optical_sar_fusion",
    "change_detection",
    "change_vqa",
    "rs_generalist",
]


def _create_tool(canonical_id: str) -> Optional[BaseTool]:
    """Instantiate only the requested canonical tool and its direct dependencies."""
    if canonical_id == "land_cover":
        from .land_cover import BigEarthNetTool
        return BigEarthNetTool()
    elif canonical_id == "building_detection":
        from .building_detection import BuildingDetectionTool
        return BuildingDetectionTool()
    elif canonical_id == "optical_sar":
        from .optical_sar import OpticalSARTool
        return OpticalSARTool()
    elif canonical_id == "change_detection":
        from .change_detection import ChangeDetectionTool
        return ChangeDetectionTool()
    elif canonical_id == "grounding":
        from .grounding import GroundingTool
        return GroundingTool()
    elif canonical_id == "caption":
        from .caption import CaptionTool
        return CaptionTool(ben_tool=get_tool("land_cover"))
    elif canonical_id == "vqa":
        from .vqa import VQATool
        return VQATool(
            ben_tool=get_tool("land_cover"),
            building_tool=get_tool("building_detection"),
        )
    elif canonical_id == "rs_generalist":
        from .generalist import GeneralistTool
        return GeneralistTool()
    return None


def get_tool(tool_id: str) -> Optional[BaseTool]:
    """Retrieve or lazily construct the requested tool singleton."""
    canonical_id = _ALIASES.get(tool_id, tool_id)
    if canonical_id not in _instances:
        inst = _create_tool(canonical_id)
        if inst is not None:
            _instances[canonical_id] = inst
    return _instances.get(canonical_id)


def _get_all() -> Dict[str, BaseTool]:
    """Construct or retrieve all tool instances mapping both canonical and alias keys."""
    for tid in CANONICAL_TOOL_IDS:
        get_tool(tid)
    res: Dict[str, BaseTool] = {}
    for tid in ALL_TOOL_IDS:
        tool = get_tool(tid)
        if tool is not None:
            res[tid] = tool
    return res


def list_all_tools() -> Dict[str, Any]:
    """Return specifications for all tools without triggering heavy model inference."""
    return {tid: tool.to_spec() for tid, tool in _get_all().items()}


def reset_registry() -> None:
    """Clear instantiated tools for testing and memory release."""
    global _instances
    _instances.clear()


# Backward-compatibility: TOOLS[key] still works but initializes lazily on first access
class _LazyToolsMapping:
    """Proxy that defers tool initialization until dict access."""
    def __getitem__(self, key: str) -> BaseTool:
        tool = get_tool(key)
        if tool is None:
            raise KeyError(key)
        return tool

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return key in ALL_TOOL_IDS

    def keys(self):  # type: ignore[override]
        return _get_all().keys()

    def values(self):  # type: ignore[override]
        return _get_all().values()

    def items(self):  # type: ignore[override]
        return _get_all().items()

    def get(self, key: str, default: Optional[BaseTool] = None) -> Optional[BaseTool]:
        tool = get_tool(key)
        return tool if tool is not None else default


TOOLS: Any = _LazyToolsMapping()
