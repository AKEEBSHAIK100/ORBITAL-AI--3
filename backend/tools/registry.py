from typing import Any, Dict, Optional

from .base import BaseTool
from .vqa import VQATool
from .caption import CaptionTool
from .grounding import GroundingTool
from .building_detection import BuildingDetectionTool
from .land_cover import BigEarthNetTool
from .optical_sar import OpticalSARTool
from .change_detection import ChangeDetectionTool

_ben_tool_instance = BigEarthNetTool()
_building_tool_instance = BuildingDetectionTool()
_optical_sar_instance = OpticalSARTool()
_change_tool_instance = ChangeDetectionTool()
_grounding_tool_instance = GroundingTool()
_caption_tool_instance = CaptionTool(ben_tool=_ben_tool_instance)
_vqa_tool_instance = VQATool(ben_tool=_ben_tool_instance, building_tool=_building_tool_instance)

TOOLS: Dict[str, BaseTool] = {
    "vqa": _vqa_tool_instance,
    "caption": _caption_tool_instance,
    "grounding": _grounding_tool_instance,
    "building_detection": _building_tool_instance,
    "land_cover": _ben_tool_instance,
    "optical_sar": _optical_sar_instance,
    "change_detection": _change_tool_instance,
    "change_vqa": _change_tool_instance,
}

def get_tool(tool_id: str) -> Optional[BaseTool]:
    return TOOLS.get(tool_id)

def list_all_tools() -> Dict[str, Any]:
    return {tid: tool.to_spec() for tid, tool in TOOLS.items()}
