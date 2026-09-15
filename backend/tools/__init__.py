"""
SatQuery AI Tools Package.

Heavy tool implementations are lazily initialized on first access via registry
to avoid blocking startup for 15–20 s while YOLO and BigEarthNet weights load.
"""
from .registry import TOOLS, get_tool, list_all_tools
from .base import BaseTool

__all__ = [
    "TOOLS",
    "get_tool",
    "list_all_tools",
    "BaseTool",
]
