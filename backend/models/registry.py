"""
Runtime specialist registry facade.

Keeps the agent orchestration layer independent from the concrete tool
implementation. Tool availability is evaluated from the actual registered
specialist and never fabricated.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from ..tools.registry import get_tool, list_all_tools


class SpecialistEntry:
    def __init__(self, tool: Any):
        self._tool = tool
        self.id = getattr(tool, "id", "")
        self.name = getattr(tool, "name", self.id)
        self.model_id = getattr(tool, "model_id", self.id)
        self.task = (getattr(tool, "supported_tasks", None) or [self.id])[0]
        self.is_available = bool(getattr(tool, "is_available", True))
        self.unavailable_reason = (
            getattr(tool, "unavailable_reason", None) if not self.is_available else None
        )

    def to_dict(self) -> Dict[str, Any]:
        if hasattr(self._tool, "to_spec"):
            spec = dict(self._tool.to_spec())
        else:
            spec = {
                "id": self.id,
                "name": self.name,
                "model_id": self.model_id,
            }
        spec.update({
            "is_available": self.is_available,
            "unavailable_reason": self.unavailable_reason,
        })
        return spec


class ModelRegistry:
    _instance: Optional["ModelRegistry"] = None

    @classmethod
    def get_instance(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_specialist(self, specialist_id: str) -> Optional[SpecialistEntry]:
        tool = get_tool(specialist_id)
        return SpecialistEntry(tool) if tool is not None else None

    def list_specialists(self):
        return [self.get_specialist(tool_id).to_dict() for tool_id in list_all_tools().keys()]

    def reset(self) -> None:
        """Testing hook; does not unload tool instances."""
        self.__class__._instance = None
