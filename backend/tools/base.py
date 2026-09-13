from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np

class BaseTool(ABC):
    id: str
    name: str
    description: str
    supported_tasks: List[str]
    modalities: List[str]
    adapter: str
    domain_adaptation: str
    model_id: str
    permitted_parameters: Dict[str, Any]

    @abstractmethod
    def run(self, inputs: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute tool on validated inputs and return structured evidence dictionary."""
        pass

    def to_spec(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "supported_tasks": self.supported_tasks,
            "modalities": self.modalities,
            "adapter": self.adapter,
            "domain_adaptation": self.domain_adaptation,
            "model_id": self.model_id,
            "permitted_parameters": self.permitted_parameters,
        }
