"""
Dataset adapters and registry for SatQuery AI.
"""
from .dataset_registry import DatasetRegistry, DatasetEntry, DatasetStatus
from .bigearthnet import BigEarthNetAdapter, BIGEARTHNET_19_CLASSES
from .vrsbench import VRSBenchAdapter
from .rsvqa import RSVQAAdapter
from .cdvqa import CDVQAAdapter

__all__ = [
    "DatasetRegistry",
    "DatasetEntry",
    "DatasetStatus",
    "BigEarthNetAdapter",
    "BIGEARTHNET_19_CLASSES",
    "VRSBenchAdapter",
    "RSVQAAdapter",
    "CDVQAAdapter",
]
