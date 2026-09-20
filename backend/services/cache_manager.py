"""
SatQuery AI — Lightweight In-Process Image Preprocessing & Query Result Cache.

Provides bounded, thread-safe, memory-conservative caching with TTL and LRU eviction
to eliminate redundant image preprocessing and repeated specialist inferences.
Pure Python — no external cache daemons or persistent database dependencies.
"""

from __future__ import annotations

import collections
import copy
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

# ── Configuration Defaults ──────────────────────────────────────────────────
MAX_RESULT_CACHE_ENTRIES = int(os.getenv("MAX_RESULT_CACHE_ENTRIES", "256"))
RESULT_CACHE_TTL_SECONDS = float(os.getenv("RESULT_CACHE_TTL_SECONDS", "3600.0"))

MAX_IMAGE_CACHE_ENTRIES = int(os.getenv("MAX_IMAGE_CACHE_ENTRIES", "32"))
IMAGE_CACHE_TTL_SECONDS = float(os.getenv("IMAGE_CACHE_TTL_SECONDS", "1800.0"))


# ── Hash Utilities ──────────────────────────────────────────────────────────

def compute_image_hash(image: Any) -> str:
    """
    Computes a stable SHA-256 hash for image inputs without unnecessary deep copies.
    Supports bytes, bytearray, numpy.ndarray, and PIL.Image.
    """
    if image is None:
        return "none"

    if isinstance(image, (bytes, bytearray)):
        return hashlib.sha256(image).hexdigest()

    if isinstance(image, np.ndarray):
        arr = np.ascontiguousarray(image)
        h = hashlib.sha256(arr.data)
        # Include shape and dtype to ensure distinct identities
        h.update(f":{arr.shape}:{arr.dtype}".encode("utf-8"))
        return h.hexdigest()

    if hasattr(image, "tobytes"):
        # PIL Image or similar
        try:
            h = hashlib.sha256(image.tobytes())
            h.update(f":{image.size}:{image.mode}".encode("utf-8"))
            return h.hexdigest()
        except Exception:
            pass

    if isinstance(image, str):
        return hashlib.sha256(image.encode("utf-8")).hexdigest()

    return hashlib.sha256(str(id(image)).encode("utf-8")).hexdigest()


def compute_paired_image_hash(image1: Any, image2: Any) -> str:
    """Computes a distinct compound hash for bi-temporal or multi-modal image pairs."""
    h1 = compute_image_hash(image1)
    h2 = compute_image_hash(image2)
    return f"pair:{h1}:{h2}"


def get_adapter_fingerprint(
    adapter_name: Optional[str],
    adapter_path: Optional[Union[str, Path]] = None
) -> str:
    """
    Generates a deterministic fingerprint for adapted models based on adapter identity
    and underlying weight file status. If adapter weights are modified, fingerprint changes.
    """
    if not adapter_name:
        return "none"

    name_clean = str(adapter_name).strip()
    if not adapter_path:
        return name_clean

    p = Path(adapter_path)
    if not p.exists():
        return f"{name_clean}:not_found"

    # Check for weights file inside adapter directory
    weights_file = p / "adapter_model.safetensors"
    if weights_file.exists() and weights_file.is_file():
        try:
            st = weights_file.stat()
            return f"{name_clean}:{st.st_size}:{st.st_mtime_ns}"
        except Exception:
            pass

    # Fallback to directory mtime
    try:
        return f"{name_clean}:{p.stat().st_mtime_ns}"
    except Exception:
        return name_clean


def normalize_query_text(query: str) -> str:
    """Normalizes natural language query by stripping and collapsing whitespace to lowercase."""
    if not query:
        return ""
    q = query.strip().lower()
    return re.sub(r"\s+", " ", q)


def build_cache_key(
    image_identity: str,
    query: str,
    task: str,
    model_id: str,
    model_version: str,
    adapter_identity: str,
    parameters: Optional[Dict[str, Any]] = None
) -> str:
    """
    Constructs a deterministic, collision-resistant query result cache key.
    Adheres strictly to Section 4 & 6:
    image_identity + normalized_query + task + model_id + model_version + adapter_identity + parameters
    """
    norm_q = normalize_query_text(query)
    params = parameters or {}

    # Canonical sorted parameter string excluding internal orchestrator-injected keys
    filtered_params = {
        k: v for k, v in sorted(params.items())
        if not k.startswith("supporting_")  # internal shared evidence references
    }
    canonical_params = json.dumps(filtered_params, sort_keys=True, default=str)

    raw_key = (
        f"img={image_identity}|q={norm_q}|task={task}|"
        f"m={model_id}|v={model_version}|adp={adapter_identity}|p={canonical_params}"
    )
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


# ── In-Process Bounded TTL LRU Cache ────────────────────────────────────────

class BoundedTTLMemoryCache:
    """
    Thread-safe, bounded, in-memory cache with TTL and LRU eviction.
    Protects against memory leaks and deep-copies stored and retrieved objects
    to ensure mutability safety across concurrent requests.
    """

    def __init__(self, max_entries: int, ttl_seconds: float, name: str = "cache"):
        self.max_entries = max(1, max_entries)
        self.ttl_seconds = max(0.1, ttl_seconds)
        self.name = name
        self._cache: collections.OrderedDict[str, Tuple[float, Any]] = collections.OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Retrieves a cached value if present and unexpired. Returns a deep copy."""
        now = time.time()
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            created_at, val = self._cache[key]
            if now - created_at > self.ttl_seconds:
                # Expired
                del self._cache[key]
                self._misses += 1
                return None

            # Mark recently used
            self._cache.move_to_end(key)
            self._hits += 1
            return copy.deepcopy(val)

    def set(self, key: str, value: Any) -> None:
        """Stores a value in the cache with LRU eviction if full. Stores a deep copy."""
        now = time.time()
        with self._lock:
            if key in self._cache:
                del self._cache[key]
            elif len(self._cache) >= self.max_entries:
                # Evict least recently used (first item)
                self._cache.popitem(last=False)

            self._cache[key] = (now, copy.deepcopy(value))

    def clear(self) -> None:
        """Clears all entries from the cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict[str, Any]:
        """Returns operational telemetry for the cache."""
        with self._lock:
            return {
                "name": self.name,
                "current_entries": len(self._cache),
                "max_entries": self.max_entries,
                "ttl_seconds": self.ttl_seconds,
                "hits": self._hits,
                "misses": self._misses,
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


# ── Global Singletons ───────────────────────────────────────────────────────

_image_cache: Optional[BoundedTTLMemoryCache] = None
_result_cache: Optional[BoundedTTLMemoryCache] = None
_cache_init_lock = threading.Lock()


def get_image_preprocessing_cache() -> BoundedTTLMemoryCache:
    """Returns singleton instance of the image preprocessing cache."""
    global _image_cache
    if _image_cache is None:
        with _cache_init_lock:
            if _image_cache is None:
                _image_cache = BoundedTTLMemoryCache(
                    max_entries=MAX_IMAGE_CACHE_ENTRIES,
                    ttl_seconds=IMAGE_CACHE_TTL_SECONDS,
                    name="image_preprocessing_cache"
                )
    return _image_cache


def get_query_result_cache() -> BoundedTTLMemoryCache:
    """Returns singleton instance of the specialist query result cache."""
    global _result_cache
    if _result_cache is None:
        with _cache_init_lock:
            if _result_cache is None:
                _result_cache = BoundedTTLMemoryCache(
                    max_entries=MAX_RESULT_CACHE_ENTRIES,
                    ttl_seconds=RESULT_CACHE_TTL_SECONDS,
                    name="query_result_cache"
                )
    return _result_cache


def clear_all_caches() -> None:
    """Utility to clear all in-process caches (useful between test runs)."""
    if _image_cache is not None:
        _image_cache.clear()
    if _result_cache is not None:
        _result_cache.clear()
