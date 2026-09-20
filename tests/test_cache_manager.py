"""
Tests for cache_manager.py — Checkpoint 3.

Covers:
- BoundedTTLMemoryCache: LRU eviction, TTL expiry, thread-safety, deep-copy isolation
- Hash utilities: compute_image_hash, compute_paired_image_hash, get_adapter_fingerprint
- Cache key builder: build_cache_key, normalize_query_text
- Singleton accessors: get_image_preprocessing_cache, get_query_result_cache, clear_all_caches

All tests are pure Python — no model loading, no GPU, no file I/O beyond tmp_path fixtures.
"""

import copy
import threading
import time

import numpy as np
import pytest

from backend.services.cache_manager import (
    BoundedTTLMemoryCache,
    build_cache_key,
    clear_all_caches,
    compute_image_hash,
    compute_paired_image_hash,
    get_adapter_fingerprint,
    get_image_preprocessing_cache,
    get_query_result_cache,
    normalize_query_text,
)


# ── BoundedTTLMemoryCache ────────────────────────────────────────────────────

class TestBoundedTTLMemoryCache:
    def setup_method(self):
        clear_all_caches()

    def _make_cache(self, max_entries=5, ttl_seconds=60.0):
        return BoundedTTLMemoryCache(max_entries=max_entries, ttl_seconds=ttl_seconds, name="test")

    def test_basic_set_get(self):
        cache = self._make_cache()
        cache.set("k1", {"val": 42})
        result = cache.get("k1")
        assert result == {"val": 42}

    def test_miss_returns_none(self):
        cache = self._make_cache()
        assert cache.get("nonexistent") is None

    def test_lru_eviction(self):
        cache = self._make_cache(max_entries=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        cache.get("a")            # promote 'a'
        cache.set("d", 4)         # evicts 'b'
        assert cache.get("b") is None
        assert cache.get("a") == 1
        assert cache.get("c") == 3
        assert cache.get("d") == 4

    def test_ttl_expiry(self):
        cache = BoundedTTLMemoryCache(max_entries=10, ttl_seconds=0.05, name="ttl_test")
        cache.set("k", "value")
        assert cache.get("k") == "value"
        time.sleep(0.1)
        assert cache.get("k") is None

    def test_update_existing_key_refreshes_lru(self):
        cache = self._make_cache(max_entries=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("a", 99)        # refresh 'a'
        cache.set("c", 3)         # evicts 'b'
        assert cache.get("b") is None
        assert cache.get("a") == 99
        assert cache.get("c") == 3

    def test_deep_copy_isolation_on_store(self):
        cache = self._make_cache()
        original = {"nested": [1, 2, 3]}
        cache.set("k", original)
        original["nested"].append(99)
        result = cache.get("k")
        assert result["nested"] == [1, 2, 3]

    def test_deep_copy_isolation_on_retrieve(self):
        cache = self._make_cache()
        cache.set("k", {"x": [10, 20]})
        retrieved = cache.get("k")
        retrieved["x"].append(30)
        second = cache.get("k")
        assert second["x"] == [10, 20]

    def test_clear(self):
        cache = self._make_cache()
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert len(cache) == 0
        assert cache.get("a") is None

    def test_stats(self):
        cache = self._make_cache(max_entries=10, ttl_seconds=60.0)
        cache.set("k", "v")
        cache.get("k")
        cache.get("missing")
        s = cache.stats()
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert s["current_entries"] == 1
        assert s["max_entries"] == 10

    def test_thread_safety(self):
        cache = self._make_cache(max_entries=50)
        errors = []

        def writer(n):
            try:
                for i in range(20):
                    cache.set(f"k{n}_{i}", {"data": n * 100 + i})
            except Exception as e:
                errors.append(str(e))

        def reader(n):
            try:
                for i in range(20):
                    cache.get(f"k{n}_{i}")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(5)]
        threads += [threading.Thread(target=reader, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []

    def test_max_entries_minimum_one(self):
        cache = BoundedTTLMemoryCache(max_entries=0, ttl_seconds=60.0, name="min_test")
        cache.set("k", "v")
        assert cache.get("k") == "v"
        assert len(cache) == 1


# ── Hash Utilities ───────────────────────────────────────────────────────────

class TestComputeImageHash:
    def test_bytes_consistent(self):
        data = b"hello world"
        assert compute_image_hash(data) == compute_image_hash(data)

    def test_different_bytes_differ(self):
        assert compute_image_hash(b"abc") != compute_image_hash(b"xyz")

    def test_numpy_consistent(self):
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        assert compute_image_hash(arr) == compute_image_hash(arr.copy())

    def test_numpy_different_content_differs(self):
        a = np.zeros((10, 10, 3), dtype=np.uint8)
        b = np.ones((10, 10, 3), dtype=np.uint8)
        assert compute_image_hash(a) != compute_image_hash(b)

    def test_numpy_different_shape_differs(self):
        a = np.zeros((10, 20, 3), dtype=np.uint8)
        b = np.zeros((20, 10, 3), dtype=np.uint8)
        assert compute_image_hash(a) != compute_image_hash(b)

    def test_numpy_different_dtype_differs(self):
        a = np.zeros((10, 10), dtype=np.uint8)
        b = np.zeros((10, 10), dtype=np.float32)
        assert compute_image_hash(a) != compute_image_hash(b)

    def test_none_returns_none_string(self):
        assert compute_image_hash(None) == "none"

    def test_string_input(self):
        h = compute_image_hash("/path/to/image.tif")
        assert isinstance(h, str) and len(h) == 64


class TestComputePairedImageHash:
    def test_order_matters(self):
        a, b = b"image_a", b"image_b"
        assert compute_paired_image_hash(a, b) != compute_paired_image_hash(b, a)

    def test_same_pair_consistent(self):
        a, b = b"img1", b"img2"
        assert compute_paired_image_hash(a, b) == compute_paired_image_hash(a, b)

    def test_contains_pair_prefix(self):
        assert compute_paired_image_hash(b"x", b"y").startswith("pair:")


class TestGetAdapterFingerprint:
    def test_no_adapter_returns_none(self):
        assert get_adapter_fingerprint(None) == "none"
        assert get_adapter_fingerprint("") == "none"

    def test_name_only_no_path(self):
        assert get_adapter_fingerprint("my_adapter") == "my_adapter"

    def test_nonexistent_path(self, tmp_path):
        fp = get_adapter_fingerprint("adapter", tmp_path / "nonexistent_dir")
        assert "not_found" in fp

    def test_existing_dir_without_weights(self, tmp_path):
        d = tmp_path / "my_adapter"
        d.mkdir()
        fp = get_adapter_fingerprint("my_adapter", d)
        assert "my_adapter" in fp

    def test_existing_dir_with_weights(self, tmp_path):
        d = tmp_path / "my_adapter"
        d.mkdir()
        w = d / "adapter_model.safetensors"
        w.write_bytes(b"fake weights data")
        fp = get_adapter_fingerprint("my_adapter", d)
        assert "my_adapter" in fp
        assert str(w.stat().st_size) in fp


# ── Query Text Normalization ─────────────────────────────────────────────────

class TestNormalizeQueryText:
    def test_lowercase(self):
        assert normalize_query_text("DESCRIBE The Scene") == "describe the scene"

    def test_strip_whitespace(self):
        assert normalize_query_text("  hello  ") == "hello"

    def test_collapse_internal_whitespace(self):
        assert normalize_query_text("too   many   spaces") == "too many spaces"

    def test_empty_string(self):
        assert normalize_query_text("") == ""

    def test_none_like_empty(self):
        assert normalize_query_text(None) == ""


# ── Cache Key Builder ────────────────────────────────────────────────────────

class TestBuildCacheKey:
    def _key(self, **overrides):
        defaults = dict(
            image_identity="img_hash_abc",
            query="What is the land cover?",
            task="land_cover",
            model_id="resnet50-s2-v0.2.0",
            model_version="v0.2.0",
            adapter_identity="none",
            parameters={"top_k": 5},
        )
        defaults.update(overrides)
        return build_cache_key(**defaults)

    def test_deterministic(self):
        assert self._key() == self._key()

    def test_different_image_hash_differs(self):
        assert self._key(image_identity="hash1") != self._key(image_identity="hash2")

    def test_different_query_differs(self):
        assert self._key(query="query A") != self._key(query="query B")

    def test_query_normalized(self):
        k1 = self._key(query="What IS the  Land Cover?")
        k2 = self._key(query="what is the land cover?")
        assert k1 == k2

    def test_different_task_differs(self):
        assert self._key(task="land_cover") != self._key(task="building_detection")

    def test_different_model_differs(self):
        assert self._key(model_id="modelA") != self._key(model_id="modelB")

    def test_different_adapter_differs(self):
        assert self._key(adapter_identity="adp1") != self._key(adapter_identity="adp2")

    def test_parameter_order_irrelevant(self):
        k1 = self._key(parameters={"a": 1, "b": 2})
        k2 = self._key(parameters={"b": 2, "a": 1})
        assert k1 == k2

    def test_supporting_keys_excluded(self):
        k1 = self._key(parameters={"top_k": 5})
        k2 = self._key(parameters={"top_k": 5, "supporting_land_cover_result": {"big": "data"}})
        assert k1 == k2

    def test_returns_sha256_hex(self):
        key = self._key()
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)


# ── Singleton Accessors & Clear ──────────────────────────────────────────────

class TestSingletons:
    def setup_method(self):
        clear_all_caches()

    def test_image_cache_is_singleton(self):
        c1 = get_image_preprocessing_cache()
        c2 = get_image_preprocessing_cache()
        assert c1 is c2

    def test_result_cache_is_singleton(self):
        c1 = get_query_result_cache()
        c2 = get_query_result_cache()
        assert c1 is c2

    def test_clear_all_caches_empties_both(self):
        img_cache = get_image_preprocessing_cache()
        result_cache = get_query_result_cache()
        img_cache.set("k", "v")
        result_cache.set("k", "v")
        assert len(img_cache) == 1
        assert len(result_cache) == 1
        clear_all_caches()
        assert len(img_cache) == 0
        assert len(result_cache) == 0
