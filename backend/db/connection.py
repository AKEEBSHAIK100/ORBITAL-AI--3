"""
backend/db/connection.py — Optional PostgreSQL connection for FastAPI (Phase 4)

Connects only when DATABASE_URL is set.
All usage is wrapped in try/except — never crashes FastAPI.
asyncpg is optional; psycopg2 is used as a synchronous fallback.
"""
from __future__ import annotations
import os
import logging

logger = logging.getLogger(__name__)

_pool = None
_sync_conn = None

DATABASE_URL = os.getenv("DATABASE_URL", "")


async def get_async_pool():
    """
    Returns an asyncpg connection pool, or None if unavailable.
    Call this from FastAPI lifespan to warm the pool.
    """
    global _pool
    if not DATABASE_URL:
        return None
    if _pool is not None:
        return _pool
    try:
        import asyncpg  # type: ignore
        _pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=1,
            max_size=int(os.getenv("DB_POOL_MAX", "5")),
            command_timeout=float(os.getenv("DB_CONNECTION_TIMEOUT_MS", "5000")) / 1000,
        )
        logger.info("[SatQuery][DB-Python] asyncpg pool initialized.")
        return _pool
    except ImportError:
        logger.warning("[SatQuery][DB-Python] asyncpg not installed — DB writes from Python backend disabled.")
        return None
    except Exception as e:
        logger.warning(f"[SatQuery][DB-Python] Pool init failed (non-fatal): {e}")
        return None


def get_sync_conn():
    """
    Returns a psycopg2 connection for synchronous use, or None.
    """
    global _sync_conn
    if not DATABASE_URL:
        return None
    if _sync_conn is not None:
        try:
            _sync_conn.cursor().execute("SELECT 1")
            return _sync_conn
        except Exception:
            _sync_conn = None
    try:
        import psycopg2  # type: ignore
        _sync_conn = psycopg2.connect(DATABASE_URL)
        return _sync_conn
    except ImportError:
        return None
    except Exception as e:
        logger.warning(f"[SatQuery][DB-Python] psycopg2 connect failed (non-fatal): {e}")
        return None


async def db_execute(sql: str, *args) -> bool:
    """
    Executes a SQL statement asynchronously. Returns True on success, False otherwise.
    """
    pool = await get_async_pool()
    if pool is None:
        return False
    try:
        await pool.execute(sql, *args)
        return True
    except Exception as e:
        logger.warning(f"[SatQuery][DB-Python] Execute error (non-fatal): {e}")
        return False


async def db_fetch(sql: str, *args) -> list:
    """
    Fetches rows asynchronously. Returns empty list on failure.
    """
    pool = await get_async_pool()
    if pool is None:
        return []
    try:
        return await pool.fetch(sql, *args)
    except Exception as e:
        logger.warning(f"[SatQuery][DB-Python] Fetch error (non-fatal): {e}")
        return []
