/**
 * db/pool.ts — Shared PostgreSQL connection pool (Phase 4)
 *
 * Fails open: if DATABASE_URL is not set or the DB is unreachable, getPool()
 * returns null and all callers must degrade gracefully to in-memory behavior.
 * The application must NEVER crash because the database is absent.
 */
import 'dotenv/config'
import pg from 'pg'

const { Pool } = pg

let _pool: pg.Pool | null = null
let _initAttempted = false

/**
 * Returns the shared Pool, or null if DB is unconfigured / unreachable.
 * Call this lazily — never at module import time so startup is non-blocking.
 */
export function getPool(): pg.Pool | null {
  if (_initAttempted) return _pool
  _initAttempted = true

  const url = process.env.DATABASE_URL
  if (!url) {
    console.warn('[SatQuery][DB] DATABASE_URL not set — running without PostgreSQL (in-memory mode).')
    return null
  }

  try {
    _pool = new Pool({
      connectionString: url,
      max: Number(process.env.DB_POOL_MAX ?? 10),
      idleTimeoutMillis: Number(process.env.DB_IDLE_TIMEOUT_MS ?? 30_000),
      connectionTimeoutMillis: Number(process.env.DB_CONNECTION_TIMEOUT_MS ?? 5_000),
      ssl: url.includes('sslmode=require') ? { rejectUnauthorized: false } : undefined,
    })

    _pool.on('error', (err) => {
      console.error('[SatQuery][DB] Pool error (non-fatal):', err.message)
    })

    console.log('[SatQuery][DB] PostgreSQL pool initialized.')
  } catch (err) {
    console.error('[SatQuery][DB] Failed to create pool (non-fatal):', (err as Error).message)
    _pool = null
  }

  return _pool
}

/**
 * Executes a query with a timeout. Returns null on any error.
 * All callers should treat null as "DB unavailable".
 */
export async function dbQuery<T extends pg.QueryResultRow = pg.QueryResultRow>(
  sql: string,
  params: unknown[] = []
): Promise<pg.QueryResult<T> | null> {
  const pool = getPool()
  if (!pool) return null
  try {
    return await pool.query<T>(sql, params)
  } catch (err) {
    console.error('[SatQuery][DB] Query error (non-fatal):', (err as Error).message, '| SQL:', sql.slice(0, 80))
    return null
  }
}

/**
 * Checks whether the DB is reachable right now.
 */
export async function dbPing(): Promise<boolean> {
  const result = await dbQuery('SELECT 1')
  return result !== null
}

/**
 * Returns pool statistics for the health endpoint.
 */
export function poolStats(): { configured: boolean; total: number; idle: number; waiting: number } {
  const pool = getPool()
  if (!pool) return { configured: false, total: 0, idle: 0, waiting: 0 }
  return {
    configured: true,
    total: pool.totalCount,
    idle: pool.idleCount,
    waiting: pool.waitingCount,
  }
}
