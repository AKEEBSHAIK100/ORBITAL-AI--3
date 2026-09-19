/**
 * db/migrate.ts — Idempotent migration runner (Phase 4)
 *
 * Runs all SQL migration files in order at server startup.
 * Non-fatal: if DB is unavailable, logs a warning and returns false.
 * Uses IF NOT EXISTS / ON CONFLICT DO NOTHING so re-runs are safe.
 */
import 'dotenv/config'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { getPool } from './pool.js'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const MIGRATIONS_DIR = path.join(__dirname, 'migrations')

/** Track whether migrations have already run this process lifetime. */
let _ran = false

/**
 * Runs all *.sql files in db/migrations/ in lexicographic order.
 * Idempotent — safe to call on every server startup.
 * Returns true if DB is available and migrations ran, false otherwise.
 */
export async function runMigrations(): Promise<boolean> {
  if (_ran) return true
  const pool = getPool()
  if (!pool) {
    console.warn('[SatQuery][DB] Skipping migrations — no database configured.')
    return false
  }

  let files: string[]
  try {
    files = fs
      .readdirSync(MIGRATIONS_DIR)
      .filter((f) => f.endsWith('.sql'))
      .sort()
  } catch {
    console.warn('[SatQuery][DB] Migrations directory not found — skipping.')
    return false
  }

  const client = await pool.connect().catch((err: Error) => {
    console.error('[SatQuery][DB] Migration: could not connect —', err.message)
    return null
  })
  if (!client) return false

  try {
    // Ensure schema_migrations tracking table exists
    await client.query(`
      CREATE TABLE IF NOT EXISTS schema_migrations (
        version TEXT PRIMARY KEY,
        applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
      )
    `)

    for (const file of files) {
      const version = path.basename(file, '.sql')

      const existing = await client.query(
        'SELECT 1 FROM schema_migrations WHERE version = $1',
        [version]
      )
      if ((existing.rowCount ?? 0) > 0) {
        console.log(`[SatQuery][DB] Migration ${version}: already applied, skipping.`)
        continue
      }

      const sql = fs.readFileSync(path.join(MIGRATIONS_DIR, file), 'utf8')
      await client.query('BEGIN')
      try {
        await client.query(sql)
        await client.query(
          'INSERT INTO schema_migrations (version) VALUES ($1) ON CONFLICT DO NOTHING',
          [version]
        )
        await client.query('COMMIT')
        console.log(`[SatQuery][DB] Migration ${version}: applied.`)
      } catch (err) {
        await client.query('ROLLBACK')
        console.error(`[SatQuery][DB] Migration ${version} FAILED (rolled back):`, (err as Error).message)
        // Continue — don't crash startup over a seeding failure
      }
    }

    _ran = true
    console.log('[SatQuery][DB] All migrations complete.')
    return true
  } catch (err) {
    console.error('[SatQuery][DB] Migration runner error (non-fatal):', (err as Error).message)
    return false
  } finally {
    client.release()
  }
}
