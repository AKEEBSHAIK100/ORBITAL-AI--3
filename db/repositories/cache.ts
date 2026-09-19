/**
 * db/repositories/cache.ts — PostgreSQL-backed analysis cache (Phase 4)
 *
 * Cache key includes: normalized query + task + image SHA-256 + model/version +
 * agent version + parameters + cache schema version + session ID.
 *
 * Never caches: errors, failed inference, demo/synthetic responses,
 * missing-model responses, or results from expired images.
 *
 * Falls back gracefully — returns null (miss) when DB is unavailable.
 */
import crypto from 'node:crypto'
import { dbQuery } from '../pool.js'

const CACHE_TTL_MINUTES = Number(process.env.CACHE_TTL_MINUTES ?? 30)
const CACHE_SCHEMA_VERSION = 1

export interface CachePayload {
  result: Record<string, unknown>
  task_type: string
  model_id?: string
  agent_version?: string
}

/**
 * Builds a deterministic, normalized cache key from all relevant dimensions.
 */
export function buildCacheKey(opts: {
  query: string
  taskType: string
  imageSha256?: string
  secondaryImageSha256?: string
  modelId?: string
  agentVersion?: string
  parameters?: Record<string, unknown>
  sessionId?: string
}): string {
  const normalized = JSON.stringify({
    q: opts.query.toLowerCase().trim(),
    t: opts.taskType,
    h1: opts.imageSha256 ?? '',
    h2: opts.secondaryImageSha256 ?? '',
    m: opts.modelId ?? '',
    av: opts.agentVersion ?? '',
    p: opts.parameters ? JSON.stringify(opts.parameters) : '',
    sv: CACHE_SCHEMA_VERSION,
    s: opts.sessionId ?? '',
  })
  return crypto.createHash('sha256').update(normalized).digest('hex')
}

/**
 * Returns true if this result should be cached.
 * Does NOT cache errors, demo/synthetic, unavailable-model, or empty results.
 */
export function isCacheable(result: Record<string, unknown>, isDemo: boolean): boolean {
  if (isDemo) return false
  if (!result || typeof result !== 'object') return false
  if (result['error']) return false
  if (result['specialist_unavailable']) return false
  if (result['model_id'] === 'heuristic-node-fallback') return false
  if (result['synthetic_fallback'] === true) return false
  // Must have an answer or result_json to be worth caching
  if (!result['answer'] && !result['result_json']) return false
  return true
}

/**
 * Looks up a cache entry. Returns the cached payload or null (miss/DB unavailable).
 */
export async function getCached(cacheKey: string): Promise<CachePayload | null> {
  const result = await dbQuery<{
    result_json: Record<string, unknown>
    task_type: string
    model_id: string | null
    agent_version: string | null
  }>(`
    UPDATE analysis_cache
    SET hit_count = hit_count + 1
    WHERE cache_key = $1 AND expires_at > NOW()
    RETURNING result_json, task_type, model_id, agent_version
  `, [cacheKey])

  if (!result || (result.rowCount ?? 0) === 0) return null

  const row = result.rows[0]
  return {
    result: row.result_json,
    task_type: row.task_type,
    model_id: row.model_id ?? undefined,
    agent_version: row.agent_version ?? undefined,
  }
}

/**
 * Writes a successful result to the cache. Non-fatal if DB unavailable.
 */
export async function putCached(
  cacheKey: string,
  payload: CachePayload,
  sessionId?: string
): Promise<void> {
  await dbQuery(`
    INSERT INTO analysis_cache
      (cache_key, session_id, task_type, result_json, model_id, agent_version, cache_schema_ver, expires_at)
    VALUES
      ($1, $2, $3, $4, $5, $6, $7, NOW() + ($8 || ' minutes')::INTERVAL)
    ON CONFLICT (cache_key) DO UPDATE
      SET result_json = EXCLUDED.result_json,
          expires_at = EXCLUDED.expires_at,
          hit_count = 0
  `, [
    cacheKey,
    sessionId ?? null,
    payload.task_type,
    JSON.stringify(payload.result),
    payload.model_id ?? null,
    payload.agent_version ?? null,
    CACHE_SCHEMA_VERSION,
    CACHE_TTL_MINUTES,
  ])
}

/**
 * Removes expired cache entries. Safe to call on a schedule.
 */
export async function pruneExpiredCache(): Promise<void> {
  await dbQuery(`DELETE FROM analysis_cache WHERE expires_at < NOW() - INTERVAL '5 minutes'`)
}
