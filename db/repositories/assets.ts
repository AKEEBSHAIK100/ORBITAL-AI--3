/**
 * db/repositories/assets.ts — Image metadata storage (Phase 4)
 *
 * Stores SHA-256, storage key, mime type, and expiry ONLY.
 * Never stores image bytes or base64 in PostgreSQL.
 * Falls back gracefully when DB is unavailable.
 */
import { randomUUID } from 'node:crypto'
import crypto from 'node:crypto'
import { dbQuery } from '../pool.js'

const ASSET_TTL_MINUTES = Number(process.env.SESSION_TTL_MINUTES ?? 30)

export interface AssetRecord {
  id: string
  session_id: string | null
  storage_key: string
  sha256: string
  mime_type: string
  size_bytes: number | null
  modality: string
  sensor_hint: string | null
  expires_at: string
}

/**
 * Computes the SHA-256 hash of a data URL's base64 payload.
 * Handles both full data URLs and raw base64 strings.
 */
export function computeImageSha256(dataUrl: string): string {
  const b64 = dataUrl.includes(',') ? dataUrl.split(',')[1] : dataUrl
  return crypto.createHash('sha256').update(b64 ?? dataUrl).digest('hex')
}

/**
 * Extracts mime type from a data URL. Defaults to image/jpeg.
 */
export function extractMimeType(dataUrl: string): string {
  const match = dataUrl.match(/^data:(image\/[a-z+]+);base64,/)
  return match?.[1] ?? 'image/jpeg'
}

/**
 * Estimates base64 payload byte size from a data URL string length.
 */
export function estimateSizeBytes(dataUrl: string): number {
  const b64 = dataUrl.includes(',') ? dataUrl.split(',')[1] : dataUrl
  return Math.floor((b64?.length ?? 0) * 0.75)
}

/**
 * Registers an image asset in the DB (metadata only).
 * If the SHA-256 already exists for this session, returns the existing record.
 * Returns the asset ID or null if DB unavailable.
 */
export async function registerAsset(
  dataUrl: string,
  sessionId: string | null,
  modality: 'optical' | 'sar' | 'multispectral' = 'optical',
  sensorHint?: string
): Promise<string | null> {
  const sha256 = computeImageSha256(dataUrl)
  const mimeType = extractMimeType(dataUrl)
  const sizeBytes = estimateSizeBytes(dataUrl)

  // Check if already registered for this session
  if (sessionId) {
    const existing = await dbQuery<{ id: string }>(`
      SELECT id FROM assets
      WHERE sha256 = $1 AND session_id = $2 AND expires_at > NOW()
      LIMIT 1
    `, [sha256, sessionId])
    if ((existing?.rowCount ?? 0) > 0) {
      return existing!.rows[0].id
    }
  }

  const assetId = randomUUID()
  const storageKey = `session/${sessionId ?? 'anon'}/${sha256.slice(0, 16)}`

  const result = await dbQuery(`
    INSERT INTO assets
      (id, session_id, storage_key, sha256, mime_type, size_bytes, modality, sensor_hint, expires_at)
    VALUES
      ($1, $2, $3, $4, $5, $6, $7, $8, NOW() + ($9 || ' minutes')::INTERVAL)
    ON CONFLICT DO NOTHING
    RETURNING id
  `, [assetId, sessionId, storageKey, sha256, mimeType, sizeBytes, modality, sensorHint ?? null, ASSET_TTL_MINUTES])

  return result?.rows[0]?.id ?? assetId
}

/**
 * Looks up an asset by SHA-256 and session. Returns null if not found or DB unavailable.
 */
export async function findAssetBySha256(sha256: string, sessionId?: string): Promise<AssetRecord | null> {
  const result = await dbQuery<AssetRecord>(`
    SELECT * FROM assets
    WHERE sha256 = $1
      ${sessionId ? 'AND session_id = $2' : ''}
      AND expires_at > NOW()
    ORDER BY created_at DESC
    LIMIT 1
  `, sessionId ? [sha256, sessionId] : [sha256])
  return result?.rows[0] ?? null
}

/**
 * Purges assets with expired TTL. Safe to call periodically.
 */
export async function pruneExpiredAssets(): Promise<void> {
  await dbQuery(`DELETE FROM assets WHERE expires_at < NOW() - INTERVAL '5 minutes'`)
}
