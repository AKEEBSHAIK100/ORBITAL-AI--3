/**
 * db/repositories/sessions.ts — Session + analysis persistence (Phase 4)
 *
 * All functions degrade gracefully to no-op when DB is unavailable.
 * Never throws — callers must not depend on DB success for correctness.
 */
import { randomUUID } from 'node:crypto'
import { dbQuery } from '../pool.js'

const SESSION_TTL_MINUTES = Number(process.env.SESSION_TTL_MINUTES ?? 30)

// ─── Types ────────────────────────────────────────────────────────────────────

export interface HistoryEntry {
  question: string
  answer: string
  task_type?: string
  timestamp?: string
}

export interface SessionRow {
  id: string
  call_count: number
  history: HistoryEntry[]
  expires_at: string
}

export interface TraceStep {
  step: number
  tool: string
  description?: string
  input_summary?: string
  output_summary?: string
  duration_ms?: number
  status?: string
  parameters?: Record<string, unknown>
}

// ─── Session CRUD ─────────────────────────────────────────────────────────────

/**
 * Gets an existing session or creates a new one. Returns null if DB unavailable.
 */
export async function upsertSession(sessionId: string): Promise<SessionRow | null> {
  const result = await dbQuery<SessionRow>(`
    INSERT INTO sessions (id, expires_at)
    VALUES ($1, NOW() + ($2 || ' minutes')::INTERVAL)
    ON CONFLICT (id) DO UPDATE
      SET updated_at = NOW(),
          expires_at = NOW() + ($2 || ' minutes')::INTERVAL
    RETURNING id, call_count, history, expires_at
  `, [sessionId, SESSION_TTL_MINUTES])

  return result?.rows[0] ?? null
}

/**
 * Increments call_count for a session. Returns new count or -1 if DB unavailable.
 */
export async function incrementSessionCalls(sessionId: string): Promise<number> {
  const result = await dbQuery<{ call_count: number }>(`
    UPDATE sessions SET call_count = call_count + 1, updated_at = NOW()
    WHERE id = $1
    RETURNING call_count
  `, [sessionId])
  return result?.rows[0]?.call_count ?? -1
}

/**
 * Appends a history entry to a session. Non-fatal.
 */
export async function appendSessionHistory(
  sessionId: string,
  entry: HistoryEntry
): Promise<void> {
  await dbQuery(`
    UPDATE sessions
    SET history = history || $2::jsonb,
        updated_at = NOW()
    WHERE id = $1
  `, [sessionId, JSON.stringify([entry])])
}

/**
 * Returns the full persisted history for a session.
 */
export async function getSessionHistory(sessionId: string): Promise<HistoryEntry[]> {
  const result = await dbQuery<{ history: HistoryEntry[] }>(`
    SELECT history FROM sessions WHERE id = $1
  `, [sessionId])
  return result?.rows[0]?.history ?? []
}

/**
 * Deletes a session and all cascaded rows (runs, results, traces, assets, cache).
 */
export async function deleteSession(sessionId: string): Promise<boolean> {
  const result = await dbQuery(`DELETE FROM sessions WHERE id = $1`, [sessionId])
  // Also delete analysis_cache rows keyed to this session
  await dbQuery(`DELETE FROM analysis_cache WHERE session_id = $1`, [sessionId])
  return (result?.rowCount ?? 0) > 0
}

/**
 * Deletes sessions expired more than 5 minutes ago. Safe to call periodically.
 */
export async function pruneExpiredSessions(): Promise<void> {
  await dbQuery(`DELETE FROM sessions WHERE expires_at < NOW() - INTERVAL '5 minutes'`)
}

// ─── Analysis Runs ────────────────────────────────────────────────────────────

export interface CreateRunOptions {
  sessionId: string
  taskType: string
  toolId?: string
  isDemo?: boolean
  metadata?: Record<string, unknown>
}

/**
 * Creates an analysis_run row. Returns run ID or null.
 */
export async function createAnalysisRun(opts: CreateRunOptions): Promise<string | null> {
  const runId = randomUUID()
  const result = await dbQuery(`
    INSERT INTO analysis_runs (id, session_id, task_type, tool_id, is_demo, metadata)
    VALUES ($1, $2, $3, $4, $5, $6)
    RETURNING id
  `, [runId, opts.sessionId, opts.taskType, opts.toolId ?? null, opts.isDemo ?? false, JSON.stringify(opts.metadata ?? {})])

  return result?.rows[0]?.id ?? null
}

/**
 * Marks an analysis_run as completed with duration and optional error tag.
 */
export async function completeAnalysisRun(
  runId: string,
  durationMs: number,
  status: 'success' | 'error' | 'unavailable' = 'success',
  errorTag?: string,
  isCached = false
): Promise<void> {
  await dbQuery(`
    UPDATE analysis_runs
    SET status = $2, completed_at = NOW(), duration_ms = $3, error_tag = $4, is_cached = $5
    WHERE id = $1
  `, [runId, status, Math.round(durationMs), errorTag ?? null, isCached])
}

// ─── Results & Inputs ─────────────────────────────────────────────────────────

export async function persistAnalysisResult(
  runId: string,
  taskType: string,
  resultJson: Record<string, unknown>,
  confidence?: string,
  confidencePct?: number
): Promise<void> {
  const id = randomUUID()
  await dbQuery(`
    INSERT INTO analysis_results (id, run_id, task_type, result_json, confidence, confidence_pct)
    VALUES ($1, $2, $3, $4, $5, $6)
  `, [id, runId, taskType, JSON.stringify(resultJson), confidence ?? null, confidencePct ?? null])
}

export async function persistAnalysisInput(
  runId: string,
  inputType: 'query' | 'image',
  queryText?: string,
  assetId?: string,
  modality?: string,
  label?: string
): Promise<void> {
  const id = randomUUID()
  await dbQuery(`
    INSERT INTO analysis_inputs (id, run_id, input_type, query_text, asset_id, modality, label)
    VALUES ($1, $2, $3, $4, $5, $6, $7)
  `, [id, runId, inputType, queryText ?? null, assetId ?? null, modality ?? null, label ?? null])
}

// ─── Trace Steps ──────────────────────────────────────────────────────────────

export async function persistTraceSteps(
  runId: string,
  steps: TraceStep[]
): Promise<void> {
  for (const step of steps) {
    await dbQuery(`
      INSERT INTO agent_trace_steps
        (run_id, step_index, tool, description, input_summary, output_summary, duration_ms, status, parameters)
      VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
    `, [
      runId,
      step.step,
      step.tool,
      step.description ?? null,
      step.input_summary ?? null,
      step.output_summary ?? null,
      step.duration_ms ?? null,
      step.status ?? 'success',
      step.parameters ? JSON.stringify(step.parameters) : null,
    ])
  }
}
