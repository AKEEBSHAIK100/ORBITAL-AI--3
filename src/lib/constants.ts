// ─── Cost-aware constants ─────────────────────────────────────────────────────
// Single source of truth for model selection, token budgets, and usage caps.
// Never hardcode these values elsewhere — always import from here.

/** The vision model used for all AI calls. Change once here to affect the entire stack. */
export const MODEL = 'remote-sensing-specialists'

/** Max output tokens for single-image analysis. The JSON schema fits comfortably under 700. */
export const MAX_TOKENS_ANALYZE = 700

/** Max output tokens for before/after comparison. Slightly higher to accommodate change_regions. */
export const MAX_TOKENS_COMPARE = 800

/**
 * Parse a raw env value into a valid positive integer session limit.
 *
 * Why not just `Number(import.meta.env.VITE_SESSION_CALL_LIMIT ?? 100)`?
 *
 * In a Vite build where VITE_SESSION_CALL_LIMIT is absent, the env object
 * exposes the key with the value `""` (empty string) — NOT undefined/null.
 * The `??` operator only catches null/undefined, so `"" ?? 100` evaluates
 * to `""`, and `Number("") === 0`.  With a limit of 0 the guard condition
 * `sessionCallCount >= SESSION_CALL_LIMIT` becomes `0 >= 0 → true` and the
 * guard fires immediately on a fresh session before any query is submitted.
 *
 * This helper defends against: undefined, null, "", "0", NaN, and negatives.
 * Any invalid or non-positive value resolves to the safe application default.
 */
function parseSessionLimit(raw: string | undefined | null, defaultLimit = 100): number {
  const n = Number(raw)
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : defaultLimit
}

/**
 * Soft per-session request guard shown to users.
 * Override via VITE_SESSION_CALL_LIMIT env var (build-time, client-side).
 * Configurable for live demo and testing environments.
 * Guaranteed to be a positive integer; falls back to 100 for any invalid value.
 */
export const SESSION_CALL_LIMIT = parseSessionLimit(
  import.meta.env?.VITE_SESSION_CALL_LIMIT,
)

