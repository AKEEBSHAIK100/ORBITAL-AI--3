// ─── Server-side cost constants ───────────────────────────────────────────────
// Mirrors src/lib/constants.ts but uses process.env (no import.meta.env).
// Used by server.ts and api/*.ts — never import the src version from server code.

/** The vision model used for all AI calls. Change once here to affect all server handlers. */
export const MODEL = process.env.OPENAI_VISION_MODEL ?? 'gpt-5.6-luna'

/** Max output tokens for single-image analysis. */
export const MAX_TOKENS_ANALYZE = 700

/** Max output tokens for before/after comparison. */
export const MAX_TOKENS_COMPARE = 800

/**
 * Parse a raw env string into a valid positive integer session limit.
 * Defends against: undefined, null, "" (empty), "0", NaN, and negatives.
 * Any invalid or non-positive value resolves to the safe application default.
 * Matches the identical helper in src/lib/constants.ts.
 */
function parseSessionLimit(raw: string | undefined | null, defaultLimit = 100): number {
  const n = Number(raw)
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : defaultLimit
}

/** Per-session soft request guard enforced server-side (backs up the client-side guard). */
export const SESSION_CALL_LIMIT = parseSessionLimit(process.env.SESSION_CALL_LIMIT)


