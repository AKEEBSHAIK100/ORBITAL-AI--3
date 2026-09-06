// ─── Cost-aware constants ─────────────────────────────────────────────────────
// Single source of truth for model selection, token budgets, and usage caps.
// Never hardcode these values elsewhere — always import from here.

/** The vision model used for all AI calls. Change once here to affect the entire stack. */
export const MODEL = 'claude-sonnet-5'

/** Max output tokens for single-image analysis. The JSON schema fits comfortably under 700. */
export const MAX_TOKENS_ANALYZE = 700

/** Max output tokens for before/after comparison. Slightly higher to accommodate change_regions. */
export const MAX_TOKENS_COMPARE = 800

/**
 * Soft per-session call limit shown to users.
 * Override via VITE_SESSION_CALL_LIMIT env var (build-time, client-side).
 * Keep high during a live demo; lower during open testing.
 */
export const SESSION_CALL_LIMIT = Number(
  import.meta.env?.VITE_SESSION_CALL_LIMIT ?? 18,
)
