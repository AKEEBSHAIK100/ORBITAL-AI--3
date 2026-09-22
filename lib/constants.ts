// ─── Server-side cost constants ───────────────────────────────────────────────
// Mirrors src/lib/constants.ts but uses process.env (no import.meta.env).
// Used by server.ts and api/*.ts — never import the src version from server code.

/** The vision model used for all AI calls. Change once here to affect all server handlers. */
export const MODEL = process.env.OPENAI_VISION_MODEL ?? 'claude-sonnet-5'

/** Max output tokens for single-image analysis. */
export const MAX_TOKENS_ANALYZE = 700

/** Max output tokens for before/after comparison. */
export const MAX_TOKENS_COMPARE = 800

/** Per-session soft request guard enforced server-side (backs up the client-side guard). */
export const SESSION_CALL_LIMIT = Number(process.env.SESSION_CALL_LIMIT ?? 100)

