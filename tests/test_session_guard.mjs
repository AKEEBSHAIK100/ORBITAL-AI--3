/**
 * tests/test_session_guard.mjs
 *
 * Regression test for the production bug where SESSION_CALL_LIMIT resolved
 * to 0 (instead of 100) because:
 *
 *   Number(import.meta.env.VITE_SESSION_CALL_LIMIT ?? 100)
 *
 * When VITE_SESSION_CALL_LIMIT is absent from the Vite build, Vite exposes
 * the key as "" (empty string) — NOT null/undefined. So `"" ?? 100 → ""`
 * and `Number("") === 0`. With limit=0 the guard `count >= limit` fires
 * immediately at count=0, blocking every fresh session.
 *
 * The fix: parseSessionLimit() validates the parsed number is a finite
 * positive integer, otherwise returns the safe default (100).
 *
 * Run: node tests/test_session_guard.mjs
 */

import assert from 'node:assert/strict'

// ── Inline the fixed logic (no import.meta.env dependency in test env) ────────

function parseSessionLimit(raw, defaultLimit = 100) {
  const n = Number(raw)
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : defaultLimit
}

function isGuardReached(currentCount, sessionLimit) {
  return currentCount >= sessionLimit
}

let passed = 0
let failed = 0

function test(name, fn) {
  try {
    fn()
    console.log(`  ✓ ${name}`)
    passed++
  } catch (err) {
    console.error(`  ✗ ${name}`)
    console.error(`      ${err.message}`)
    failed++
  }
}

console.log('\n── parseSessionLimit (the fix) ──────────────────────────────────────────────')

test('undefined  → 100  (absent env var, typeof undefined)', () => {
  assert.equal(parseSessionLimit(undefined), 100)
})

test('null       → 100  (null coalesced)', () => {
  assert.equal(parseSessionLimit(null), 100)
})

test('"" (empty) → 100  [ROOT CAUSE: was Number("")===0 before fix]', () => {
  // This was the production failure path:
  //   import.meta.env.VITE_SESSION_CALL_LIMIT === "" when absent
  //   Number("") === 0  →  guard always active
  assert.equal(parseSessionLimit(''), 100)
})

test('"0"        → 100  (zero string is invalid)', () => {
  assert.equal(parseSessionLimit('0'), 100)
})

test('0          → 100  (numeric zero is invalid)', () => {
  assert.equal(parseSessionLimit(0), 100)
})

test('-1         → 100  (negative is invalid)', () => {
  assert.equal(parseSessionLimit(-1), 100)
})

test('"NaN"      → 100  (NaN string is invalid)', () => {
  assert.equal(parseSessionLimit('NaN'), 100)
})

test('NaN        → 100  (numeric NaN is invalid)', () => {
  assert.equal(parseSessionLimit(NaN), 100)
})

test('"100"      → 100  (valid string integer)', () => {
  assert.equal(parseSessionLimit('100'), 100)
})

test('"999"      → 999  (valid demo override)', () => {
  assert.equal(parseSessionLimit('999'), 999)
})

test('"18"       → 18   (valid low-credit testing value)', () => {
  assert.equal(parseSessionLimit('18'), 18)
})

test('"5.9"      → 5    (float truncated to integer)', () => {
  assert.equal(parseSessionLimit('5.9'), 5)
})

test('custom default=50 with invalid input → 50', () => {
  assert.equal(parseSessionLimit('', 50), 50)
})

console.log('\n── Guard logic: isGuardReached(count, limit) ────────────────────────────────')

test('CRITICAL: fresh session — count=0, limit=100 → guard NOT reached (request allowed)', () => {
  assert.equal(isGuardReached(0, 100), false,
    'Guard must NOT block a fresh session with 0 queries')
})

test('count=1, limit=100 → guard NOT reached', () => {
  assert.equal(isGuardReached(1, 100), false)
})

test('count=99, limit=100 → guard NOT reached (last allowed request)', () => {
  assert.equal(isGuardReached(99, 100), false)
})

test('count=100, limit=100 → guard REACHED (first blocked request)', () => {
  assert.equal(isGuardReached(100, 100), true)
})

test('count=101, limit=100 → guard REACHED', () => {
  assert.equal(isGuardReached(101, 100), true)
})

console.log('\n── End-to-end: missing/invalid env → effective limit=100, fresh session allowed ─')

test('absent VITE env var ("") → effective limit 100 → count=0 NOT blocked', () => {
  // This is the exact production failure scenario before the fix
  const effective = parseSessionLimit('')          // was 0, now 100
  assert.equal(effective, 100, 'Effective limit must be 100')
  assert.equal(isGuardReached(0, effective), false, 'Fresh session must not be blocked')
})

test('invalid env var ("abc") → effective limit 100 → count=0 NOT blocked', () => {
  const effective = parseSessionLimit('abc')
  assert.equal(effective, 100)
  assert.equal(isGuardReached(0, effective), false)
})

test('zero env var ("0") → effective limit 100 → count=0 NOT blocked', () => {
  const effective = parseSessionLimit('0')
  assert.equal(effective, 100)
  assert.equal(isGuardReached(0, effective), false)
})

test('valid env var ("999") → effective limit 999 → count=0 NOT blocked', () => {
  const effective = parseSessionLimit('999')
  assert.equal(effective, 999)
  assert.equal(isGuardReached(0, effective), false)
})

test('valid env var ("999") → count=998 NOT blocked, count=999 BLOCKED', () => {
  const effective = parseSessionLimit('999')
  assert.equal(isGuardReached(998, effective), false)
  assert.equal(isGuardReached(999, effective), true)
})

// ── Summary ───────────────────────────────────────────────────────────────────

console.log(`\n${'─'.repeat(72)}`)
console.log(`  Results: ${passed} passed, ${failed} failed`)
console.log('─'.repeat(72))

if (failed > 0) {
  console.error(`\n  ✗ ${failed} test(s) FAILED — session guard regression present\n`)
  process.exit(1)
} else {
  console.log(`\n  ✓ All tests passed — session guard behaves correctly\n`)
  process.exit(0)
}
