import React, { useState, useEffect } from 'react'

export interface CookiePreferences {
  essential: boolean
  timestamp: string
  policyVersion: string
}

const STORAGE_KEY = 'satquery_cookie_preferences'
const POLICY_VERSION = '2026.1-strict-essential-only'

export function getStoredCookiePreferences(): CookiePreferences | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return null
    return JSON.parse(raw) as CookiePreferences
  } catch {
    return null
  }
}

export function saveCookiePreferences(): CookiePreferences {
  const full: CookiePreferences = {
    essential: true,
    timestamp: new Date().toISOString(),
    policyVersion: POLICY_VERSION,
  }
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(full))
  } catch {
    // ignore
  }
  return full
}

interface CookieConsentProps {
  onOpenPolicy?: () => void
}

export default function CookieConsentBanner({ onOpenPolicy }: CookieConsentProps) {
  const [visible, setVisible] = useState(false)
  const [showDetails, setShowDetails] = useState(false)

  useEffect(() => {
    const existing = getStoredCookiePreferences()
    if (!existing) {
      const timer = setTimeout(() => setVisible(true), 800)
      return () => clearTimeout(timer)
    }
  }, [])

  const handleAcknowledge = () => {
    saveCookiePreferences()
    setVisible(false)
    setShowDetails(false)
  }

  if (!visible && !showDetails) return null

  return (
    <>
      {/* Strict Essential-Only Bottom Banner */}
      {visible && !showDetails && (
        <aside
          role="region"
          aria-label="Required Data Notice"
          className="fixed bottom-4 left-4 right-4 md:left-auto md:right-6 md:max-w-xl z-[9000] p-5 rounded-2xl border shadow-2xl backdrop-blur-xl transition-all duration-300"
          style={{
            background: 'rgba(7, 17, 31, 0.96)',
            borderColor: 'rgba(56, 189, 248, 0.28)',
            boxShadow: '0 20px 50px rgba(0, 0, 0, 0.75), 0 0 30px rgba(32, 217, 255, 0.08)',
          }}
        >
          <div className="flex items-start gap-3.5 mb-3">
            <div
              className="w-8 h-8 rounded-xl flex items-center justify-center shrink-0 text-base"
              style={{ background: 'rgba(53, 224, 184, 0.15)', border: '1px solid rgba(53, 224, 184, 0.35)', color: '#35E0B8' }}
            >
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
                <path d="M7 11V7a5 5 0 0 1 10 0v4" />
              </svg>
            </div>
            <div>
              <h2 className="text-sm font-bold text-slate-100 mb-1" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
                Strictly Required Data Policy
              </h2>
              <p className="text-xs text-slate-300 leading-relaxed">
                SatQuery AI collects <strong className="text-slate-100">strictly required technical data only</strong> (session ID and rate-limiting counters)
                essential for satellite inference workflows. We do <strong className="text-emerald-400">not</strong> collect tracking, advertising,
                or third-party analytical cookies.
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center justify-between gap-2.5 pt-2 border-t border-slate-800">
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setShowDetails(true)}
                className="text-[11px] font-mono font-medium text-cyan-400 hover:text-cyan-300 transition-colors underline cursor-pointer"
              >
                View Required Data
              </button>
              <span className="text-slate-600 text-xs">·</span>
              {onOpenPolicy && (
                <button
                  type="button"
                  onClick={onOpenPolicy}
                  className="text-[11px] font-mono text-slate-400 hover:text-slate-200 transition-colors cursor-pointer"
                >
                  Full Cookie Policy
                </button>
              )}
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleAcknowledge}
                className="px-4 py-1.5 rounded-lg text-xs font-semibold text-slate-950 bg-cyan-400 hover:bg-cyan-300 transition-all shadow-md shadow-cyan-950 cursor-pointer"
              >
                Acknowledge Required Only
              </button>
            </div>
          </div>
        </aside>
      )}

      {/* Details Modal on Exactly What Required Data is Stored */}
      {showDetails && (
        <div
          className="fixed inset-0 z-[9990] flex items-center justify-center p-4"
          style={{ background: 'rgba(2, 8, 16, 0.85)', backdropFilter: 'blur(8px)' }}
          onClick={() => setShowDetails(false)}
        >
          <div
            role="dialog"
            aria-modal="true"
            aria-labelledby="required-data-title"
            className="w-full max-w-lg rounded-2xl p-6 border shadow-2xl space-y-4"
            style={{
              background: '#07111F',
              borderColor: 'rgba(56, 189, 248, 0.28)',
              boxShadow: '0 25px 60px rgba(0,0,0,0.85)',
            }}
            onClick={e => e.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center gap-2">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-emerald-400 shrink-0">
                  <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                </svg>
                <h3 id="required-data-title" className="text-base font-bold text-slate-100" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
                  Strictly Required Data Disclosures
                </h3>
              </div>
              <button
                type="button"
                onClick={() => setShowDetails(false)}
                className="text-slate-400 hover:text-slate-200 text-sm font-mono cursor-pointer"
              >
                ✕
              </button>
            </div>

            <p className="text-xs text-slate-300 leading-relaxed">
              In accordance with our minimal data collection principle, the following is an exhaustive list of all items stored in your browser:
            </p>

            <div className="space-y-2 text-xs font-mono">
              <div className="p-3 rounded-xl border border-slate-800 bg-slate-900/70 space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-cyan-300 font-bold">1. Session Identifier</span>
                  <span className="text-[10px] text-emerald-400 font-semibold">Strictly Required</span>
                </div>
                <p className="text-[11px] font-sans text-slate-300">
                  A random UUID generated per browser tab to associate multi-turn queries with your uploaded satellite scene and enforce the 18-call rate limit.
                </p>
              </div>

              <div className="p-3 rounded-xl border border-slate-800 bg-slate-900/70 space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-cyan-300 font-bold">2. Consent Acknowledgment</span>
                  <span className="text-[10px] text-emerald-400 font-semibold">Strictly Required</span>
                </div>
                <p className="text-[11px] font-sans text-slate-300">
                  Records that you have acknowledged the required data notice so the banner does not interrupt subsequent visits.
                </p>
              </div>

              <div className="p-3 rounded-xl border border-slate-800 bg-slate-900/70 space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-cyan-300 font-bold">3. Active Analysis Mode</span>
                  <span className="text-[10px] text-emerald-400 font-semibold">Strictly Required</span>
                </div>
                <p className="text-[11px] font-sans text-slate-300">
                  Preserves whether you are viewing Single Image, Optical–SAR Fusion, or Bi-Temporal mode during tab switching.
                </p>
              </div>
            </div>

            <div className="p-3 rounded-xl border border-emerald-500/30 bg-emerald-500/10 text-emerald-200 text-xs flex items-start gap-2 leading-relaxed">
              <span className="text-emerald-400 mt-0.5">✓</span>
              <span>
                <strong>Zero Tracking Policy:</strong> We do not set analytics cookies, third-party advertising IDs, or behavioral profiling trackers.
              </span>
            </div>

            <div className="flex items-center justify-end gap-2 pt-3 border-t border-slate-800">
              <button
                type="button"
                onClick={handleAcknowledge}
                className="px-5 py-2 rounded-xl text-xs font-semibold text-slate-950 bg-cyan-400 hover:bg-cyan-300 transition-all cursor-pointer shadow-md shadow-cyan-950"
              >
                Confirm & Accept Required Only
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
