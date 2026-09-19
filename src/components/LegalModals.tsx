import React from 'react'

export type LegalModalType = 'privacy' | 'terms' | 'cookies'

interface LegalModalsProps {
  isOpen: boolean
  activeTab: LegalModalType
  onClose: () => void
  onSelectTab: (tab: LegalModalType) => void
  onOpenCookieSettings?: () => void
}

export default function LegalModals({
  isOpen,
  activeTab,
  onClose,
  onSelectTab,
  onOpenCookieSettings,
}: LegalModalsProps) {
  if (!isOpen) return null

  return (
    <div
      className="fixed inset-0 z-[9980] flex items-center justify-center p-4 sm:p-6"
      style={{ background: 'rgba(2, 8, 16, 0.88)', backdropFilter: 'blur(12px)' }}
      onClick={onClose}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="legal-modal-title"
        className="w-full max-w-4xl max-h-[88vh] flex flex-col rounded-2xl border shadow-2xl overflow-hidden"
        style={{
          background: '#07111F',
          borderColor: 'rgba(56, 189, 248, 0.25)',
          boxShadow: '0 25px 60px rgba(0, 0, 0, 0.85), 0 0 40px rgba(32, 217, 255, 0.08)',
        }}
        onClick={e => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-[#050D18]">
          <div className="flex items-center gap-3">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 shadow-[0_0_8px_#20D9FF]" />
            <h2 id="legal-modal-title" className="text-base font-bold text-slate-100 font-mono tracking-wide">
              COOKIE & STORAGE DISCLOSURES
            </h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="w-8 h-8 rounded-lg flex items-center justify-center text-slate-400 hover:text-white hover:bg-slate-800 transition-colors cursor-pointer text-sm"
          >
            ✕
          </button>
        </div>

        {/* Tab switcher */}
        <div className="flex border-b border-slate-800 bg-[#081322] px-6 gap-2 pt-2">
          <button
            type="button"
            onClick={() => onSelectTab('cookies')}
            className="px-4 py-2.5 text-xs font-mono font-medium rounded-t-lg transition-all cursor-pointer border-b-2"
            style={{
              borderColor: activeTab === 'cookies' ? '#20D9FF' : 'transparent',
              color: activeTab === 'cookies' ? '#20D9FF' : '#94A3B8',
              background: activeTab === 'cookies' ? 'rgba(32, 217, 255, 0.08)' : 'transparent',
            }}
          >
            Cookie & Storage Policy
          </button>
          <a
            href="/privacy"
            className="px-4 py-2.5 text-xs font-mono font-medium rounded-t-lg transition-all cursor-pointer text-slate-400 hover:text-cyan-300"
          >
            Privacy Policy ↗
          </a>
          <a
            href="/terms"
            className="px-4 py-2.5 text-xs font-mono font-medium rounded-t-lg transition-all cursor-pointer text-slate-400 hover:text-cyan-300"
          >
            Terms & Conditions ↗
          </a>
        </div>

        {/* Modal Body Content (Scrollable) */}
        <div className="p-6 overflow-y-auto space-y-6 text-slate-300 text-sm leading-relaxed max-h-[64vh]">
          {activeTab === 'privacy' && (
            <div className="py-8 text-center space-y-4">
              <p className="text-slate-300">The SatQuery AI Privacy Policy is now hosted on its own dedicated page.</p>
              <a
                href="/privacy"
                className="inline-block px-5 py-2.5 rounded-xl font-mono text-xs font-semibold bg-cyan-400 text-slate-950 hover:bg-cyan-300 transition-all"
              >
                Go to Privacy Policy Page →
              </a>
            </div>
          )}

          {activeTab === 'terms' && (
            <div className="py-8 text-center space-y-4">
              <p className="text-slate-300">The SatQuery AI Terms & Conditions are now hosted on their own dedicated page.</p>
              <a
                href="/terms"
                className="inline-block px-5 py-2.5 rounded-xl font-mono text-xs font-semibold bg-cyan-400 text-slate-950 hover:bg-cyan-300 transition-all"
              >
                Go to Terms & Conditions Page →
              </a>
            </div>
          )}

          {activeTab === 'cookies' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-xl font-bold text-slate-100 mb-2" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
                  Strict Required-Data & Storage Policy
                </h3>
                <p className="text-xs font-mono text-slate-400">
                  Strictly required data only · Zero advertising or behavioral tracking
                </p>
              </div>

              <section className="space-y-2">
                <h4 className="font-semibold text-slate-100 text-sm">1. Strict Minimal Data Principle</h4>
                <p className="text-xs text-slate-300">
                  SatQuery AI enforces a <strong className="text-slate-100">strictly required data collection standard</strong>. We do not collect non-essential,
                  advertising, or analytical cookies. Rather than third-party tracking scripts, the platform uses local browser storage solely to maintain
                  your active satellite analysis session and prevent quota abuse.
                </p>
              </section>

              <section className="space-y-3">
                <h4 className="font-semibold text-slate-100 text-sm">2. Exhaustive List of Strictly Required Data</h4>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs text-left border-collapse border border-slate-800">
                    <thead>
                      <tr className="bg-slate-900/80 text-slate-200">
                        <th className="p-2 border border-slate-800">Key / Storage Item</th>
                        <th className="p-2 border border-slate-800">Status</th>
                        <th className="p-2 border border-slate-800">Technical Necessity</th>
                        <th className="p-2 border border-slate-800">Lifespan</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800 text-slate-300">
                      <tr>
                        <td className="p-2 font-mono text-cyan-300">satquery_cookie_preferences</td>
                        <td className="p-2 text-emerald-400 font-semibold">Strictly Required</td>
                        <td className="p-2">Records user acknowledgment of required data notice.</td>
                        <td className="p-2">Stored in browser localStorage until cleared</td>
                      </tr>
                      <tr>
                        <td className="p-2 font-mono text-cyan-300">sessionId</td>
                        <td className="p-2 text-emerald-400 font-semibold">Strictly Required</td>
                        <td className="p-2">In-memory session UUID to correlate multi-turn queries with uploaded imagery in temporary server cache.</td>
                        <td className="p-2">Session only (in-memory)</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </section>

              <section className="space-y-2">
                <h4 className="font-semibold text-slate-100 text-sm">3. What We Do NOT Use</h4>
                <p className="text-xs text-slate-300">
                  We explicitly declare that SatQuery AI does <strong className="text-slate-100">not</strong> use:
                </p>
                <ul className="text-xs text-slate-300 list-disc list-inside space-y-1">
                  <li>No advertising or retargeting pixels (Facebook, Google AdSense, etc.)</li>
                  <li>No cross-site tracking or fingerprinting algorithms</li>
                  <li>No sale or commercial distribution of browser behavioral data</li>
                </ul>
              </section>

              <section className="space-y-2">
                <h4 className="font-semibold text-slate-100 text-sm">4. Managing Your Preferences</h4>
                <p className="text-xs text-slate-300">
                  You can change your consent preferences or reset your stored preferences at any time:
                </p>
                {onOpenCookieSettings && (
                  <button
                    type="button"
                    onClick={() => {
                      onClose()
                      onOpenCookieSettings()
                    }}
                    className="mt-2 px-4 py-2 rounded-xl text-xs font-mono font-semibold text-cyan-400 border border-cyan-400/40 bg-cyan-950/30 hover:bg-cyan-900/40 transition-all cursor-pointer flex items-center gap-1.5"
                  >
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <circle cx="12" cy="12" r="3" />
                      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
                    </svg>
                    <span>Open Cookie & Storage Settings</span>
                  </button>
                )}
              </section>
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-3.5 border-t border-slate-800 bg-[#050D18] flex items-center justify-between text-xs font-mono text-slate-400">
          <span>© 2026 SatQuery AI Project · All rights reserved</span>
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 transition-colors cursor-pointer"
          >
            Close Document
          </button>
        </div>
      </div>
    </div>
  )
}
