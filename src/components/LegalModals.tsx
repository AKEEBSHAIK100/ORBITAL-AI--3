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
              LEGAL & GOVERNANCE DOCUMENTATION
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
          {[
            { id: 'privacy' as const, label: 'Privacy Policy' },
            { id: 'terms' as const, label: 'Terms & Conditions' },
            { id: 'cookies' as const, label: 'Cookie Policy' },
          ].map(tab => (
            <button
              key={tab.id}
              type="button"
              onClick={() => onSelectTab(tab.id)}
              className="px-4 py-2.5 text-xs font-mono font-medium rounded-t-lg transition-all cursor-pointer border-b-2"
              style={{
                borderColor: activeTab === tab.id ? '#20D9FF' : 'transparent',
                color: activeTab === tab.id ? '#20D9FF' : '#94A3B8',
                background: activeTab === tab.id ? 'rgba(32, 217, 255, 0.08)' : 'transparent',
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Modal Body Content (Scrollable) */}
        <div className="p-6 overflow-y-auto space-y-6 text-slate-300 text-sm leading-relaxed max-h-[64vh]">
          {activeTab === 'privacy' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-xl font-bold text-slate-100 mb-2" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
                  SatQuery AI Privacy Policy
                </h3>
                <p className="text-xs font-mono text-slate-400">
                  Last Updated: September 13, 2026 · Project: ISRO/SAC Remote-Sensing VLM Challenge
                </p>
              </div>

              <section className="space-y-2">
                <h4 className="font-semibold text-slate-100 text-sm">1. Introduction & Scope</h4>
                <p className="text-xs text-slate-300">
                  SatQuery AI is an agentic vision-language platform designed for scientific evaluation of single,
                  bi-temporal, and optical–SAR satellite imagery. We respect your privacy and are committed to protecting
                  any operational or research data processed through this platform. This policy outlines how information is handled
                  by <span className="text-cyan-400 font-mono">[SatQuery AI Project Team / ISRO SAC VLM Challenge]</span>.
                </p>
              </section>

              <section className="space-y-2">
                <h4 className="font-semibold text-slate-100 text-sm">2. Information We Process</h4>
                <ul className="text-xs text-slate-300 list-disc list-inside space-y-1">
                  <li>
                    <strong className="text-slate-100">Uploaded Geospatial Imagery:</strong> Satellite or aerial imagery (GeoTIFF, TIFF, PNG, JPEG)
                    uploaded for instance segmentation, land-cover classification, change detection, or cross-modal fusion.
                    Images are processed in temporary server memory or client-side canvas CV memory and are not stored in permanent public repositories.
                  </li>
                  <li>
                    <strong className="text-slate-100">User Natural-Language Queries:</strong> Prompts submitted to evaluate land-use,
                    water bodies, building counts, or multi-temporal shifts.
                  </li>
                  <li>
                    <strong className="text-slate-100">Session Metadata:</strong> Ephemeral session identifiers, API rate limit counters, and
                    inference timestamps for fair use enforcement and observable execution trace logging.
                  </li>
                </ul>
              </section>

              <section className="space-y-2">
                <h4 className="font-semibold text-slate-100 text-sm">3. Local Storage & Cookies</h4>
                <p className="text-xs text-slate-300">
                  We use browser <code className="text-cyan-300 font-mono">localStorage</code> strictly for functional purposes: preserving your
                  analysis history within the current session, storing your cookie consent choices, and caching telemetry thumbnails to eliminate redundant uploads.
                  We do <strong className="text-slate-100">not</strong> use advertising cookies, marketing pixels, or third-party behavioral trackers.
                </p>
              </section>

              <section className="space-y-2">
                <h4 className="font-semibold text-slate-100 text-sm">4. Data Security & Retention</h4>
                <p className="text-xs text-slate-300">
                  Uploaded image buffers and temporary tile segmentations are purged upon session termination or when you click "Reset" or "New Image".
                  API communication with backend microservices uses encrypted channels. Users maintain full ownership of all uploaded remote sensing imagery.
                </p>
              </section>

              <section className="space-y-2">
                <h4 className="font-semibold text-slate-100 text-sm">5. Third-Party Service Providers</h4>
                <p className="text-xs text-slate-300">
                  When deployed with live LLM/VLM providers (Anthropic Claude, OpenAI, or HuggingFace inference endpoints), queries and normalized image data
                  are transmitted according to their respective enterprise scientific privacy agreements. No imagery is used to train public commercial models.
                </p>
              </section>

              <section className="space-y-2">
                <h4 className="font-semibold text-slate-100 text-sm">6. User Rights & Data Deletion</h4>
                <p className="text-xs text-slate-300">
                  You may clear your entire in-browser session history, cached telemetry, and detection bounding boxes at any time by clicking "Reset Session"
                  in the Dashboard or clearing your browser local storage.
                </p>
              </section>

              <section className="space-y-2">
                <h4 className="font-semibold text-slate-100 text-sm">7. Project Contact Information</h4>
                <p className="text-xs text-slate-300">
                  For scientific inquiry, data removal requests, or institutional inquiries, contact:
                  <br />
                  <span className="text-cyan-300 font-mono">[contact@satquery.ai]</span> · <span className="text-slate-400">[SatQuery AI Project / Space Applications Centre (SAC), ISRO]</span>
                </p>
              </section>
            </div>
          )}

          {activeTab === 'terms' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-xl font-bold text-slate-100 mb-2" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
                  Terms & Conditions of Use
                </h3>
                <p className="text-xs font-mono text-slate-400">
                  Effective Date: September 13, 2026 · Governing Guidelines: ISRO/SAC AI Challenge Framework
                </p>
              </div>

              <section className="space-y-2">
                <h4 className="font-semibold text-slate-100 text-sm">1. Acceptance of Terms</h4>
                <p className="text-xs text-slate-300">
                  By accessing or using the SatQuery AI web application and associated API services, you agree to comply with and be bound by
                  these Terms & Conditions. If you disagree with any portion of these terms, please discontinue use immediately.
                </p>
              </section>

              <section className="space-y-2">
                <h4 className="font-semibold text-slate-100 text-sm">2. Permitted Scientific & Research Use</h4>
                <p className="text-xs text-slate-300">
                  SatQuery AI is provided for research, disaster evaluation, agricultural monitoring, urban planning analysis, and academic benchmark evaluation.
                  Users agree not to utilize the platform for unlawful surveillance, unauthorized military targeting, or attempts to circumvent API quotas.
                </p>
              </section>

              <section className="space-y-2">
                <h4 className="font-semibold text-slate-100 text-sm">3. Intellectual Property in Imagery</h4>
                <p className="text-xs text-slate-300">
                  You retain all existing intellectual property rights and proprietary ownership of any satellite imagery (Cartosat, RISAT, Sentinel, Landsat, or commercial aerial passes)
                  uploaded to the platform. SatQuery AI claims no ownership over your source imagery or derived vector shapefiles.
                </p>
              </section>

              <section className="space-y-2">
                <h4 className="font-semibold text-slate-100 text-sm">4. AI Output Disclaimer & Accuracy Limitations</h4>
                <div className="p-3.5 rounded-xl border border-amber-500/30 bg-amber-500/10 text-amber-200 text-xs leading-relaxed space-y-1">
                  <div className="font-semibold flex items-center gap-1.5">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="shrink-0">
                      <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
                      <line x1="12" y1="9" x2="12" y2="13" />
                      <line x1="12" y1="17" x2="12.01" y2="17" />
                    </svg>
                    <span>IMPORTANT NOTICE ON REMOTE SENSING ACCURACY</span>
                  </div>
                  <p>
                    All outputs generated by SatQuery AI, including BigEarthNet multi-label classifications, building footprint counts,
                    NDVI vegetation proxies, change percentages, and optical-SAR cross-correlations, are analytical machine-learning estimates.
                    While developed against peer-reviewed benchmarks (VRSBench, RSVQA, CDVQA, reBEN), results must be ground-truthed and verified
                    prior to use in life-critical disaster mitigation, flood rescue, or legal land-title decisions.
                  </p>
                </div>
              </section>

              <section className="space-y-2">
                <h4 className="font-semibold text-slate-100 text-sm">5. Service Availability & Rate Limits</h4>
                <p className="text-xs text-slate-300">
                  To protect computational resources during live evaluations, session rate limits (up to 18 calls per session by default)
                  are enforced. We reserve the right to throttle or temporarily restrict sessions displaying anomalous automation behavior.
                </p>
              </section>

              <section className="space-y-2">
                <h4 className="font-semibold text-slate-100 text-sm">6. Limitation of Liability</h4>
                <p className="text-xs text-slate-300">
                  Under no circumstances shall the project contributors, academic sponsors, or affiliated institutions be held liable for any direct,
                  indirect, incidental, or consequential damages resulting from reliance on model inferences or system unavailability.
                </p>
              </section>

              <section className="space-y-2">
                <h4 className="font-semibold text-slate-100 text-sm">7. Applicable Guidelines</h4>
                <p className="text-xs text-slate-300">
                  These terms are governed in accordance with applicable open-source scientific software guidelines and the institutional framework of
                  <span className="text-slate-100 font-mono"> [Indian Space Research Organisation (ISRO) / Space Applications Centre (SAC)]</span>.
                </p>
              </section>
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
                        <td className="p-2">1 year</td>
                      </tr>
                      <tr>
                        <td className="p-2 font-mono text-cyan-300">sessionId (crypto UUID)</td>
                        <td className="p-2 text-emerald-400 font-semibold">Strictly Required</td>
                        <td className="p-2">Tracks active session rate limit (18 calls cap) and binds follow-up queries to the uploaded image.</td>
                        <td className="p-2">Session only</td>
                      </tr>
                      <tr>
                        <td className="p-2 font-mono text-cyan-300">activeAppMode</td>
                        <td className="p-2 text-emerald-400 font-semibold">Strictly Required</td>
                        <td className="p-2">Maintains user's selected mode (Single, SAR Fusion, or Bi-Temporal) during navigation.</td>
                        <td className="p-2">Session only</td>
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
