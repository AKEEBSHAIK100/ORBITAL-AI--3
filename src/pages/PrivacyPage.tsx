import React from 'react'

export default function PrivacyPage() {
  return (
    <div className="min-h-screen text-slate-200" style={{ background: '#030712' }}>
      {/* Top Navigation */}
      <header className="sticky top-0 z-50 border-b backdrop-blur-md" style={{ background: 'rgba(7, 17, 31, 0.85)', borderColor: 'rgba(56, 189, 248, 0.16)' }}>
        <div className="max-w-4xl mx-auto px-5 h-16 flex items-center justify-between">
          <a
            href="/"
            className="flex items-center gap-2 text-xs font-mono font-medium text-cyan-400 hover:text-cyan-300 transition-colors"
          >
            <span>←</span>
            <span>Back to ORBITAL-AI</span>
          </a>
          <nav className="flex items-center gap-4 text-xs font-mono">
            <span className="text-slate-400 font-semibold">Privacy Policy</span>
            <span className="text-slate-600">·</span>
            <a href="/terms" className="text-slate-400 hover:text-slate-200 transition-colors">
              Terms & Conditions
            </a>
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-4xl mx-auto px-5 py-12 sm:py-16">
        <div className="mb-10 pb-6 border-b" style={{ borderColor: 'rgba(56, 189, 248, 0.16)' }}>
          <div className="inline-block text-[10px] font-mono tracking-widest uppercase mb-3 px-2.5 py-1 rounded" style={{ color: '#20D9FF', background: 'rgba(32, 217, 255, 0.08)', border: '1px solid rgba(56, 189, 248, 0.24)' }}>
            Legal Documentation
          </div>
          <h1 className="text-3xl sm:text-4xl font-bold text-slate-100 mb-3" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
            ORBITAL-AI Privacy Policy
          </h1>
          <p className="text-xs font-mono text-slate-400">
            Last Updated: September 17, 2026
          </p>
        </div>

        <div className="space-y-10 text-sm leading-relaxed text-slate-300">
          {/* Section 1 */}
          <section className="space-y-3">
            <h2 className="text-lg font-bold text-slate-100" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
              1. Introduction
            </h2>
            <p>
              This Privacy Policy describes how ORBITAL-AI handles information submitted through the web application. It outlines what data is processed when you interact with the platform, how that information is utilized during analysis sessions, and the choices available to you.
            </p>
          </section>

          {/* Section 2 */}
          <section className="space-y-3">
            <h2 className="text-lg font-bold text-slate-100" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
              2. Information Processed
            </h2>
            <p>
              ORBITAL-AI processes only the data necessary to perform the requested remote-sensing image analysis and maintain application functionality:
            </p>
            <ul className="list-disc list-inside space-y-1.5 pl-2 text-slate-300">
              <li>
                <strong className="text-slate-100">Uploaded Satellite and Aerial Imagery:</strong> Images provided by users to perform visual question answering, land-cover assessment, structural detection, change analysis, or optical-SAR fusion.
              </li>
              <li>
                <strong className="text-slate-100">Natural-Language Queries:</strong> Questions, task prompts, and conversation follow-ups entered to interrogate uploaded imagery.
              </li>
              <li>
                <strong className="text-slate-100">Session Identifier (sessionId):</strong> An ephemeral identifier generated in the browser to correlate consecutive requests with the active session.
              </li>
              <li>
                <strong className="text-slate-100">Operational Request Information:</strong> Basic request parameters, inference timestamps, and rate-limiting counters required to operate and protect the service.
              </li>
            </ul>
            <p className="text-xs text-slate-400">
              The application does not collect personal identity information such as names, physical addresses, phone numbers, or user account registrations.
            </p>
          </section>

          {/* Section 3 */}
          <section className="space-y-3">
            <h2 className="text-lg font-bold text-slate-100" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
              3. Image Processing and Temporary Cache
            </h2>
            <p>
              Uploaded imagery used by the single-image analysis workflow may be placed in a server-side in-memory session cache associated with a session identifier. The current application sets this cache to expire after up to 30 minutes.
            </p>
            <p>
              The application uses this cache so follow-up questions can reuse the uploaded image without retransmitting the full image on every request. This in-memory cache is temporary and volatile; it does not constitute permanent storage, persistent archiving, or a backup service.
            </p>
          </section>

          {/* Section 4 */}
          <section className="space-y-3">
            <h2 className="text-lg font-bold text-slate-100" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
              4. Bi-Temporal Comparisons
            </h2>
            <p>
              In the comparison workflow, users submit an earlier baseline image and a later observation image. These before and after images are transmitted to the application backend specifically to calculate structural, vegetative, and hydrological differences for that comparison request.
            </p>
          </section>

          {/* Section 5 */}
          <section className="space-y-3">
            <h2 className="text-lg font-bold text-slate-100" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
              5. Browser Storage and Cookies
            </h2>
            <p>
              The application uses browser <code className="text-cyan-300 font-mono text-xs px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800">localStorage</code> to record your cookie-consent acknowledgment so the notification banner does not repeatedly appear across visits.
            </p>
            <p>
              The application does not currently use that storage entry as an advertising profile or for cross-site behavioral tracking. Hosting environments, content delivery networks, or external infrastructure providers may deploy standard network-level headers or technical cookies necessary for routing and security.
            </p>
          </section>

          {/* Section 6 */}
          <section className="space-y-3">
            <h2 className="text-lg font-bold text-slate-100" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
              6. Model and Service Providers
            </h2>
            <p>
              Depending on deployment configuration, analysis requests may be processed by the application's backend and a configured model/inference provider. When third-party model endpoints are active, query text and formatted image payloads are transmitted to the designated provider solely to execute the requested inference.
            </p>
          </section>

          {/* Section 7 */}
          <section className="space-y-3">
            <h2 className="text-lg font-bold text-slate-100" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
              7. Data Retention
            </h2>
            <p>
              Data retention across ORBITAL-AI components is structured as follows:
            </p>
            <ul className="list-disc list-inside space-y-1.5 pl-2 text-slate-300">
              <li>
                <strong className="text-slate-100">Single-Image Session Cache:</strong> Held in temporary server-side memory for up to 30 minutes, or until the server process restarts or the cache entry is replaced.
              </li>
              <li>
                <strong className="text-slate-100">Browser Cookie-Consent Preference:</strong> Stored locally by the browser in <code className="text-cyan-300 font-mono text-xs">localStorage</code> until cleared by the user or modified via settings.
              </li>
              <li>
                <strong className="text-slate-100">Comparison Request Data:</strong> Processed for the active comparison request without an independent persistent storage lifecycle.
              </li>
            </ul>
          </section>

          {/* Section 8 */}
          <section className="space-y-3">
            <h2 className="text-lg font-bold text-slate-100" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
              8. Security
            </h2>
            <p>
              The application is designed to process uploaded imagery through its configured backend services. Users should avoid uploading information they are not authorized to submit, or imagery containing sensitive, confidential, or restricted proprietary data.
            </p>
          </section>

          {/* Section 9 */}
          <section className="space-y-3">
            <h2 className="text-lg font-bold text-slate-100" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
              9. User Choices
            </h2>
            <p>
              Users maintain direct control over their participation:
            </p>
            <ul className="list-disc list-inside space-y-1.5 pl-2 text-slate-300">
              <li>You may discontinue use of the application at any time.</li>
              <li>You can clear local browser data and cookies through your browser settings.</li>
              <li>You should avoid submitting imagery for which you do not possess valid operational or distribution permissions.</li>
              <li>You may reach out with inquiries through the contact mechanism shown in the application.</li>
            </ul>
          </section>

          {/* Section 10 */}
          <section className="space-y-3">
            <h2 className="text-lg font-bold text-slate-100" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
              10. Changes to this Policy
            </h2>
            <p>
              This Privacy Policy may be updated periodically to reflect modifications to application behavior, technical infrastructure, or operational capabilities. Revisions will be published on this page with an updated revision date.
            </p>
          </section>

          {/* Section 11 */}
          <section className="space-y-3">
            <h2 className="text-lg font-bold text-slate-100" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
              11. Contact
            </h2>
            <p>
              For project or data-handling questions, use the contact channel provided in the application.
            </p>
          </section>
        </div>

        {/* Bottom Navigation */}
        <div className="mt-16 pt-8 border-t flex flex-col sm:flex-row items-center justify-between gap-4 text-xs font-mono text-slate-400" style={{ borderColor: 'rgba(56, 189, 248, 0.16)' }}>
          <a href="/" className="text-cyan-400 hover:text-cyan-300 transition-colors">
            ← Back to ORBITAL-AI
          </a>
          <div className="flex items-center gap-4">
            <a href="/terms" className="hover:text-slate-200 transition-colors">
              Terms & Conditions →
            </a>
          </div>
        </div>
      </main>
    </div>
  )
}
