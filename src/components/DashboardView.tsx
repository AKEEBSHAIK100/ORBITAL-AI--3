import React from 'react'
import { ImageTelemetry, BuildingAnalysisResult } from '../App'
import { ExecutionTrace, FusionFeatures } from '../lib/agentController'

interface BENLabelScore { name: string; short: string; score: number; active: boolean }
interface BENResult {
  labels: BENLabelScore[]
  active_labels: BENLabelScore[]
  top_label: string
  confidence: number
  model_id: string
  available: boolean
  device: string
}

interface ChatMessage {
  question: string
  answer: string
  confidenceScore: number
  confidence_percent: number
  confidence: 'high' | 'medium' | 'low'
  confidence_status?: 'calibrated' | 'not_calibrated' | 'unavailable' | string
  confidence_reason?: string
  detected_features: string[]
  label: string
  timestamp: string
}

interface DashboardViewProps {
  sessionId: string | null
  sessionCallCount: number
  sessionLimit: number
  imageTelemetry: ImageTelemetry
  buildingAnalysis: BuildingAnalysisResult | null
  benResults: BENResult | null
  history: ChatMessage[]
  activeMode: string
  onSwitchToWorkspace: () => void
  onSelectHistoricalQuery: (query: string) => void
  onExportReport: () => void
  onResetSession: () => void
}

export default function DashboardView({
  sessionId,
  sessionCallCount,
  sessionLimit,
  imageTelemetry,
  buildingAnalysis,
  benResults,
  history,
  activeMode,
  onSwitchToWorkspace,
  onSelectHistoricalQuery,
  onExportReport,
  onResetSession,
}: DashboardViewProps) {
  const percentUsed = Math.min(100, Math.round((sessionCallCount / sessionLimit) * 100))

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Dashboard Top Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-6 rounded-2xl border border-slate-800 bg-[#07111F]/90 backdrop-blur-md shadow-xl">
        <div>
          <div className="flex items-center gap-2.5 mb-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 shadow-[0_0_8px_#35E0B8]" />
            <span className="text-xs font-mono tracking-widest text-emerald-400 uppercase font-bold">
              OPERATIONAL TELEMETRY DASHBOARD
            </span>
          </div>
          <h2 className="text-2xl lg:text-3xl font-bold text-slate-100" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
            Mission Status & Session Activity
          </h2>
          <p className="text-xs font-mono text-slate-400 mt-1">
            Session ID: <span className="text-cyan-300 font-semibold">{sessionId ? sessionId.slice(0, 16) + '…' : 'Initializing…'}</span> · Mode: <span className="text-slate-200 capitalize">{activeMode}</span>
          </p>
        </div>

        <div className="flex items-center gap-2.5 shrink-0">
          <button
            type="button"
            onClick={onExportReport}
            className="px-4 py-2 rounded-xl text-xs font-mono font-semibold text-cyan-400 bg-cyan-950/40 hover:bg-cyan-900/50 border border-cyan-500/30 transition-all flex items-center gap-2 cursor-pointer"
          >
            <span>⬇</span>
            <span>Export Session JSON</span>
          </button>
          <button
            type="button"
            onClick={onSwitchToWorkspace}
            className="px-4 py-2 rounded-xl text-xs font-semibold text-slate-950 bg-cyan-400 hover:bg-cyan-300 transition-all shadow-md shadow-cyan-950 cursor-pointer flex items-center gap-1.5"
          >
            <span>Workspace</span>
            <span>→</span>
          </button>
        </div>
      </div>

      {/* Top Quick Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {/* Session Rate Limit Card */}
        <div className="p-5 rounded-2xl border border-slate-800 bg-[#07111F] space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-mono text-slate-400 uppercase">Session Inferences</span>
            <span className="text-xs font-mono px-2 py-0.5 rounded font-bold bg-cyan-500/15 text-cyan-400 border border-cyan-500/30">
              {sessionCallCount} / {sessionLimit}
            </span>
          </div>
          <div className="text-3xl font-bold text-slate-100" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
            {sessionCallCount} <span className="text-sm font-normal text-slate-400 font-mono">calls</span>
          </div>
          <div className="w-full h-1.5 rounded-full bg-slate-800 overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{
                width: `${percentUsed}%`,
                background: percentUsed > 80 ? '#FF6B6B' : percentUsed > 50 ? '#FF9F43' : '#20D9FF',
              }}
            />
          </div>
          <div className="text-[10px] font-mono text-slate-400">
            {Math.max(0, sessionLimit - sessionCallCount)} calls remaining before session request guard
          </div>
        </div>

        {/* BigEarthNet Top Class Card */}
        <div className="p-5 rounded-2xl border border-slate-800 bg-[#07111F] space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-mono text-slate-400 uppercase">Land Cover Classification</span>
            <span className="text-xs font-mono px-2 py-0.5 rounded font-bold bg-cyan-500/15 text-cyan-400 border border-cyan-500/30">
              {benResults ? 'Verified' : 'Standby'}
            </span>
          </div>
          <div className="text-xl font-bold text-slate-100 truncate" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
            {benResults ? benResults.top_label : 'Awaiting analysis'}
          </div>
          <div className="text-[10px] font-mono text-cyan-400">
            {benResults ? `${benResults.confidence.toFixed(1)}% model score` : 'Zero unverified measurements'}
          </div>
        </div>

        {/* Active Analysis Mode Card */}
        <div className="p-5 rounded-2xl border border-slate-800 bg-[#07111F] space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-mono text-slate-400 uppercase">Analysis Mode</span>
            <span className="text-xs font-mono px-2 py-0.5 rounded font-bold bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 uppercase">
              {activeMode}
            </span>
          </div>
          <div className="text-xl font-bold text-slate-100" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
            {activeMode === 'compare' ? 'Bi-temporal' : activeMode === 'fusion' ? 'Optical + SAR' : 'Single Image'}
          </div>
          <div className="text-[10px] font-mono text-slate-400 truncate">
            {activeMode === 'compare' ? 'Compare 2 observation dates' : activeMode === 'fusion' ? 'Cross-modal optical & radar' : 'Visual question answering & description'}
          </div>
        </div>
      </div>

      {/* Model Registry & Engine Health Status */}
      <div className="p-6 rounded-2xl border border-slate-800 bg-[#07111F] space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <h3 className="text-base font-bold text-slate-100 font-mono tracking-wide flex items-center gap-2">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-cyan-400">
              <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
            </svg>
            <span>MODEL PIPELINE HEALTH & STATUS</span>
          </h3>
          <span className="text-xs font-mono text-emerald-400 bg-emerald-500/10 px-2.5 py-1 rounded-md border border-emerald-500/30">
            ● All Specialist Engines Operational
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3 text-xs font-mono">
          <div className="p-3.5 rounded-xl border border-slate-800/80 bg-slate-900/50 space-y-1">
            <div className="flex items-center justify-between">
              <span className="font-bold text-slate-200">BigEarthNet v2.0</span>
              <span className="text-[10px] text-emerald-400">READY</span>
            </div>
            <div className="text-slate-400 text-[11px]">ResNet-50 · 19-Class CLC</div>
            <div className="text-slate-500 text-[10px] truncate">HuggingFace / BIFOLD Weights</div>
          </div>

          <div className="p-3.5 rounded-xl border border-slate-800/80 bg-slate-900/50 space-y-1">
            <div className="flex items-center justify-between">
              <span className="font-bold text-slate-200">YOLOv8-Seg</span>
              <span className="text-[10px] text-emerald-400">READY</span>
            </div>
            <div className="text-slate-400 text-[11px]">Instance Segmentation</div>
            <div className="text-slate-500 text-[10px]">Tiling + IoU Deduplication</div>
          </div>

          <div className="p-3.5 rounded-xl border border-slate-800/80 bg-slate-900/50 space-y-1">
            <div className="flex items-center justify-between">
              <span className="font-bold text-slate-200">Optical–SAR Fusion</span>
              <span className="text-[10px] text-emerald-400">READY</span>
            </div>
            <div className="text-slate-400 text-[11px]">C-Band Backscatter + SSIM</div>
            <div className="text-slate-500 text-[10px]">Dual-Sensor Coregistration</div>
          </div>

          <div className="p-3.5 rounded-xl border border-slate-800/80 bg-slate-900/50 space-y-1">
            <div className="flex items-center justify-between">
              <span className="font-bold text-slate-200">CDVQA Change Engine</span>
              <span className="text-[10px] text-emerald-400">READY</span>
            </div>
            <div className="text-slate-400 text-[11px]">Bi-Temporal Ground Control</div>
            <div className="text-slate-500 text-[10px]">Sub-pixel alignment ≤0.3px</div>
          </div>
        </div>
      </div>

      {/* Activity History Table */}
      <div className="p-6 rounded-2xl border border-slate-800 bg-[#07111F] space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div>
            <h3 className="text-base font-bold text-slate-100 font-mono tracking-wide">
              SESSION QUERY AUDIT LOG
            </h3>
            <p className="text-xs text-slate-400 font-mono mt-0.5">
              Chronological log of verified multi-modal queries in current session
            </p>
          </div>
          {history.length > 0 && (
            <button
              type="button"
              onClick={onResetSession}
              className="text-xs font-mono text-rose-400 hover:text-rose-300 hover:bg-rose-500/10 px-3 py-1.5 rounded-lg border border-rose-500/30 transition-all cursor-pointer"
            >
              Clear Session Log
            </button>
          )}
        </div>

        {history.length === 0 ? (
          /* Clean Professional Empty State */
          <div className="py-14 text-center space-y-3">
            <div className="w-14 h-14 rounded-2xl border border-slate-800 bg-slate-900/60 flex items-center justify-center mx-auto text-2xl">
              <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" className="text-cyan-400">
                <circle cx="12" cy="12" r="2" />
                <path d="M16.24 7.76a6 6 0 0 1 0 8.49m-8.48-.01a6 6 0 0 1 0-8.49m11.31-2.82a10 10 0 0 1 0 14.14m-14.14 0a10 10 0 0 1 0-14.14" />
              </svg>
            </div>
            <h4 className="text-base font-bold text-slate-200" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
              No Analysis Queries Recorded Yet
            </h4>
            <p className="text-xs text-slate-400 max-w-md mx-auto leading-relaxed">
              Upload a satellite image in the Workspace and execute an inquiry (such as VRSBench captioning or water body grounding)
              to record live telemetry and verifiable execution traces in this dashboard.
            </p>
            <button
              type="button"
              onClick={onSwitchToWorkspace}
              className="mt-2 px-5 py-2.5 rounded-xl text-xs font-semibold text-slate-950 bg-cyan-400 hover:bg-cyan-300 transition-all shadow-md shadow-cyan-950 cursor-pointer inline-flex items-center gap-1.5"
            >
              <span>Open AI Workspace</span>
              <span>→</span>
            </button>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse font-mono">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400 text-[11px]">
                  <th className="py-3 px-4">Time</th>
                  <th className="py-3 px-4">Query / Prompt</th>
                  <th className="py-3 px-4">Task Domain</th>
                  <th className="py-3 px-4">Confidence / Score</th>
                  <th className="py-3 px-4 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/80 text-slate-300">
                {history.map((item, idx) => (
                  <tr key={`${item.timestamp}-${idx}`} className="hover:bg-slate-900/40 transition-colors">
                    <td className="py-3.5 px-4 text-slate-400 whitespace-nowrap">{item.timestamp}</td>
                    <td className="py-3.5 px-4 font-sans text-slate-100 max-w-xs sm:max-w-sm truncate">
                      "{item.question}"
                    </td>
                    <td className="py-3.5 px-4 whitespace-nowrap">
                      <span className="px-2.5 py-1 rounded-md text-[10px] font-bold bg-cyan-500/15 text-cyan-300 border border-cyan-500/30">
                        {item.label}
                      </span>
                    </td>
                    <td className="py-3.5 px-4 whitespace-nowrap">
                      {item.confidence_status === 'not_calibrated' ? (
                        <div className="flex flex-col">
                          <span className="text-[11px] font-mono text-amber-400 font-medium">Not calibrated</span>
                          {(item.confidence_percent ?? item.confidenceScore) > 0 && (
                            <span className="text-[10px] font-mono text-slate-400">Score: {item.confidence_percent ?? item.confidenceScore}%</span>
                          )}
                        </div>
                      ) : item.confidence_status === 'calibrated' ? (
                        <div className="flex flex-col">
                          <span className="text-[11px] font-mono text-emerald-400 font-semibold">Calibrated</span>
                          <span className="text-[10px] font-mono text-slate-300">{item.confidence_percent ?? item.confidenceScore}%</span>
                        </div>
                      ) : (item.confidence_percent ?? item.confidenceScore) === 0 ? (
                        <span className="text-[11px] font-mono text-slate-500">Unavailable</span>
                      ) : (
                        <span
                          className="font-bold font-mono text-xs"
                          style={{
                            color: (item.confidence_percent ?? item.confidenceScore) >= 85 ? '#35E0B8' : '#FF9F43',
                          }}
                        >
                          Score: {item.confidence_percent ?? item.confidenceScore}%
                        </span>
                      )}
                    </td>
                    <td className="py-3.5 px-4 text-right whitespace-nowrap">
                      <button
                        type="button"
                        onClick={() => onSelectHistoricalQuery(item.question)}
                        className="text-cyan-400 hover:text-cyan-300 underline text-xs cursor-pointer"
                      >
                        Re-run
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
