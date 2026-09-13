import React, { useState } from 'react'
import { ExecutionTrace, formatTaskLabel, TOOL_REGISTRY } from '../lib/agentController'

const CYN = '#20D9FF'
const ORG = '#FF9F43'
const MNT = '#35E0B8'
const WHT = '#F4F7FA'
const GRY = '#9AA9B8'
const PNL = '#060D1A'
const CB = 'rgba(32,217,255,0.18)'

interface Props {
  isOpen: boolean
  onClose: () => void
  trace: ExecutionTrace | null
  queryTitle?: string
}

export default function AgentTraceModal({ isOpen, onClose, trace, queryTitle }: Props) {
  const [copied, setCopied] = useState(false)

  if (!isOpen || !trace) return null

  const handleCopyJson = () => {
    navigator.clipboard.writeText(JSON.stringify(trace, null, 2))
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const primaryTool = TOOL_REGISTRY[trace.model_registry_entry?.model_id] ||
    Object.values(TOOL_REGISTRY).find(t => trace.tools_invoked?.includes(t.id)) ||
    TOOL_REGISTRY.rs_vqa

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: 'rgba(2,8,16,0.82)', backdropFilter: 'blur(8px)' }}
      onClick={onClose}
    >
      <div
        className="relative w-full max-w-2xl max-h-[88vh] rounded-2xl flex flex-col overflow-hidden border shadow-2xl"
        style={{
          background: 'linear-gradient(180deg, #071022 0%, #030814 100%)',
          borderColor: CB,
          boxShadow: '0 0 40px rgba(32,217,255,0.15)',
        }}
        onClick={e => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div
          className="flex items-center justify-between px-6 py-4 border-b"
          style={{ borderColor: 'rgba(32,217,255,0.15)', background: 'rgba(6,13,26,0.7)' }}
        >
          <div className="flex items-center gap-3">
            <div
              className="w-8 h-8 rounded-lg flex items-center justify-center font-mono text-xs font-bold"
              style={{ background: `${CYN}22`, color: CYN, border: `1px solid ${CYN}66` }}
            >
              AI
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-sm font-bold tracking-wider uppercase font-mono" style={{ color: WHT }}>
                  Auditable Agent Execution Trace
                </h3>
                <span
                  className="px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase"
                  style={{ background: `${MNT}22`, color: MNT, border: `1px solid ${MNT}55` }}
                >
                  {trace.agent_version}
                </span>
              </div>
              <p className="text-[11px] font-mono" style={{ color: GRY }}>
                Deterministic Task Routing & Specialist Tool Sequencing
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleCopyJson}
              className="px-2.5 py-1 rounded text-xs font-mono transition-all flex items-center gap-1.5 cursor-pointer"
              style={{
                background: copied ? `${MNT}33` : 'rgba(255,255,255,0.06)',
                color: copied ? MNT : WHT,
                border: `1px solid ${copied ? MNT : 'rgba(255,255,255,0.15)'}`,
              }}
              title="Copy raw JSON trace for benchmark evaluation"
            >
              <span>{copied ? '✓ COPIED JSON' : 'COPY JSON TRACE'}</span>
            </button>

            <button
              onClick={onClose}
              className="w-7 h-7 rounded-lg flex items-center justify-center font-mono text-sm hover:opacity-80 transition-opacity cursor-pointer"
              style={{ background: 'rgba(255,255,255,0.06)', color: GRY, border: '1px solid rgba(255,255,255,0.1)' }}
            >
              ✕
            </button>
          </div>
        </div>

        {/* Modal Scrollable Body */}
        <div className="flex-1 overflow-y-auto p-6 space-y-5">
          {/* Query context & Task summary */}
          <div
            className="p-3.5 rounded-xl border flex items-center justify-between"
            style={{ background: 'rgba(32,217,255,0.04)', borderColor: 'rgba(32,217,255,0.2)' }}
          >
            <div>
              <span className="text-[10px] font-mono tracking-wider uppercase" style={{ color: GRY }}>
                CLASSIFIED TASK ROUTE
              </span>
              <h4 className="text-sm font-semibold mt-0.5" style={{ color: CYN }}>
                {formatTaskLabel(trace.task_type)}
              </h4>
              {queryTitle && (
                <p className="text-xs font-mono mt-1 line-clamp-1 italic" style={{ color: 'rgba(244,247,250,0.7)' }}>
                  "{queryTitle}"
                </p>
              )}
            </div>

            <div className="text-right">
              <span className="text-[10px] font-mono tracking-wider uppercase" style={{ color: GRY }}>
                TOTAL LATENCY
              </span>
              <div className="text-base font-mono font-bold" style={{ color: MNT }}>
                {trace.total_duration_ms} ms
              </div>
              <span className="text-[10px] font-mono" style={{ color: GRY }}>
                {trace.steps?.length || 0} stages executed
              </span>
            </div>
          </div>

          {/* Input Validation & Modality Checks */}
          <div
            className="p-3.5 rounded-xl border space-y-2"
            style={{ background: 'rgba(255,255,255,0.02)', borderColor: 'rgba(255,255,255,0.08)' }}
          >
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-mono tracking-wider uppercase font-bold" style={{ color: WHT }}>
                INPUT VALIDATION & SENSOR MODALITIES
              </span>
              <span
                className="px-2 py-0.5 rounded text-[10px] font-mono uppercase font-bold"
                style={{
                  background: trace.input_validation?.compatibility === 'valid' ? `${MNT}22` : `${ORG}22`,
                  color: trace.input_validation?.compatibility === 'valid' ? MNT : ORG,
                  border: `1px solid ${trace.input_validation?.compatibility === 'valid' ? MNT : ORG}66`,
                }}
              >
                {trace.input_validation?.compatibility || 'VALID'}
              </span>
            </div>

            <div className="grid grid-cols-3 gap-2 text-xs font-mono pt-1">
              <div>
                <span className="text-[10px] block" style={{ color: GRY }}>Images Provided</span>
                <span style={{ color: WHT }}>{trace.input_validation?.images_provided ?? 1} Observation(s)</span>
              </div>
              <div>
                <span className="text-[10px] block" style={{ color: GRY }}>Sensors / Modalities</span>
                <span style={{ color: WHT }}>{trace.input_validation?.modalities?.join(', ') || 'Optical RGB'}</span>
              </div>
              <div>
                <span className="text-[10px] block" style={{ color: GRY }}>Format Compatibility</span>
                <span style={{ color: WHT }}>{trace.input_validation?.formats?.join(', ') || 'JPEG/PNG/TIFF'}</span>
              </div>
            </div>

            {trace.input_validation?.notes && trace.input_validation.notes.length > 0 && (
              <div className="pt-2 border-t border-white/5 space-y-1">
                {trace.input_validation.notes.map((note, idx) => (
                  <p key={idx} className="text-[11px] font-mono flex items-start gap-1.5" style={{ color: GRY }}>
                    <span style={{ color: CYN }}>›</span> {note}
                  </p>
                ))}
              </div>
            )}
          </div>

          {/* Sequential Execution Steps */}
          <div>
            <span className="text-[10px] font-mono tracking-wider uppercase font-bold block mb-2.5" style={{ color: WHT }}>
              AGENT EXECUTION PIPELINE
            </span>

            <div className="space-y-2">
              {trace.steps?.map((st) => (
                <div
                  key={st.step}
                  className="p-3 rounded-xl border flex flex-col gap-1.5 transition-colors"
                  style={{
                    background: 'rgba(6,13,26,0.6)',
                    borderColor: st.status === 'success' ? 'rgba(32,217,255,0.2)' : 'rgba(255,107,107,0.3)',
                  }}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span
                        className="w-5 h-5 rounded-full flex items-center justify-center font-mono text-[10px] font-bold"
                        style={{ background: `${CYN}22`, color: CYN, border: `1px solid ${CYN}55` }}
                      >
                        {st.step}
                      </span>
                      <span className="text-xs font-mono font-bold uppercase" style={{ color: WHT }}>
                        {st.tool}
                      </span>
                      <span className="text-[10px] font-mono" style={{ color: GRY }}>
                        — {st.description}
                      </span>
                    </div>

                    <div className="flex items-center gap-2 font-mono text-[11px]">
                      <span style={{ color: MNT }}>{st.duration_ms} ms</span>
                      <span
                        className="px-1.5 py-0.2 rounded text-[9px] uppercase font-bold"
                        style={{ background: `${MNT}22`, color: MNT }}
                      >
                        {st.status}
                      </span>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-[11px] font-mono bg-black/20 p-2 rounded-lg mt-0.5">
                    <div>
                      <span className="text-[10px] block font-bold" style={{ color: GRY }}>INPUT</span>
                      <span style={{ color: 'rgba(244,247,250,0.85)' }}>{st.input_summary}</span>
                    </div>
                    <div>
                      <span className="text-[10px] block font-bold" style={{ color: GRY }}>OUTPUT</span>
                      <span style={{ color: CYN }}>{st.output_summary}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Model Registry & Domain Adaptation Info */}
          <div
            className="p-4 rounded-xl border space-y-2"
            style={{ background: 'rgba(53,224,184,0.03)', borderColor: 'rgba(53,224,184,0.2)' }}
          >
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-mono tracking-wider uppercase font-bold" style={{ color: MNT }}>
                MODEL REGISTRY & REMOTE-SENSING ADAPTATION
              </span>
              <span className="text-[10px] font-mono" style={{ color: GRY }}>
                ISRO / SAC Criteria Compliance
              </span>
            </div>

            <div className="space-y-1.5 text-xs font-mono">
              <div className="flex items-start gap-2">
                <span className="text-white/40 w-24 shrink-0">Model ID:</span>
                <span style={{ color: WHT }}>{trace.model_registry_entry?.model_id || primaryTool.model_id}</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-white/40 w-24 shrink-0">Adapter:</span>
                <span style={{ color: CYN }}>{trace.model_registry_entry?.adapter || primaryTool.adapter}</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-white/40 w-24 shrink-0">Adaptation:</span>
                <span style={{ color: 'rgba(244,247,250,0.85)' }}>
                  {trace.model_registry_entry?.domain_adaptation || primaryTool.domain_adaptation}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Modal Footer */}
        <div
          className="flex items-center justify-between px-6 py-3 border-t text-xs font-mono"
          style={{ borderColor: 'rgba(255,255,255,0.08)', background: 'rgba(6,13,26,0.8)' }}
        >
          <span style={{ color: GRY }}>
            Auditable Trace generated in compliance with ISRO/SAC evaluation benchmarks
          </span>
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg text-xs font-mono font-bold transition-all cursor-pointer"
            style={{ background: `${CYN}22`, color: CYN, border: `1px solid ${CYN}88` }}
          >
            CLOSE HUD
          </button>
        </div>
      </div>
    </div>
  )
}
