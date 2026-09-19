import React, { useState, useEffect } from 'react'

const CYN = '#20D9FF'
const ORG = '#FF9F43'
const MNT = '#35E0B8'
const RED = '#FF5252'
const WHT = '#F4F7FA'
const GRY = '#9AA9B8'
const CB = 'rgba(32,217,255,0.18)'

interface DatasetInfo {
  id: string
  name: string
  version: string
  local_path: string
  modality: string
  task: string
  split: string
  status: 'NOT_DOWNLOADED' | 'AVAILABLE' | 'INVALID' | 'PROCESSING' | 'READY'
  download_url: string
  license_attribution: string
  notes?: string
}

interface SpecialistInfo {
  id: string
  name: string
  task: string
  model_id: string
  version: string
  modality: string[]
  supported_input_types: string[]
  checkpoint_location?: string | null
  is_available: boolean
  unavailable_reason?: string | null
}

interface EvaluationRun {
  dataset: string
  dataset_version: string
  split: string
  task: string
  model: string
  model_version: string
  timestamp: string
  metrics?: Record<string, any>
  sample_count: number
  status: 'NOT_RUN' | 'DATASET_UNAVAILABLE' | 'RUNNING' | 'COMPLETED' | 'FAILED'
}

interface Props {
  isOpen: boolean
  onClose: () => void
}

export default function ModelStatusModal({ isOpen, onClose }: Props) {
  const [tab, setTab] = useState<'specialists' | 'datasets' | 'benchmarks'>('specialists')
  const [datasets, setDatasets] = useState<DatasetInfo[]>([])
  const [specialists, setSpecialists] = useState<SpecialistInfo[]>([])
  const [evaluations, setEvaluations] = useState<EvaluationRun[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!isOpen) return
    setLoading(true)
    fetch('/api/model-status')
      .then(res => res.json())
      .then(data => {
        setDatasets(data.datasets || [])
        setSpecialists(data.specialists || [])
        setEvaluations(data.evaluations || [])
        setLoading(false)
      })
      .catch(() => {
        // Fallback default structure if backend offline
        setLoading(false)
      })
  }, [isOpen])

  if (!isOpen) return null

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: 'rgba(2,8,16,0.85)', backdropFilter: 'blur(8px)' }}
      onClick={onClose}
    >
      <div
        className="relative w-full max-w-4xl max-h-[90vh] rounded-2xl flex flex-col overflow-hidden border shadow-2xl"
        style={{
          background: '#07111F',
          borderColor: CB,
          boxShadow: '0 0 50px rgba(32,217,255,0.18)',
        }}
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div
          className="flex items-center justify-between px-6 py-4 border-b"
          style={{ borderColor: 'rgba(32,217,255,0.15)', background: 'rgba(6,13,26,0.8)' }}
        >
          <div className="flex items-center gap-3">
            <div
              className="w-8 h-8 rounded-lg flex items-center justify-center font-mono text-xs font-bold"
              style={{ background: `${CYN}22`, color: CYN, border: `1px solid ${CYN}66` }}
            >
              VLM
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-sm font-bold tracking-wider uppercase font-mono" style={{ color: WHT }}>
                  Model & Dataset Provenance Dashboard
                </h3>
                <span
                  className="px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase"
                  style={{ background: `${MNT}22`, color: MNT, border: `1px solid ${MNT}55` }}
                >
                  GROUND TRUTH AUDIT
                </span>
              </div>
              <p className="text-[11px] font-mono" style={{ color: GRY }}>
                Real-time inspection of specialist models, local dataset presence, and verified benchmark runs.
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="w-7 h-7 rounded-lg flex items-center justify-center font-mono text-sm hover:opacity-80 transition-opacity cursor-pointer"
            style={{ background: 'rgba(255,255,255,0.06)', color: GRY, border: '1px solid rgba(255,255,255,0.1)' }}
          >
            ✕
          </button>
        </div>

        {/* Tab Navigation */}
        <div className="flex border-b px-6 py-2 gap-3 text-xs font-mono" style={{ borderColor: 'rgba(32,217,255,0.12)', background: '#050E1C' }}>
          {[
            { id: 'specialists', label: `Specialist Models (${specialists.length})` },
            { id: 'datasets', label: `Dataset Registry (${datasets.length})` },
            { id: 'benchmarks', label: `Benchmark Results (${evaluations.length})` },
          ].map(t => (
            <button
              key={t.id}
              onClick={() => setTab(t.id as any)}
              className="px-3 py-1.5 rounded-lg transition-all cursor-pointer"
              style={{
                background: tab === t.id ? `${CYN}22` : 'transparent',
                color: tab === t.id ? CYN : GRY,
                border: tab === t.id ? `1px solid ${CYN}66` : '1px solid transparent',
              }}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* Content Body */}
        <div className="p-6 overflow-y-auto space-y-4 text-xs font-mono" style={{ color: WHT }}>
          {loading ? (
            <div className="py-12 text-center text-sm" style={{ color: GRY }}>
              Querying model registry and disk assets…
            </div>
          ) : tab === 'specialists' ? (
            <div className="space-y-3">
              <div className="text-[11px]" style={{ color: GRY }}>
                8 Domain Specialists registered. Models without installed checkpoints degrade gracefully with controlled explanations.
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {specialists.map(spec => (
                  <div
                    key={spec.id}
                    className="p-3.5 rounded-xl border flex flex-col justify-between"
                    style={{ background: 'rgba(10,22,40,0.6)', borderColor: spec.is_available ? `${MNT}44` : `${ORG}44` }}
                  >
                    <div>
                      <div className="flex items-center justify-between gap-2 mb-1.5">
                        <span className="font-bold text-sm" style={{ color: WHT }}>{spec.name}</span>
                        <span
                          className="px-2 py-0.5 rounded text-[10px] font-bold uppercase"
                          style={{
                            background: spec.is_available ? `${MNT}22` : `${ORG}22`,
                            color: spec.is_available ? MNT : ORG,
                            border: `1px solid ${spec.is_available ? MNT : ORG}55`
                          }}
                        >
                          {spec.is_available ? 'AVAILABLE' : 'UNAVAILABLE'}
                        </span>
                      </div>
                      <div className="text-[11px] mb-1" style={{ color: CYN }}>
                        ID: {spec.model_id} (v{spec.version})
                      </div>
                      <div className="text-[11px] mb-1" style={{ color: GRY }}>
                        Task: <span className="text-white">{spec.task}</span> | Modalities: {spec.modality.join(', ')}
                      </div>
                      {spec.checkpoint_location && (
                        <div className="text-[10px] truncate" style={{ color: GRY }}>
                          Checkpoint: {spec.checkpoint_location}
                        </div>
                      )}
                    </div>
                    {spec.unavailable_reason && (
                      <div className="mt-2 text-[10px] p-2 rounded" style={{ background: `${ORG}11`, color: ORG }}>
                        Notice: {spec.unavailable_reason}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          ) : tab === 'datasets' ? (
            <div className="space-y-3">
              <div className="text-[11px]" style={{ color: GRY }}>
                Datasets default to NOT_DOWNLOADED to avoid bandwidth bloat. Normal website functions zero-shot / heuristics when datasets are absent.
              </div>
              <div className="space-y-2">
                {datasets.map(ds => {
                  const isAvail = ds.status === 'AVAILABLE' || ds.status === 'READY'
                  return (
                    <div
                      key={ds.id}
                      className="p-3 rounded-xl border flex items-center justify-between gap-4"
                      style={{ background: 'rgba(10,22,40,0.6)', borderColor: isAvail ? `${MNT}44` : 'rgba(255,255,255,0.08)' }}
                    >
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="font-bold text-sm" style={{ color: WHT }}>{ds.name}</span>
                          <span className="text-[11px]" style={{ color: CYN }}>v{ds.version}</span>
                          <span
                            className="px-2 py-0.5 rounded text-[10px] font-bold"
                            style={{
                              background: isAvail ? `${MNT}22` : 'rgba(255,255,255,0.08)',
                              color: isAvail ? MNT : GRY,
                              border: `1px solid ${isAvail ? MNT : 'rgba(255,255,255,0.15)'}55`
                            }}
                          >
                            {ds.status}
                          </span>
                        </div>
                        <div className="text-[11px] mt-1" style={{ color: GRY }}>
                          Task: {ds.task} | Modality: {ds.modality} | Format: {ds.split} ({ds.local_path})
                        </div>
                        {ds.notes && (
                          <div className="text-[10px] mt-0.5" style={{ color: GRY }}>
                            {ds.notes}
                          </div>
                        )}
                      </div>
                      <div className="text-right shrink-0">
                        <a
                          href={ds.download_url}
                          target="_blank"
                          rel="noreferrer"
                          className="px-2.5 py-1 rounded text-[11px] inline-block hover:opacity-80 transition-opacity"
                          style={{ background: `${CYN}22`, color: CYN, border: `1px solid ${CYN}55` }}
                        >
                          Source URL ↗
                        </a>
                        <div className="text-[9px] mt-1" style={{ color: GRY }}>
                          {ds.license_attribution.slice(0, 30)}…
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          ) : (
            <div className="space-y-3">
              <div className="text-[11px]" style={{ color: GRY }}>
                Historical benchmark evaluation runs stored in <code className="text-white">evaluation/results/</code>.
                Scores are never fabricated; if an evaluation has not been run, it is explicitly shown as <code className="text-amber-400">NOT_RUN</code>.
              </div>
              {evaluations.length === 0 ? (
                <div className="py-8 px-4 rounded-xl border text-center space-y-2" style={{ borderColor: 'rgba(255,255,255,0.1)', background: 'rgba(10,22,40,0.4)' }}>
                  <div className="text-sm font-bold" style={{ color: ORG }}>No Completed Benchmark Runs Found</div>
                  <div className="text-[11px]" style={{ color: GRY }}>
                    Benchmark scores will appear here after executing evaluation scripts on downloaded datasets:
                  </div>
                  <div className="text-[11px] p-2 rounded inline-block font-mono" style={{ background: '#050E1C', color: CYN }}>
                    python evaluation/vrsbench/evaluate_captioning.py<br/>
                    python evaluation/rsvqa/evaluate.py --variant lr<br/>
                    python evaluation/cdvqa/evaluate.py
                  </div>
                </div>
              ) : (
                <div className="space-y-2">
                  {evaluations.map((ev, i) => (
                    <div
                      key={i}
                      className="p-3 rounded-xl border flex flex-col gap-2"
                      style={{ background: 'rgba(10,22,40,0.6)', borderColor: `${CYN}33` }}
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-bold text-sm" style={{ color: WHT }}>{ev.dataset} — {ev.task}</span>
                        <span className="text-[10px] px-2 py-0.5 rounded font-bold" style={{ background: `${MNT}22`, color: MNT }}>
                          {ev.status}
                        </span>
                      </div>
                      <div className="text-[11px]" style={{ color: GRY }}>
                        Model: {ev.model} (v{ev.model_version}) | Evaluated on {ev.sample_count} samples ({ev.timestamp})
                      </div>
                      {ev.metrics && (
                        <div className="p-2 rounded text-[11px] font-mono" style={{ background: '#050E1C', color: MNT }}>
                          {JSON.stringify(ev.metrics, null, 2)}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
