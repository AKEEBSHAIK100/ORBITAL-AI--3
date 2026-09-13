import React, { useState } from 'react'
import { BIGEARTHNET_19_CLASSES, BENCHMARK_METRICS } from '../data/BigEarthNet'

const CYN = '#20D9FF'
const ORG = '#FF9F43'
const MNT = '#35E0B8'
const WHT = '#F4F7FA'
const GRY = '#9AA9B8'
const CB = 'rgba(32,217,255,0.18)'

interface Props {
  isOpen: boolean
  onClose: () => void
}

export default function EvaluationCriteriaModal({ isOpen, onClose }: Props) {
  const [activeTab, setActiveTab] = useState<'benchmarks' | 'taxonomy' | 'datasets'>('benchmarks')

  if (!isOpen) return null

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: 'rgba(2,8,16,0.85)', backdropFilter: 'blur(8px)' }}
      onClick={onClose}
    >
      <div
        className="relative w-full max-w-3xl max-h-[90vh] rounded-2xl flex flex-col overflow-hidden border shadow-2xl"
        style={{
          background: 'linear-gradient(180deg, #071022 0%, #030814 100%)',
          borderColor: CB,
          boxShadow: '0 0 50px rgba(32,217,255,0.18)',
        }}
        onClick={e => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div
          className="flex items-center justify-between px-6 py-4 border-b"
          style={{ borderColor: 'rgba(32,217,255,0.15)', background: 'rgba(6,13,26,0.8)' }}
        >
          <div className="flex items-center gap-3">
            <div
              className="w-8 h-8 rounded-lg flex items-center justify-center font-mono text-xs font-bold"
              style={{ background: `${CYN}22`, color: CYN, border: `1px solid ${CYN}66` }}
            >
              ISRO
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-sm font-bold tracking-wider uppercase font-mono" style={{ color: WHT }}>
                  Evaluation & Judging Criteria
                </h3>
                <span
                  className="px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase"
                  style={{ background: `${MNT}22`, color: MNT, border: `1px solid ${MNT}55` }}
                >
                  SAC BENCHMARK SPEC
                </span>
              </div>
              <p className="text-[11px] font-mono" style={{ color: GRY }}>
                Public Benchmark Test Subsets + ISRO/SAC Evaluation Protocol
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

        {/* Tab Selector */}
        <div className="flex border-b px-6 py-2 gap-3 text-xs font-mono" style={{ borderColor: 'rgba(32,217,255,0.12)', background: '#050E1C' }}>
          {[
            { id: 'benchmarks', label: 'Evaluation Metrics & Datasets' },
            { id: 'taxonomy', label: 'BigEarthNet Domain Taxonomy' },
            { id: 'datasets', label: 'Sensors (Cartosat & RISAT)' },
          ].map(t => (
            <button
              key={t.id}
              onClick={() => setActiveTab(t.id as any)}
              className="px-3 py-1.5 rounded-lg transition-all cursor-pointer"
              style={{
                background: activeTab === t.id ? `${CYN}22` : 'transparent',
                color: activeTab === t.id ? CYN : GRY,
                border: activeTab === t.id ? `1px solid ${CYN}66` : '1px solid transparent',
              }}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* Modal Body */}
        <div className="p-6 overflow-y-auto space-y-6 text-xs leading-relaxed" style={{ color: WHT }}>
          {activeTab === 'benchmarks' && (
            <div className="space-y-4">
              <div className="p-3.5 rounded-xl border" style={{ background: 'rgba(32,217,255,0.05)', borderColor: `${CYN}33` }}>
                <div className="font-semibold text-sm mb-1 font-mono" style={{ color: CYN }}>
                  Normalized Evaluation Protocol
                </div>
                <p style={{ color: GRY }}>
                  Final evaluation uses prescribed public benchmark test subsets and an ISRO/SAC evaluation dataset. Scores are normalized across task metrics before aggregation:
                </p>
              </div>

              {/* Table */}
              <div className="rounded-xl overflow-hidden border" style={{ borderColor: 'rgba(255,255,255,0.1)' }}>
                <table className="w-full text-left border-collapse text-[11px] font-mono">
                  <thead>
                    <tr style={{ background: 'rgba(255,255,255,0.06)', color: CYN }}>
                      <th className="p-3 border-b border-r" style={{ borderColor: 'rgba(255,255,255,0.1)' }}>Benchmark / Task</th>
                      <th className="p-3 border-b border-r" style={{ borderColor: 'rgba(255,255,255,0.1)' }}>Target Task</th>
                      <th className="p-3 border-b border-r" style={{ borderColor: 'rgba(255,255,255,0.1)' }}>Primary Evaluation Metrics</th>
                      <th className="p-3 border-b" style={{ borderColor: 'rgba(255,255,255,0.1)' }}>Data Test Subset</th>
                    </tr>
                  </thead>
                  <tbody>
                    {BENCHMARK_METRICS.map((m, i) => (
                      <tr key={i} className="hover:bg-white/5 transition-colors" style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                        <td className="p-3 font-bold border-r" style={{ color: MNT, borderColor: 'rgba(255,255,255,0.1)' }}>{m.benchmark}</td>
                        <td className="p-3 border-r text-slate-300" style={{ borderColor: 'rgba(255,255,255,0.1)' }}>{m.task}</td>
                        <td className="p-3 border-r font-mono text-cyan-300" style={{ borderColor: 'rgba(255,255,255,0.1)' }}>{m.metrics}</td>
                        <td className="p-3 text-slate-400">{m.dataset_split}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-2">
                <div className="p-3 rounded-xl border" style={{ background: 'rgba(6,13,26,0.6)', borderColor: CB }}>
                  <div className="font-bold text-xs font-mono mb-1" style={{ color: ORG }}>ISRO/SAC Test Protocol</div>
                  <p className="text-[11px] text-slate-300">
                    The ISRO/SAC evaluation set contains pre-georeferenced and co-registered Cartosat-2S optical (0.65m) and RISAT SAR (1.0m) image pairs with reference annotations, masks, and bounding boxes.
                  </p>
                </div>
                <div className="p-3 rounded-xl border" style={{ background: 'rgba(6,13,26,0.6)', borderColor: CB }}>
                  <div className="font-bold text-xs font-mono mb-1" style={{ color: MNT }}>Agentic Scoring Criteria</div>
                  <p className="text-[11px] text-slate-300">
                    Evaluated on task routing accuracy, radiometric input validation precision, specialized tool sequencing, and observable execution trace completeness.
                  </p>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'taxonomy' && (
            <div className="space-y-4">
              <div className="p-3.5 rounded-xl border" style={{ background: 'rgba(53,224,184,0.06)', borderColor: `${MNT}44` }}>
                <div className="font-semibold text-sm font-mono mb-1" style={{ color: MNT }}>
                  BigEarthNet 43-Class / 19-Class Domain Adaptation
                </div>
                <p style={{ color: GRY }}>
                  SatQuery AI adapts vision-language representations to multi-sensor satellite imagery using the hierarchical BigEarthNet Corine Land Cover taxonomy (BigEarthNet.txt):
                </p>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 font-mono text-[11px]">
                {BIGEARTHNET_19_CLASSES.map((cls, idx) => (
                  <div
                    key={idx}
                    className="p-2.5 rounded-lg border flex items-center gap-2"
                    style={{ background: 'rgba(255,255,255,0.03)', borderColor: 'rgba(255,255,255,0.08)' }}
                  >
                    <span className="text-[9px] px-1.5 py-0.5 rounded font-bold" style={{ background: `${CYN}22`, color: CYN }}>
                      {String(idx + 1).padStart(2, '0')}
                    </span>
                    <span className="text-slate-200 truncate" title={cls}>{cls}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeTab === 'datasets' && (
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 rounded-xl border space-y-2" style={{ background: 'rgba(6,13,26,0.8)', borderColor: CB }}>
                  <div className="flex items-center justify-between font-mono font-bold" style={{ color: CYN }}>
                    <span>CARTOSAT-2S (OPTICAL)</span>
                    <span className="text-[10px] px-2 py-0.5 rounded" style={{ background: `${CYN}22` }}>0.65m GSD</span>
                  </div>
                  <p className="text-[11px] text-slate-300">
                    High-resolution panchromatic and multi-spectral imagery. Sensitive to optical albedo, rooftop geometries, chlorophyll absorption (NDVI), and hydrological boundaries.
                  </p>
                  <div className="text-[10px] font-mono text-cyan-400">
                    Bands: Pan (0.65m), B, G, R, NIR (2.0m)
                  </div>
                </div>

                <div className="p-4 rounded-xl border space-y-2" style={{ background: 'rgba(6,13,26,0.8)', borderColor: `${ORG}55` }}>
                  <div className="flex items-center justify-between font-mono font-bold" style={{ color: ORG }}>
                    <span>RISAT-1A / SENTINEL-1 (SAR)</span>
                    <span className="text-[10px] px-2 py-0.5 rounded" style={{ background: `${ORG}22` }}>C-Band Radar</span>
                  </div>
                  <p className="text-[11px] text-slate-300">
                    Synthetic Aperture Radar operating in C-band. Penetrates clouds and canopy, detecting surface roughness, structural dielectric properties, and double-bounce returns from built structures.
                  </p>
                  <div className="text-[10px] font-mono text-amber-400">
                    Polarization: VV, VH, HH, HV (Hybrid-Pol)
                  </div>
                </div>
              </div>

              <div className="p-3.5 rounded-xl border font-mono text-[11px] space-y-1" style={{ background: 'rgba(255,255,255,0.02)', borderColor: 'rgba(255,255,255,0.08)' }}>
                <div className="text-slate-400 font-bold">Supported Image Formats:</div>
                <div className="text-slate-300">
                  • <strong>GeoTIFF (.tif / .tiff / .geotiff)</strong>: Native geospatial coordinate systems, multi-band 16-bit radiometric calibration.
                </div>
                <div className="text-slate-300">
                  • <strong>PNG / JPEG (.png / .jpg)</strong>: Approved public benchmark datasets (VRSBench, RSVQA, CDVQA).
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div
          className="flex items-center justify-between px-6 py-3 border-t text-xs font-mono"
          style={{ borderColor: 'rgba(32,217,255,0.15)', background: 'rgba(6,13,26,0.8)' }}
        >
          <span style={{ color: GRY }}>SatQuery AI · ISRO/SAC VLM Challenge Specification</span>
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg text-xs font-mono font-medium transition-all cursor-pointer"
            style={{ background: `${CYN}22`, color: CYN, border: `1px solid ${CYN}55` }}
          >
            CLOSE
          </button>
        </div>
      </div>
    </div>
  )
}
