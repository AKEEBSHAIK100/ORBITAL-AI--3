import React, { useState } from 'react'

interface DocumentationModalProps {
  isOpen: boolean
  onClose: () => void
}

type DocTab = 'architecture' | 'tasks' | 'taxonomy' | 'api' | 'citations'

export default function DocumentationModal({ isOpen, onClose }: DocumentationModalProps) {
  const [activeTab, setActiveTab] = useState<DocTab>('architecture')

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
        aria-labelledby="doc-modal-title"
        className="w-full max-w-5xl max-h-[90vh] flex flex-col rounded-2xl border shadow-2xl overflow-hidden"
        style={{
          background: '#07111F',
          borderColor: 'rgba(56, 189, 248, 0.25)',
          boxShadow: '0 25px 60px rgba(0, 0, 0, 0.85), 0 0 40px rgba(32, 217, 255, 0.08)',
        }}
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-[#050D18]">
          <div className="flex items-center gap-3">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 shadow-[0_0_8px_#20D9FF]" />
            <div>
              <h2 id="doc-modal-title" className="text-base font-bold text-slate-100 font-mono tracking-wide">
                ORBITAL-AI · TECHNICAL DOCUMENTATION & PROTOCOL
              </h2>
              <p className="text-[11px] font-mono text-slate-400">
                Agentic Remote-Sensing System Specification · SIH 2026 Demo
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="w-8 h-8 rounded-lg flex items-center justify-center text-slate-400 hover:text-white hover:bg-slate-800 transition-colors cursor-pointer text-sm"
          >
            ✕
          </button>
        </div>

        {/* Tab Navigation */}
        <div className="flex border-b border-slate-800 bg-[#081322] px-6 gap-2 pt-2 overflow-x-auto">
          {[
            { id: 'architecture' as const, label: '1. Agent Architecture' },
            { id: 'tasks' as const, label: '2. ISRO/SAC Task Workflows' },
            { id: 'taxonomy' as const, label: '3. BigEarthNet v2.0 Taxonomy' },
            { id: 'api' as const, label: '4. REST API Reference' },
            { id: 'citations' as const, label: '5. Academic References' },
          ].map(tab => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setActiveTab(tab.id)}
              className="px-4 py-2.5 text-xs font-mono font-medium rounded-t-lg transition-all cursor-pointer whitespace-nowrap border-b-2"
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

        {/* Tab Content */}
        <div className="p-6 overflow-y-auto space-y-6 text-slate-300 text-sm leading-relaxed max-h-[66vh]">
          {activeTab === 'architecture' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-xl font-bold text-slate-100 mb-2" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
                  Deterministic Agentic Orchestration Framework
                </h3>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Unlike opaque monolithic chatbots, SatQuery AI implements an observable pipeline separating query intent
                  classification, multi-sensor input validation, specialist tool execution, and transparent trace generation.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
                {[
                  { step: '01', name: 'Task Classifier', desc: 'Routes queries deterministically into captioning, grounding, bi-temporal change, or SAR fusion workflows.' },
                  { step: '02', name: 'Input Validator', desc: 'Checks sensor modality, image dimensions, radiometric scaling, and co-registration compatibility.' },
                  { step: '03', name: 'Specialist Models', desc: 'Executes BigEarthNet ResNet-50, YOLOv8-Seg instance tiler, or OpenCV cross-modal correlation.' },
                  { step: '04', name: 'Trace Generator', desc: 'Outputs machine-readable execution traces with exact latencies, parameters, and confidence rationale.' },
                ].map(s => (
                  <div key={s.step} className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 space-y-1.5">
                    <span className="text-[10px] font-mono text-cyan-400 font-bold">{s.step} · MODULE</span>
                    <h4 className="text-sm font-bold text-slate-100">{s.name}</h4>
                    <p className="text-xs text-slate-400">{s.desc}</p>
                  </div>
                ))}
              </div>

              <div className="p-4 rounded-xl border border-slate-800 bg-slate-950/60 font-mono text-xs text-slate-300 space-y-2">
                <div className="text-cyan-400 font-bold">Execution Trace Schema Guarantee:</div>
                <pre className="text-[11px] text-slate-400 overflow-x-auto p-2 bg-slate-900 rounded">
{`{
  "agent_version": "SatQuery-Agent-v2.1",
  "task_type": "caption" | "grounding" | "change_detection" | "sar_optical_fusion" | "building_audit",
  "tools_invoked": ["rs_task_classifier", "rs_input_validator", "ben_resnet50_v2", "yolo_segmenter"],
  "steps": [{ "step": 1, "tool": "...", "status": "success", "duration_ms": 12 }],
  "total_duration_ms": 45,
  "input_validation": { "valid": true, "notes": ["Dual-sensor alignment verified"] }
}`}
                </pre>
              </div>
            </div>
          )}

          {activeTab === 'tasks' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-xl font-bold text-slate-100 mb-2" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
                  SIH 2026 Remote-Sensing Task Workflows
                </h3>
                <p className="text-xs text-slate-400">
                  Workflow coverage mapped to the project problem requirements; benchmark metrics are reported only when independently verified.
                </p>
              </div>

              <div className="space-y-4">
                {[
                  {
                    title: '1. VRSBench Scene Captioning & Attribute Description',
                    badge: 'CAPTION',
                    color: '#20D9FF',
                    desc: 'Generates multi-attribute natural language descriptions of the scene, identifying continuous urban fabric, road transport corridors, agricultural plots, and natural vegetation.',
                    metric: 'Evaluation plan: CIDEr, BLEU-4 and multi-label F1 can be reported only from a reproducible VRSBench test run; no unverified score is shown here.',
                  },
                  {
                    title: '2. RSVQA Text-Guided Region Grounding',
                    badge: 'GROUNDING',
                    color: '#35E0B8',
                    desc: 'Localizes requested entities (such as "highlight the water body") into normalized bounding coordinates [x_percent, y_percent, w_percent, h_percent] rendered dynamically onto the image canvas.',
                    metric: 'Evaluation plan: mIoU on an explicitly identified RSVQA-HR split; production results are labelled as uncalibrated unless ground-truth evaluation has been completed.',
                  },
                  {
                    title: '3. CDVQA Bi-Temporal Change Detection',
                    badge: 'CHANGE-VQA',
                    color: '#FF9F43',
                    desc: 'Co-registers pairs of satellite images acquired at different passes, detects structural additions or environmental deltas, and answers comparative questions regarding urban expansion and deforestation.',
                    metric: 'Evaluation plan: change classification accuracy and spatial agreement on a reproducible CDVQA split; no fabricated metric is presented.',
                  },
                  {
                    title: '4. Optical–SAR Cross-Modal Joint Analysis',
                    badge: 'CROSS-MODAL',
                    color: '#20D9FF',
                    desc: 'Combines optical reflectance (chlorophyll absorption, spectral indices) with microwave SAR backscatter (dielectric properties, double-bounce scattering) for cloud penetration and all-weather structural audit.',
                    metric: 'Evaluation plan: modality-specific structural/correlation diagnostics after input co-registration; these are not presented as benchmark scores.',
                  },
                  {
                    title: '5. Deep Learning Instance Building Segmentation',
                    badge: 'YOLO-SEG',
                    color: '#35E0B8',
                    desc: 'Splits ultra-high-resolution scenes into overlapping 512px tiles, performs polygon instance segmentation, and filters overlaps via IoU Non-Maximum Suppression (NMS) with unique IDs (B001, B002).',
                    metric: 'Evaluation plan: Precision, Recall and mAP@0.5 on a documented building-footprint ground-truth set.',
                  },
                ].map(t => (
                  <div key={t.badge} className="p-4 rounded-xl border border-slate-800 bg-slate-900/50 space-y-2">
                    <div className="flex items-center justify-between">
                      <h4 className="text-sm font-bold text-slate-100">{t.title}</h4>
                      <span
                        className="text-[10px] font-mono px-2 py-0.5 rounded font-bold"
                        style={{ background: `${t.color}15`, color: t.color, border: `1px solid ${t.color}35` }}
                      >
                        {t.badge}
                      </span>
                    </div>
                    <p className="text-xs text-slate-300 leading-relaxed">{t.desc}</p>
                    <div className="text-[11px] font-mono text-slate-400 pt-2 border-t border-slate-800/80">
                      Standard: <span className="text-slate-300">{t.metric}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeTab === 'taxonomy' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-xl font-bold text-slate-100 mb-2" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
                  BigEarthNet v2.0 (reBEN) 19-Class Taxonomy
                </h3>
                <p className="text-xs text-slate-400">
                  Standardized land-cover nomenclature adapted from the Corine Land Cover (CLC) hierarchy
                </p>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2.5 font-mono text-xs">
                {[
                  { id: '01', name: 'Urban fabric', cat: 'Artificial Surfaces' },
                  { id: '02', name: 'Industrial or commercial units', cat: 'Artificial Surfaces' },
                  { id: '03', name: 'Arable land', cat: 'Agricultural' },
                  { id: '04', name: 'Permanent crops', cat: 'Agricultural' },
                  { id: '05', name: 'Pastures', cat: 'Agricultural' },
                  { id: '06', name: 'Complex cultivation patterns', cat: 'Agricultural' },
                  { id: '07', name: 'Land principally occupied by agriculture + natural veg', cat: 'Agricultural' },
                  { id: '08', name: 'Agro-forestry areas', cat: 'Forest / Scrub' },
                  { id: '09', name: 'Broad-leaved forest', cat: 'Forest / Scrub' },
                  { id: '10', name: 'Coniferous forest', cat: 'Forest / Scrub' },
                  { id: '11', name: 'Mixed forest', cat: 'Forest / Scrub' },
                  { id: '12', name: 'Natural grassland & sparsely vegetated areas', cat: 'Natural' },
                  { id: '13', name: 'Moors, heathland & sclerophyllous vegetation', cat: 'Natural' },
                  { id: '14', name: 'Transitional woodland/shrub', cat: 'Natural' },
                  { id: '15', name: 'Beaches, dunes, sands', cat: 'Natural' },
                  { id: '16', name: 'Inland wetlands', cat: 'Wetlands' },
                  { id: '17', name: 'Coastal wetlands', cat: 'Wetlands' },
                  { id: '18', name: 'Inland waters', cat: 'Water' },
                  { id: '19', name: 'Marine waters', cat: 'Water' },
                ].map(c => (
                  <div key={c.id} className="p-3 rounded-lg border border-slate-800 bg-slate-900/40 flex items-start gap-2.5">
                    <span className="text-[10px] text-cyan-400 font-bold shrink-0">{c.id}</span>
                    <div className="min-w-0">
                      <div className="text-slate-200 font-sans font-medium text-xs leading-snug">{c.name}</div>
                      <div className="text-[10px] text-slate-500 mt-0.5">{c.cat}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeTab === 'api' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-xl font-bold text-slate-100 mb-2" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
                  REST API Endpoint Reference
                </h3>
                <p className="text-xs text-slate-400">
                  Unified endpoints supported across FastAPI (Port 8000), Express (Port 8787), and Vercel Serverless
                </p>
              </div>

              <div className="space-y-4">
                {[
                  {
                    method: 'POST',
                    path: '/classify',
                    desc: 'Runs BigEarthNet v2.0 ResNet-50 19-class land-cover classification on an uploaded satellite image.',
                    payload: '{"image": "data:image/jpeg;base64,...", "top_k": 8, "threshold": 0.2}',
                  },
                  {
                    method: 'POST',
                    path: '/api/analyze',
                    desc: 'Core vision-language reasoning endpoint for single image VQA, scene captioning, and spatial grounding.',
                    payload: '{"question": "Describe the land-cover...", "image": "data:image/jpeg;base64,...", "sessionId": "..."}',
                  },
                  {
                    method: 'POST',
                    path: '/api/compare',
                    desc: 'CDVQA bi-temporal change detection comparing earlier baseline pass against recent satellite observation.',
                    payload: '{"question": "What changed between these dates?", "beforeImage": "...", "afterImage": "..."}',
                  },
                  {
                    method: 'POST',
                    path: '/api/fuse',
                    desc: 'Joint optical and SAR cross-modal feature extraction with SSIM coregistration and backscatter analysis.',
                    payload: '{"opticalImage": "...", "sarImage": "...", "question": "Identify built-up regions..."}',
                  },
                  {
                    method: 'POST',
                    path: '/analyze/buildings',
                    desc: 'Tiled YOLOv8 rooftop instance segmentation returning polygon coordinates and IoU deduplicated counts.',
                    payload: '{"image": "data:image/jpeg;base64,..."} or multipart/form-data',
                  },
                ].map(ep => (
                  <div key={ep.path} className="p-4 rounded-xl border border-slate-800 bg-slate-900/50 space-y-2">
                    <div className="flex items-center gap-2 font-mono text-xs">
                      <span className="px-2 py-0.5 rounded font-bold bg-cyan-400/20 text-cyan-400 border border-cyan-400/30">
                        {ep.method}
                      </span>
                      <span className="text-slate-100 font-bold">{ep.path}</span>
                    </div>
                    <p className="text-xs text-slate-300">{ep.desc}</p>
                    <div className="p-2 rounded bg-slate-950 text-[11px] font-mono text-slate-400 overflow-x-auto">
                      Payload: <code className="text-slate-200">{ep.payload}</code>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeTab === 'citations' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-xl font-bold text-slate-100 mb-2" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
                  Academic Literature & Citations
                </h3>
                <p className="text-xs text-slate-400">
                  Key foundational papers and dataset benchmarks incorporated into SatQuery AI
                </p>
              </div>

              <div className="space-y-3 text-xs text-slate-300 font-mono">
                <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/40 space-y-1">
                  <div className="text-cyan-400 font-bold">[1] BigEarthNet v2.0 (reBEN)</div>
                  <p className="text-slate-300 font-sans text-xs">
                    Clasen, K., Hackel, L., Burgert, T., et al. (2025). <em>"reBEN: Refined BigEarthNet Dataset for Remote Sensing Image Analysis"</em>.
                    IEEE International Geoscience and Remote Sensing Symposium (IGARSS 2025). arXiv:2407.03653.
                  </p>
                </div>

                <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/40 space-y-1">
                  <div className="text-cyan-400 font-bold">[2] VRSBench Benchmark</div>
                  <p className="text-slate-300 font-sans text-xs">
                    Li, K., et al. (2024). <em>"VRSBench: A Versatile Vision-Language Benchmark for Remote Sensing"</em>.
                    Comprehensive scene captioning and multi-attribute localization protocol.
                  </p>
                </div>

                <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/40 space-y-1">
                  <div className="text-cyan-400 font-bold">[3] RSVQA High Resolution</div>
                  <p className="text-slate-300 font-sans text-xs">
                    Lobry, S., Marcos, D., Murray, J., & Tuia, D. (2020). <em>"RSVQA: Visual Question Answering for Remote Sensing Data"</em>.
                    IEEE Transactions on Geoscience and Remote Sensing.
                  </p>
                </div>

                <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/40 space-y-1">
                  <div className="text-cyan-400 font-bold">[4] CDVQA Change Detection</div>
                  <p className="text-slate-300 font-sans text-xs">
                    Yuan, Z., et al. (2023). <em>"Change Detection Visual Question Answering for Remote Sensing"</em>.
                    Standardized bi-temporal evaluation framework for satellite change reasoning.
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-3.5 border-t border-slate-800 bg-[#050D18] flex items-center justify-between text-xs font-mono text-slate-400">
          <span>ISRO/SAC Remote-Sensing VLM Challenge · SatQuery AI</span>
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 transition-colors cursor-pointer"
          >
            Close Documentation
          </button>
        </div>
      </div>
    </div>
  )
}
