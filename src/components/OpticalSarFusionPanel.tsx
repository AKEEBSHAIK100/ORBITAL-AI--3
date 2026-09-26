import React, { useState, useRef } from 'react'
import { FusionFeatures, ExecutionTrace } from '../lib/agentController'
import { compressToJpeg } from '../utils/imageUtils'

const CYN = '#20D9FF'
const ORG = '#FF9F43'
const MNT = '#35E0B8'
const WHT = '#F4F7FA'
const GRY = '#9AA9B8'
const PNL = '#060D1A'
const CB = 'rgba(32,217,255,0.18)'


interface Props {
  onRunFusion: (opticalDataUrl: string, sarDataUrl: string, query?: string) => Promise<void>
  busy: boolean
  fusionFeatures: FusionFeatures | null
  lastFusionTrace: ExecutionTrace | null
  onOpenTrace: (trace: ExecutionTrace) => void
}

export default function OpticalSarFusionPanel({
  onRunFusion,
  busy,
  fusionFeatures,
  lastFusionTrace,
  onOpenTrace,
}: Props) {
  const [opticalImage, setOpticalImage] = useState<string | null>(null)
  const [sarImage, setSarImage] = useState<string | null>(null)
  const [opticalLabel, setOpticalLabel] = useState<string>('Cartosat-2S (Optical RGB)')
  const [sarLabel, setSarLabel] = useState<string>('RISAT-1A / Sentinel-1 (C-Band SAR)')
  const [customQuery, setCustomQuery] = useState<string>('In simple language, what does each image show, and what additional information does the SAR image provide?')
  const [uploadError, setUploadError] = useState<string | null>(null)

  const optInputRef = useRef<HTMLInputElement>(null)
  const sarInputRef = useRef<HTMLInputElement>(null)

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>, isSar: boolean) => {
    const file = e.target.files?.[0]
    if (!file) return
    setUploadError(null)
    try {
      // compressToJpeg always outputs data:image/jpeg;base64,... — safe for parseDataUrl()
      // on the server and handles GeoTIFF / TIFF via canvas fallback.
      const jpegDataUrl = await compressToJpeg(file, isSar ? 'SAR' : 'Optical')
      if (isSar) {
        setSarImage(jpegDataUrl)
        setSarLabel(file.name)
      } else {
        setOpticalImage(jpegDataUrl)
        setOpticalLabel(file.name)
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to load image.'
      setUploadError(msg)
    }
    // Reset input so the same file can be re-selected after clearing
    e.target.value = ''
  }

  const handleExecute = () => {
    if (!opticalImage || !sarImage) return
    onRunFusion(opticalImage, sarImage, customQuery)
  }

  return (
    <div className="w-full space-y-4">
      {/* Header bar */}
      <div
        className="p-4 rounded-xl border flex flex-col sm:flex-row sm:items-center justify-between gap-3"
        style={{ background: 'rgba(6,13,26,0.75)', borderColor: CB }}
      >
        <div className="flex items-center gap-3">
          <div
            className="w-10 h-10 rounded-lg flex items-center justify-center font-mono font-bold text-sm"
            style={{ background: `${CYN}22`, color: CYN, border: `1px solid ${CYN}66` }}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
            </svg>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-bold tracking-wider uppercase font-mono" style={{ color: WHT }}>
                Optical–SAR Multi-Modal Fusion Engine
              </h3>
              <span
                className="px-2 py-0.5 rounded text-[10px] font-mono uppercase font-bold"
                style={{ background: `${MNT}22`, color: MNT, border: `1px solid ${MNT}55` }}
              >
                OPTICAL + SAR INPUTS
              </span>
            </div>
            <p className="text-xs font-mono" style={{ color: GRY }}>
              Joint analysis: optical imagery + microwave radar observations
            </p>
          </div>
        </div>

        <button
          onClick={() => {} }
          className="px-3 py-1.5 rounded-lg text-xs font-mono font-semibold transition-all cursor-default flex items-center gap-1.5 self-start sm:self-auto opacity-70" disabled
          style={{
            background: 'rgba(32,217,255,0.08)',
            color: CYN,
            border: `1px solid ${CYN}44`,
          }}
          title="Upload two observations of the same area; co-registration is verified by the backend when metadata is available."
        >
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="2" />
            <path d="M16.24 7.76a6 6 0 0 1 0 8.49m-8.48-.01a6 6 0 0 1 0-8.49m11.31-2.82a10 10 0 0 1 0 14.14m-14.14 0a10 10 0 0 1 0-14.14" />
          </svg>
          <span>REAL INPUTS ONLY</span>
        </button>
      </div>

      {/* Dual Upload Slots */}
      {uploadError && (
        <div
          className="px-3 py-2 rounded-lg text-xs font-mono flex items-center gap-2"
          style={{ background: 'rgba(255,99,71,0.12)', color: '#FF6347', border: '1px solid rgba(255,99,71,0.3)' }}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="shrink-0">
            <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
            <line x1="12" y1="9" x2="12" y2="13" />
            <line x1="12" y1="17" x2="12.01" y2="17" />
          </svg>
          <span>{uploadError}</span>
        </div>
      )}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Slot 1: Optical */}
        <div
          className="p-4 rounded-xl border flex flex-col justify-between relative overflow-hidden group min-h-[220px]"
          style={{
            background: 'rgba(7,16,34,0.6)',
            borderColor: opticalImage ? 'rgba(32,217,255,0.4)' : 'rgba(255,255,255,0.1)',
          }}
        >
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-[11px] font-mono font-bold uppercase tracking-wider" style={{ color: CYN }}>
                MODALITY 1: OPTICAL / MULTI-SPECTRAL
              </span>
              <span className="text-[10px] font-mono text-white/50">Optical / multispectral</span>
            </div>

            {opticalImage ? (
              <div className="relative rounded-lg overflow-hidden border border-white/10 h-36 bg-black/40">
                <img src={opticalImage} alt="Optical" className="w-full h-full object-cover" />
                <div className="absolute bottom-0 inset-x-0 bg-black/70 px-2 py-1 text-[10px] font-mono text-white/80 truncate">
                  {opticalLabel}
                </div>
              </div>
            ) : (
              <div
                onClick={() => optInputRef.current?.click()}
                className="h-36 rounded-lg border-2 border-dashed flex flex-col items-center justify-center p-4 text-center cursor-pointer transition-colors"
                style={{ borderColor: 'rgba(32,217,255,0.3)', background: 'rgba(32,217,255,0.02)' }}
              >
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" className="mb-1 text-cyan-400">
                  <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
                  <circle cx="12" cy="13" r="4" />
                </svg>
                <span className="text-xs font-mono font-semibold" style={{ color: WHT }}>
                  Upload Optical RGB / Multi-spectral
                </span>
                <span className="text-[10px] font-mono text-white/40 mt-0.5">
                  GeoTIFF, TIFF, JPEG, PNG (Max 15MB)
                </span>
              </div>
            )}
          </div>

          <div className="flex items-center justify-between mt-3 pt-2 border-t border-white/5">
            <button
              onClick={() => optInputRef.current?.click()}
              className="text-[11px] font-mono text-white/60 hover:text-white transition-colors cursor-pointer"
            >
              {opticalImage ? 'Replace Optical Image' : 'Select File'}
            </button>
            <input
              ref={optInputRef}
              type="file"
              accept="image/jpeg,image/png,image/webp,image/tiff,.tif,.tiff"
              className="hidden"
              onChange={e => handleFileUpload(e, false)}
            />
          </div>
        </div>

        {/* Slot 2: SAR */}
        <div
          className="p-4 rounded-xl border flex flex-col justify-between relative overflow-hidden group min-h-[220px]"
          style={{
            background: 'rgba(7,16,34,0.6)',
            borderColor: sarImage ? 'rgba(53,224,184,0.4)' : 'rgba(255,255,255,0.1)',
          }}
        >
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-[11px] font-mono font-bold uppercase tracking-wider" style={{ color: MNT }}>
                MODALITY 2: SYNTHETIC APERTURE RADAR (SAR)
              </span>
              <span className="text-[10px] font-mono text-white/50">SAR backscatter</span>
            </div>

            {sarImage ? (
              <div className="relative rounded-lg overflow-hidden border border-white/10 h-36 bg-black/40">
                <img src={sarImage} alt="SAR" className="w-full h-full object-cover filter contrast-125" />
                <div className="absolute bottom-0 inset-x-0 bg-black/70 px-2 py-1 text-[10px] font-mono text-white/80 truncate">
                  {sarLabel}
                </div>
              </div>
            ) : (
              <div
                onClick={() => sarInputRef.current?.click()}
                className="h-36 rounded-lg border-2 border-dashed flex flex-col items-center justify-center p-4 text-center cursor-pointer transition-colors"
                style={{ borderColor: 'rgba(53,224,184,0.3)', background: 'rgba(53,224,184,0.02)' }}
              >
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" className="mb-1 text-emerald-400">
                  <circle cx="12" cy="12" r="2" />
                  <path d="M4.93 4.93a10 10 0 0 1 14.14 0" />
                  <path d="M7.76 7.76a6 6 0 0 1 8.48 0" />
                  <line x1="12" y1="14" x2="12" y2="22" />
                </svg>
                <span className="text-xs font-mono font-semibold" style={{ color: WHT }}>
                  Upload Co-Registered SAR Observation
                </span>
                <span className="text-[10px] font-mono text-white/40 mt-0.5">
                  Backscatter Intensity / GeoTIFF / TIFF
                </span>
              </div>
            )}
          </div>

          <div className="flex items-center justify-between mt-3 pt-2 border-t border-white/5">
            <button
              onClick={() => sarInputRef.current?.click()}
              className="text-[11px] font-mono text-white/60 hover:text-white transition-colors cursor-pointer"
            >
              {sarImage ? 'Replace SAR Image' : 'Select File'}
            </button>
            <input
              ref={sarInputRef}
              type="file"
              accept="image/jpeg,image/png,image/webp,image/tiff,.tif,.tiff"
              className="hidden"
              onChange={e => handleFileUpload(e, true)}
            />
          </div>
        </div>
      </div>

      {/* Query Bar & Trigger */}
      <div
        className="p-3.5 rounded-xl border flex flex-col sm:flex-row items-center gap-3"
        style={{ background: 'rgba(6,13,26,0.85)', borderColor: CB }}
      >
        <div className="flex-1 w-full">
          <input
            type="text"
            value={customQuery}
            onChange={e => setCustomQuery(e.target.value)}
            placeholder="Specify cross-modal question or leave default..."
            className="w-full bg-black/30 border border-white/10 rounded-lg px-3 py-2 text-xs font-mono text-white placeholder-white/30 focus:outline-none focus:border-[#20D9FF]"
          />
        </div>

        <button
          onClick={handleExecute}
          disabled={busy || !opticalImage || !sarImage}
          className="w-full sm:w-auto px-5 py-2 rounded-lg text-xs font-mono font-bold uppercase transition-all flex items-center justify-center gap-2 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed hover:brightness-110"
          style={{
            background: CYN,
            color: '#020810',
            boxShadow: '0 0 20px rgba(32,217,255,0.25)',
          }}
        >
          {busy ? (
            <>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className="animate-spin">
                <circle cx="12" cy="12" r="10" strokeOpacity="0.25" />
                <path d="M12 2a10 10 0 0 1 10 10" />
              </svg>
              <span>COMPUTING FUSION...</span>
            </>
          ) : (
            <>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
              </svg>
              <span>RUN JOINT FUSION ANALYSIS</span>
            </>
          )}
        </button>
      </div>

      {/* Real-time Multi-Modal Telemetry Gauges */}
      {fusionFeatures && (
        <div
          className="p-4 rounded-xl border space-y-3"
          style={{ background: 'rgba(7,16,34,0.7)', borderColor: 'rgba(53,224,184,0.3)' }}
        >
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono font-bold uppercase tracking-wider" style={{ color: MNT }}>
              EXTRACTED CROSS-MODAL TELEMETRY MATRIX
            </span>
            {lastFusionTrace && (
              <button
                onClick={() => onOpenTrace(lastFusionTrace)}
                className="px-2.5 py-1 rounded text-[10px] font-mono font-bold transition-all cursor-pointer flex items-center gap-1.5"
                style={{ background: `${CYN}22`, color: CYN, border: `1px solid ${CYN}88` }}
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="11" cy="11" r="8" />
                  <line x1="21" y1="21" x2="16.65" y2="16.65" />
                </svg>
                <span>INSPECT AGENT TRACE</span>
              </button>
            )}
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
            {/* Optical NDVI */}
            <div className="p-2.5 rounded-lg bg-black/30 border border-white/5">
              <span className="text-[10px] block text-white/40">OPTICAL NDVI PROXY</span>
              <span className="text-sm font-bold" style={{ color: MNT }}>
                {Math.round(fusionFeatures.optical.vegetation_fraction * 100)}%
              </span>
              <span className="text-[10px] block text-white/50">Canopy Chlorophyll</span>
            </div>

            {/* SAR Backscatter */}
            <div className="p-2.5 rounded-lg bg-black/30 border border-white/5">
              <span className="text-[10px] block text-white/40">SAR BACKSCATTER</span>
              <span className="text-sm font-bold" style={{ color: CYN }}>
                {fusionFeatures.sar.mean_backscatter_db} dB
              </span>
              <span className="text-[10px] block text-white/50">Microwave Return</span>
            </div>

            {/* Speckle Noise */}
            <div className="p-2.5 rounded-lg bg-black/30 border border-white/5">
              <span className="text-[10px] block text-white/40">SPECKLE INDEX (σ²/μ²)</span>
              <span className="text-sm font-bold" style={{ color: ORG }}>
                {fusionFeatures.sar.speckle_index}
              </span>
              <span className="text-[10px] block text-white/50">Radar Noise Model</span>
            </div>

            {/* Cross-Correlation */}
            <div className="p-2.5 rounded-lg bg-black/30 border border-white/5">
              <span className="text-[10px] block text-white/40">CROSS-CORRELATION</span>
              <span className="text-sm font-bold" style={{ color: WHT }}>
                {fusionFeatures.cross_modal.cross_correlation}
              </span>
              <span className="text-[10px] block" style={{ color: MNT }}>
                {fusionFeatures.cross_modal.fusion_confidence.toUpperCase()} FIDELITY
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
