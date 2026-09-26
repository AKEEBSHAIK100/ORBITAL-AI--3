import React, { useState, useEffect, useRef, useCallback } from 'react'
import Globe from './components/Globe'
import { SESSION_CALL_LIMIT } from './lib/constants'
import AgentTraceModal from './components/AgentTraceModal'
import OpticalSarFusionPanel from './components/OpticalSarFusionPanel'
import EvaluationCriteriaModal from './components/EvaluationCriteriaModal'
import CookieConsentBanner from './components/CookieConsentBanner'
import LegalModals, { LegalModalType } from './components/LegalModals'
import DocumentationModal from './components/DocumentationModal'
import ContactModal from './components/ContactModal'
import DashboardView from './components/DashboardView'
import { ExecutionTrace, ExecutionTraceStep, FusionFeatures, classifyTask } from './lib/agentController'
import { createAnalysisSession, recordAnalysisRun } from './lib/supabase'

export const API_BASE = (() => {
  const envUrl = (import.meta.env.VITE_API_BASE_URL || import.meta.env.VITE_API_URL || '').replace(/\/$/, '')
  if (typeof window !== 'undefined' && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
    if (envUrl.includes('localhost') || envUrl.includes('127.0.0.1')) {
      return ''
    }
  }
  return envUrl
})()

// ── Design tokens (WCAG compliant high-contrast developer grade) ──────────────
const C = {
  cyan:        '#20D9FF',
  orange:      '#FF9F43',
  mint:        '#35E0B8',
  white:       '#F8FAFC',
  muted:       '#94A3B8',
  dim:         '#64748B',
  danger:      '#FF6B6B',
  base:        '#030712',
  surface:     '#07111F',
  card:        '#0F1E36',
  border:      'rgba(56,189,248,0.16)',
  borderHover: 'rgba(56,189,248,0.32)',
} as const

// ── Representative Natural-Language Queries ──────────────────────────────────
export const OFFICIAL_REPRESENTATIVE_QUERIES = [
  { query: 'Describe this scene.', task: 'Scene overview & land cover', badge: 'SCENE', color: C.cyan },
  { query: 'What type of land is present?', task: 'Dominant land-cover classification', badge: 'LAND COVER', color: C.mint },
  { query: 'Is there vegetation present?', task: 'Vegetation assessment', badge: 'VEGETATION', color: C.mint },
  { query: 'What changed between these images?', task: 'Bi-temporal change detection', badge: 'CHANGE', color: C.orange },
  { query: 'Compare the optical and radar observations.', task: 'Optical + SAR cross-modal analysis', badge: 'OPTICAL+SAR', color: C.cyan },
]

// ── Types ─────────────────────────────────────────────────────────────────────
type HiddenLayer = 'drought' | 'harvest' | 'flood' | 'urban' | 'roads'
type AppMode = 'single' | 'compare' | 'fusion'
type Terrain = 'vegetation' | 'water' | 'urban' | 'arid'

export type Region = { x_percent: number; y_percent: number; w_percent: number; h_percent: number }

type Analysis = {
  answer: string
  confidence: 'high' | 'medium' | 'low' | 'unavailable'
  confidence_percent: number | null
  confidenceScore: number | null
  confidence_status?: 'calibrated' | 'not_calibrated' | 'unavailable' | string
  confidence_reason?: string
  detected_features: string[]
  suggested_followups: string[]
  label: string
  revealed_layer?: HiddenLayer
  count_estimate?: { low: number; high: number; best_estimate: number } | null
  count_uncertainty_factors?: string[]
  region?: Region | null
  execution_trace?: ExecutionTrace | null
  fusion_features?: FusionFeatures | null
  mode?: string
  is_synthetic?: boolean
}

type ChatMessage = {
  question: string
  answer: string
  confidenceScore: number
  confidence_percent: number
  confidence: 'high' | 'medium' | 'low' | 'unavailable'
  confidence_status?: 'calibrated' | 'not_calibrated' | 'unavailable' | string
  confidence_reason?: string
  detected_features: string[]
  label: string
  timestamp: string
  region?: Region | null
  execution_trace?: ExecutionTrace | null
  fusion_features?: FusionFeatures | null
  mode?: string
  is_synthetic?: boolean
}

export interface ImageTelemetry {
  landClass: string; landClassPct: string; buildingCount: string; waterPct: string
  vegetationPct: string; terrain: Terrain; locationTag: string
  landSub: string; buildingSub: string; waterSub: string; vegSub: string
  raster_info?: {
    format?: string
    crs?: string
    resolution_m?: number | null
    bands?: number
    is_geotiff?: boolean
  }
}

export interface BuildingDetection {
  id: string; confidence: number; confidence_tier: 'high' | 'medium' | 'low'
  bbox: [number,number,number,number]; bbox_pct: [number,number,number,number]
  polygon: [number,number][]; polygon_pct: [number,number][]
  centroid: [number,number]; centroid_pct: [number,number]
  area: number; is_partial: boolean
}

export interface BuildingAnalysisResult {
  building_count: number; high_confidence_count: number; medium_confidence_count: number
  low_confidence_count: number; partial_count: number; confidence?: number | null
  confidence_level?: 'High' | 'Medium' | 'Low' | null; validation_status?: string | null
  detections: BuildingDetection[]; image_dimensions?: { width: number; height: number }
  mode?: string
  error?: string
}

// BigEarthNet 19-class result
interface BENLabelScore { name: string; short: string; score: number; active: boolean }
interface BENResult {
  labels: BENLabelScore[]; active_labels: BENLabelScore[]; top_label: string
  confidence: number | null; model_id: string | null; available: boolean; device: string; note: string; citation?: string | null
  mode?: string
}

// Toast system
type ToastType = 'success' | 'error' | 'info' | 'warning'
interface Toast { id: string; message: string; type: ToastType; duration?: number }



// ── Default telemetry ─────────────────────────────────────────────────────────
export const DEFAULT_TELEMETRY: ImageTelemetry = {
  landClass: '—', landClassPct: '—', buildingCount: '—', waterPct: '—',
  vegetationPct: '—', terrain: 'urban',
  locationTag: '',
  landSub: 'Awaiting analysis', buildingSub: '',
  waterSub: '', vegSub: '',
}

// ── Image analysis caches ─────────────────────────────────────────────────────
const imageTerrainCache = new Map<string, Terrain>()
const imageTelemetryCache = new Map<string, ImageTelemetry>()

// ── Image compression + telemetry ────────────────────────────────────────────
async function compressImage(file: File): Promise<{ dataUrl: string; telemetry: ImageTelemetry }> {
  if (!file.type.startsWith('image/')) {
    throw new Error('Please choose a browser-decodable image (JPG, PNG, or WEBP).');
  }
  if (file.size > 20 * 1024 * 1024) throw new Error('Image exceeds 20 MB limit. Please resize.');

  let source: ImageBitmap;
  try {
    source = await createImageBitmap(file);
  } catch {
    throw new Error('This image format could not be decoded in the browser. TIFF/GeoTIFF files require a raster-aware backend.');
  }

  const scale = Math.min(1, 1600 / Math.max(source.width, source.height));
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(source.width * scale));
  canvas.height = Math.max(1, Math.round(source.height * scale));
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    source.close();
    throw new Error('Could not prepare the uploaded image.');
  }
  ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
  source.close();

  const telemetry: ImageTelemetry = {
    ...DEFAULT_TELEMETRY,
    locationTag: `${canvas.width}×${canvas.height} px · ${file.name}`,
  };
  return { dataUrl: canvas.toDataURL('image/jpeg', 0.82), telemetry };
}

// ── Scroll reveal hook ────────────────────────────────────────────────────────
function useScrollReveal(threshold = 0.1) {
  const ref = useRef<HTMLDivElement>(null)
  const [visible, setVisible] = useState(false)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const obs = new IntersectionObserver(([e]) => { if (e.isIntersecting) setVisible(true) }, { threshold })
    obs.observe(el)
    return () => obs.disconnect()
  }, [])
  return { ref, visible }
}

// ── Toast hook ────────────────────────────────────────────────────────────────
function useToast() {
  const [toasts, setToasts] = useState<Toast[]>([])
  const add = useCallback((message: string, type: ToastType = 'info', duration = 4000) => {
    const id = Math.random().toString(36).slice(2)
    setToasts(prev => [...prev, { id, message, type, duration }])
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), duration + 400)
  }, [])
  const remove = useCallback((id: string) => setToasts(prev => prev.filter(t => t.id !== id)), [])
  return { toasts, addToast: add, removeToast: remove }
}

// ── Toast Container ───────────────────────────────────────────────────────────
function ToastContainer({ toasts, onRemove }: { toasts: Toast[], onRemove: (id: string) => void }) {
  const icons = { success: '✓', error: '✕', info: 'ℹ', warning: '⚠' }
  const colors = {
    success: { bg: 'rgba(53,224,184,0.12)', border: 'rgba(53,224,184,0.3)', icon: C.mint },
    error:   { bg: 'rgba(255,107,107,0.12)', border: 'rgba(255,107,107,0.3)', icon: C.danger },
    info:    { bg: 'rgba(32,217,255,0.10)', border: 'rgba(32,217,255,0.25)', icon: C.cyan },
    warning: { bg: 'rgba(255,159,67,0.12)', border: 'rgba(255,159,67,0.3)', icon: C.orange },
  }
  return (
    <div className="fixed top-5 right-5 z-[9999] flex flex-col gap-2.5 pointer-events-none">
      {toasts.map(t => {
        const col = colors[t.type]
        return (
          <div key={t.id} className="toast-enter pointer-events-auto flex items-start gap-3 px-4 py-3 rounded-xl max-w-sm"
            style={{ background: col.bg, border: `1px solid ${col.border}`, backdropFilter: 'blur(20px)', boxShadow: '0 8px 32px rgba(0,0,0,0.4)' }}>
            <span className="mt-0.5 text-sm font-bold shrink-0" style={{ color: col.icon }}>{icons[t.type]}</span>
            <span className="text-sm leading-snug flex-1" style={{ color: C.white }}>{t.message}</span>
            <button onClick={() => onRemove(t.id)} className="shrink-0 opacity-50 hover:opacity-100 transition-opacity text-xs" style={{ color: C.muted }}>✕</button>
          </div>
        )
      })}
    </div>
  )
}

// ── Confidence badge ──────────────────────────────────────────────────────────
function ConfidenceBadge({
  confidence,
  percent,
  confidence_status,
  reason,
  mode,
}: {
  confidence?: 'high' | 'medium' | 'low' | 'unavailable'
  percent?: number | null
  confidence_status?: 'calibrated' | 'not_calibrated' | 'unavailable' | string
  reason?: string
  mode?: string
}) {
  const [tip, setTip] = useState(false)
  const isCalibrated = confidence_status === 'calibrated'
  const isUnavailable = confidence_status === 'unavailable' || confidence === 'unavailable' || (percent === 0 && !isCalibrated)

  let badgeColor: string = C.cyan
  let mainLabel = 'Confidence: Not calibrated'
  let subLabel = ''

  if (isUnavailable) {
    badgeColor = C.danger
    mainLabel = 'Confidence: Unavailable'
  } else if (isCalibrated && percent != null) {
    badgeColor = percent >= 85 ? C.mint : percent >= 65 ? C.orange : C.danger
    mainLabel = 'Calibrated Confidence'
    subLabel = `${percent}%`
  } else {
    // Uncalibrated model or heuristic score
    badgeColor = C.cyan
    mainLabel = 'Confidence: Not calibrated'
    if (percent != null && percent > 0) {
      subLabel = `Model score: ${percent}%`
    }
  }

  return (
    <div className="relative inline-flex items-center gap-1.5">
      <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-lg text-[11px] font-mono"
        style={{ background: `${badgeColor}14`, border: `1px solid ${badgeColor}44`, color: badgeColor }}>
        <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ background: badgeColor, boxShadow: `0 0 6px ${badgeColor}` }} />
        <span className="font-semibold">{mainLabel}</span>
        {subLabel && (
          <>
            <span className="opacity-60">·</span>
            <span className="font-bold">{subLabel}</span>
          </>
        )}
        {percent != null && percent > 0 && !isUnavailable && (
          <div className="w-8 h-1 rounded-full overflow-hidden shrink-0" style={{ background: 'rgba(0,0,0,0.3)' }}>
            <div className="h-full rounded-full" style={{ width: `${percent}%`, background: badgeColor, transition: 'width 0.5s ease' }} />
          </div>
        )}
      </div>
      <div className="relative" onMouseEnter={() => setTip(true)} onMouseLeave={() => setTip(false)}>
        <span className="w-4 h-4 rounded-full flex items-center justify-center text-[10px] cursor-default"
          style={{ border: `1px solid ${C.border}`, color: C.muted, background: 'rgba(6,13,26,0.7)' }}>ⓘ</span>
        {tip && (
          <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 p-2.5 rounded-lg text-[10px] leading-snug z-50 pointer-events-none"
            style={{ background: '#071022F5', border: `1px solid ${C.borderHover}`, color: '#D8F6FF', boxShadow: '0 4px 20px rgba(0,0,0,0.6)' }}>
            <div className="font-semibold text-white mb-0.5">
              {isCalibrated ? 'Calibrated Empirical Confidence' : isUnavailable ? 'Service Unavailable' : 'Uncalibrated Model Output'}
            </div>
            <div>
              {isCalibrated
                ? 'Only displayed when a specialist explicitly reports calibrated confidence backed by empirical validation.'
                : isUnavailable
                ? 'Specialist model service is currently offline or unreachable.'
                : 'Raw model prediction score/probability. Empirical error-rate calibration pending.'}
            </div>
            {reason && <div className="mt-1 pt-1 text-[9px] text-slate-400 border-t border-slate-700/50">Factor: {reason}</div>}
          </div>
        )}
      </div>
    </div>
  )
}

// ── Skeleton loader ───────────────────────────────────────────────────────────
function Skeleton({ className = '', style = {} }: { className?: string, style?: React.CSSProperties }) {
  return <div className={`shimmer rounded-lg ${className}`} style={{ background: 'rgba(255,255,255,0.04)', ...style }} />
}

// ── BigEarthNet Classification Panel ─────────────────────────────────────────
function BENPanel({ results, loading }: { results: BENResult | null, loading: boolean }) {
  if (!results && !loading) return null
  const topLabels = results?.labels?.slice(0,8) ?? []
  const maxScore = Math.max(...topLabels.map(l => l.score), 0.01)
  return (
    <div className="rounded-2xl overflow-hidden" style={{ background: C.surface, border: `1px solid ${C.border}` }}>
      <div className="flex items-center justify-between px-5 py-3 border-b" style={{ borderColor: C.border }}>
        <div className="flex items-center gap-2.5">
          <div className="w-6 h-6 rounded-lg flex items-center justify-center text-[10px] font-mono font-bold"
            style={{ background: `${C.mint}22`, color: C.mint, border: `1px solid ${C.mint}44` }}>BEN</div>
          <div>
            <div className="text-xs font-semibold font-mono tracking-wide" style={{ color: C.white }}>BigEarthNet v2.0 Classification</div>
            <div className="text-[10px] font-mono" style={{ color: C.muted }}>
              {loading ? 'Running inference…' : results?.available ? `ResNet-50 · ${results.device?.toUpperCase()} · 19 Classes` : 'Specialist unavailable'}
            </div>
          </div>
        </div>
        {results && (
          <div className="text-right">
            <div className="text-xs font-mono font-bold" style={{ color: C.mint }}>{results.top_label}</div>
            <div className="text-[10px] font-mono" style={{ color: C.muted }}>
              {results.available ? `${results.confidence?.toFixed(1)}% model score` : 'No result returned'}
            </div>
          </div>
        )}
      </div>
      <div className="p-4 space-y-2.5">
        {loading ? (
          Array.from({length: 5}).map((_,i) => (
            <div key={i} className="flex items-center gap-3">
              <Skeleton className="w-28 h-3" />
              <div className="flex-1 h-3 rounded-full" style={{ background: 'rgba(255,255,255,0.04)' }} />
              <Skeleton className="w-10 h-3" />
            </div>
          ))
        ) : (
          topLabels.map((label, idx) => {
            const barW = Math.round((label.score / maxScore) * 100)
            const col = label.active ? (idx === 0 ? C.mint : idx < 3 ? C.cyan : C.muted) : C.dim
            return (
              <div key={label.name} className="flex items-center gap-3">
                <div className="text-[10px] font-mono truncate shrink-0 w-36" style={{ color: label.active ? C.white : C.muted }}>{label.short}</div>
                <div className="flex-1 h-2 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.04)' }}>
                  <div className="h-full rounded-full ben-bar" style={{
                    '--bar-width': `${barW}%`, width: `${barW}%`, background: col,
                    '--idx': idx, boxShadow: label.active ? `0 0 8px ${col}66` : 'none'
                  } as any} />
                </div>
                <div className="text-[10px] font-mono font-bold shrink-0 w-9 text-right" style={{ color: col }}>
                  {(label.score * 100).toFixed(0)}%
                </div>
                {label.active && <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ background: col }} />}
              </div>
            )
          })
        )}
      </div>
      {results && (
        <div className="px-4 pb-3">
          <div className="flex flex-wrap gap-1.5">
            {results.active_labels?.map(l => (
              <span key={l.name} className="text-[9px] font-mono px-2 py-0.5 rounded-md"
                style={{ background: `${C.mint}18`, color: C.mint, border: `1px solid ${C.mint}33` }}>{l.short}</span>
            ))}
          </div>
          <div className="mt-2 text-[9px] font-mono" style={{ color: C.dim }}>
            BIFOLD-BigEarthNetv2-0 · Clasen et al., IGARSS 2025
          </div>
        </div>
      )}
    </div>
  )
}

// ── Navbar ────────────────────────────────────────────────────────────────────
function Navbar({
  scrolled,
  onAnalyze,
  onEval,
  onUpload,
  appMode,
  setAppMode,
  activeView,
  setActiveView,
  onOpenDocs,
  onOpenContact,
}: {
  scrolled: boolean
  onAnalyze: () => void
  onEval: () => void
  onUpload: () => void
  appMode: AppMode
  setAppMode: (m: AppMode) => void
  activeView: 'workspace' | 'dashboard'
  setActiveView: (v: 'workspace' | 'dashboard') => void
  onOpenDocs: () => void
  onOpenContact: () => void
}) {
  const [mobileOpen, setMobileOpen] = useState(false)
  const scrollTo = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    setMobileOpen(false)
  }

  return (
    <header className="fixed top-0 left-0 right-0 z-50 transition-all duration-500"
      style={{
        background: scrolled ? 'rgba(3,7,18,0.95)' : 'transparent',
        backdropFilter: scrolled ? 'blur(24px)' : 'none',
        borderBottom: scrolled ? `1px solid ${C.border}` : '1px solid transparent',
        boxShadow: scrolled ? '0 4px 40px rgba(0,0,0,0.6)' : 'none',
      }}>
      <div className="max-w-7xl mx-auto px-5 h-16 flex items-center gap-6">
        {/* Logo */}
        <button
          onClick={() => { setActiveView('workspace'); scrollTo('explore') }}
          className="flex items-center gap-2.5 shrink-0 group cursor-pointer"
          style={{ background: 'none', border: 'none' }}
        >
          <div className="relative w-8 h-8 flex items-center justify-center">
            <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
              <circle cx="14" cy="14" r="12" stroke={C.cyan} strokeWidth="1.5" />
              <ellipse cx="14" cy="14" rx="5" ry="12" stroke={C.cyan} strokeWidth="1.5" />
              <line x1="2" y1="14" x2="26" y2="14" stroke={C.cyan} strokeWidth="1.5" />
              <circle cx="14" cy="14" r="2.5" fill={C.cyan} />
            </svg>
          </div>
          <div className="text-left">
            <span className="text-sm font-bold tracking-wide" style={{ fontFamily: "'Space Grotesk', sans-serif", color: C.white }}>
              ORBITAL<span style={{ color: C.cyan }}>-AI</span>
            </span>
            <div className="text-[9px] font-mono tracking-widest leading-none" style={{ color: C.muted }}>GEOSPATIAL INTELLIGENCE</div>
          </div>
        </button>

        {/* Desktop links */}
        <nav aria-label="Main Navigation" className="hidden lg:flex items-center gap-2 flex-1 ml-6">
          <button
            onClick={() => { setActiveView('workspace'); scrollTo('analyze') }}
            className={`px-3 py-1.5 rounded-lg text-xs transition-all cursor-pointer font-medium ${
              activeView === 'workspace' ? 'text-cyan-400 font-semibold bg-cyan-950/40 border border-cyan-500/30' : 'text-slate-300 hover:text-white hover:bg-slate-800/40'
            }`}
          >
            Analyze
          </button>
          <button
            onClick={() => { setActiveView('dashboard'); scrollTo('analyze') }}
            className={`px-3 py-1.5 rounded-lg text-xs transition-all cursor-pointer font-medium ${
              activeView === 'dashboard' ? 'text-cyan-400 font-semibold bg-cyan-950/40 border border-cyan-500/30' : 'text-slate-300 hover:text-white hover:bg-slate-800/40'
            }`}
          >
            History
          </button>
          <button
            onClick={onOpenDocs}
            className="px-3 py-1.5 rounded-lg text-xs transition-all cursor-pointer font-medium text-slate-300 hover:text-white hover:bg-slate-800/40"
          >
            Documentation
          </button>
          <button
            onClick={() => scrollTo('about')}
            className="px-3 py-1.5 rounded-lg text-xs transition-all cursor-pointer font-medium text-slate-300 hover:text-white hover:bg-slate-800/40"
          >
            About
          </button>
        </nav>

        {/* Right actions */}
        <div className="flex items-center gap-2 ml-auto">
          <button
            onClick={() => { setActiveView('workspace'); onAnalyze() }}
            className="btn-primary text-xs px-4 py-2 font-semibold shadow-md shadow-cyan-950"
          >
            Launch Workspace
          </button>

          {/* Mobile hamburger */}
          <button
            onClick={() => setMobileOpen(v => !v)}
            className="lg:hidden p-2 rounded-lg ml-1 text-slate-300"
            style={{ background: 'rgba(255,255,255,0.06)', border: `1px solid ${C.border}`, cursor: 'pointer' }}
          >
            {mobileOpen ? '✕' : '☰'}
          </button>
        </div>
      </div>

      {/* Mobile menu */}
      {mobileOpen && (
        <nav aria-label="Mobile Navigation" className="lg:hidden border-t px-5 py-4 space-y-1.5" style={{ borderColor: C.border, background: 'rgba(3,7,18,0.98)', backdropFilter: 'blur(24px)' }}>
          <button
            onClick={() => { setActiveView('workspace'); scrollTo('analyze'); setMobileOpen(false) }}
            className="w-full text-left px-3 py-2 rounded-lg text-xs text-slate-200 hover:bg-slate-800/50"
          >
            Analyze
          </button>
          <button
            onClick={() => { setActiveView('dashboard'); scrollTo('analyze'); setMobileOpen(false) }}
            className="w-full text-left px-3 py-2 rounded-lg text-xs text-slate-200 hover:bg-slate-800/50"
          >
            History
          </button>
          <button
            onClick={() => { onOpenDocs(); setMobileOpen(false) }}
            className="w-full text-left px-3 py-2 rounded-lg text-xs text-slate-200 hover:bg-slate-800/50"
          >
            Documentation
          </button>
          <button
            onClick={() => { scrollTo('about'); setMobileOpen(false) }}
            className="w-full text-left px-3 py-2 rounded-lg text-xs text-slate-200 hover:bg-slate-800/50"
          >
            About
          </button>
          <div className="pt-2 flex gap-2">
            <button onClick={() => { onEval(); setMobileOpen(false) }} className="btn-outline-cyan text-xs px-3 py-2 font-mono flex-1">
              EVAL CRITERIA
            </button>
            <button onClick={() => { onUpload(); setMobileOpen(false) }} className="btn-ghost text-xs px-3 py-2 flex-1">
              Upload Image
            </button>
          </div>
        </nav>
      )}
    </header>
  )
}

// ── Starfield background ──────────────────────────────────────────────────────
function Starfield() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const stars: { x: number; y: number; r: number; opacity: number; speed: number; phase: number }[] = []
    const resize = () => { canvas.width = canvas.offsetWidth; canvas.height = canvas.offsetHeight }
    resize()
    for (let i = 0; i < 220; i++) {
      stars.push({
        x: Math.random() * canvas.width, y: Math.random() * canvas.height,
        r: Math.random() * 1.2 + 0.2,
        opacity: Math.random() * 0.7 + 0.1, speed: Math.random() * 0.5 + 0.1,
        phase: Math.random() * Math.PI * 2,
      })
    }
    let raf = 0, t = 0
    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      t += 0.01
      stars.forEach(s => {
        const alpha = s.opacity * (0.5 + 0.5 * Math.sin(t * s.speed + s.phase))
        ctx.beginPath()
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(220,240,255,${alpha})`
        ctx.fill()
      })
      raf = requestAnimationFrame(draw)
    }
    draw()
    window.addEventListener('resize', resize)
    return () => { cancelAnimationFrame(raf); window.removeEventListener('resize', resize) }
  }, [])
  return <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none" style={{ opacity: 0.8 }} />
}

// ── Hero section ──────────────────────────────────────────────────────────────
function HeroSection({
  onUpload, onAnalyze, onQueryChip, imageTelemetry, buildingAnalysis
}: {
  onUpload: () => void; onAnalyze: () => void; onQueryChip: (q: string) => void
  imageTelemetry: ImageTelemetry; buildingAnalysis: BuildingAnalysisResult | null
}) {
  return (
    <section id="explore" className="relative min-h-screen flex items-center overflow-hidden" style={{ background: C.base }}>
      <Starfield />

      {/* Grid background */}
      <div className="absolute inset-0 grid-bg opacity-40 pointer-events-none" />

      <div className="relative max-w-7xl mx-auto px-5 pt-24 pb-16 w-full z-10">
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_1.1fr] gap-12 items-center min-h-[88vh]">

          {/* Left: Copy */}
          <div className="py-10 fade-up">
            {/* Badge */}
            <div className="inline-flex items-center gap-2 mb-8 px-3 py-1.5 rounded-md"
              style={{ background: 'rgba(32,217,255,0.08)', border: `1px solid ${C.borderHover}` }}>
              <span className="w-2 h-2 rounded-full pulse-dot" style={{ background: C.cyan }} />
              <span className="text-xs font-mono tracking-[0.2em] uppercase" style={{ color: C.cyan }}>GEOSPATIAL VISION-LANGUAGE SYSTEM</span>
            </div>

            {/* Headline */}
            <h1 className="text-5xl lg:text-6xl xl:text-7xl font-bold leading-[1.02] mb-6" style={{ fontFamily: "'Space Grotesk', sans-serif", letterSpacing: '-0.03em' }}>
              Ask One Question About Satellite Imagery.
              <br />
              <span style={{ color: 'var(--cyan)' }}>The System Routes the Workflow.</span>
            </h1>

            <p className="text-lg leading-relaxed mb-10 max-w-lg" style={{ color: C.muted, lineHeight: 1.8 }}>
              ORBITAL-AI turns natural-language questions into validated remote-sensing workflows across single-image analysis, bi-temporal change understanding, and optical-SAR reasoning. Every run exposes what was executed, what evidence was available, and where uncertainty remains.
            </p>

            {/* CTA buttons */}
            <div className="flex flex-wrap gap-3 mb-12">
              <button onClick={onAnalyze} className="btn-primary flex items-center gap-2 px-6 py-3 text-sm">
                Open Analysis Workspace →
              </button>
            </div>

            {/* Stats */}
            <div className="flex flex-wrap gap-8 pt-6 mb-10" style={{ borderTop: `1px solid ${C.border}` }}>
              {[
                { val: '3', label: 'Analysis Input Modes', col: C.cyan },
                { val: 'MODULAR', label: 'Specialist Routing', col: C.mint },
                { val: 'RULE', label: 'Query-Driven Routing', col: C.orange },
                { val: 'TRACE', label: 'Observable Verification', col: C.white },
              ].map(s => (
                <div key={s.label}>
                  <div className="text-2xl font-bold" style={{ fontFamily: "'Space Grotesk', sans-serif", color: s.col }}>{s.val}</div>
                  <div className="text-[11px] mt-0.5 font-mono" style={{ color: C.muted }}>{s.label}</div>
                </div>
              ))}
            </div>

            {/* Representative Queries */}
            <div className="space-y-2">
              <div className="text-[10px] font-mono tracking-widest uppercase mb-3" style={{ color: C.dim }}>EXAMPLE NATURAL-LANGUAGE QUERIES</div>
              <div className="flex flex-wrap gap-2">
                {OFFICIAL_REPRESENTATIVE_QUERIES.map(q => (
                  <button key={q.badge} onClick={() => onQueryChip(q.query)}
                    className="query-chip text-left text-[11px] px-3 py-1.5 rounded-lg font-mono"
                    style={{ background: `${q.color}10`, border: `1px solid ${q.color}33`, color: q.color }}>
                    <span className="opacity-60 mr-1">"{q.query}"</span>
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Right: Globe + telemetry status */}
          <div className="relative flex items-center justify-center float-anim" style={{ minHeight: 540 }}>
            {/* Orbital ring decoration */}
            <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
              <svg viewBox="0 0 600 600" className="absolute w-full h-full" style={{ opacity: 0.12 }}>
                <ellipse cx="300" cy="300" rx="270" ry="100" fill="none" stroke={C.cyan} strokeWidth="0.8" transform="rotate(-20 300 300)" />
                <ellipse cx="300" cy="300" rx="235" ry="75" fill="none" stroke={C.orange} strokeWidth="0.5" transform="rotate(40 300 300)" />
                <ellipse cx="300" cy="300" rx="200" ry="55" fill="none" stroke={C.mint} strokeWidth="0.4" transform="rotate(10 300 300)" strokeDasharray="4 6" />
              </svg>
            </div>

            {/* Globe */}
            <Globe style={{ width: '100%', height: 520, maxWidth: 560 }} />

            {/* Clean Live Status indicator */}
            <div className="absolute bottom-6 left-0 glass card-border rounded-xl px-4 py-3 text-xs font-mono">
              <div className="text-[9px] uppercase tracking-widest mb-1" style={{ color: C.muted }}>PIPELINE STATUS</div>
              <div className="flex items-center gap-2"><span className="w-2 h-2 rounded-full glow-pulse" style={{ background: C.mint }} /><span style={{ color: C.mint, fontWeight: 600 }}>Interface Ready</span></div>
              <div className="text-[10px] mt-0.5" style={{ color: C.muted }}>Awaiting imagery and a natural-language query</div>
            </div>
          </div>
        </div>
      </div>

      {/* Scroll indicator */}
      <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 opacity-40">
        <div className="text-[10px] font-mono tracking-widest" style={{ color: C.muted }}>SCROLL</div>
        <div className="w-0.5 h-8 rounded-full overflow-hidden" style={{ background: `${C.border}` }}>
          <div className="w-full h-1/2 rounded-full progress-indeterminate" style={{ background: C.cyan }} />
        </div>
      </div>
    </section>
  )
}

// ── Workspace / Analyze Panel ─────────────────────────────────────────────────
// ── Response Normalizer ────────────────────────────────────────────────────────
// Adapts the FastAPI unified response schema to the frontend's Analysis type
function normalizeAnalyzeResponse(payload: Record<string, any>, fallbackPrompt: string): Analysis {
  const confStatus = payload.confidence_status || (payload.available === false ? 'unavailable' : 'not_calibrated')
  const rawConf = typeof payload.confidence === 'number' ? payload.confidence : (typeof payload.confidence_score === 'number' ? payload.confidence_score : null)
  const confPct = rawConf !== null ? Math.round(rawConf * 100) : null
  const taskType = payload.task_type || 'vqa'

  let label = 'Analysis'
  if (taskType === 'building_segmentation' || payload.building_analysis) label = 'Building Footprint Audit'
  else if (taskType === 'change_detection') label = 'Bi-Temporal Change Detection'
  else if (taskType === 'sar_optical_fusion') label = 'Optical–SAR Fusion'
  else if (taskType === 'grounding') label = 'Visual Grounding'

  let region = undefined
  if (payload.grounding?.bounding_box_percent) {
    const [x, y, w, h] = payload.grounding.bounding_box_percent
    region = { x_percent: x, y_percent: y, w_percent: w, h_percent: h }
  }

  const detected_features: string[] = []
  if (payload.building_analysis?.building_count) {
    detected_features.push(payload.building_analysis.building_count + ' Footprints')
  }
  if (payload.classification?.top_label) {
    detected_features.push(payload.classification.top_label)
  }
  if (payload.metrics?.iou) {
    detected_features.push('IoU: ' + (payload.metrics.iou * 100).toFixed(1) + '%')
  }

  return {
    answer: payload.answer || 'Analysis complete.',
    confidence: payload.confidence_level ? String(payload.confidence_level).toLowerCase() as Analysis['confidence'] : 'unavailable',
    confidence_percent: confPct,
    confidenceScore: confPct,
    confidence_status: confStatus,
    confidence_reason: payload.reasoning || (confStatus === 'not_calibrated' ? 'Model output score; empirical calibration pending.' : undefined),
    detected_features: detected_features.length > 0 ? detected_features : [label],
    label,
    suggested_followups: payload.suggested_followups || ['What is the main feature in this image?', 'Where is the most important area to inspect?', 'Can you explain that in simpler terms?'],
    execution_trace: payload.execution_trace || null,
    region,
    mode: payload.mode || (payload.available === false ? 'unavailable' : undefined),
    is_synthetic: payload.is_synthetic === true,
  }
}

export default function App() {
  // ── State ─────────────────────────────────────────────────────────────────
  const [scrolled, setScrolled] = useState(false)

  const [activeRegion, setActiveRegion] = useState<{ region: Region; label: string; confidence_percent?: number } | null>(null)
  const [beforePct, setBeforePct] = useState(50)
  const [dragging, setDragging] = useState(false)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [imageTelemetry, setImageTelemetry] = useState<ImageTelemetry>(DEFAULT_TELEMETRY)
  const [beforeImage, setBeforeImage] = useState<string | null>(null)
  const [afterImage, setAfterImage] = useState<string | null>(null)
  const [question, setQuestion] = useState('')
  const [analysis, setAnalysis] = useState<Analysis | null>(null)
  const [history, setHistory] = useState<ChatMessage[]>([])
  const [suggestions, setSuggestions] = useState(OFFICIAL_REPRESENTATIVE_QUERIES.map(q => q.query))
  const [busy, setBusy] = useState(false)
  const [status, setStatus] = useState('Ready · Upload an image to begin')
  const [error, setError] = useState('')
  const [showInfoModal, setShowInfoModal] = useState(false)
  const [showEvalModal, setShowEvalModal] = useState(false)
  const [originalFile, setOriginalFile] = useState<File | null>(null)
  const [buildingAnalysis, setBuildingAnalysis] = useState<BuildingAnalysisResult | null>(null)
  const [showBuildingsOverlay, setShowBuildingsOverlay] = useState(false)
  const [isDetectingBuildings, setIsDetectingBuildings] = useState(false)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [sessionCallCount, setSessionCallCount] = useState(0)
  const [appMode, setAppMode] = useState<AppMode>('single')
  const [activeTrace, setActiveTrace] = useState<ExecutionTrace | null>(null)
  const [traceModalOpen, setTraceModalOpen] = useState(false)
  const [fusionFeatures, setFusionFeatures] = useState<FusionFeatures | null>(null)
  const [lastFusionTrace, setLastFusionTrace] = useState<ExecutionTrace | null>(null)
  const [activeOverlay, setActiveOverlay] = useState<HiddenLayer | null>(null)
  const [revealedLayers, setRevealedLayers] = useState<HiddenLayer[]>([])
  const [benResults, setBenResults] = useState<BENResult | null>(null)
  const [classifyingBEN, setClassifyingBEN] = useState(false)
  const [isDragOver, setIsDragOver] = useState(false)
  const [temporalResult, setTemporalResult] = useState<{
    question: string; answer: string; confidenceScore: number
    confidence_status?: string
    features: string[]; execution_trace?: ExecutionTrace | null
    mode?: string
  } | null>(null)

  // ── Navigation, Dashboard & Legal Modals ─────────────────────────────────
  const [activeView, setActiveView] = useState<'workspace' | 'dashboard'>('workspace')
  const [docModalOpen, setDocModalOpen] = useState(false)
  const [contactModalOpen, setContactModalOpen] = useState(false)
  const [legalModalOpen, setLegalModalOpen] = useState(false)
  const [activeLegalTab, setActiveLegalTab] = useState<LegalModalType>('cookies')

  const { toasts, addToast, removeToast } = useToast()
  const atCap = sessionCallCount >= SESSION_CALL_LIMIT

  const fileInputRef = useRef<HTMLInputElement>(null)
  const beforeInputRef = useRef<HTMLInputElement>(null)
  const afterInputRef = useRef<HTMLInputElement>(null)
  const compareRef = useRef<HTMLDivElement>(null)
  const chatBottomRef = useRef<HTMLDivElement>(null)

  const r1 = useScrollReveal(); const r2 = useScrollReveal(); const r3 = useScrollReveal()
  const r4 = useScrollReveal(); const r5 = useScrollReveal(); const r6 = useScrollReveal()

  useEffect(() => {
    const fn = () => setScrolled(window.scrollY > 60)
    window.addEventListener('scroll', fn, { passive: true })
    return () => window.removeEventListener('scroll', fn)
  }, [])

  const scrollToSection = (id: string) => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  const focusWorkspace = () => scrollToSection('analyze')
  const openUploader = () => fileInputRef.current?.click()

  // ── BigEarthNet classification ─────────────────────────────────────────────
  const classifyWithBEN = useCallback(async (dataUrl: string) => {
    setClassifyingBEN(true)
    try {
      const res = await fetch(`${API_BASE}/classify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataUrl, top_k: 8, threshold: 0.2 }),
      })
      if (!res.ok) throw new Error('trained BigEarthNet specialist unavailable')
      const data = await res.json()
      if (!data?.available || !data?.top_label) {
        setBenResults(null)
        return
      }
      setBenResults(data)
    } catch {
      // Never fabricate a land-cover result when the trained specialist is unavailable.
      setBenResults(null)
    } finally {
      setClassifyingBEN(false)
    }
  }, [])
  // ── Building detection ─────────────────────────────────────────────────────
  const runBuildingDetection = useCallback(async (fileOverride?: File) => {
    setIsDetectingBuildings(true)
    setError('')
    setStatus('Tiling image for instance segmentation…')
    const steps = ['Tiling image…', 'Running YOLO segmentation…', 'Detecting rooftops…', 'Merging duplicates (NMS/IoU)…', 'Counting footprints…']
    let stepIdx = 0
    const timer = setInterval(() => { stepIdx = (stepIdx+1) % steps.length; setStatus(steps[stepIdx]) }, 2800)
    const fileToSend = fileOverride ?? originalFile
    try {
      let data: BuildingAnalysisResult | null = null
      const buildingEndpoints = API_BASE
        ? [`${API_BASE}/analyze/buildings`, `${API_BASE}/api/buildings`]
        : ['/analyze/buildings', '/api/buildings']

      for (const ep of buildingEndpoints) {
        try {
          let res: Response
          if (fileToSend) {
            const fd = new FormData(); fd.append('file', fileToSend)
            res = await fetch(ep, { method: 'POST', body: fd })
          } else {
            res = await fetch(ep, { method: 'POST', headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ image: imagePreview ?? '' }) })
          }
          if (res.ok) {
            const parsed = await res.json().catch(() => null)
            if (parsed && typeof parsed.building_count === 'number') { data = parsed; break }
          } else if (res.status === 202) {
            const sig = await res.json().catch(() => null)
            if (sig?.custom_analysis_required) {
              throw new Error('Building detection specialist is busy. No client-side estimate is generated.')
            }
          }
        } catch { /* try next */ }
      }

      if (!data) {
        throw new Error('Building detection unavailable: specialist YOLO model service is not reachable.')
      }

      setBuildingAnalysis(data)
      setShowBuildingsOverlay(true)
      setActiveOverlay('urban')
      setImageTelemetry(prev => ({
        ...prev,
        buildingCount: String(data!.building_count),
        buildingSub: `High: ${data!.high_confidence_count} · Med: ${data!.medium_confidence_count} · Partial: ${data!.partial_count}`,
      }))
      setStatus(`Detected ${data.building_count} buildings (${data.confidence_level})`)
      addToast(`Found ${data.building_count} building footprints (${data.confidence_level} confidence)`, 'success')
      return data
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Specialist YOLO building detection service is not reachable.'
      setError(msg)
      setStatus('Detection unavailable')
      addToast(msg, 'error')
      return null
    } finally {
      clearInterval(timer); setIsDetectingBuildings(false)
    }
  }, [originalFile, imagePreview, addToast])

  // ── File handlers ──────────────────────────────────────────────────────────
  const handleFile = async (file?: File) => {
    if (!file) return
    setError(''); setStatus('Processing image…'); setOriginalFile(file)
    setBuildingAnalysis(null); setShowBuildingsOverlay(false); setBenResults(null)
    try {
      const { dataUrl, telemetry } = await compressImage(file)
      setImagePreview(dataUrl)
      setImageTelemetry(telemetry)
      setAnalysis(null); setHistory([]); setSuggestions(OFFICIAL_REPRESENTATIVE_QUERIES.map(q => q.query))
      setActiveOverlay(null); setActiveRegion(null); setRevealedLayers([])
      setSessionId(null); setSessionCallCount(0); setTemporalResult(null)
      setStatus('Image loaded · Ask a question')
      focusWorkspace()
      addToast(`Image loaded: ${file.name} (${(file.size/1024).toFixed(0)}KB)`, 'info')
      // Run BigEarthNet classification in background
      classifyWithBEN(dataUrl)
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Could not load image.'
      setError(msg); setStatus('Upload error'); addToast(msg, 'error')
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault(); setIsDragOver(false)
    const file = e.dataTransfer.files?.[0]
    if (file) handleFile(file)
  }

  const handleBeforeFile = async (file?: File) => {
    if (!file) return
    try { const { dataUrl } = await compressImage(file); setBeforeImage(dataUrl); addToast('Earlier image loaded', 'info') }
    catch (e) { addToast(e instanceof Error ? e.message : 'Failed', 'error') }
  }

  const handleAfterFile = async (file?: File) => {
    if (!file) return
    try { const { dataUrl } = await compressImage(file); setAfterImage(dataUrl); addToast('Later image loaded', 'info') }
    catch (e) { addToast(e instanceof Error ? e.message : 'Failed', 'error') }
  }

  // ── Ask question ───────────────────────────────────────────────────────────
  const askQuestion = useCallback(async (preset?: string) => {
    const prompt = (preset ?? question).trim()
    if (!prompt || busy || atCap) return
    let activeSessionId = sessionId
    setQuestion(''); setError(''); setBusy(true)
    const detectedLayer = detectHiddenLayer(prompt)
    if (detectedLayer) { setActiveOverlay(detectedLayer); setRevealedLayers(prev => Array.from(new Set([...prev, detectedLayer]))) }

    try {
      setStatus('Analyzing scene…')
      if (!activeSessionId) {
        try {
          const session = await createAnalysisSession({ query: prompt, modality: appMode === 'single' ? 'optical' : appMode, taskType: classifyTask(prompt, 1, ['optical']) })
          activeSessionId = session?.id || null
          if (activeSessionId) setSessionId(activeSessionId)
        } catch (sessionErr) {
          console.warn('[Orbital-AI] Supabase session persistence unavailable:', sessionErr)
        }
      }
      let result: Analysis

      const lowerPrompt = prompt.toLowerCase()
      const isBuildingQuery = lowerPrompt.includes('building') || lowerPrompt.includes('structure') ||
        lowerPrompt.includes('how many') || (lowerPrompt.includes('count') && !lowerPrompt.includes('water')) ||
        lowerPrompt.includes('footprint') || lowerPrompt.includes('house')

      if (isBuildingQuery) {
        setStatus('Running instance segmentation…')
        try {
          const bRes = await runBuildingDetection()
          if (bRes) {
            result = {
              answer: `Instance segmentation returned ${bRes.building_count} building footprints.${bRes.confidence_level ? ` Confidence tier: ${bRes.confidence_level}.` : ''}${bRes.confidence != null ? ` Model score: ${Math.round(bRes.confidence * 100)}% (not independently calibrated).` : ''}${bRes.high_confidence_count != null ? ` Breakdown: ${bRes.high_confidence_count} high, ${bRes.medium_confidence_count} medium, ${bRes.partial_count} partial perimeter.` : ''}${bRes.validation_status ? ` ${bRes.validation_status}.` : ''}`,
              confidence: bRes.confidence_level ? bRes.confidence_level.toLowerCase() as Analysis['confidence'] : 'unavailable',
              confidence_percent: bRes.confidence != null ? Math.round(bRes.confidence * 100) : null, confidenceScore: bRes.confidence != null ? Math.round(bRes.confidence * 100) : null,
              confidence_status: bRes.confidence != null ? 'not_calibrated' : 'unavailable',
              confidence_reason: bRes.confidence != null ? 'Detector-reported model score; independent calibration is not established.' : 'Detector did not provide a confidence score.',
              detected_features: [`${bRes.building_count} Detected Footprints`, `${bRes.high_confidence_count} High Confidence`, `${bRes.medium_confidence_count} Medium`, `${bRes.partial_count} Partial Edge`],
              label: 'Building Footprint Audit', revealed_layer: 'urban',
              suggested_followups: ['Total roof area for solar?','Buildings closest to flood zone?','Density distribution?'],
            }
          } else {
            result = {
              answer: 'Building detection unavailable for this scene. The specialist YOLO segmentation service did not return footprints.',
              confidence: 'unavailable',
              confidence_percent: null,
              confidenceScore: null,
              confidence_status: 'unavailable',
              confidence_reason: 'Building detector service returned null or no valid footprints.',
              detected_features: ['Footprint Audit Unavailable'],
              label: 'Building Audit Unavailable',
              suggested_followups: ['Retry building audit with high-resolution imagery', 'Use a supported high-resolution optical product'],
            }
          }
        } catch (bErr: any) {
          result = {
            answer: `Building footprint analysis failed: ${bErr?.message || 'Specialist model service unreachable.'}`,
            confidence: 'unavailable',
            confidence_percent: null,
            confidenceScore: null,
            confidence_status: 'unavailable',
            confidence_reason: bErr?.message || 'Building detection model service failed.',
            detected_features: ['Model Service Unavailable'],
            label: 'Building Footprint Error',
            suggested_followups: ['Verify the specialist backend is running', 'Retry analysis'],
          }
        }
      } else {
        try {
          const task_type = isBuildingQuery ? 'building_segmentation' : 'vqa'
          const body: Record<string, unknown> = {
            query: prompt,
            question: prompt,
            task_type,
            image: imagePreview || undefined,
            history: history.map(h => ({ question: h.question, answer: h.answer })),
            sessionId: activeSessionId || undefined,
          }
          const res = await fetch(`${API_BASE}/api/analyze`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
          })
          const payload = await res.json().catch(() => ({}))
          // A 200 response can still be an honest "unavailable" result. Detect that
          // BEFORE accepting payload.answer, otherwise the UI renders the unavailable
          // message and never reaches the browser visual baseline.
          const specialistUnavailable =
            payload?.confidence_status === 'unavailable' ||
            payload?.label === 'Analysis unavailable' ||
            /serverless deployment|no executable specialist|no configured vlm/i.test(String(payload?.answer || ''))

          if (res.ok && (payload.answer || payload.building_analysis) && !specialistUnavailable) {
            result = normalizeAnalyzeResponse(payload, prompt)
            if (payload.building_analysis) {
              setBuildingAnalysis(payload.building_analysis)
            }
            if (payload.execution_trace) {
              result.execution_trace = payload.execution_trace
              setActiveTrace(payload.execution_trace)
            }
            if (payload.revealed_layer) {
              setActiveOverlay(payload.revealed_layer)
              setRevealedLayers(prev => Array.from(new Set([...prev, payload.revealed_layer])))
            }
          } else {
            const errorMsg = typeof payload.detail === 'string'
              ? payload.detail
              : (payload.error || `Specialist analysis failed (HTTP ${res.status})`)
            // Production integrity guard: an unavailable specialist remains unavailable.
            // A browser RGB heuristic is only allowed when explicitly opted in for development.
            result = {
                answer: `Remote sensing analysis unavailable: ${errorMsg}`,
                confidence: 'low',
                confidence_percent: null,
                confidenceScore: null,
                confidence_status: 'unavailable',
                confidence_reason: `API response HTTP ${res.status}: ${errorMsg}`,
                detected_features: ['Analysis Unavailable'],
                label: 'Analysis Error',
                suggested_followups: ['Configure the production VLM provider', 'Use the adapted specialist through the Python backend'],
                execution_trace: payload.execution_trace || null,
              }
          }
        } catch (fetchErr: any) {
          result = {
            answer: `Analysis unavailable: could not contact specialist analysis service (${fetchErr?.message || 'Network error'}).`,
            confidence: 'low',
            confidence_percent: null,
            confidenceScore: null,
            confidence_status: 'unavailable',
            confidence_reason: fetchErr?.message || 'Network connection to analysis API failed.',
            detected_features: ['Service Unreachable'],
            label: 'Connection Error',
            suggested_followups: ['Ensure FastAPI backend is running', 'Retry query'],
          }
        }
      }

      // Never manufacture confidence from qualitative labels. Only a calibrated
      // source may populate a numeric confidence value in the judge-facing UI.
      const isCalibrated = result.confidence_status === 'calibrated'
      const confPct = isCalibrated && typeof result.confidence_percent === 'number'
        ? Math.max(0, Math.min(100, Math.round(result.confidence_percent)))
        : 0
      result.confidence_percent = confPct
      result.confidenceScore = confPct

      if (result.region && typeof result.region.x_percent === 'number') {
        setActiveRegion({ region: result.region, label: result.label || 'Analysis Target', confidence_percent: confPct })
      } else setActiveRegion(null)

      setAnalysis(result)
      if (activeSessionId) {
        try {
          await recordAnalysisRun({
            sessionId: activeSessionId,
            taskType: classifyTask(prompt, 1, ['optical']),
            modelName: result.execution_trace?.steps?.find((step: any) => step?.tool)?.tool || null,
            modelAdaptation: result.is_synthetic ? 'not_applicable_or_synthetic' : null,
            result,
            executionTrace: result.execution_trace,
            confidence: typeof result.confidence_percent === 'number' ? result.confidence_percent / 100 : null,
            confidenceStatus: result.confidence_status || 'not_calibrated',
            status: result.confidence_status === 'unavailable' ? 'unavailable' : 'completed',
          })
        } catch (persistErr) {
          console.warn('[Orbital-AI] Supabase result persistence unavailable:', persistErr)
        }
      }
      setHistory(prev => [...prev, {
        question: prompt, answer: result.answer, confidence_percent: confPct, confidenceScore: confPct,
        confidence: result.confidence, confidence_status: result.confidence_status, confidence_reason: result.confidence_reason,
        detected_features: result.detected_features || [], label: result.label || 'Analysis',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        region: result.region ?? null,
        execution_trace: result.execution_trace ?? null, fusion_features: result.fusion_features ?? null,
        is_synthetic: result.is_synthetic,
        mode: result.mode,
      }])
      setSuggestions(result.suggested_followups?.length ? result.suggested_followups : OFFICIAL_REPRESENTATIVE_QUERIES.map(q => q.query))
      setSessionCallCount(prev => prev+1)
      setStatus('Analysis complete')
      setTimeout(() => chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' }), 100)
    } catch (err: any) {
      const errAnswer = `Analysis error: ${err?.message || 'Remote sensing tool pipeline failed.'}`
      setError(errAnswer)
      setStatus('Analysis failed')
      setHistory(prev => [...prev, {
        question: prompt, answer: errAnswer,
        confidence_percent: null, confidenceScore: null,
        confidence: 'unavailable', confidence_status: 'unavailable', confidence_reason: 'Pipeline execution threw an unhandled exception.',
        detected_features: ['Pipeline Failure'], label: 'Execution Error',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        region: null, execution_trace: null, fusion_features: null,
      }])
      setTimeout(() => chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' }), 100)
    } finally { setBusy(false) }
  }, [question, busy, atCap, imagePreview, history, sessionId, runBuildingDetection])

  // ── Comparison ────────────────────────────────────────────────────────────
  const runComparison = async (customPrompt?: string) => {
    setError('')
    const queryPrompt = customPrompt || 'Compare earlier and later satellite passes'
    setStatus('Running change detection…')
    compareRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' })
    let compAnswer = (beforeImage && afterImage)
      ? 'Change detection analysis requires the backend specialist service. Submit your image pair to receive a real bi-temporal comparison. (Backend service did not respond.)'
      : 'Change detection requires the backend specialist service. Upload a before/after image pair and ensure the analysis backend is running to receive real results. (Backend service did not respond.)'
    let confScore = 0
    let confStatus: string = 'unavailable'
    let features: string[] = ['Change Detection Unavailable']
    let traceData: ExecutionTrace | null = null
    let mode: string = 'unavailable'
    try {
      const res = await fetch(`${API_BASE}/api/analyze/change`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: queryPrompt,
          image: beforeImage || undefined,
          secondary_image: afterImage || undefined,
          task_type: 'change_detection',
          modality: 'optical',
          secondary_modality: 'optical',
        }),
      })
      if (res.ok) {
        const data = await res.json()
        if (data.answer) {
          compAnswer = data.answer
          confScore = typeof data.confidence === 'number' ? Math.round(data.confidence * 100) : (typeof data.confidenceScore === 'number' ? data.confidenceScore : 0)
          confStatus = data.confidence_status || 'not_calibrated'
          if (data.detected_features?.length) features = data.detected_features
          if (data.execution_trace) { traceData = data.execution_trace; setActiveTrace(data.execution_trace) }
          mode = data.mode || 'model'
        }
      } else {
        const legacyRes = await fetch(`${API_BASE}/api/compare`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ beforeImage: beforeImage || undefined, afterImage: afterImage || undefined, question: queryPrompt, beforeLabel: beforeImage ? 'Uploaded Earlier' : '2024 Baseline', afterLabel: afterImage ? 'Uploaded Later' : '2026 Pass' }),
        })
        if (legacyRes.ok) {
          const data = await legacyRes.json()
          if (data.answer) {
            compAnswer = data.answer; confScore = typeof data.confidenceScore === 'number' ? data.confidenceScore : 0
            confStatus = data.confidence_status || 'not_calibrated'
            if (data.detected_features?.length) features = data.detected_features
            if (data.execution_trace) { traceData = data.execution_trace; setActiveTrace(data.execution_trace) }
            mode = data.mode || 'model'
          }
        } else {
          mode = 'unavailable'
          confStatus = 'unavailable'
        }
      }
    } catch {
      mode = 'unavailable'
      confStatus = 'unavailable'
    }

    setTemporalResult({ question: queryPrompt, answer: compAnswer, confidenceScore: confScore, confidence_status: confStatus, features, execution_trace: traceData, mode })
    setHistory(prev => [...prev, {
      question: queryPrompt, answer: compAnswer, confidence_percent: confScore, confidenceScore: confScore,
      confidence: confStatus === 'unavailable' ? 'unavailable' : (confScore >= 85 ? 'high' : 'medium'), confidence_status: confStatus, detected_features: features,
      label: 'Bi-Temporal Change Detection', timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      region: null, execution_trace: traceData, fusion_features: null,
      is_synthetic: false,
      mode,
    }])
    setStatus('Change detection complete')
    addToast('Change detection complete: results below', 'success')
    setTimeout(() => chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' }), 200)
  }

  // ── Fusion ────────────────────────────────────────────────────────────────
  const handleRunFusion = async (
    opticalDataUrl: string,
    sarDataUrl: string,
    fusionQuery: string = 'Identify built-up and water-covered regions using joint optical and SAR data'
  ) => {
    setBusy(true); setError('')
    setStatus('Running optical–SAR fusion…')
    try {
      const res = await fetch(`${API_BASE}/api/analyze/optical-sar`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: fusionQuery,
          image: opticalDataUrl,
          secondary_image: sarDataUrl,
          task_type: 'sar_optical_fusion',
          modality: 'optical',
          secondary_modality: 'sar',
        }),
      })
      const data = await res.json()
      if (res.ok && data.answer) {
        if (data.fusion_features) setFusionFeatures(data.fusion_features)
        if (data.execution_trace) { setLastFusionTrace(data.execution_trace); setActiveTrace(data.execution_trace) }
        const confVal = typeof data.confidence === 'number' ? Math.round(data.confidence * 100) : (typeof data.confidence_percent === 'number' ? data.confidence_percent : 0)
        setHistory(prev => [...prev, {
          question: fusionQuery, answer: data.answer, confidence_percent: confVal, confidenceScore: confVal,
          confidence: data.confidence_level ? data.confidence_level.toLowerCase() : (data.confidence_status === 'unavailable' || data.confidence == null ? 'unavailable' : data.confidence),
          confidence_reason: data.reasoning || data.confidence_reason,
          detected_features: data.detected_features || ['Optical–SAR Joint Fusion'], label: 'Optical–SAR Fusion',
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          execution_trace: data.execution_trace ?? null, fusion_features: data.fusion_features ?? null,
          mode: data.mode || 'model',
        }])
        setAnalysis(normalizeAnalyzeResponse(data, fusionQuery)); setStatus('Fusion complete')
        addToast('Optical–SAR fusion complete', 'success')
      } else {
        const legacyRes = await fetch(`${API_BASE}/api/fuse`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ opticalImage: opticalDataUrl, sarImage: sarDataUrl, question: fusionQuery }),
        })
        const legacyData = await legacyRes.json()
        if (legacyRes.ok && legacyData.answer) {
          if (legacyData.fusion_features) setFusionFeatures(legacyData.fusion_features)
          if (legacyData.execution_trace) { setLastFusionTrace(legacyData.execution_trace); setActiveTrace(legacyData.execution_trace) }
          const confVal = typeof legacyData.confidence_percent === 'number' ? legacyData.confidence_percent : 0
          setHistory(prev => [...prev, {
            question: fusionQuery, answer: legacyData.answer, confidence_percent: confVal, confidenceScore: confVal,
            confidence: legacyData.confidence || 'unavailable', confidence_reason: legacyData.confidence_reason, confidence_status: legacyData.confidence_status || 'not_calibrated',
            detected_features: legacyData.detected_features || [], label: legacyData.label || 'Optical–SAR Fusion',
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
            execution_trace: legacyData.execution_trace ?? null, fusion_features: legacyData.fusion_features ?? null,
            mode: legacyData.mode || 'model',
          }])
          setAnalysis(legacyData as Analysis); setStatus('Fusion complete')
          addToast('Optical–SAR fusion complete', 'success')
        } else throw new Error(legacyData.error || 'Fusion failed')
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Fusion error'
      setError(msg); setStatus('Fusion error'); addToast(msg, 'error')
    } finally { setBusy(false) }
  }

  // ── Export report ─────────────────────────────────────────────────────────
  const downloadReport = () => {
    const data = {
      generator: 'ORBITAL-AI: Agentic Remote-Sensing Analysis', version: '3.1.0',
      exported_at: new Date().toISOString(), session_id: sessionId, active_mode: appMode,
      telemetry: imageTelemetry, bigearth_classification: benResults,
      latest_analysis: analysis, building_audit: buildingAnalysis,
      optical_sar_fusion: fusionFeatures, latest_execution_trace: activeTrace || lastFusionTrace,
      conversation_history: history,
    }
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = `orbital_ai_report_${Date.now()}.json`
    document.body.appendChild(a); a.click(); document.body.removeChild(a); URL.revokeObjectURL(url)
    addToast('Report exported as JSON', 'success')
  }

  // ── Download GeoJSON building footprints ────────────────────────────────
  const downloadGeoJSON = () => {
    if (!buildingAnalysis?.detections?.length) {
      addToast('Run building footprint detection first', 'warning')
      return
    }
    const dims = buildingAnalysis.image_dimensions ?? { width: 1000, height: 1000 }
    const features = buildingAnalysis.detections.map(d => ({
      type: 'Feature' as const,
      properties: {
        id: d.id,
        confidence: d.confidence,
        confidence_tier: d.confidence_tier,
        area_px: d.area,
        is_partial: d.is_partial,
        centroid_x_pct: d.centroid_pct?.[0],
        centroid_y_pct: d.centroid_pct?.[1],
      },
      geometry: {
        type: 'Polygon' as const,
        coordinates: [[
          ...d.polygon_pct.map(([xp, yp]) => [
            parseFloat(((xp / 100) * dims.width).toFixed(2)),
            parseFloat(((yp / 100) * dims.height).toFixed(2)),
          ]),
        ]],
      },
    }))
    const geojson = {
      type: 'FeatureCollection',
      metadata: {
        generator: 'ORBITAL-AI: Building Footprint Extractor',
        version: '3.0.0',
        exported_at: new Date().toISOString(),
        total_buildings: buildingAnalysis.building_count,
        high_confidence: buildingAnalysis.high_confidence_count,
        medium_confidence: buildingAnalysis.medium_confidence_count,
        confidence_level: buildingAnalysis.confidence_level,
      },
      features,
    }
    const blob = new Blob([JSON.stringify(geojson, null, 2)], { type: 'application/geo+json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = `satquery_buildings_${Date.now()}.geojson`
    document.body.appendChild(a); a.click(); document.body.removeChild(a); URL.revokeObjectURL(url)
    addToast(`GeoJSON exported (${buildingAnalysis.building_count} footprints)`, 'success')
  }

  // ── Download Markdown analysis report ──────────────────────────────────
  const downloadMarkdownReport = () => {
    const ts = new Date().toISOString()
    const trace = activeTrace || lastFusionTrace
    const lines: string[] = [
      '# SatQuery AI: Remote Sensing Mission Analysis Report',
      `**Generated:** ${ts} · **Version:** SatQuery-Agent-v3.0`,
      '---',
      '## 1. Session Overview',
      `- **Active Mode:** \`${appMode}\``,
      `- **Session ID:** \`${sessionId ?? 'not created'}\``,
      `- **Terrain Classification:** ${imageTelemetry.landClass} (${imageTelemetry.landClassPct})`,
      '',
      '## 2. BigEarthNet v2.0 Land-Cover Classification',
    ]
    if (benResults) {
      lines.push(`- **Top Label:** ${benResults.top_label}`)
      lines.push(`- **Model:** \`${benResults.model_id}\``)
      lines.push(`- **Active Labels:** ${benResults.active_labels?.map(l => l.name).join(', ')}`)
    } else { lines.push('- *Not yet run*') }
    lines.push('', '## 3. Latest Analysis')
    if (analysis) {
      lines.push(`> ${analysis.answer}`)
      lines.push(`- **Confidence:** ${analysis.confidence_percent ?? '—'}%`)
    } else { lines.push('- *No analysis run yet*') }
    if (buildingAnalysis) {
      lines.push('', '## 4. Building Footprint Audit')
      lines.push(`- **Total Structures:** ${buildingAnalysis.building_count}`)
      lines.push(`- **High Confidence:** ${buildingAnalysis.high_confidence_count}`)
      lines.push(`- **Medium Confidence:** ${buildingAnalysis.medium_confidence_count}`)
      lines.push(`- **Partial Edge:** ${buildingAnalysis.partial_count}`)
      lines.push(`- **Confidence Level:** ${buildingAnalysis.confidence_level} (${Math.round(buildingAnalysis.confidence * 100)}%)`)
      lines.push(`- **Validation:** ${buildingAnalysis.validation_status}`)
    }
    if (trace) {
      lines.push('', '## 5. Observable Execution Trace')
      lines.push(`- **Task:** ${trace.task_type}`)
      lines.push(`- **Total Duration:** ${trace.total_duration_ms?.toFixed(0) ?? '—'} ms`)
      ;(trace.steps ?? []).forEach((s: ExecutionTraceStep) => {
        lines.push(`  - Step ${s.step}: **${String(s.tool)}**: ${String(s.output_summary ?? '')} (${typeof s.duration_ms === 'number' ? s.duration_ms.toFixed(0) : '—'} ms)`)
      })
    }
    if (history.length) {
      lines.push('', '## 6. Conversation History')
      history.forEach((m, i) => {
        lines.push(`### Q${i+1}: ${m.question}`)
        lines.push(`> ${m.answer}`)
        lines.push(`- Confidence: ${m.confidence_percent}% · Label: ${m.label}`)
      })
    }
    lines.push('', '---', '*Report certified by SatQuery AI Agentic Remote-Sensing Platform.*')
    const blob = new Blob([lines.join('\n')], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = `orbital_ai_analysis_report_${Date.now()}.md`
    document.body.appendChild(a); a.click(); document.body.removeChild(a); URL.revokeObjectURL(url)
    addToast('Analysis report exported as Markdown', 'success')
  }



  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div style={{ background: C.base, color: C.white, fontFamily: "'Inter', sans-serif" }} className="min-h-full">
      {/* Hidden file inputs */}
      <input ref={fileInputRef} type="file" accept="image/*,.tif,.tiff,.geotiff" className="hidden"
        onClick={e => { (e.target as HTMLInputElement).value = '' }}
        onChange={e => { handleFile(e.target.files?.[0]); (e.target as HTMLInputElement).value = '' }} />
      <input ref={beforeInputRef} type="file" accept="image/*" className="hidden"
        onClick={e => { (e.target as HTMLInputElement).value = '' }}
        onChange={e => { handleBeforeFile(e.target.files?.[0]); (e.target as HTMLInputElement).value = '' }} />
      <input ref={afterInputRef} type="file" accept="image/*" className="hidden"
        onClick={e => { (e.target as HTMLInputElement).value = '' }}
        onChange={e => { handleAfterFile(e.target.files?.[0]); (e.target as HTMLInputElement).value = '' }} />

      {/* Modals */}
      {showInfoModal && (
        <div className="fixed inset-0 z-[200] flex items-center justify-center p-4" style={{ background: 'rgba(2,8,16,0.88)', backdropFilter: 'blur(12px)' }} onClick={() => setShowInfoModal(false)}>
          <div className="max-w-lg w-full rounded-2xl p-7 fade-in" style={{ background: C.surface, border: `1px solid ${C.borderHover}`, boxShadow: '0 0 60px rgba(32,217,255,0.10)' }} onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-5">
              <div className="flex items-center gap-2"><span className="w-2 h-2 rounded-full" style={{ background: C.cyan, boxShadow: `0 0 8px ${C.cyan}` }} /><span className="text-xs font-mono tracking-widest" style={{ color: C.cyan }}>BUILDING DETECTION ENGINE</span></div>
              <button onClick={() => setShowInfoModal(false)} className="btn-ghost px-2 py-1 text-xs rounded">✕ Close</button>
            </div>
            <h3 className="text-xl font-bold mb-3" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>How Building Audit Works</h3>
            <p className="text-sm leading-relaxed mb-3" style={{ color: C.muted }}>Orbital-AI tiles high-resolution imagery into overlapping windows, runs <strong style={{ color: C.white }}>YOLO instance segmentation</strong> on each tile to detect rooftop boundaries, then remaps coordinates globally.</p>
            <p className="text-sm leading-relaxed mb-3" style={{ color: C.muted }}>An <strong style={{ color: C.white }}>IoU Non-Maximum Suppression</strong> filter merges duplicate detections across tile seams, assigns unique IDs (B001, B002…), and separates complete structures from partial edge footprints.</p>
            <div className="rounded-xl p-3 text-xs font-mono" style={{ background: `${C.cyan}08`, border: `1px solid ${C.border}` }}>
              <div className="mb-1" style={{ color: C.cyan }}>Confidence tiers:</div>
              <div style={{ color: C.muted }}>
                <span style={{ color: C.cyan }}>High</span> ≥0.45 · <span style={{ color: C.mint }}>Medium</span> 0.28–0.45 · <span style={{ color: C.orange }}>Partial</span> edge objects
              </div>
            </div>
          </div>
        </div>
      )}
      {showEvalModal && <EvaluationCriteriaModal isOpen={showEvalModal} onClose={() => setShowEvalModal(false)} />}
      <AgentTraceModal isOpen={traceModalOpen} onClose={() => setTraceModalOpen(false)} trace={activeTrace || lastFusionTrace} queryTitle={analysis?.label || 'Remote Sensing Agent Trace'} />
      <ToastContainer toasts={toasts} onRemove={removeToast} />

      {/* Technical Documentation Modal */}
      <DocumentationModal isOpen={docModalOpen} onClose={() => setDocModalOpen(false)} />

      {/* Research & Institutional Contact Modal */}
      <ContactModal
        isOpen={contactModalOpen}
        onClose={() => setContactModalOpen(false)}
        onSubmitSuccess={(msg) => addToast(msg, 'success')}
      />

      {/* Legal & Governance Modals (Privacy, Terms, Strict Required Data Policy) */}
      <LegalModals
        isOpen={legalModalOpen}
        activeTab={activeLegalTab}
        onClose={() => setLegalModalOpen(false)}
        onSelectTab={setActiveLegalTab}
        onOpenCookieSettings={() => {
          setActiveLegalTab('cookies')
          setLegalModalOpen(true)
        }}
      />

      {/* Strictly Required Data Policy Consent Banner */}
      <CookieConsentBanner
        onOpenPolicy={() => {
          setActiveLegalTab('cookies')
          setLegalModalOpen(true)
        }}
      />

      {/* Navbar */}
      <Navbar
        scrolled={scrolled}
        onAnalyze={focusWorkspace}
        onEval={() => setShowEvalModal(true)}
        onUpload={openUploader}
        appMode={appMode}
        setAppMode={setAppMode}
        activeView={activeView}
        setActiveView={setActiveView}
        onOpenDocs={() => setDocModalOpen(true)}
        onOpenContact={() => setContactModalOpen(true)}
      />

      {/* Hero */}
      <HeroSection onUpload={openUploader} onAnalyze={focusWorkspace}
        onQueryChip={q => { focusWorkspace(); setTimeout(() => askQuestion(q), 400) }}
        imageTelemetry={imageTelemetry} buildingAnalysis={buildingAnalysis} />

      {/* ── WORKSPACE ───────────────────────────────────────────────────────── */}
      <section id="analyze" className="pt-8 pb-16" style={{ background: C.base }}>
        <div className="max-w-7xl mx-auto px-5">

          {/* View toggle (Workspace vs Live Dashboard & Audit) */}
          <div className="flex flex-wrap items-center justify-between gap-4 mb-6 pb-4" style={{ borderBottom: `1px solid ${C.border}` }}>
            <div className="flex items-center gap-2 p-1 rounded-xl" style={{ background: C.surface, border: `1px solid ${C.border}` }}>
              <button
                type="button"
                onClick={() => setActiveView('workspace')}
                className="px-3.5 py-1.5 rounded-lg text-xs font-mono font-medium transition-all cursor-pointer"
                style={{
                  background: activeView === 'workspace' ? `${C.cyan}20` : 'transparent',
                  color: activeView === 'workspace' ? C.cyan : C.muted,
                  border: activeView === 'workspace' ? `1px solid ${C.cyan}44` : '1px solid transparent',
                  boxShadow: activeView === 'workspace' ? `0 0 14px ${C.cyan}22` : 'none',
                }}
              >
                ⊞ Satellite Workspace
              </button>
              <button
                type="button"
                onClick={() => setActiveView('dashboard')}
                className="px-3.5 py-1.5 rounded-lg text-xs font-mono font-medium transition-all cursor-pointer flex items-center gap-1.5"
                style={{
                  background: activeView === 'dashboard' ? `${C.cyan}20` : 'transparent',
                  color: activeView === 'dashboard' ? C.cyan : C.muted,
                  border: activeView === 'dashboard' ? `1px solid ${C.cyan}44` : '1px solid transparent',
                  boxShadow: activeView === 'dashboard' ? `0 0 14px ${C.cyan}22` : 'none',
                }}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" /><path d="M3 9h18" /><path d="M9 21V9" /></svg>
                <span>Audit & Telemetry Dashboard</span>
              </button>
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setDocModalOpen(true)}
                className="btn-ghost px-3 py-1.5 text-xs font-mono rounded-lg flex items-center gap-1.5"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z" /><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z" /></svg><span>Docs & API</span>
              </button>
              <button
                type="button"
                onClick={downloadReport}
                className="btn-ghost px-3 py-1.5 text-xs font-mono flex items-center gap-1.5 rounded-lg"
                title="Download session audit report"
              >
                <span>⬇</span><span className="hidden sm:inline">Export JSON</span>
              </button>
              <button
                type="button"
                onClick={downloadMarkdownReport}
                className="btn-ghost px-3 py-1.5 text-xs font-mono flex items-center gap-1.5 rounded-lg"
                title="Download analysis report as Markdown"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><polyline points="14 2 14 8 20 8" /><line x1="16" y1="13" x2="8" y2="13" /><line x1="16" y1="17" x2="8" y2="17" /></svg><span className="hidden sm:inline">Report.md</span>
              </button>
              {buildingAnalysis && (
                <button
                  type="button"
                  onClick={downloadGeoJSON}
                  className="btn-ghost px-3 py-1.5 text-xs font-mono flex items-center gap-1.5 rounded-lg"
                  title="Download building footprints as GeoJSON"
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6" /><line x1="8" y1="2" x2="8" y2="18" /><line x1="16" y1="6" x2="16" y2="22" /></svg><span className="hidden sm:inline">GeoJSON</span>
                </button>
              )}
              {(activeTrace || lastFusionTrace) && (
                <button
                  type="button"
                  onClick={() => setTraceModalOpen(true)}
                  className="btn-outline-cyan px-3 py-1.5 text-xs font-mono flex items-center gap-1.5 rounded-lg"
                >
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" /></svg><span>Agent Trace</span>
                </button>
              )}
            </div>
          </div>

          {activeView === 'dashboard' ? (
            <DashboardView
              sessionId={sessionId}
              sessionCallCount={sessionCallCount}
              sessionLimit={SESSION_CALL_LIMIT}
              imageTelemetry={imageTelemetry}
              buildingAnalysis={buildingAnalysis}
              benResults={benResults}
              history={history}
              activeMode={appMode}
              onSwitchToWorkspace={() => setActiveView('workspace')}
              onSelectHistoricalQuery={(q) => {
                setActiveView('workspace')
                setQuestion(q)
                askQuestion(q)
              }}
              onExportReport={downloadReport}
              onResetSession={() => {
                setHistory([])
                setSessionCallCount(0)
                setSessionId(null)
                setAnalysis(null)
                setActiveRegion(null)
                setTemporalResult(null)
                addToast('Session state & telemetry cleared', 'info')
              }}
            />
          ) : (
            <>
              {/* Mode switcher */}
              <div className="flex flex-wrap items-center gap-2 mb-6">
                <div className="text-xs font-mono uppercase tracking-widest mr-3" style={{ color: C.dim }}>Analysis Mode</div>
                {([
                  { id: 'single', label: 'Single Image', col: C.cyan },
                  { id: 'fusion', label: 'Optical-SAR Fusion', col: C.mint },
                  { id: 'compare', label: 'Bi-Temporal Compare', col: C.orange },
                ] as const).map(m => (
                  <button key={m.id} onClick={() => { setAppMode(m.id); if (m.id !== 'single') scrollToSection(m.id === 'compare' ? 'compare-section' : 'analyze') }}
                    className="px-4 py-2 rounded-xl text-xs font-mono font-bold tracking-wide transition-all"
                    style={{
                      background: appMode === m.id ? `${m.col}20` : 'transparent',
                      color: appMode === m.id ? m.col : C.muted,
                      border: appMode === m.id ? `1px solid ${m.col}55` : `1px solid ${C.border}`,
                      boxShadow: appMode === m.id ? `0 0 20px ${m.col}22` : 'none',
                      cursor: 'pointer',
                    }}>
                    {m.label}
                  </button>
                ))}
              </div>

              {appMode === 'fusion' ? (
            <OpticalSarFusionPanel onRunFusion={handleRunFusion} busy={busy} fusionFeatures={fusionFeatures}
              lastFusionTrace={lastFusionTrace} onOpenTrace={trace => { setActiveTrace(trace); setTraceModalOpen(true) }} />
          ) : (
            <div className="grid grid-cols-1 xl:grid-cols-[1fr_380px] gap-5">

              {/* Left: Image viewer + BigEarthNet panel */}
              <div className="space-y-4">

                {/* Image viewer card */}
                <div className="rounded-2xl overflow-hidden" style={{ background: C.surface, border: `1px solid ${C.border}`, boxShadow: '0 0 60px rgba(32,217,255,0.04)' }}>
                  {/* Image Viewer Toolbar */}
                  <div className="flex items-center gap-3 px-4 py-2.5 text-xs font-mono" style={{ background: `${C.base}BB`, borderBottom: `1px solid ${C.border}` }}>
                    <div className="flex items-center gap-2 flex-1 min-w-0">
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke={C.muted} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
                      <span className="truncate text-[10px]" style={{ color: C.muted }}>
                        {imagePreview ? imageTelemetry.locationTag || 'Image loaded' : 'No image loaded'}
                      </span>
                    </div>
                    <div className="ml-auto flex items-center gap-2 shrink-0">
                      <span className="text-[10px]" style={{ color: busy ? C.orange : C.mint }}>● {busy ? 'ANALYZING' : status.split('·')[0].trim().toUpperCase()}</span>
                    </div>
                  </div>

                  {/* Image with overlays — drag-drop enabled */}
                  <div className="relative"
                    style={{ minHeight: 360, background: C.base, cursor: imagePreview ? 'default' : 'pointer' }}
                    onDragOver={e => { e.preventDefault(); setIsDragOver(true) }}
                    onDragLeave={() => setIsDragOver(false)}
                    onDrop={handleDrop}
                    onClick={!imagePreview ? openUploader : undefined}
                  >
                    {imagePreview ? (
                      <img src={imagePreview}
                        alt="Uploaded satellite image"
                        className="w-full h-full object-cover absolute inset-0 transition-opacity duration-300"
                        style={{ opacity: showBuildingsOverlay ? 0.94 : 0.7, minHeight: 360 }} />
                    ) : (
                      <div className={`absolute inset-0 flex flex-col items-center justify-center gap-4 transition-all ${isDragOver ? 'drop-active' : ''}`}
                        style={{ border: `2px dashed ${isDragOver ? C.cyan : C.border}`, borderRadius: 0, background: isDragOver ? `${C.cyan}06` : 'transparent' }}>
                        <div className="w-16 h-16 rounded-2xl flex items-center justify-center" style={{ background: `${C.cyan}12`, border: `1px solid ${C.borderHover}` }}>
                          <svg width="28" height="28" viewBox="0 0 28 28" fill="none"><path d="M14 4v16M5 13l9-9 9 9M4 24h20" stroke={C.cyan} strokeWidth="2" strokeLinecap="round"/></svg>
                        </div>
                        <div className="text-center">
                          <div className="text-sm font-semibold mb-1" style={{ color: C.white }}>Drop satellite image here</div>
                          <div className="text-xs" style={{ color: C.muted }}>JPEG · PNG · GeoTIFF · WebP · up to 20 MB</div>
                        </div>
                        <button onClick={e => { e.stopPropagation(); openUploader() }} className="btn-outline-cyan px-5 py-2.5 text-sm font-medium rounded-xl">Browse files</button>
                        <div className="text-[10px] font-mono text-center max-w-xs" style={{ color: C.dim }}>
                          Supports Cartosat-2S, Sentinel-1 (SAR), Sentinel-2, Landsat, and standard aerial imagery
                        </div>
                      </div>
                    )}

                    {!showBuildingsOverlay && imagePreview && <div className="absolute inset-0" style={{ background: `${C.base}28` }} />}

                    {/* Grid overlay */}
                    {imagePreview && (
                      <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox="0 0 700 360" preserveAspectRatio="none">
                        {[1,2,3,4].map(i => <line key={`h${i}`} x1="0" y1={i*72} x2="700" y2={i*72} stroke={C.cyan} strokeOpacity="0.06" strokeWidth="0.5" />)}
                        {[1,2,3,4,5,6,7,8,9].map(i => <line key={`v${i}`} x1={i*78} y1="0" x2={i*78} y2="360" stroke={C.cyan} strokeOpacity="0.06" strokeWidth="0.5" />)}

                        {/* Building footprints */}
                        {showBuildingsOverlay && buildingAnalysis?.detections && (
                          <g>
                            {buildingAnalysis.detections.map(d => {
                              const sc = d.is_partial ? '#FFB86C' : d.confidence_tier === 'high' ? C.cyan : C.mint
                              const fc = d.is_partial ? 'rgba(255,184,108,0.18)' : d.confidence_tier === 'high' ? 'rgba(32,217,255,0.20)' : 'rgba(53,224,184,0.18)'
                              const cx = (d.centroid_pct[0]/100)*700, cy = (d.centroid_pct[1]/100)*360
                              return (
                                <g key={d.id}>
                                  {d.polygon_pct && d.polygon_pct.length >= 3
                                    ? <polygon points={d.polygon_pct.map(([px,py]) => `${(px/100)*700},${(py/100)*360}`).join(' ')} fill={fc} stroke={sc} strokeWidth="1.2" />
                                    : <rect x={(d.bbox_pct[0]/100)*700} y={(d.bbox_pct[1]/100)*360} width={(d.bbox_pct[2]/100)*700} height={(d.bbox_pct[3]/100)*360} fill={fc} stroke={sc} strokeWidth="1.2" rx="2" />
                                  }
                                  <text x={cx} y={cy+2.5} fill="#FFF" fontSize="5" fontFamily="JetBrains Mono" fontWeight="bold" textAnchor="middle">{d.id}</text>
                                </g>
                              )
                            })}
                          </g>
                        )}

                        {/* Analysis region */}
                        {activeRegion && activeRegion.label !== 'Building Footprint Audit' && (
                          <g>
                            <rect x={(activeRegion.region.x_percent/100)*700} y={(activeRegion.region.y_percent/100)*360}
                              width={(activeRegion.region.w_percent/100)*700} height={(activeRegion.region.h_percent/100)*360}
                              fill="rgba(32,217,255,0.07)" stroke={C.cyan} strokeWidth="1.5" strokeDasharray="6 3" rx="3" />
                            <circle cx={(activeRegion.region.x_percent/100)*700} cy={(activeRegion.region.y_percent/100)*360} r="3" fill={C.cyan} />
                            <rect x={(activeRegion.region.x_percent/100)*700}
                              y={Math.max(4, (activeRegion.region.y_percent/100)*360 - 22)}
                              width={Math.max(120, activeRegion.label.length*7.2+24)} height="18"
                              fill={`${C.surface}F5`} stroke={C.cyan} strokeWidth="1" rx="4" />
                            <text x={(activeRegion.region.x_percent/100)*700+8}
                              y={Math.max(15, (activeRegion.region.y_percent/100)*360 - 9)}
                              fill={C.cyan} fontSize="8" fontWeight="bold" fontFamily="JetBrains Mono">
                              ● {activeRegion.label.toUpperCase()}
                            </text>
                          </g>
                        )}
                        <text x="8" y="16" fill={C.muted} fontSize="7.5" fontFamily="JetBrains Mono" fillOpacity="0.6">{imageTelemetry.locationTag}</text>
                      </svg>
                    )}

                    {/* Analysis region badge */}
                    {imagePreview && activeRegion && (
                      <div className="absolute top-3 left-3 z-10 flex items-center gap-2 px-3 py-1.5 rounded-lg text-[11px] font-mono glass-strong" style={{ border: `1px solid ${C.cyan}`, color: C.cyan }}>
                        <span className="w-2 h-2 rounded-full" style={{ background: C.cyan, boxShadow: `0 0 6px ${C.cyan}` }} />
                        <span className="font-semibold">{activeRegion.label}</span>
                      </div>
                    )}

                    {/* Zoom controls */}
                    <div className="absolute bottom-3 left-3 flex flex-col gap-1 rounded-xl p-1" style={{ background: `${C.surface}CC`, border: `1px solid ${C.border}` }}>
                      {['+','⊙','−'].map(z => (
                        <button key={z} className="w-7 h-7 rounded-lg text-xs flex items-center justify-center" style={{ color: C.muted, cursor: 'default' }}>{z}</button>
                      ))}
                    </div>

                    {/* Upload new image button (when image is loaded) */}
                    {imagePreview && (
                      <button onClick={openUploader} className="absolute bottom-3 right-3 px-3 py-1.5 rounded-lg text-xs font-mono glass-strong cursor-pointer transition-all hover:border-cyan-400"
                        style={{ border: `1px solid ${C.border}`, color: C.muted }}>
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-1 inline-block"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" /></svg>New Image
                      </button>
                    )}
                  </div>

                  {/* Provenance bar — only shown after real analysis */}
                  {history.length > 0 && (
                    <div className="px-5 py-3 flex items-center gap-4 text-[10px] font-mono flex-wrap" style={{ borderTop: `1px solid ${C.border}`, background: `${C.base}66` }}>
                      <span style={{ color: C.dim }}>LAST ANALYSIS</span>
                      <span style={{ color: C.cyan }}>{history[history.length - 1]?.label || 'Query complete'}</span>
                      <span style={{ color: C.dim }}>·</span>
                      {(() => {
                        const last = history[history.length - 1]
                        const cs = last?.confidence_status
                        if (cs === 'calibrated' && typeof last?.confidence_percent === 'number' && last.confidence_percent > 0) {
                          return <span style={{ color: C.mint }}>Confidence: {last.confidence_percent}%</span>
                        } else if (cs === 'unavailable' || (typeof last?.confidence_percent === 'number' && last.confidence_percent === 0)) {
                          return <span style={{ color: C.danger }}>Confidence: Unavailable</span>
                        } else {
                          return <span style={{ color: C.orange }}>Confidence: Not calibrated</span>
                        }
                      })()}
                      <span style={{ color: C.dim }}>·</span>
                      <span style={{ color: C.muted }}>{history[history.length - 1]?.timestamp}</span>
                    </div>
                  )}
                </div>

                {/* BigEarthNet Classification Panel */}
                <BENPanel results={benResults} loading={classifyingBEN} />
              </div>

              {/* Right: AI Chat Panel */}
              <div className="flex flex-col rounded-2xl overflow-hidden" style={{ background: C.surface, border: `1px solid ${C.border}`, minHeight: 600 }}>
                {/* Panel header */}
                <div className="px-4 py-3.5 flex items-center justify-between" style={{ borderBottom: `1px solid ${C.border}`, background: `${C.base}88` }}>
                  <div className="flex items-center gap-2.5">
                    <div className="w-7 h-7 rounded-lg flex items-center justify-center" style={{ background: `${C.cyan}18`, border: `1px solid ${C.borderHover}` }}>
                      <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><circle cx="7" cy="7" r="6" stroke={C.cyan} strokeWidth="1.2"/><circle cx="7" cy="7" r="2" fill={C.cyan}/><line x1="7" y1="1" x2="7" y2="4" stroke={C.cyan} strokeWidth="1.2"/><line x1="7" y1="10" x2="7" y2="13" stroke={C.cyan} strokeWidth="1.2"/></svg>
                    </div>
                    <div>
                      <div className="text-xs font-semibold font-mono" style={{ color: C.white }}>AI ANALYSIS PANEL</div>
                      <div className="text-[10px] font-mono" style={{ color: busy ? C.orange : C.mint }}>● {busy ? status : 'Ready'}</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <button onClick={() => setShowInfoModal(true)} className="w-6 h-6 rounded-lg flex items-center justify-center text-[11px] btn-ghost">?</button>
                    {(activeTrace || lastFusionTrace) && (
                      <button onClick={() => setTraceModalOpen(true)} className="px-2 py-1 rounded-lg text-[10px] font-mono flex items-center gap-1 btn-outline-cyan">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" /></svg><span>Trace</span>
                      </button>
                    )}
                  </div>
                </div>

                {/* Chat history */}
                <div className="flex-1 p-4 space-y-4 overflow-y-auto" style={{ maxHeight: 400 }}>
                  {history.length === 0 && !busy && (
                    <div className="rounded-xl p-4 text-xs leading-relaxed text-center" style={{ background: `${C.card}88`, border: `1px solid ${C.border}`, color: C.muted }}>
                      <div className="mb-2 opacity-40 flex justify-center">
                        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="2" /><path d="M16.24 7.76a6 6 0 0 1 0 8.49m-8.48-.01a6 6 0 0 1 0-8.49m11.31-2.82a10 10 0 0 1 0 14.14m-14.14 0a10 10 0 0 1 0-14.14" /></svg>
                      </div>
                      <div className="font-medium mb-1" style={{ color: C.white }}>Upload an image to begin</div>
                      <div>Ask in natural language. SatQuery AI routes the request to the relevant analysis pipeline and returns the available result with an execution trace.</div>
                    </div>
                  )}

                  {history.map((m, i) => (
                    <div key={`${m.question}-${i}`} className="space-y-2 fade-in">
                      {/* User message */}
                      <div className="flex justify-end">
                        <div className="max-w-[88%] rounded-xl px-3.5 py-2 text-xs" style={{ background: `${C.cyan}14`, border: `1px solid ${C.cyan}28`, color: C.cyan }}>
                          {m.question}
                        </div>
                      </div>

                      {/* AI response */}
                      <div className="rounded-xl px-4 py-3.5 space-y-2.5" style={{ background: C.card, border: `1px solid ${C.border}` }}>

                        <div className="flex items-center justify-between text-[10px] font-mono flex-wrap gap-1">
                          <ConfidenceBadge confidence={m.confidence} percent={m.confidence_percent ?? m.confidenceScore} confidence_status={m.confidence_status} reason={m.confidence_reason} mode={m.mode} />
                          <span style={{ color: C.dim }}>{m.timestamp}</span>
                        </div>
                        <div className="text-xs leading-relaxed" style={{ color: '#D0E0F0', lineHeight: 1.75 }}>{m.answer}</div>

                        {/* Features */}
                        {m.detected_features?.length > 0 && (
                          <div className="flex flex-wrap gap-1">
                            {m.detected_features.map(f => (
                              <span key={f} className="text-[9px] px-2 py-0.5 rounded-md font-mono" style={{ background: `${C.cyan}0A`, color: C.muted, border: `1px solid ${C.border}` }}>{f}</span>
                            ))}
                          </div>
                        )}

                        {/* Building count */}
                        {buildingAnalysis && (m.question.toLowerCase().includes('building') || m.question.toLowerCase().includes('how many') || m.question.toLowerCase().includes('count')) && (
                          <div className="rounded-xl px-3 py-2.5 text-xs" style={{ background: `${C.cyan}0C`, border: `1px solid ${C.cyan}28` }}>
                            <div className="font-mono font-bold text-sm mb-1" style={{ color: C.cyan }}>
                              DETECTED: <span style={{ color: C.white }}>{buildingAnalysis.building_count}</span> buildings
                            </div>
                            <div className="font-mono text-[10px]" style={{ color: C.muted }}>
                              High: <span style={{ color: C.mint }}>{buildingAnalysis.high_confidence_count}</span> · Med: <span style={{ color: C.orange }}>{buildingAnalysis.medium_confidence_count}</span> · Partial: <span style={{ color: '#FFB86C' }}>{buildingAnalysis.partial_count}</span>
                            </div>
                            <div className="text-[9px] mt-1 italic" style={{ color: C.dim }}>{buildingAnalysis.validation_status}</div>
                          </div>
                        )}

                        {/* Execution trace button */}
                        {m.execution_trace && (
                          <button onClick={() => { setActiveTrace(m.execution_trace!); setTraceModalOpen(true) }}
                            className="px-2.5 py-1 rounded-lg text-[10px] font-mono btn-outline-cyan flex items-center gap-1.5 w-fit">
                            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" /></svg>
                            <span>View Execution Trace ({m.execution_trace.task_type.toUpperCase()})</span>
                          </button>
                        )}

                        {/* Fusion telemetry */}
                        {m.fusion_features && (
                          <div className="rounded-lg p-2.5 text-[10px] font-mono space-y-1" style={{ background: `${C.mint}0A`, border: `1px solid ${C.mint}28` }}>
                            <div className="flex justify-between font-bold" style={{ color: C.mint }}>
                              <span>CROSS-MODAL TELEMETRY</span><span>SSIM: {m.fusion_features.cross_modal.structural_similarity}</span>
                            </div>
                            <div className="flex gap-3 text-[9px]" style={{ color: C.muted }}>
                              <span>Veg: {Math.round(m.fusion_features.optical.vegetation_fraction*100)}%</span>
                              <span>SAR: {m.fusion_features.sar.mean_backscatter_db} dB</span>
                              <span>Speckle: {m.fusion_features.sar.speckle_index}</span>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}

                  {busy && (
                    <div className="space-y-2">
                      <div className="flex justify-end"><div className="h-6 rounded-xl px-4" style={{ background: `${C.cyan}14`, border: `1px solid ${C.cyan}20`, width: '70%' }}><Skeleton style={{ height: '100%', background: 'transparent' }} /></div></div>
                      <div className="rounded-xl p-4 space-y-2" style={{ background: C.card, border: `1px solid ${C.border}` }}>
                        <Skeleton style={{ height: 10, width: '60%' }} />
                        <Skeleton style={{ height: 10, width: '90%' }} />
                        <Skeleton style={{ height: 10, width: '75%' }} />
                      </div>
                    </div>
                  )}

                  {error && <div className="rounded-xl px-4 py-3 text-xs leading-relaxed" style={{ background: 'rgba(255,107,107,0.10)', border: `1px solid rgba(255,107,107,0.3)`, color: C.danger }}>{error}</div>}
                  {atCap && (
                    <div className="rounded-xl px-4 py-3 text-xs font-medium flex items-center gap-2" style={{ background: `${C.orange}10`, border: `1px solid ${C.orange}50`, color: C.orange }}>
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="shrink-0">
                        <circle cx="12" cy="12" r="10" />
                        <line x1="12" y1="8" x2="12" y2="12" />
                        <line x1="12" y1="16" x2="12.01" y2="16" />
                      </svg>
                      <span>Session request guard reached ({SESSION_CALL_LIMIT} queries). Refresh the page or use the dashboard → Clear Session Log to continue analysis.</span>
                    </div>
                  )}
                  <div ref={chatBottomRef} />
                </div>

                {/* Suggestions */}
                <div className="px-3 py-2" style={{ borderTop: `1px solid ${C.border}` }}>
                  <div className="text-[9px] font-mono tracking-widest mb-2 px-1" style={{ color: C.dim }}>SUGGESTED QUERIES</div>
                  <div className="flex gap-1.5 flex-wrap">
                    {suggestions.slice(0,4).map(s => (
                      <button key={s} onClick={() => askQuestion(s)} disabled={busy || atCap}
                        className="text-left px-2.5 py-1.5 rounded-lg text-[10px] transition-all disabled:opacity-40 query-chip font-mono"
                        style={{ color: C.cyan, border: `1px solid ${C.cyan}28`, background: `${C.cyan}08` }}>
                        <span className="w-1.5 h-1.5 rounded-full inline-block mr-1.5" style={{ background: C.cyan }} />
                        <span className="line-clamp-1">{s.slice(0,42)}{s.length > 42 ? '…' : ''}</span>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Input */}
                <div className="p-3" style={{ borderTop: `1px solid ${C.border}` }}>
                  <form onSubmit={e => { e.preventDefault(); askQuestion() }} className="flex items-center gap-2">
                    <div className="flex-1 flex items-center gap-2 rounded-xl px-3 py-2.5 input-field text-xs">
                      <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ background: atCap ? C.orange : C.cyan, boxShadow: `0 0 6px ${atCap ? C.orange : C.cyan}` }} />
                      <input value={question} onChange={e => setQuestion(e.target.value)}
                        placeholder={atCap ? `Session limit reached (${SESSION_CALL_LIMIT} queries). Refresh the page to continue.` : 'Ask anything about this image…'}
                        disabled={atCap} className="flex-1 bg-transparent outline-none disabled:opacity-50 text-xs"
                        style={{ color: C.white, fontFamily: 'Inter, sans-serif' }} />
                    </div>
                    <button type="submit" disabled={busy || !question.trim() || atCap}
                      className="btn-primary px-4 py-2.5 text-xs font-semibold rounded-xl disabled:opacity-40">
                      ASK
                    </button>
                  </form>
                </div>
              </div>
            </div>
          )}
          </>
          )}
        </div>
      </section>

      {/* ── FEATURES SECTION ─────────────────────────────────────────────────── */}
      <section id="features" ref={r2.ref} className="py-28 section-reveal" style={{ borderTop: `1px solid ${C.border}`, background: C.base }}>
        <div className="max-w-7xl mx-auto px-5">
          <div className="text-center mb-16">
            <div className="inline-block text-xs font-mono tracking-widest mb-4 px-3 py-1.5 rounded-md" style={{ color: C.orange, background: `${C.orange}0E`, border: `1px solid ${C.orange}22` }}>CORE CAPABILITIES</div>
            <h2 className="text-4xl lg:text-5xl font-bold mb-4" style={{ fontFamily: "'Space Grotesk', sans-serif", letterSpacing: '-0.02em' }}>Intelligence Across<br />Every Modality</h2>
            <p className="text-base max-w-xl mx-auto" style={{ color: C.muted }}>Query-driven specialist workflows with observable execution traces and explicit capability limits.</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            {[
              { icon: '◉', color: C.cyan, badge: 'VQA', title: 'Single-Image VQA', desc: 'Multi-attribute natural-language visual question answering and scene descriptions for land-cover, dominant objects, and spatial layout.', detail: 'BigEarthNet taxonomy · specialist model path · provenance' },
              { icon: '⊕', color: C.mint, badge: 'Land Cover', title: 'Land Cover Classification', desc: 'Uses the BigEarthNet taxonomy when the trained classification specialist is available; raw model scores are labelled as uncalibrated unless empirical calibration is verified.', detail: 'Normalized region output · grounding workflow' },
              { icon: '△', color: C.orange, badge: 'Change', title: 'Bi-Temporal Change Detection', desc: 'Compares two observations when a compatible change-analysis specialist is available. The system validates the pair and distinguishes qualitative change understanding from calibrated quantitative measurements.', detail: 'Co-registration checks · bi-temporal change workflow · provenance trace' },
              { icon: '⬡', color: C.mint, badge: 'Fusion', title: 'Optical-SAR Analysis', desc: 'Combines supplied optical/multispectral evidence with SAR backscatter when both inputs are valid and co-registered. RGB-only uploads are never treated as multispectral.', detail: 'Cross-modal reasoning · modality validation · execution trace' },
              { icon: '◎', color: C.orange, badge: 'Agentic', title: 'Traceable Task Routing', desc: 'Every query passes through a transparent task classifier → tool registry → specialist model pipeline. Full execution trace viewable and exportable.', detail: 'Observable trace · Permitted parameters · JSON export' },
            ].map((f, i) => (
              <div key={i} className="rounded-2xl p-6 transition-all duration-300 cursor-default group stat-card"
                style={{ background: C.surface, border: `1px solid ${C.border}` }}>
                <div className="flex items-start justify-between mb-5">
                  <div className="w-11 h-11 rounded-xl flex items-center justify-center text-xl" style={{ background: `${f.color}18`, border: `1px solid ${f.color}33`, color: f.color }}>{f.icon}</div>
                  <span className="text-[9px] font-mono font-bold px-2 py-0.5 rounded-md" style={{ background: `${f.color}18`, color: f.color, border: `1px solid ${f.color}33` }}>{f.badge}</span>
                </div>
                <h3 className="text-base font-bold mb-2" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>{f.title}</h3>
                <p className="text-sm leading-relaxed mb-4" style={{ color: C.muted }}>{f.desc}</p>
                <div className="text-[10px] font-mono pt-3" style={{ color: C.dim, borderTop: `1px solid ${C.border}` }}>{f.detail}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── COMPARE SECTION ─────────────────────────────────────────────────── */}
      <section id="compare-section" ref={r3.ref} className="py-28 scroll-mt-10"
        style={{ background: C.surface, borderTop: `1px solid ${C.border}`, borderBottom: `1px solid ${C.border}` }}>
        <div className="max-w-7xl mx-auto px-5">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
            <div>
              <div className="inline-block text-xs font-mono tracking-widest mb-5 px-3 py-1.5 rounded-md" style={{ color: C.mint, background: `${C.mint}0E`, border: `1px solid ${C.mint}22` }}>BI-TEMPORAL ANALYSIS · CDVQA</div>
              <h2 className="text-4xl lg:text-5xl font-bold mb-5" style={{ fontFamily: "'Space Grotesk', sans-serif", letterSpacing: '-0.02em' }}>Compare Any Two<br />Points in Time</h2>
              <p className="text-base leading-relaxed mb-8" style={{ color: C.muted, lineHeight: 1.8 }}>Upload earlier and later observations. ORBITAL-AI compares the supplied scenes when an executable change specialist is available, and reports limitations when it is not.</p>

              <div className="flex flex-wrap gap-2.5 mb-8">
                <button onClick={() => runComparison()} className="btn-outline-cyan px-5 py-2.5 text-sm font-medium rounded-xl flex items-center gap-2">
                  <span>▸</span><span>Run Change Detection</span>
                </button>
                <button onClick={() => beforeInputRef.current?.click()} className="btn-ghost px-4 py-2.5 text-xs font-mono rounded-xl flex items-center gap-2">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" /></svg><span>{beforeImage ? 'Earlier Loaded' : 'Upload Earlier Image'}</span>
                </button>
                <button onClick={() => afterInputRef.current?.click()} className="btn-ghost px-4 py-2.5 text-xs font-mono rounded-xl flex items-center gap-2">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" /></svg><span>{afterImage ? 'Later Loaded' : 'Upload Later Image'}</span>
                </button>
                {(beforeImage || afterImage) && (
                  <button onClick={() => { setBeforeImage(null); setAfterImage(null); addToast('Images reset', 'info') }}
                    className="btn-ghost px-3 py-2.5 text-xs font-mono rounded-xl" style={{ color: C.danger }}>Reset</button>
                )}
              </div>

              {/* Quick query chips */}
              <div className="space-y-2.5">
                {[
                  { q: 'What changed between these two dates, and where did the change occur?', col: C.orange, badge: 'CDVQA' },
                  { q: 'Has the built-up area increased, decreased, or remained unchanged?', col: C.mint, badge: 'CHANGE-VQA' },
                  { q: 'Has vegetation increased or decreased?', col: C.cyan, badge: 'VEGETATION' },
                ].map(({ q, col, badge }) => (
                  <button key={q} onClick={() => runComparison(q)}
                    className="w-full text-left flex items-center gap-3 text-sm py-3 px-4 rounded-xl transition-all group stat-card"
                    style={{ background: C.card, border: `1px solid ${C.border}` }}>
                    <span className="text-[9px] font-mono font-bold px-2 py-0.5 rounded shrink-0" style={{ background: `${col}18`, color: col }}>{badge}</span>
                    <span className="flex-1 text-xs" style={{ color: C.white }}>"{q}"</span>
                    <span className="opacity-0 group-hover:opacity-100 transition-opacity shrink-0" style={{ color: col }}>▸</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Comparison viewer */}
            <div ref={compareRef} className="relative rounded-2xl overflow-hidden" style={{ background: C.card, border: `1px solid ${C.border}`, minHeight: 360 }}>
              {/* Before/After slider */}
              <div className="relative h-80 overflow-hidden" onMouseMove={e => { if (!dragging) return; const r = e.currentTarget.getBoundingClientRect(); setBeforePct(Math.max(5, Math.min(95, ((e.clientX-r.left)/r.width)*100))) }}
                onMouseDown={() => setDragging(true)} onMouseUp={() => setDragging(false)} onMouseLeave={() => setDragging(false)}>
                {/* After */}
                <div className="absolute inset-0">
                  <img src={afterImage ?? ''} alt="After" className="w-full h-full object-cover" style={{ opacity: 0.7 }} />
                  <div className="absolute top-2 right-2 px-2 py-1 rounded-lg text-[10px] font-mono" style={{ background: `${C.surface}DD`, color: C.mint, border: `1px solid ${C.mint}40` }}>2026 · LATER</div>
                </div>
                {/* Before (clipped) */}
                <div className="absolute inset-0 overflow-hidden" style={{ width: `${beforePct}%` }}>
                  <img src={beforeImage ?? ''} alt="Before" className="w-full h-full object-cover" style={{ minWidth: `${100/beforePct*100}%`, opacity: 0.7 }} />
                  <div className="absolute top-2 left-2 px-2 py-1 rounded-lg text-[10px] font-mono" style={{ background: `${C.surface}DD`, color: C.orange, border: `1px solid ${C.orange}40` }}>2024 · EARLIER</div>
                </div>
                {/* Slider line */}
                <div className="absolute inset-y-0 flex items-center z-10 pointer-events-none" style={{ left: `${beforePct}%`, transform: 'translateX(-50%)' }}>
                  <div className="w-0.5 h-full" style={{ background: C.white, boxShadow: '0 0 12px rgba(255,255,255,0.4)' }} />
                  <div className="absolute w-8 h-8 rounded-full flex items-center justify-center cursor-ew-resize pointer-events-auto"
                    style={{ background: C.white, boxShadow: '0 0 20px rgba(255,255,255,0.4)' }}>
                    <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M5 4L2 7l3 3M9 4l3 3-3 3" stroke={C.base} strokeWidth="1.5" strokeLinecap="round"/></svg>
                  </div>
                </div>
              </div>

              {/* Result */}
              {temporalResult && (
                <div className="p-4 border-t space-y-3" style={{ borderColor: C.border }}>
                  <div className="flex items-center justify-between text-xs font-mono flex-wrap gap-2">
                    <ConfidenceBadge
                      confidence={temporalResult.confidenceScore >= 85 ? 'high' : 'medium'}
                      percent={temporalResult.confidenceScore}
                      confidence_status={temporalResult.confidence_status || (temporalResult.'not_calibrated')}
                      mode={temporalResult.mode}
                    />
                    <span className="text-[10px]" style={{ color: C.dim }}>Bi-Temporal CDVQA</span>
                  </div>
                  <p className="text-xs leading-relaxed" style={{ color: '#D0E0F0', lineHeight: 1.75 }}>{temporalResult.answer}</p>
                  <div className="flex flex-wrap gap-1">
                    {temporalResult.features.map(f => (
                      <span key={f} className="text-[9px] px-2 py-0.5 rounded-full font-mono" style={{ background: `${C.mint}10`, color: C.mint, border: `1px solid ${C.mint}28` }}>{f}</span>
                    ))}
                  </div>
                  {temporalResult.execution_trace && (
                    <button onClick={() => { setActiveTrace(temporalResult.execution_trace!); setTraceModalOpen(true) }}
                      className="btn-outline-cyan px-3 py-1.5 text-[10px] font-mono flex items-center gap-1.5 rounded-lg">
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" /></svg><span>View Execution Trace</span>
                    </button>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* ── ABOUT / QUERY EXAMPLES ──────────────────────────────────────────── */}
      <section id="about" ref={r5.ref} className="py-28"
        style={{ background: C.base }}>
        <div className="max-w-7xl mx-auto px-5">
          <div className="text-center mb-16">
            <div className="inline-block text-xs font-mono tracking-widest mb-4 px-3 py-1.5 rounded-md" style={{ color: C.cyan, background: `${C.cyan}0E`, border: `1px solid ${C.cyan}22` }}>NATURAL LANGUAGE INTERFACE</div>
            <h2 className="text-4xl lg:text-5xl font-bold mb-4" style={{ fontFamily: "'Space Grotesk', sans-serif", letterSpacing: '-0.02em' }}>Ask in Plain Language</h2>
            <p className="text-base max-w-xl mx-auto" style={{ color: C.muted }}>No GIS expertise needed. SatQuery AI translates everyday questions into precise remote-sensing analysis with full provenance.</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            {[
              { q: 'How much of this area is covered by water?', a: 'The system will identify and describe any water bodies present, their extent, and relevant hydrological features.', col: C.cyan, badge: 'HYDROLOGY' },
              { q: 'Are there any new roads or buildings since last year?', a: 'Bi-temporal comparison will detect and quantify structural additions, removals, and area changes between the two images.', col: C.orange, badge: 'CHANGE' },
              { q: 'Has vegetation increased or decreased?', a: 'Vegetation coverage is assessed using spectral analysis and the change specialist returns directional and magnitude estimates.', col: C.mint, badge: 'VEGETATION' },
              { q: 'Identify the dominant land use in this image.', a: 'The land cover specialist classifies the scene using the BigEarthNet 19-class taxonomy, returning top labels and confidence.', col: C.orange, badge: 'LAND COVER' },
              { q: 'Use optical and SAR images to identify built-up regions.', a: 'Cross-modal analysis correlates optical reflectance and SAR backscatter to highlight built-up areas penetrating cloud or canopy cover.', col: C.mint, badge: 'CROSS-MODAL' },
              { q: 'Describe this satellite image.', a: 'The caption specialist produces a natural-language scene description covering terrain type, land cover, and salient objects.', col: C.cyan, badge: 'SCENE' },
            ].map((item, i) => (
              <div key={i} className="rounded-2xl p-5 stat-card" style={{ background: C.surface, border: `1px solid ${C.border}` }}>
                <div className="flex items-center justify-between mb-3">
                  <span className="text-[9px] font-mono font-bold px-2 py-0.5 rounded-md" style={{ background: `${item.col}18`, color: item.col, border: `1px solid ${item.col}28` }}>{item.badge}</span>
                  <span className="text-[10px] font-mono" style={{ color: C.dim }}>USER QUERY</span>
                </div>
                <p className="text-sm font-semibold mb-3 leading-relaxed" style={{ color: C.white, fontFamily: "'Space Grotesk', sans-serif" }}>"{item.q}"</p>
                <div className="h-px mb-3" style={{ background: C.border }} />
                <div className="text-[10px] font-mono mb-1.5" style={{ color: C.dim }}>SATQUERY AI RESPONSE</div>
                <p className="text-xs leading-relaxed" style={{ color: C.muted }}>{item.a}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA ────────────────────────────────────────────────────────────── */}
      <section ref={r6.ref} className="py-36 text-center relative overflow-hidden"
        style={{ background: C.surface, borderTop: `1px solid ${C.border}` }}>
        <div className="absolute inset-0 pointer-events-none grid-bg opacity-30" />
        <div className="relative max-w-3xl mx-auto px-5">
          <div className="inline-block text-xs font-mono tracking-widest mb-6 px-3 py-1.5 rounded-md" style={{ color: C.cyan, background: `${C.cyan}0E`, border: `1px solid ${C.cyan}22` }}>START ANALYZING</div>
          <h2 className="text-5xl lg:text-6xl font-bold mb-6" style={{ fontFamily: "'Space Grotesk', sans-serif", letterSpacing: '-0.03em' }}>
            Upload. Query.<br /><span style={{ color: 'var(--cyan)' }}>Inspect the result.</span>
          </h2>
          <p className="text-lg mb-12 max-w-lg mx-auto" style={{ color: C.muted, lineHeight: 1.8 }}>Explore satellite imagery through natural-language questions, image comparison, and optical-SAR analysis. Review the analysis output and inspect the execution trace behind each request.</p>
          <div className="flex flex-wrap justify-center gap-4">
            <button onClick={focusWorkspace} className="btn-primary px-8 py-4 text-sm font-semibold rounded-xl">Upload Image &amp; Start →</button>
          </div>

          {/* Tech stack badges */}
          <div className="flex flex-wrap justify-center gap-2.5 mt-12">
            {['BigEarthNet v2.0', 'ResNet-50', 'VRSBench', 'RSVQA', 'CDVQA', 'Cartosat-2S', 'RISAT-1A', 'Sentinel-1/2'].map(t => (
              <span key={t} className="text-[10px] font-mono px-3 py-1 rounded-md" style={{ background: `${C.border}`, color: C.dim, border: `1px solid ${C.border}` }}>{t}</span>
            ))}
          </div>
        </div>
      </section>

      {/* ── FOOTER ──────────────────────────────────────────────────────────── */}
      <footer style={{ background: C.base, borderTop: `1px solid ${C.border}` }}>
        <div className="max-w-7xl mx-auto px-5 py-16">
          <div className="grid grid-cols-1 md:grid-cols-[1.8fr_1fr_1fr_1fr] gap-12 mb-12">
            {/* Brand */}
            <div>
              <div className="flex items-center gap-2.5 mb-4">
                <svg width="22" height="22" viewBox="0 0 22 22" fill="none">
                  <circle cx="11" cy="11" r="9" stroke={C.cyan} strokeWidth="1.5" />
                  <ellipse cx="11" cy="11" rx="4.5" ry="9" stroke={C.cyan} strokeWidth="1.5" />
                  <line x1="2" y1="11" x2="20" y2="11" stroke={C.cyan} strokeWidth="1.5" />
                  <circle cx="11" cy="11" r="2" fill={C.cyan} />
                </svg>
                <span className="font-bold tracking-wide text-sm" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>SATQUERY<span style={{ color: C.cyan }}>AI</span></span>
              </div>
              <p className="text-sm leading-relaxed mb-5" style={{ color: C.muted }}>Agentic remote-sensing vision-language assistant for satellite imagery analysis. BigEarthNet domain adaptation, multi-modal analysis, deterministic task routing.</p>
              <div className="flex flex-wrap gap-1.5">
                {['BigEarthNet v2.0', 'ResNet-50', 'BIFOLD/TU Berlin'].map(t => (
                  <span key={t} className="text-[9px] font-mono px-2 py-0.5 rounded-md" style={{ background: `${C.cyan}0A`, color: C.dim, border: `1px solid ${C.border}` }}>{t}</span>
                ))}
              </div>
            </div>

            {/* Column 1: Platform & Workflows */}
            <div>
              <div className="text-[10px] font-mono tracking-widest mb-4 uppercase" style={{ color: C.dim }}>Platform Workflows</div>
              <div className="space-y-2.5 text-sm">
                <button
                  type="button"
                  onClick={() => { setActiveView('workspace'); setAppMode('single'); focusWorkspace() }}
                  className="block text-left transition-colors cursor-pointer"
                  style={{ color: C.muted }}
                  onMouseEnter={e => (e.currentTarget as HTMLElement).style.color = C.white}
                  onMouseLeave={e => (e.currentTarget as HTMLElement).style.color = C.muted}
                >
                  Single-Image VQA
                </button>
                <button
                  type="button"
                  onClick={() => { setActiveView('workspace'); setAppMode('compare'); scrollToSection('compare-section') }}
                  className="block text-left transition-colors cursor-pointer"
                  style={{ color: C.muted }}
                  onMouseEnter={e => (e.currentTarget as HTMLElement).style.color = C.white}
                  onMouseLeave={e => (e.currentTarget as HTMLElement).style.color = C.muted}
                >
                  Bi-Temporal Change Detection
                </button>
                <button
                  type="button"
                  onClick={() => { setActiveView('workspace'); setAppMode('fusion'); focusWorkspace() }}
                  className="block text-left transition-colors cursor-pointer"
                  style={{ color: C.muted }}
                  onMouseEnter={e => (e.currentTarget as HTMLElement).style.color = C.white}
                  onMouseLeave={e => (e.currentTarget as HTMLElement).style.color = C.muted}
                >
                  Optical–SAR Fusion
                </button>

                <button
                  type="button"
                  onClick={() => { setActiveView('dashboard'); focusWorkspace() }}
                  className="block text-left transition-colors cursor-pointer text-cyan-400 font-medium"
                  onMouseEnter={e => (e.currentTarget as HTMLElement).style.color = C.cyan}
                  onMouseLeave={e => (e.currentTarget as HTMLElement).style.color = '#38BDF8'}
                >
                  Audit & Telemetry Dashboard →
                </button>
              </div>
            </div>

            {/* Column 2: Benchmarks & Protocols */}
            <div>
              <div className="text-[10px] font-mono tracking-widest mb-4 uppercase" style={{ color: C.dim }}>Benchmarks & Tasks</div>
              <div className="space-y-2.5 text-sm">
                <button
                  type="button"
                  onClick={() => setDocModalOpen(true)}
                  className="block text-left transition-colors cursor-pointer"
                  style={{ color: C.muted }}
                  onMouseEnter={e => (e.currentTarget as HTMLElement).style.color = C.white}
                  onMouseLeave={e => (e.currentTarget as HTMLElement).style.color = C.muted}
                >
                  VRSBench Scene Captioning
                </button>
                <button
                  type="button"
                  onClick={() => setDocModalOpen(true)}
                  className="block text-left transition-colors cursor-pointer"
                  style={{ color: C.muted }}
                  onMouseEnter={e => (e.currentTarget as HTMLElement).style.color = C.white}
                  onMouseLeave={e => (e.currentTarget as HTMLElement).style.color = C.muted}
                >
                  RSVQA Text Grounding
                </button>
                <button
                  type="button"
                  onClick={() => setDocModalOpen(true)}
                  className="block text-left transition-colors cursor-pointer"
                  style={{ color: C.muted }}
                  onMouseEnter={e => (e.currentTarget as HTMLElement).style.color = C.white}
                  onMouseLeave={e => (e.currentTarget as HTMLElement).style.color = C.muted}
                >
                  CDVQA Bi-Temporal Change Detection
                </button>
                <button
                  type="button"
                  onClick={() => setDocModalOpen(true)}
                  className="block text-left transition-colors cursor-pointer"
                  style={{ color: C.muted }}
                  onMouseEnter={e => (e.currentTarget as HTMLElement).style.color = C.white}
                  onMouseLeave={e => (e.currentTarget as HTMLElement).style.color = C.muted}
                >
                  BigEarthNet 19-Class Taxonomy
                </button>

              </div>
            </div>

            {/* Column 3: Docs, Research & Legal */}
            <div>
              <div className="text-[10px] font-mono tracking-widest mb-4 uppercase" style={{ color: C.dim }}>Documentation & Legal</div>
              <div className="space-y-2.5 text-sm">
                <button
                  type="button"
                  onClick={() => setDocModalOpen(true)}
                  className="block text-left transition-colors cursor-pointer"
                  style={{ color: C.muted }}
                  onMouseEnter={e => (e.currentTarget as HTMLElement).style.color = C.white}
                  onMouseLeave={e => (e.currentTarget as HTMLElement).style.color = C.muted}
                >
                  System Docs & REST API
                </button>
                <button
                  type="button"
                  onClick={() => setContactModalOpen(true)}
                  className="block text-left transition-colors cursor-pointer"
                  style={{ color: C.muted }}
                  onMouseEnter={e => (e.currentTarget as HTMLElement).style.color = C.white}
                  onMouseLeave={e => (e.currentTarget as HTMLElement).style.color = C.muted}
                >
                  Contact Research Team
                </button>
                <button
                  type="button"
                  onClick={() => { setActiveLegalTab('cookies'); setLegalModalOpen(true) }}
                  className="block text-left transition-colors cursor-pointer font-medium"
                  style={{ color: '#35E0B8' }}
                  onMouseEnter={e => (e.currentTarget as HTMLElement).style.color = C.white}
                  onMouseLeave={e => (e.currentTarget as HTMLElement).style.color = '#35E0B8'}
                >
                  Strict Required Data Policy
                </button>
                <a
                  href="/privacy"
                  className="block text-left transition-colors cursor-pointer"
                  style={{ color: C.muted }}
                  onMouseEnter={e => (e.currentTarget as HTMLElement).style.color = C.white}
                  onMouseLeave={e => (e.currentTarget as HTMLElement).style.color = C.muted}
                >
                  Privacy Policy
                </a>
                <a
                  href="/terms"
                  className="block text-left transition-colors cursor-pointer"
                  style={{ color: C.muted }}
                  onMouseEnter={e => (e.currentTarget as HTMLElement).style.color = C.white}
                  onMouseLeave={e => (e.currentTarget as HTMLElement).style.color = C.muted}
                >
                  Terms & Conditions
                </a>
              </div>
            </div>
          </div>

          <div className="flex flex-col md:flex-row items-center justify-between gap-4 pt-8" style={{ borderTop: `1px solid ${C.border}` }}>
            <div className="text-xs font-mono" style={{ color: C.dim }}>© 2026 SatQuery AI · Remote-Sensing Vision-Language Assistant · All rights reserved</div>
            <div className="flex items-center gap-4 text-xs font-mono flex-wrap" style={{ color: C.dim }}>
              <a
                href="https://arxiv.org/abs/2407.03653"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-white transition-colors"
                style={{ color: 'inherit' }}
              >
                Clasen et al. arXiv:2407.03653 ↗
              </a>
              <div className="w-1 h-1 rounded-full" style={{ background: C.dim }} />
              <button
                type="button"
                onClick={() => { setActiveLegalTab('cookies'); setLegalModalOpen(true) }}
                className="hover:text-cyan-400 transition-colors cursor-pointer"
                style={{ background: 'none', border: 'none', color: 'inherit' }}
              >
                Strict Data Policy
              </button>

            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}
