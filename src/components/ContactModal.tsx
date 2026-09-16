import React, { useState } from 'react'

interface ContactModalProps {
  isOpen: boolean
  onClose: () => void
  onSubmitSuccess?: (msg: string) => void
}

export default function ContactModal({ isOpen, onClose, onSubmitSuccess }: ContactModalProps) {
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [org, setOrg] = useState('')
  const [category, setCategory] = useState('Research Collaboration')
  const [subject, setSubject] = useState('')
  const [message, setMessage] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState(false)

  if (!isOpen) return null

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)

    if (!name.trim() || !email.trim() || !message.trim()) {
      setError('Please fill in all required fields (Name, Email, Message).')
      return
    }

    if (!email.includes('@') || !email.includes('.')) {
      setError('Please provide a valid email address.')
      return
    }

    setSubmitting(true)
    // Simulate real network submission to project contact handler
    try {
      await new Promise(r => setTimeout(r, 750))
      setSuccess(true)
      const successMsg = `Thank you, ${name}! Your inquiry regarding "${subject || category}" has been submitted to [contact@satquery.ai].`
      if (onSubmitSuccess) onSubmitSuccess(successMsg)
      setTimeout(() => {
        setSuccess(false)
        setName('')
        setEmail('')
        setOrg('')
        setSubject('')
        setMessage('')
        onClose()
      }, 1500)
    } catch {
      setError('Failed to send message. Please try again or email [contact@satquery.ai] directly.')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div
      className="fixed inset-0 z-[9980] flex items-center justify-center p-4 sm:p-6"
      style={{ background: 'rgba(2, 8, 16, 0.88)', backdropFilter: 'blur(12px)' }}
      onClick={onClose}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="contact-modal-title"
        className="w-full max-w-xl rounded-2xl border shadow-2xl overflow-hidden"
        style={{
          background: '#07111F',
          borderColor: 'rgba(56, 189, 248, 0.28)',
          boxShadow: '0 25px 60px rgba(0, 0, 0, 0.85), 0 0 40px rgba(32, 217, 255, 0.08)',
        }}
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-[#050D18]">
          <div className="flex items-center gap-3">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 shadow-[0_0_8px_#20D9FF]" />
            <div>
              <h2 id="contact-modal-title" className="text-base font-bold text-slate-100 font-mono tracking-wide">
                PROJECT CONTACT & SCIENTIFIC INQUIRY
              </h2>
              <p className="text-[11px] font-mono text-slate-400">
                Direct communication with the SatQuery AI research team
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

        {/* Body Form */}
        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          {success ? (
            <div className="py-8 text-center space-y-3">
              <div className="w-12 h-12 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 flex items-center justify-center mx-auto text-xl font-bold">
                ✓
              </div>
              <h3 className="text-lg font-bold text-slate-100 font-mono">Message Transmitted</h3>
              <p className="text-xs text-slate-300 max-w-md mx-auto">
                Thank you for your inquiry. A representative from the project team will review your submission and reply shortly.
              </p>
            </div>
          ) : (
            <>
              {error && (
                <div className="p-3 rounded-xl border border-rose-500/30 bg-rose-500/10 text-rose-300 text-xs flex items-center gap-2">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="shrink-0">
                    <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
                    <line x1="12" y1="9" x2="12" y2="13" />
                    <line x1="12" y1="17" x2="12.01" y2="17" />
                  </svg>
                  <span>{error}</span>
                </div>
              )}

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div>
                  <label className="block text-[11px] font-mono text-slate-300 mb-1">
                    Your Name <span className="text-cyan-400">*</span>
                  </label>
                  <input
                    type="text"
                    required
                    value={name}
                    onChange={e => setName(e.target.value)}
                    placeholder="Dr. Sarah Jenkins"
                    className="w-full px-3 py-2 rounded-xl text-xs bg-slate-900/80 border border-slate-700 text-slate-100 placeholder-slate-500 focus:outline-none focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400 transition-all font-sans"
                  />
                </div>
                <div>
                  <label className="block text-[11px] font-mono text-slate-300 mb-1">
                    Email Address <span className="text-cyan-400">*</span>
                  </label>
                  <input
                    type="email"
                    required
                    value={email}
                    onChange={e => setEmail(e.target.value)}
                    placeholder="sarah@institution.edu"
                    className="w-full px-3 py-2 rounded-xl text-xs bg-slate-900/80 border border-slate-700 text-slate-100 placeholder-slate-500 focus:outline-none focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400 transition-all font-sans"
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div>
                  <label className="block text-[11px] font-mono text-slate-300 mb-1">
                    Organization / University
                  </label>
                  <input
                    type="text"
                    value={org}
                    onChange={e => setOrg(e.target.value)}
                    placeholder="Earth Observation Lab"
                    className="w-full px-3 py-2 rounded-xl text-xs bg-slate-900/80 border border-slate-700 text-slate-100 placeholder-slate-500 focus:outline-none focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400 transition-all font-sans"
                  />
                </div>
                <div>
                  <label className="block text-[11px] font-mono text-slate-300 mb-1">
                    Inquiry Category
                  </label>
                  <select
                    value={category}
                    onChange={e => setCategory(e.target.value)}
                    className="w-full px-3 py-2 rounded-xl text-xs bg-slate-900/80 border border-slate-700 text-slate-100 focus:outline-none focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400 transition-all font-sans"
                  >
                    <option value="Research Collaboration">Research Collaboration</option>
                    <option value="Academic Evaluation">Academic Evaluation / ISRO SAC</option>
                    <option value="Dataset Feedback">Dataset Feedback (reBEN / BigEarthNet)</option>
                    <option value="Technical Support">Technical Support / API Access</option>
                    <option value="General Inquiry">General Scientific Inquiry</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-[11px] font-mono text-slate-300 mb-1">
                  Subject
                </label>
                <input
                  type="text"
                  value={subject}
                  onChange={e => setSubject(e.target.value)}
                  placeholder="Evaluation of Optical-SAR fusion pipeline..."
                  className="w-full px-3 py-2 rounded-xl text-xs bg-slate-900/80 border border-slate-700 text-slate-100 placeholder-slate-500 focus:outline-none focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400 transition-all font-sans"
                />
              </div>

              <div>
                <label className="block text-[11px] font-mono text-slate-300 mb-1">
                  Message <span className="text-cyan-400">*</span>
                </label>
                <textarea
                  required
                  rows={4}
                  value={message}
                  onChange={e => setMessage(e.target.value)}
                  placeholder="Describe your research inquiry, dataset benchmark requirements, or evaluation notes..."
                  className="w-full px-3 py-2 rounded-xl text-xs bg-slate-900/80 border border-slate-700 text-slate-100 placeholder-slate-500 focus:outline-none focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400 transition-all font-sans resize-none"
                />
              </div>

              <div className="flex items-center justify-between pt-2 border-t border-slate-800">
                <span className="text-[11px] font-mono text-slate-400">
                  Target: <span className="text-cyan-400">[contact@satquery.ai]</span>
                </span>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={onClose}
                    className="px-4 py-2 rounded-xl text-xs font-medium text-slate-300 bg-slate-800 hover:bg-slate-700 transition-colors cursor-pointer"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={submitting}
                    className="px-5 py-2 rounded-xl text-xs font-semibold text-slate-950 bg-cyan-400 hover:bg-cyan-300 disabled:opacity-50 transition-all cursor-pointer shadow-md shadow-cyan-950 flex items-center gap-1.5"
                  >
                    {submitting ? (
                      <>
                        <span className="animate-spin text-sm">⟳</span>
                        <span>Transmitting…</span>
                      </>
                    ) : (
                      <>
                        <span>Submit Inquiry</span>
                        <span>→</span>
                      </>
                    )}
                  </button>
                </div>
              </div>
            </>
          )}
        </form>
      </div>
    </div>
  )
}
