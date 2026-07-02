import { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { UploadCloud } from 'lucide-react';
import { useStore } from '../store';
import { analyzeDocument, queryDocument, saveDocumentSession, adaptDocAnalysis, getToken } from '../api';
import { IconDocument, IconUpload, IconSend, IconCopy, IconCheck, IconScale } from '../icons';

function CopyBtn({ text }) {
  const [copied, setCopied] = useState(false);
  return (
    <button 
      style={{ position: 'absolute', top: 8, right: 8, opacity: 0, transition: 'opacity 150ms', background: 'var(--c-surface)', border: '1px solid var(--c-border)', borderRadius: 4, padding: 4, cursor: 'pointer', color: 'var(--c-text2)' }}
      className="copy-btn"
      onClick={() => {
        navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      }}
    >
      {copied ? <IconCheck size={14} /> : <IconCopy size={14} />}
    </button>
  );
}

function parseText(text) {
  if (!text) return null;
  let html = text.replace(/\[(IPC|BNS)\s*§\d+[a-zA-Z]*\]|Section\s+\d+[a-zA-Z]*\s+(IPC|BNS)/gi, match => `<span class="citation-badge">${match}</span>`);
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/--(.*?)--/g, ''); 
  
  const blocks = html.split(/\n\s*\n/);
  return blocks.map((block, i) => {
    const lines = block.split('\n');
    if (lines[0].match(/^(\*|-|•)\s/)) {
      return <ul key={i} style={{ margin: '8px 0', paddingLeft: 24, lineHeight: 1.7 }}>
        {lines.map((l, j) => {
          const content = l.replace(/^(\*|-|•)\s/, '');
          return <li key={j} dangerouslySetInnerHTML={{ __html: content }} />;
        })}
      </ul>;
    }
    if (lines[0].match(/^\d+\.\s/)) {
      return <ol key={i} style={{ margin: '8px 0', paddingLeft: 24, lineHeight: 1.7 }}>
        {lines.map((l, j) => {
          const content = l.replace(/^\d+\.\s/, '');
          return <li key={j} dangerouslySetInnerHTML={{ __html: content }} />;
        })}
      </ol>;
    }
    return <p key={i} dangerouslySetInnerHTML={{ __html: block.replace(/\n/g, '<br/>') }} style={{ marginBottom: 12, lineHeight: 1.6 }} />;
  });
}

export default function DocumentView() {
  const store = useStore();
  const { toast, setCurrentDoc, currentDoc, user, refreshSessions, language } = store;
  const [file, setFile] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [tab, setTab] = useState('text');
  const [docChat, setDocChat] = useState([]);
  const [docQ, setDocQ] = useState('');
  const [docLoading, setDocLoading] = useState(false);
  const [docSession, setDocSession] = useState(null);
  const [extractedText, setExtractedText] = useState('');
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef(null);

  const handleFile = (f) => {
    if (f.size > 10 * 1024 * 1024) { toast('File exceeds 10MB limit', 'error'); return; }
    if (f.size > 5 * 1024 * 1024) toast('Large file — analysis may take longer', 'warning');
    setFile(f); setCurrentDoc(null); setExtractedText(''); setDocChat([]);
  };

  const handleDrop = (e) => { e.preventDefault(); setDragOver(false); if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]); };

  const handleAnalyze = async () => {
    if (!file) return;
    setAnalyzing(true); setProgress(0);
    const iv = setInterval(() => setProgress(p => Math.min(p + Math.random() * 12, 88)), 400);
    try {
      const raw = await analyzeDocument(file);
      const analysis = adaptDocAnalysis(raw);
      setCurrentDoc(analysis);
      setExtractedText(raw.text || raw.extracted_text || '');
      setProgress(100);
      toast('Document analysed successfully');

      try {
        const saved = await saveDocumentSession({
          filename: analysis.filename, doc_type: analysis.docType,
          risk_level: analysis.riskLevel, risk_score: analysis.riskScore,
          summary: analysis.riskSummary || `${analysis.docType} document analysed.`,
          confidence: analysis.confidence, session_id: docSession || null,
          user_id: user?.id || null,
        });
        if (saved?.session_id) { setDocSession(saved.session_id); refreshSessions(); }
      } catch { /* non-fatal */ }
    } catch (err) { toast(err.message || 'Analysis failed', 'error'); }
    finally { clearInterval(iv); setAnalyzing(false); setTimeout(() => setProgress(0), 600); }
  };

  const handleDocQuery = async () => {
    const q = docQ.trim(); if (!q || !extractedText) return;
    setDocQ('');
    setDocChat(c => [...c, { role: 'user', content: q }]);
    setDocLoading(true);
    try {
      // Direct raw fetch explicitly requested by the user prompt for document Q&A bug
      const token = getToken();
      const rawRes = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/v1/document/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          doc_text: extractedText,
          question: q,
          session_id: docSession,
          language: language || 'en'
        })
      });
      if (!rawRes.ok) throw new Error('Query failed');
      const res = await rawRes.json();
      
      if (res?.session_id) setDocSession(res.session_id);
      setDocChat(c => [...c, {
        role: 'assistant',
        content: res?.answer || 'No answer received.',
        sections: res?.applicable_sections || [],
        riskNote: res?.risk_note || '',
      }]);
    } catch (err) {
      setDocChat(c => [...c, { role: 'assistant', content: `Error: ${err.message}` }]);
    } finally { setDocLoading(false); }
  };

  const doc = currentDoc;

  const entityColors = {
    'IPC sections': { bg: 'rgba(139,92,246,0.12)', text: '#A78BFA' },
    'BNS sections': { bg: 'rgba(99,102,241,0.12)', text: '#818CF8' },
    'Acts': { bg: 'rgba(59,130,246,0.12)', text: '#60A5FA' },
    'Persons': { bg: 'rgba(34,197,94,0.12)', text: '#4ADE80' },
    'Monetary': { bg: 'rgba(234,179,8,0.12)', text: '#FCD34D' },
    'Dates': { bg: 'rgba(100,116,139,0.12)', text: '#94A3B8' },
    'Case numbers': { bg: 'rgba(6,182,212,0.12)', text: '#22D3EE' },
  };

  return (
    <div className="view-enter doc-panels" style={{ height: '100%' }}>
      {/* LEFT PANEL */}
      <div className="doc-left" style={{ overflowY: 'auto' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
          <div style={{ width: 28, height: 1, background: 'var(--c-gold)' }} />
          <span style={{ fontFamily: 'var(--f-mono)', fontSize: 10, fontWeight: 700, letterSpacing: '0.2em', color: 'var(--c-gold)', textTransform: 'uppercase' }}>
            LexShield · Legal Intelligence
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 24 }}>
          <IconDocument color="var(--c-gold)" size={28} />
          <h2 style={{ fontFamily: 'var(--f-head)', fontSize: 22, color: 'var(--c-text)', margin: 0 }}>Document Analysis</h2>
        </div>

        {!doc && (
          <>
            <div 
              className="upload-zone-box"
              style={{
                border: `1.5px dashed ${dragOver ? 'var(--c-gold)' : 'var(--c-border)'}`,
                borderRadius: 'var(--r-md)',
                textAlign: 'center',
                cursor: 'pointer',
                transition: 'all 200ms ease',
                background: dragOver ? 'var(--c-gold-dim)' : 'transparent',
                color: dragOver ? 'var(--c-gold)' : 'inherit'
              }}
              onMouseEnter={(e) => { e.currentTarget.style.borderColor = 'var(--c-gold)'; e.currentTarget.style.background = 'var(--c-gold-dim)'; }}
              onMouseLeave={(e) => { if (!dragOver) { e.currentTarget.style.borderColor = 'var(--c-border)'; e.currentTarget.style.background = 'transparent'; } }}
              onClick={() => fileRef.current?.click()}
              onDragOver={e => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
            >
              <motion.div
                animate={{ y: [0, -5, 0] }}
                transition={{ duration: 2.4, repeat: Infinity, ease: 'easeInOut' }}
                style={{ color: 'var(--c-gold)' }}
              >
                <UploadCloud size={32} strokeWidth={1.4} />
              </motion.div>
              <div style={{ fontFamily: 'var(--f-head)', fontSize: 18, color: 'var(--c-text2)', marginTop: 12 }}>Upload Legal Document</div>
              <div style={{ fontSize: 12, color: 'var(--c-text3)', marginTop: 6 }}>PDF, JPG, PNG — max 10MB</div>
              <button className="btn-ghost" style={{ marginTop: 16 }}>Browse Files</button>
            </div>
            <input ref={fileRef} type="file" accept=".pdf,.jpg,.jpeg,.png" hidden onChange={e => e.target.files[0] && handleFile(e.target.files[0])} />
          </>
        )}

        <AnimatePresence>
          {file && !doc && (
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              style={{
                marginTop: 16,
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                background: 'var(--c-surface)',
                padding: '12px 16px',
                borderRadius: 'var(--r-md)',
                border: '1px solid var(--c-border)',
              }}
            >
              <div style={{ minWidth: 0 }}>
                <div style={{ fontWeight: 600, fontSize: 14, color: 'var(--c-text)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{file.name}</div>
                <div style={{ fontSize: 12, color: 'var(--c-text2)', marginTop: 2 }}>{(file.size / 1024).toFixed(0)} KB</div>
              </div>
              <button className="btn-gold" onClick={handleAnalyze} disabled={analyzing} style={{ padding: '6px 12px', fontSize: 13 }}>
                {analyzing ? 'Analysing…' : 'Analyse'}
              </button>
            </motion.div>
          )}
        </AnimatePresence>

        {progress > 0 && progress < 100 && (
          <div style={{ marginTop: 12, height: 4, background: 'var(--c-border)', borderRadius: 99, overflow: 'hidden' }}>
            <div style={{ width: `${progress}%`, background: 'var(--c-gold)', height: '100%', transition: 'width 400ms' }} />
          </div>
        )}

        {doc && (
          <div className="fade-in">
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 20, minWidth: 0 }}>
              <IconDocument color="var(--c-gold)" size={20} style={{ flexShrink: 0 }} />
              <span style={{ fontWeight: 600, fontSize: 14, color: 'var(--c-text)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{doc.filename}</span>
            </div>

            {/* Step 1 */}
            <div style={{ marginBottom: 24, animation: 'fadeIn 300ms ease forwards', opacity: 0 }}>
              <span className="badge badge-gold" style={{ fontSize: 13, padding: '4px 14px' }}>
                {(doc.docType || '').replace(/_/g, ' ')}
              </span>
              <div style={{ fontSize: 11, color: 'var(--c-text3)', marginTop: 4 }}>
                Classified using: InLegalBERT Fine-tuned
              </div>
              <div style={{ height: 3, background: 'var(--c-border)', overflow: 'hidden', borderRadius: 99, marginTop: 6 }}>
                <div style={{ height: '100%', background: 'var(--c-gold)', animation: 'slideRight 600ms ease-out forwards' }} />
              </div>
            </div>

            {doc.docType === 'non_legal' ? (
              <div style={{ marginBottom: 24, animation: 'fadeIn 300ms ease forwards 150ms', opacity: 0 }}>
                <div style={{ padding: '16px 20px', background: 'var(--c-surface)', borderLeft: '3px solid var(--c-gold)', borderRadius: '0 var(--r-sm) var(--r-sm) 0' }}>
                  <div style={{ fontSize: 16, fontWeight: 600, color: 'var(--c-text)', marginBottom: 4 }}>Not a Legal Document</div>
                  <div style={{ fontSize: 13, color: 'var(--c-text2)', lineHeight: 1.5 }}>
                    The AI did not detect any legal terminology in this document. Risk assessment and legal extraction have been bypassed.
                  </div>
                </div>
              </div>
            ) : (
              <div style={{ marginBottom: 24, animation: 'fadeIn 300ms ease forwards 150ms', opacity: 0 }}>
                <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--c-gold)', marginBottom: 4 }}>AI RISK ASSESSMENT</div>
                <div style={{ fontSize: 52, fontWeight: 700, color: `var(--c-${doc.riskLevel === 'medium' ? 'medium' : doc.riskLevel === 'high' || doc.riskLevel === 'critical' ? 'high' : 'low'})` }}>
                  {Math.round(doc.riskScore)}
                </div>
                <div style={{ fontSize: 14, color: 'var(--c-text2)', textTransform: 'capitalize' }}>
                  / 100 — {doc.riskLevel} Risk
                </div>
                <div style={{ height: 8, borderRadius: 99, background: 'var(--c-border)', marginTop: 8, overflow: 'hidden' }}>
                  <div style={{ height: '100%', background: `var(--c-${doc.riskLevel === 'medium' ? 'medium' : doc.riskLevel === 'high' || doc.riskLevel === 'critical' ? 'high' : 'low'})`, width: `${Math.round(doc.riskScore)}%`, animation: 'slideRight 800ms ease-out forwards' }} />
                </div>
              </div>
            )}

            {/* Step 3 */}
            {doc.riskSummary && doc.docType !== 'non_legal' && (
              <div style={{ marginBottom: 24, animation: 'fadeIn 300ms ease forwards 300ms', opacity: 0 }}>
                <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--c-text3)', marginBottom: 8 }}>DOCUMENT SUMMARY</div>
                <div style={{ background: 'var(--c-surface)', padding: '12px 16px', borderRadius: 'var(--r-sm)', fontSize: 13, color: 'var(--c-text2)', lineHeight: 1.6 }}>
                  {doc.riskSummary}
                </div>
              </div>
            )}

            {/* Step 4 */}
            {doc.rightsAlerts?.length > 0 && (
              <div style={{ marginBottom: 24, animation: 'fadeIn 300ms ease forwards 450ms', opacity: 0 }}>
                <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--c-medium)', marginBottom: 8 }}>RIGHTS ALERTS</div>
                {doc.rightsAlerts.map((a, i) => {
                  const sColor = a.severity === 'critical' ? 'var(--c-high)' : a.severity === 'high' ? 'var(--c-medium)' : 'var(--c-low)';
                  return (
                    <div key={i} style={{ padding: '10px 14px', borderLeft: `2px solid ${sColor}`, background: 'var(--c-elevated)', borderRadius: '0 var(--r-sm) var(--r-sm) 0', marginBottom: 8 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                        <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--c-text)' }}>{a.right}</div>
                        {a.section && <span className="citation-badge" style={{ marginLeft: 8 }}>{a.section}</span>}
                      </div>
                      <div style={{ fontSize: 12, color: 'var(--c-text2)', marginTop: 2 }}>{a.violation}</div>
                    </div>
                  );
                })}
              </div>
            )}

            {/* Step 5 */}
            {doc.entities && Object.keys(doc.entities).length > 0 && (
              <div style={{ marginBottom: 24, animation: 'fadeIn 300ms ease forwards 600ms', opacity: 0 }}>
                <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--c-text3)', marginBottom: 8 }}>EXTRACTED ENTITIES</div>
                <div>
                  {Object.entries(doc.entities).flatMap(([cat, items]) => 
                    Array.isArray(items) ? items.map((item, idx) => {
                      const colorDef = entityColors[cat] || { bg: 'var(--c-elevated)', text: 'var(--c-text2)' };
                      return (
                        <span key={`${cat}-${idx}`} className="chipPop" style={{ background: colorDef.bg, color: colorDef.text, padding: '3px 10px', borderRadius: 99, fontSize: 12, display: 'inline-flex', margin: '3px 3px', animationDelay: `${600 + (idx * 30)}ms` }}>
                          {item}
                        </span>
                      );
                    }) : []
                  )}
                </div>
              </div>
            )}

            <button className="btn-outline" style={{ marginTop: 12, padding: '6px 12px', fontSize: 12 }} onClick={() => { setFile(null); setCurrentDoc(null); setDocChat([]); setExtractedText(''); }}>
              Analyse Another
            </button>
          </div>
        )}
      </div>

      {/* RIGHT PANEL */}
      <div className="doc-right" style={{ flex: 1, display: 'flex', flexDirection: 'column', background: 'var(--c-bg)' }}>
        <div style={{ borderBottom: '1px solid var(--c-border2)', display: 'flex', position: 'relative', padding: '0 10px' }}>
          {[
            { key: 'text', label: 'Extracted Text' },
            { key: 'qa',   label: 'Ask Questions'  }
          ].map((t) => (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              style={{
                padding: '14px 24px',
                fontSize: 13,
                fontWeight: 600,
                color: tab === t.key ? 'var(--c-text)' : 'var(--c-text3)',
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                position: 'relative',
                transition: 'color 150ms',
              }}
            >
              {t.label}
              {tab === t.key && (
                <motion.span
                  layoutId="doc-tab-indicator"
                  style={{
                    position: 'absolute',
                    bottom: -1,
                    left: 0,
                    right: 0,
                    height: 2,
                    background: 'var(--c-gold)',
                    borderRadius: 1,
                  }}
                  transition={{ type: 'spring', stiffness: 400, damping: 32 }}
                />
              )}
            </button>
          ))}
        </div>

        <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', height: '100%' }}>
          <AnimatePresence mode="wait">
            <motion.div
              key={tab}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.22 }}
              style={{ flex: 1, display: 'flex', flexDirection: 'column', height: '100%' }}
            >
              {tab === 'text' && (
                <div style={{ padding: 24, overflowY: 'auto', flex: 1 }}>
                  {doc?.text || extractedText ? (
                    <div style={{ position: 'relative' }}>
                      <CopyBtn text={doc?.text || extractedText} />
                      <pre style={{ fontFamily: 'var(--f-mono)', fontSize: 13, color: 'var(--c-text2)', lineHeight: 1.6, background: 'var(--c-elevated)', padding: 16, borderRadius: 'var(--r-md)', overflowX: 'auto', whiteSpace: 'pre-wrap' }}>
                        {doc?.text || extractedText}
                      </pre>
                    </div>
                  ) : (
                    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 12 }}>
                      <div style={{ opacity: 0.15 }}><IconDocument color="var(--c-gold)" size={48} className="empty-icon" /></div>
                      <div style={{ fontFamily: 'var(--f-head)', fontSize: 18, fontWeight: 600, color: 'var(--c-text)' }}>No text extracted</div>
                      <div style={{ fontSize: 14, color: 'var(--c-text2)' }}>Upload and analyse a document to see its contents here.</div>
                    </div>
                  )}
                </div>
              )}

              {tab === 'qa' && (
                <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
                  <div className="doc-qa-messages" style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 20 }}>
                    {!extractedText && (
                      <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 12 }}>
                        <div style={{ opacity: 0.15 }}><IconDocument color="var(--c-gold)" size={48} className="empty-icon" /></div>
                        <div style={{ fontFamily: 'var(--f-head)', fontSize: 18, fontWeight: 600, color: 'var(--c-text)' }}>No document loaded</div>
                        <div style={{ fontSize: 14, color: 'var(--c-text2)' }}>Analyse a document first to ask questions about it.</div>
                      </div>
                    )}
                    {docChat.map((m, i) => (
                      <div key={i} className="msg-enter" style={{ display: 'flex', gap: 10, alignItems: 'flex-start', alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start', maxWidth: m.role === 'user' ? '65%' : '82%' }}>
                        {m.role === 'assistant' && (
                          <div style={{ background: 'var(--c-gold-dim)', border: '1px solid var(--c-gold)', color: 'var(--c-gold)', fontSize: 16, borderRadius: '50%', width: 32, height: 32, flexShrink: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                            <IconScale size={16} />
                          </div>
                        )}
                        <div style={{ position: 'relative', background: m.role === 'user' ? 'var(--c-gold-dim)' : 'var(--c-surface)', border: `1px solid ${m.role === 'user' ? 'rgba(196,149,42,0.18)' : 'var(--c-border)'}`, borderRadius: m.role === 'user' ? '16px 16px 4px 16px' : '4px 16px 16px 16px', padding: '16px 20px', flex: 1 }} className="msg-bubble-wrap">
                          <style>{`.msg-bubble-wrap:hover .copy-btn { opacity: 1 !important; }`}</style>
                          <div style={{ fontSize: 14, color: m.role === 'user' ? 'var(--c-text)' : 'var(--c-text2)', lineHeight: 1.75 }}>
                            {m.role === 'assistant' ? parseText(m.content) : m.content}
                          </div>
                          {m.role === 'assistant' && <CopyBtn text={m.content} />}
                        </div>
                      </div>
                    ))}
                    {docLoading && (
                      <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start', alignSelf: 'flex-start' }}>
                        <div style={{ background: 'var(--c-gold-dim)', border: '1px solid var(--c-gold)', color: 'var(--c-gold)', fontSize: 16, borderRadius: '50%', width: 32, height: 32, flexShrink: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                          <IconScale size={16} />
                        </div>
                        <div className="typing" style={{ display: 'flex', gap: 5, padding: '16px 20px', alignSelf: 'flex-start' }}>
                          <span className="typing-dot" style={{ width: 7, height: 7, borderRadius: '50%', background: 'var(--c-gold)', animation: 'typePulse 1.2s ease-in-out infinite' }} />
                          <span className="typing-dot" style={{ width: 7, height: 7, borderRadius: '50%', background: 'var(--c-gold)', animation: 'typePulse 1.2s ease-in-out infinite 0.2s' }} />
                          <span className="typing-dot" style={{ width: 7, height: 7, borderRadius: '50%', background: 'var(--c-gold)', animation: 'typePulse 1.2s ease-in-out infinite 0.4s' }} />
                        </div>
                      </div>
                    )}
                  </div>
                  
                  {extractedText && (
                    <div className="doc-qa-input" style={{ borderTop: '1px solid var(--c-border2)', background: 'var(--c-bg)', flexShrink: 0 }}>
                      <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end' }}>
                        <textarea className="textarea" value={docQ} onChange={e => { setDocQ(e.target.value); e.target.style.height = 'auto'; e.target.style.height = Math.min(Math.max(e.target.scrollHeight, 44), 140) + 'px'; }}
                          onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleDocQuery(); } }}
                          placeholder="Ask anything about this document..." style={{ flex: 1, minHeight: 44, maxHeight: 140, resize: 'none' }} rows={1} />
                        <button className="btn-send" onClick={handleDocQuery} disabled={docLoading || !docQ.trim()}>
                          <IconSend color="#0A0B0F" size={16} />
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </motion.div>
          </AnimatePresence>
        </div>
      </div>
      <style>{`
        @keyframes slideRight {
          0% { width: 0%; }
          100% { width: 100%; }
        }
        @keyframes chipPop {
          0% { transform: scale(0.8); opacity: 0; }
          100% { transform: scale(1); opacity: 1; }
        }
        .chipPop { animation: chipPop 200ms cubic-bezier(0.34, 1.56, 0.64, 1) forwards; opacity: 0; }
      `}</style>
    </div>
  );
}
