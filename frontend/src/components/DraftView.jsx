import { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useStore } from '../store';
import { sendQuery, adaptQueryResponse, deleteSession } from '../api';
import { IconWage, IconHome, IconCheque, IconCart, IconPhone, IconHeart, IconBriefcase, IconDollar, IconSend, IconArrowBack, IconDraft, IconScale, IconCopy, IconCheck } from '../icons';

const STAGES = ['Describe', 'Clarify', 'Sections', 'Authority', 'Confirm', 'Generate'];

const DRAFT_CATEGORIES = [
  { id: 'wage_theft',    Icon: IconWage,      name: 'Wage Theft',         desc: 'Unpaid salary, delayed wages, bonus disputes',      law: 'Sec 33C(2) IDA' },
  { id: 'eviction',      Icon: IconHome,      name: 'Illegal Eviction',   desc: 'Unlawful eviction, security deposit disputes',      law: 'Sec 6 SRA' },
  { id: 'cheque_bounce', Icon: IconCheque,    name: 'Cheque Bounce',      desc: 'Dishonoured cheque, NI Act Section 138',           law: 'Sec 138 NI Act' },
  { id: 'consumer',      Icon: IconCart,      name: 'Consumer Complaint', desc: 'Defective product, service deficiency',            law: 'Sec 35 CPA' },
  { id: 'fir',           Icon: IconPhone,     name: 'FIR Complaint',      desc: 'Criminal complaint to police, cognizable offence', law: 'Sec 154 CrPC' },
  { id: 'dv',            Icon: IconHeart,     name: 'Domestic Violence',  desc: 'Protection under PWDVA 2005',                      law: 'Sec 12 PWDVA' },
  { id: 'employment',    Icon: IconBriefcase, name: 'Employment Issue',   desc: 'Wrongful termination, workplace rights',           law: 'Misc Labour Law' },
  { id: 'loan',          Icon: IconDollar,    name: 'Loan Default',       desc: 'Loan harassment, recovery agent issues',          law: 'Sec 13 SARFAESI' },
];

function stageIndex(stage) {
  return { CLARIFY: 1, SECTIONS: 2, AUTHORITY: 3, CONFIRM: 4, DONE: 5 }[stage] ?? 0;
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

function CategorySelector({ onSelect }) {
  return (
    <div className="view-enter" style={{ padding: '48px 40px', maxWidth: 960, margin: '0 auto', overflowY: 'auto', height: '100%' }}>
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
          <div style={{ width: 28, height: 1, background: 'var(--c-gold)' }} />
          <span style={{ fontFamily: 'var(--f-mono)', fontSize: 10, fontWeight: 700, letterSpacing: '0.2em', color: 'var(--c-gold)', textTransform: 'uppercase' }}>
            LexShield · Legal Intelligence
          </span>
        </div>
        <h2 style={{ fontFamily: 'var(--f-head)', fontSize: 36, fontWeight: 700, color: 'var(--c-text)', margin: 0 }}>What type of complaint do you need?</h2>
        <p style={{ fontSize: 15, color: 'var(--c-text2)', marginTop: 8, lineHeight: 1.6 }}>
          Select a category and I'll guide you through drafting your legal complaint step by step.
        </p>
      </div>
      <motion.div
        initial="hidden"
        animate="visible"
        variants={{ visible: { transition: { staggerChildren: 0.06 } } }}
        style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 20, marginTop: 40 }}
      >
        {DRAFT_CATEGORIES.map((cat, i) => (
          <motion.div
            key={cat.id}
            variants={{ hidden: { opacity: 0, y: 14 }, visible: { opacity: 1, y: 0 } }}
            whileHover={{ y: -4, transition: { duration: 0.2 } }}
            className="draft-cat-card"
            onClick={() => onSelect(cat)}
            style={{
              background: 'var(--c-surface)', border: '1px solid var(--c-border)', borderRadius: 'var(--r-md)', padding: '32px 28px', cursor: 'pointer', position: 'relative', overflow: 'hidden', transition: 'all 200ms cubic-bezier(0.4,0,0.2,1)'
            }}
          >
            <style>{`
              .draft-cat-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: var(--c-gold); opacity: 0; transition: opacity 200ms ease; }
              .draft-cat-card:hover { border-color: rgba(196,149,42,0.35); background: var(--c-elevated); box-shadow: 0 8px 24px rgba(0,0,0,0.4); }
              .draft-cat-card:hover::before { opacity: 1; }
              .draft-cat-card:active { transform: scale(0.97); transition: transform 100ms; }
            `}</style>
            <span className="citation-badge" style={{ marginBottom: 16, display: 'inline-block' }}>{cat.law}</span>
            <cat.Icon color="var(--c-gold)" size={28} style={{ display: 'block', marginBottom: 12, strokeWidth: 1.5 }} />
            <div style={{ fontFamily: 'var(--f-head)', fontSize: 18, fontWeight: 600, color: 'var(--c-text)', marginBottom: 8 }}>{cat.name}</div>
            <div style={{ fontSize: 13, color: 'var(--c-text2)', lineHeight: 1.6 }}>{cat.desc}</div>
          </motion.div>
        ))}
      </motion.div>
    </div>
  );
}

export default function DraftView() {
  const { toast, draftCategory, setDraftCategory } = useStore();
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [currentStage, setCurrentStage] = useState(null);
  const inputValRef = useRef('');
  const endRef = useRef(null);
  const didAutoSend = useRef(false);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages, loading]);

  useEffect(() => {
    if (draftCategory && messages.length === 0 && !didAutoSend.current) {
      didAutoSend.current = true;
      handleSend(`I need help drafting a ${draftCategory.name} complaint`);
    }
  }, [draftCategory, messages.length]);

  const handleSend = async (textOverride) => {
    const q = (textOverride ?? inputValRef.current).trim();
    if (!q || loading) return;
    setInput(''); inputValRef.current = '';
    setMessages(m => [...m, { role: 'user', content: q }]);
    setLoading(true);
    try {
      const raw = await sendQuery(q, sessionId, 'en');
      if (!raw) return;
      const r = adaptQueryResponse(raw);
      if (r.sessionId) setSessionId(r.sessionId);
      if (r.stage) setCurrentStage(r.stage);
      setMessages(m => [...m, {
        role: 'assistant', content: r.answer || r.draft || r.summary || 'Processing…',
        stage: r.stage, draft: r.draft, outline: r.outline,
        supportingDocuments: r.supportingDocuments, filingAuthority: r.filingAuthority,
      }]);
    } catch (err) {
      if (err.name === 'AbortError' || err.message?.includes('superseded')) return;
      toast(err.message, 'error');
    } finally { setLoading(false); }
  };

  const onKey = (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); } };
  const stageIdx = stageIndex(currentStage);

  const handleChangeCategory = async () => {
    if (window.confirm("Start over? Current draft will be lost.")) {
      if (sessionId) {
        try { await deleteSession(sessionId); } catch { /* non-fatal */ }
      }
      setMessages([]); setSessionId(null); setCurrentStage(null);
      setDraftCategory(null); didAutoSend.current = false;
    }
  };

  if (!draftCategory) {
    return <CategorySelector onSelect={cat => { setDraftCategory(cat); didAutoSend.current = false; }} />;
  }

  return (
    <div className="view-enter" style={{ display: 'flex', flexDirection: 'column', height: '100%', background: 'var(--c-bg)' }}>
      {/* Header */}
      <div style={{ padding: '20px 40px 16px', borderBottom: '1px solid var(--c-border2)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 16 }}>
          <button className="btn-ghost" onClick={handleChangeCategory} style={{ fontSize: 12, display: 'flex', alignItems: 'center', gap: 6, padding: '6px 12px' }}>
            <IconArrowBack size={14} /> Change Category
          </button>
          <div style={{ flex: 1 }} />
          <h1 style={{ fontFamily: 'var(--f-head)', fontSize: 22, color: 'var(--c-text)', margin: 0 }}>Draft {draftCategory.name}</h1>
          <draftCategory.Icon color="var(--c-gold)" size={24} />
        </div>
        {/* Stepper */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {STAGES.map((s, i) => (
            <div key={s} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: i < stageIdx ? 'var(--c-low)' : i === stageIdx ? 'var(--c-gold)' : 'var(--c-text3)', transition: 'color 300ms', display: 'flex', alignItems: 'center', gap: 6 }}>
                {i < stageIdx && (
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" style={{ strokeDasharray: 20, strokeDashoffset: 20, animation: 'checkDraw 300ms ease forwards' }}>
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                )}
                {s}
              </div>
              {i < STAGES.length - 1 && <div style={{ width: 20, height: 1, background: 'var(--c-border)' }} />}
            </div>
          ))}
        </div>
      </div>

      {/* Messages */}
      <div className="messages-area" style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', scrollBehavior: 'smooth' }}>
        {messages.map((m, i) => {
          if (m.role === 'user') return (
            <div key={i} className="msg-enter" style={{ alignSelf: 'flex-end', maxWidth: '65%', background: 'var(--c-gold-dim)', border: '1px solid rgba(196,149,42,0.18)', borderRadius: '16px 16px 4px 16px', padding: '12px 16px', fontSize: 14, color: 'var(--c-text)', lineHeight: 1.6 }}>
              {m.content}
            </div>
          );

          // DONE stage
          if (m.stage === 'DONE' && m.draft) {
            return (
              <div key={i} className="msg-enter" style={{ background: 'var(--c-surface)', border: '1px solid var(--c-border)', borderRadius: 'var(--r-md)', padding: 24, alignSelf: 'center', width: '100%', maxWidth: 760 }}>
                <div style={{ fontFamily: 'var(--f-head)', fontSize: 20, color: 'var(--c-gold)', marginBottom: 16 }}>Your Legal Draft is Ready</div>
                <div style={{ maxHeight: 400, overflowY: 'auto', background: 'var(--c-elevated)', borderRadius: 'var(--r-sm)', padding: 16, border: '1px solid var(--c-border)' }}>
                  <pre style={{ fontFamily: 'var(--f-mono)', fontSize: 13, lineHeight: 1.7, color: 'var(--c-text2)', whiteSpace: 'pre-wrap' }}>{m.draft}</pre>
                </div>
                <div style={{ display: 'flex', gap: 12, marginTop: 16 }}>
                  <button className="btn-ghost" onClick={() => { navigator.clipboard.writeText(m.draft); toast('Copied Draft to Clipboard'); }}>Copy Draft</button>
                  <button className="btn-gold pulse-btn" onClick={() => { const b = new Blob([m.draft], { type: 'text/plain' }); const a = document.createElement('a'); a.href = URL.createObjectURL(b); a.download = 'legal_draft.txt'; a.click(); }}>
                    Download .txt
                  </button>
                </div>
                <style>{`
                  @keyframes singlePulse { 0% { box-shadow: 0 0 0 12px var(--c-gold-dim); } 100% { box-shadow: 0 0 0 0 var(--c-gold-dim); } }
                  .pulse-btn { animation: singlePulse 600ms ease-out forwards; }
                `}</style>
                {m.supportingDocuments?.length > 0 && (
                  <div style={{ marginTop: 20 }}>
                    <div style={{ fontSize: 12, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--c-text3)', marginBottom: 8 }}>SUPPORTING DOCUMENTS NEEDED</div>
                    <ul style={{ margin: 0, paddingLeft: 20, color: 'var(--c-text2)', fontSize: 13, lineHeight: 1.6 }}>
                      {m.supportingDocuments.map((d, j) => <li key={j}>{d}</li>)}
                    </ul>
                  </div>
                )}
                {m.filingAuthority && (
                  <div style={{ marginTop: 16, padding: 12, background: 'var(--c-elevated)', borderLeft: '2px solid var(--c-gold)', borderRadius: '0 var(--r-sm) var(--r-sm) 0' }}>
                    <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--c-text)' }}>Filing Authority</div>
                    <div style={{ fontSize: 13, color: 'var(--c-text2)', marginTop: 4 }}>{m.filingAuthority}</div>
                  </div>
                )}
              </div>
            );
          }

          // CONFIRM stage special card
          if (m.stage === 'CONFIRM') {
            return (
              <div key={i} style={{ animation: 'slideUpConfirm 300ms ease forwards', opacity: 0, background: 'var(--c-elevated)', border: '1px solid rgba(196,149,42,0.2)', borderRadius: 'var(--r-md)', padding: '20px 24px', alignSelf: 'flex-start', maxWidth: '82%' }}>
                <style>{`
                  @keyframes slideUpConfirm {
                    0% { transform: translateY(20px); opacity: 0; }
                    100% { transform: translateY(0); opacity: 1; }
                  }
                `}</style>
                <div style={{ fontFamily: 'var(--f-head)', fontSize: 18, color: 'var(--c-gold)', marginBottom: 16 }}>Review Your Draft Outline</div>
                <div style={{ fontSize: 14, color: 'var(--c-text2)', lineHeight: 1.7 }}>
                  {m.outline ? parseText(m.outline) : parseText(m.content)}
                </div>
                <div style={{ display: 'flex', gap: 12, marginTop: 20 }}>
                  <button className="btn-gold" onClick={() => handleSend("Looks good, please generate the draft.")}>✓ Confirm & Generate</button>
                  <button className="btn-ghost" onClick={() => handleSend("I'd like to make some changes.")}>✗ Make Changes</button>
                </div>
              </div>
            );
          }

          // Default assistant message
          return (
            <div key={i} className="msg-enter" style={{ display: 'flex', gap: 10, alignItems: 'flex-start', alignSelf: 'flex-start', maxWidth: '82%' }}>
              <div style={{ background: 'var(--c-gold-dim)', border: '1px solid var(--c-gold)', color: 'var(--c-gold)', fontSize: 16, borderRadius: '50%', width: 32, height: 32, flexShrink: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <IconScale size={16} />
              </div>
              <div style={{ background: 'var(--c-surface)', border: '1px solid var(--c-border)', borderRadius: '4px 16px 16px 16px', padding: '16px 20px', flex: 1 }}>
                <div style={{ fontSize: 14, color: 'var(--c-text2)', lineHeight: 1.75 }}>
                  {parseText(m.content)}
                </div>
              </div>
            </div>
          );
        })}
        {loading && (
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
        <div ref={endRef} />
      </div>

      {/* Input */}
      <div className="chat-input-area">
        <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end' }}>
          <textarea className="textarea" value={input}
            onChange={e => { setInput(e.target.value); inputValRef.current = e.target.value; e.target.style.height = 'auto'; e.target.style.height = Math.min(Math.max(e.target.scrollHeight, 44), 140) + 'px'; }}
            onKeyDown={onKey}
            placeholder={messages.length === 0 ? 'Describe your legal issue…' : 'Provide the requested details…'}
            style={{ flex: 1, minHeight: 44, maxHeight: 140, resize: 'none' }} rows={1} />
          <button className="btn-send" onClick={() => handleSend()} disabled={loading || !input.trim()}>
            <IconSend />
          </button>
        </div>
      </div>
      <style>{`
        @keyframes checkDraw {
          to { stroke-dashoffset: 0; }
        }
      `}</style>
    </div>
  );
}
