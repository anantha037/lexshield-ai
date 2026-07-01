import React, { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowUpRight } from 'lucide-react';
import { useStore } from '../store';
import { sendQuery, adaptQueryResponse } from '../api';
import { DraftComplete } from './DraftComplete';
import { IconScale, IconSend, IconCopy, IconCheck, IconArrowDown, IconGavel, IconExternalLink, IconCheckCircle, IconWarning, IconXCircle } from '../icons';

// const LANGS = [
//   { code: 'en', label: 'EN' }, { code: 'ml', label: 'ML' },
//   { code: 'hi', label: 'HI' }, { code: 'ta', label: 'TA' }, { code: 'te', label: 'TE' },
// ];

const QUICK = [
  { q: 'What is Section 302 IPC and its punishment?' },
  { q: 'What are tenant rights under the Transfer of Property Act?' },
  { q: 'How do I file an FIR? What are my rights?' },
  { q: 'Rights of an arrested person under BNSS 2023' },
];

const CASE_LAW_QUICK = [
  { q: 'Find landmark cases on Section 302 IPC' },
  { q: 'Show me judgments about cheque bounce Section 138 NI Act' },
  { q: 'Kesavananda Bharati case and basic structure doctrine' },
  { q: 'Landmark judgments on right to privacy' },
];

const DOC_RE = /analys[ei]s?\s+(a\s+)?document|upload\s+document|check\s+this\s+(document|file)|review\s+this\s+file/i;
const DRAFT_RE = /draft\s+a?\s+complaint|write\s+a?\s+complaint|help\s+me\s+file|draft\s+an?\s+(fir|legal\s+notice)/i;
const RIGHT_RE = /what\s+are\s+my\s+rights|know\s+my\s+rights|my\s+rights\s+as/i;

function detectRedirect(text) {
  if (DOC_RE.test(text)) return { view: 'document', label: 'Document Analysis' };
  if (DRAFT_RE.test(text)) return { view: 'draft', label: 'Draft Complaint' };
  if (RIGHT_RE.test(text)) return { view: 'rights', label: 'Know Your Rights' };
  return null;
}

function timeAgo(ts) {
  if (!ts) return '';
  const s = Math.floor(Date.now() / 1000 - ts);
  if (s < 60) return 'just now';
  if (s < 3600) return Math.floor(s / 60) + 'm ago';
  if (s < 86400) return Math.floor(s / 3600) + 'h ago';
  return Math.floor(s / 86400) + 'd ago';
}

function parseText(text) {
  if (!text) return null;
  let html = text.replace(/\[(IPC|BNS|CrPC|BNSS|IEA|BSA)\s*§\d+[a-zA-Z]*\]|Section\s+\d+[a-zA-Z]*(\s+of\s+the)?\s+([A-Za-z\s]+Act|IPC|BNS|CrPC|BNSS|IEA|BSA)/gi, match => `<span class="citation-badge" style="margin:0 4px">${match}</span>`);
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong style="color:var(--c-text);font-weight:600">$1</strong>');
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

function FallbackBanner() {
  return (
    <div style={{
      display: 'flex', alignItems: 'flex-start', gap: 8,
      background: 'rgba(245, 158, 11, 0.08)',
      color: '#F59E0B',
      padding: '10px 14px', borderRadius: 8, fontSize: 13,
      border: '1px solid rgba(245, 158, 11, 0.2)', marginBottom: 12
    }}>
      <IconWarning size={16} style={{ flexShrink: 0, marginTop: 2 }} />
      <div>
        <strong style={{ display: 'block', marginBottom: 2 }}>
          Outside Knowledge Base
        </strong>
        This query could not be matched to any provision in our legal corpus.
        The response below is a general suggestion only — not legal advice.
      </div>
    </div>
  );
}

function TrustBadge({ status }) {
  if (status === 'cited') {
    return (
      <div style={{ display: 'inline-flex', alignItems: 'center', gap: 4, background: 'rgba(16, 185, 129, 0.1)', color: '#10B981', padding: '2px 8px', borderRadius: 99, fontSize: 11, fontWeight: 600, border: '1px solid rgba(16, 185, 129, 0.2)', marginBottom: 8 }}>
        <IconCheckCircle size={12} /> Sourced
      </div>
    );
  }
  if (status === 'partial') {
    return (
      <div style={{ display: 'inline-flex', alignItems: 'center', gap: 4, background: 'rgba(245, 158, 11, 0.1)', color: '#F59E0B', padding: '2px 8px', borderRadius: 99, fontSize: 11, fontWeight: 600, border: '1px solid rgba(245, 158, 11, 0.2)', marginBottom: 8 }} title="Some content in this response may not be fully cited. Verify with primary sources.">
        <IconWarning size={12} /> Partially sourced
      </div>
    );
  }
  return null;
}

function UnverifiedBanner() {
  return (
    <div style={{ display: 'flex', alignItems: 'flex-start', gap: 8, background: 'rgba(239, 68, 68, 0.1)', color: '#EF4444', padding: '10px 14px', borderRadius: 8, fontSize: 13, border: '1px solid rgba(239, 68, 68, 0.2)', marginBottom: 12 }}>
      <IconXCircle size={16} style={{ flexShrink: 0, marginTop: 2 }} />
      <div>
        <strong style={{ display: 'block', marginBottom: 2 }}>Unverified</strong>
        This response could not be grounded in a retrieved source. Do not rely on this as legal advice.
      </div>
    </div>
  );
}

function LLMBadge() {
  return (
    <div style={{ display: 'inline-flex', alignItems: 'center', gap: 4, background: 'rgba(139, 92, 246, 0.1)', color: '#8B5CF6', padding: '2px 8px', borderRadius: 99, fontSize: 11, fontWeight: 600, border: '1px solid rgba(139, 92, 246, 0.2)', marginBottom: 8 }} title="This response was generated directly by the AI model, not retrieved from the legal corpus.">
      <IconScale size={12} /> LLM Generated — Not from legal corpus
    </div>
  );
}

/* ── Case Law Cards Component ─────────────────────────────────────────────── */

function CaseLawCards({ cases }) {
  if (!cases || cases.length === 0) {
    return (
      <div className="case-law-container">
        <div className="case-law-empty">
          <IconGavel size={32} color="var(--c-text3)" />
          <div className="case-law-empty-title" style={{ marginTop: 12 }}>No Judgments Found</div>
          <div className="case-law-empty-hint">
            Try a more specific query using the section number, act name, and key legal issue.
            <br />
            Example: <em>"Section 302 IPC murder culpable homicide"</em>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="case-law-container">
      <div className="case-law-header">
        <div className="case-law-header-title">
          <IconGavel size={18} color="var(--c-gold)" />
          Case Law Results
        </div>
        <div className="case-law-header-meta">
          <span className="case-law-source-badge">Indian Kanoon</span>
          <span>{cases.length} judgment{cases.length !== 1 ? 's' : ''} found</span>
        </div>
      </div>

      {cases.map((c, i) => (
        <div
          key={i}
          className="case-law-card"
          style={{ animationDelay: `${i * 80}ms` }}
        >
          <div className="case-law-card-number">#{i + 1}</div>
          <div className="case-law-card-title">{c.title}</div>

          <div className="case-law-card-pills">
            {c.court && <span className="case-law-pill court">{c.court}</span>}
            {c.date && <span className="case-law-pill date">{c.date}</span>}
          </div>

          {c.citation && (
            <div className="case-law-citation">{c.citation}</div>
          )}

          {c.summary && (
            <div className="case-law-summary">{c.summary}</div>
          )}

          {c.url && (
            <a
              href={c.url}
              target="_blank"
              rel="noopener noreferrer"
              className="case-law-link"
            >
              <IconExternalLink size={12} />
              Read full judgment
            </a>
          )}
        </div>
      ))}

      <div className="case-law-disclaimer">
        ⚖️ Source: Indian Kanoon (indiankanoon.org). Case summaries are AI-generated
        for reference only. Always verify judgments directly from the official source
        before relying on them in legal proceedings.
      </div>
    </div>
  );
}

function TypewriterText({ content = '', isNew }) {
  const safeContent = content || '';
  const [displayed, setDisplayed] = useState(isNew ? '' : safeContent);
  const [done, setDone] = useState(!isNew);

  useEffect(() => {
    if (!isNew || done) return;
    if (displayed.length >= safeContent.length) { setDone(true); return; }
    const timeout = setTimeout(() => {
      setDisplayed(safeContent.slice(0, displayed.length + 2));
    }, 8);
    return () => clearTimeout(timeout);
  }, [displayed, safeContent, isNew, done]);

  return (
    <div style={{ fontSize: 14, color: 'var(--c-text2)', lineHeight: 1.75 }}>
      {parseText(done ? safeContent : displayed)}
    </div>
  );
}

/* ── Main ChatView ────────────────────────────────────────────────────────── */

export default function ChatView() {
  const { activeSession, setActiveSession, refreshSessions, toast, setLastResponse, chatMessages, setChatMessages, prefillInput, setPrefillInput, setActiveView, language, setLanguage, caseLawMode, setCaseLawMode } = useStore();

  const [loading, setLoading] = useState(false);
  const [input, setInput] = useState('');
  const [showScrollBtn, setShowScrollBtn] = useState(false);
  const endRef = useRef(null);
  const areaRef = useRef(null);
  const inputRef = useRef(null);
  const inputValRef = useRef('');
  const sessionRef = useRef(activeSession);
  const langRef = useRef(language);
  const abortRef = useRef(null);
  // BUG2 fix: always hold latest handleSend so prefillInput effect never uses a stale closure
  const handleSendRef = useRef(null);

  useEffect(() => { sessionRef.current = activeSession; }, [activeSession]);
  useEffect(() => { langRef.current = language; }, [language]);
  // BUG2 fix: keep ref in sync with latest handleSend so prefillInput effect is never stale
  useEffect(() => { handleSendRef.current = handleSend; });

  const scrollToBottom = () => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    if (!areaRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = areaRef.current;
    if (scrollHeight - scrollTop - clientHeight < 150) {
      scrollToBottom();
    }
  }, [chatMessages, loading]);

  const handleScroll = (e) => {
    const { scrollTop, scrollHeight, clientHeight } = e.target;
    setShowScrollBtn(scrollHeight - scrollTop - clientHeight > 200);
  };

  // BUG2 fix: use the ref so the effect always calls the latest handleSend,
  // preventing stale-closure double-message and "unable to process" failures.
  useEffect(() => {
    if (prefillInput) {
      const q = prefillInput;
      setPrefillInput('');
      // Use ref to guarantee we call the up-to-date handleSend
      if (handleSendRef.current) {
        handleSendRef.current(q);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prefillInput]);

  const handleSend = useCallback(async (textOverride, suppressRedirect = false) => {
    const q = (textOverride ?? inputValRef.current).trim();
    if (!q || loading) return;
    setInput(''); inputValRef.current = '';

    if (!suppressRedirect) {
      const redirect = detectRedirect(q);
      if (redirect) {
        setChatMessages(m => [...m,
        { role: 'user', content: q, ts: Date.now() / 1000 },
        { role: 'assistant', content: q, ts: Date.now() / 1000, redirect },
        ]);
        return;
      }
    }

    if (abortRef.current) abortRef.current.abort(new Error('superseded'));
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    setChatMessages(m => [...m, { role: 'user', content: q, ts: Date.now() / 1000 }]);
    setLoading(true);
    try {
      const raw = await sendQuery(q, sessionRef.current, langRef.current, ctrl.signal);
      if (!raw) return;
      const r = adaptQueryResponse(raw);
      if (!r) return;
      if (!sessionRef.current && r.sessionId) { setActiveSession(r.sessionId); refreshSessions(); }
      setChatMessages(m => [...m, {
        role: 'assistant',
        content: r.answer || r.draft || r.summary || '',
        fallback: raw.fallback || false,
        intent: r.intent,
        riskLevel: r.riskLevel,
        riskScore: r.riskScore,
        citations: r.citations,
        caseLawResults: r.caseLawResults || [],
        citationStatus: r.citationStatus || 'unverified', // ← FIX: was missing, always showed "Unverified"
        scopeStatus: r.scopeStatus || 'in_scope',
        scopeMessage: r.scopeMessage || '',
        source: raw.source || 'default',
        draft: r.draft,
        supportingDocuments: r.supportingDocuments,
        filingAuthority: r.filingAuthority,
        ts: Date.now() / 1000,
        isNew: true,
      }]);
      setLastResponse(r);
    } catch (err) {
      if (err.name === 'AbortError' || err.message === 'Request superseded' || err.message?.includes('superseded')) return;
      toast(err.message || 'Query failed', 'error');
      setChatMessages(m => [...m, { role: 'assistant', content: `Error: ${err.message}`, ts: Date.now() / 1000 }]);
    } finally { setLoading(false); abortRef.current = null; }
  }, [loading, setActiveSession, refreshSessions, setChatMessages, setLastResponse, toast]);

  const onInput = (e) => {
    setInput(e.target.value);
    inputValRef.current = e.target.value;
    e.target.style.height = 'auto';
    e.target.style.height = Math.min(Math.max(e.target.scrollHeight, 44), 140) + 'px';
  };

  const onKey = (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); } };

  // Determine which quick prompts to show
  const quickPrompts = caseLawMode ? CASE_LAW_QUICK : QUICK;
  const emptyTitle = caseLawMode ? 'Search Case Law' : 'Ask a Legal Question';
  const emptySubtitle = caseLawMode
    ? 'Search Indian Kanoon for Supreme Court & High Court judgments, precedents, and rulings.'
    : 'Grounded in Indian statutes, case law, and constitutional provisions.';
  const EmptyIcon = caseLawMode ? IconGavel : IconScale;

  const latestAssistantIdx = (chatMessages || []).reduce((acc, m, i) =>
    m.role === 'assistant' ? i : acc, -1);

  return (
    <div className="chat-container view-enter" style={{ display: 'flex', flexDirection: 'column', height: '100%', background: 'var(--c-bg)', position: 'relative' }}>
      <div className="view-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
          <div style={{ width: 28, height: 1, background: 'var(--c-gold)' }} />
          <span style={{ fontFamily: 'var(--f-mono)', fontSize: 10, fontWeight: 700, letterSpacing: '0.2em', color: 'var(--c-gold)', textTransform: 'uppercase' }}>
            LexShield · Legal Intelligence
          </span>
        </div>
        <h1 style={{ fontFamily: 'var(--f-head)', fontSize: 36, fontWeight: 700, color: 'var(--c-text)', letterSpacing: '-0.01em', margin: 0 }}>
          {caseLawMode ? 'Case Law Search' : 'Legal Q&A'}
        </h1>
        <p style={{ fontSize: 15, color: 'var(--c-text2)', marginTop: 8, margin: 0 }}>
          {caseLawMode ? 'Search Indian Kanoon for court judgments and precedents' : 'Ask any question grounded in Indian law'}
        </p>
      </div>

      <div className="messages-area" ref={areaRef} onScroll={handleScroll} style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', scrollBehavior: 'smooth' }}>
        {(!chatMessages || chatMessages.length === 0) && !loading && (
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 16, padding: '0 40px', minHeight: 0 }}>
            <EmptyIcon size={48} color="var(--c-gold-dim)" />
            <h2 style={{ fontFamily: 'var(--f-head)', fontSize: 28, fontWeight: 600, color: 'var(--c-text)', margin: 0 }}>{emptyTitle}</h2>
            <p style={{ fontSize: 14, color: 'var(--c-text2)', textAlign: 'center', maxWidth: 400, margin: 0 }}>
              {emptySubtitle}
            </p>
            <motion.div
              className="quick-grid"
              initial="hidden"
              animate="visible"
              variants={{ visible: { transition: { staggerChildren: 0.08 } } }}
              style={{
                marginTop: 8,
                maxWidth: 600,
                width: '100%',
              }}
            >
              {quickPrompts.map((q, i) => (
                <motion.div
                  key={i}
                  variants={{
                    hidden: { opacity: 0, y: 14 },
                    visible: { opacity: 1, y: 0 }
                  }}
                  whileHover={{ y: -3 }}
                  transition={{ duration: 0.25 }}
                  onClick={() => handleSend(q.q)}
                  className="quick-prompt-card"
                  onHoverStart={(_, info) => { }}
                  onHoverEnd={(_, info) => { }}
                  data-card="true"
                >
                  <span>{q.q}</span>
                  <ArrowUpRight
                    size={16}
                    style={{
                      flexShrink: 0,
                      marginTop: 2,
                      color: 'var(--c-text3)',
                      transition: 'color 200ms',
                    }}
                  />
                </motion.div>
              ))}
            </motion.div>
          </div>
        )}

        {(chatMessages || []).map((m, i) => {
          if (m.redirect) return (
            <div key={i} className="msg-enter" style={{ display: 'flex', gap: 10, alignItems: 'flex-start', alignSelf: 'flex-start', maxWidth: '82%' }}>
              <div style={{ background: 'var(--c-gold-dim)', border: '1px solid var(--c-gold)', color: 'var(--c-gold)', fontSize: 16, borderRadius: '50%', width: 32, height: 32, flexShrink: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <IconScale size={16} />
              </div>
              <div style={{ background: 'var(--c-surface)', border: '1px solid var(--c-border)', borderRadius: '4px 16px 16px 16px', padding: '16px 20px', flex: 1 }}>
                <div style={{ fontSize: 14, color: 'var(--c-text2)', marginBottom: 12 }}>
                  Looks like you want to use <strong style={{ color: 'var(--c-text)' }}>{m.redirect.label}</strong>.
                </div>
                <div style={{ display: 'flex', gap: 12 }}>
                  <button className="btn-gold" onClick={() => setActiveView(m.redirect.view)}>Go to {m.redirect.label} →</button>
                  <button className="btn-ghost" onClick={() => {
                    setChatMessages(msgs => msgs.filter((_, j) => j !== i && j !== i - 1));
                    handleSend(m.content.trim(), true);
                  }}>I have a legal question instead</button>
                </div>
              </div>
            </div>
          );

          if (m.role === 'user') return (
            <div key={i} className="msg-enter" style={{ alignSelf: 'flex-end', maxWidth: '65%', background: 'var(--c-gold-dim)', border: '1px solid rgba(196,149,42,0.18)', borderRadius: '16px 16px 4px 16px', padding: '12px 16px', fontSize: 14, color: 'var(--c-text)', lineHeight: 1.6 }}>
              {m.content}
              <div style={{ fontSize: 11, color: 'var(--c-text3)', textAlign: 'right', marginTop: 6 }}>{timeAgo(m.ts)}</div>
            </div>
          );

          // ── Case Law intent → render rich cards ──
          const isCaseLaw = m.intent === 'case_law_search' && m.caseLawResults && m.caseLawResults.length > 0;
          const isLegalIntent = ['legal_query', 'case_law_search', 'document_analysis', 'rights', 'draft'].includes(m.intent);
          const cStatus = m.citationStatus || 'unverified';

          if (m.role === 'assistant' && m.scopeStatus === 'out_of_scope') {
            return (
              <div key={i} className="msg-enter" style={{ display: 'flex', gap: 10, alignItems: 'flex-start', alignSelf: 'flex-start', maxWidth: '88%' }}>
                <div style={{ background: 'var(--c-gold-dim)', border: '1px solid var(--c-gold)', color: 'var(--c-gold)', fontSize: 16, borderRadius: '50%', width: 32, height: 32, flexShrink: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <IconScale size={16} />
                </div>
                <div className="scope-warning-block">
                  <div className="scope-warning-heading">
                    <IconWarning size={16} /> Out of Scope Request
                  </div>
                  <div className="scope-warning-body">
                    {m.scopeMessage || 'This request falls outside the supported scope of LexShield AI.'}
                  </div>
                  <div className="scope-warning-footer">
                    LexShield is optimized for Indian jurisdiction, case law search, and supported legal templates.
                  </div>
                </div>
              </div>
            );
          }

          if (m.draft && m.draft.length > 200) {
            return (
              <DraftComplete key={i} msg={m} sessionId={activeSession} toast={toast} />
            );
          }

          return (
            <div key={i} className="msg-enter" style={{ display: 'flex', gap: 10, alignItems: 'flex-start', alignSelf: 'flex-start', maxWidth: '88%' }}>
              <div style={{ background: isCaseLaw ? 'rgba(6,182,212,0.10)' : 'var(--c-gold-dim)', border: `1px solid ${isCaseLaw ? 'rgba(6,182,212,0.25)' : 'var(--c-gold)'}`, color: isCaseLaw ? '#06B6D4' : 'var(--c-gold)', fontSize: 16, borderRadius: '50%', width: 32, height: 32, flexShrink: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                {isCaseLaw ? <IconGavel size={16} /> : <IconScale size={16} />}
              </div>
              <div style={{ position: 'relative', background: 'var(--c-surface)', border: `1px solid ${isCaseLaw ? 'rgba(6,182,212,0.15)' : 'var(--c-border)'}`, borderRadius: '4px 16px 16px 16px', padding: '16px 20px', flex: 1 }} className="msg-bubble-wrap">
                <style>{`.msg-bubble-wrap:hover .copy-btn { opacity: 1 !important; }`}</style>

                {m.fallback && <FallbackBanner />}
                {!m.fallback && isLegalIntent && cStatus === 'unverified' && m.source !== 'llm_only' && <UnverifiedBanner />}
                {!m.fallback && isLegalIntent && cStatus !== 'unverified' && m.source !== 'llm_only' && <TrustBadge status={cStatus} />}
                {!m.fallback && m.source === 'llm_only' && <LLMBadge />}

                {isCaseLaw ? (
                  <CaseLawCards cases={m.caseLawResults} />
                ) : (
                  <TypewriterText
                    content={m.content}
                    isNew={m.isNew && !isCaseLaw}
                  />
                )}

                <div style={{ fontSize: 11, color: 'var(--c-text3)', textAlign: 'right', marginTop: 6 }}>{timeAgo(m.ts)}</div>
                <CopyBtn text={m.content} />
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

      {showScrollBtn && (
        <button
          onClick={scrollToBottom}
          style={{ position: 'absolute', bottom: 120, right: 60, width: 36, height: 36, borderRadius: '50%', background: 'var(--c-surface)', border: '1px solid var(--c-border)', color: 'var(--c-gold)', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', animation: 'fadeIn 150ms ease forwards', zIndex: 10 }}
        >
          <IconArrowDown size={18} />
        </button>
      )}

      <div className="chat-input-area">
        {/* <div style={{ display: 'flex', gap: 6, marginBottom: 10 }}>
          {LANGS.map(l => (
            <button key={l.code}
              style={{ padding: '3px 12px', borderRadius: 99, fontSize: 12, fontWeight: 600, border: `1px solid ${language === l.code ? 'var(--c-gold)' : 'var(--c-border)'}`, color: language === l.code ? 'var(--c-gold)' : 'var(--c-text3)', background: language === l.code ? 'var(--c-gold-dim)' : 'transparent', cursor: 'pointer', transition: 'all 150ms' }}
              onMouseDown={(e) => e.currentTarget.style.transform = 'scale(0.95)'}
              onMouseUp={(e) => e.currentTarget.style.transform = 'scale(1)'}
              onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}
              onClick={() => setLanguage(l.code)}>
              {l.label}
            </button>
          ))}
        </div> */}
        <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end' }}>
          <textarea ref={inputRef} className="textarea" value={input} onChange={onInput} onKeyDown={onKey}
            placeholder={caseLawMode ? 'Search for case law, judgments, or precedents...' : 'Ask a legal question...'} style={{ flex: 1, minHeight: 44, maxHeight: 140, resize: 'none' }} rows={1} />
          <motion.button
            className="btn-send"
            whileHover={{ scale: 1.06 }}
            whileTap={{ scale: 0.92 }}
            onClick={() => handleSend()}
            disabled={loading || !input.trim()}
          >
            <IconSend />
          </motion.button>
        </div>
      </div>
    </div>
  );
}