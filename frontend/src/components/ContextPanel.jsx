import { useState, useEffect } from 'react';
import { useStore } from '../store';
import { IconScale } from '../icons';

const CONF_TEXT = {
  5: 'High confidence — well-supported by Indian law',
  4: 'Good confidence — grounded in statute',
  3: 'Moderate — verify with a qualified lawyer',
  2: 'Low — limited sources found',
  1: 'Low — limited sources found',
};

const CONF_COLOR = { 
  5: 'var(--c-low)', 
  4: 'var(--c-low)', 
  3: 'var(--c-medium)', 
  2: 'var(--c-high)', 
  1: 'var(--c-high)' 
};

export default function ContextPanel() {
  const { activeSession, lastResponse } = useStore();
  const [show, setShow] = useState(true);
  const [r, setR] = useState(lastResponse || {});

  // Fade out old, fade in new logic
  useEffect(() => {
    if (lastResponse && lastResponse !== r) {
      setShow(false);
      const t = setTimeout(() => {
        setR(lastResponse);
        setShow(true);
      }, 100);
      return () => clearTimeout(t);
    }
  }, [lastResponse, r]);

  if (!activeSession) return null;

  if (!r.intent) {
    return (
      <div style={{ width: 280, minWidth: 280, background: 'var(--c-bg2)', borderLeft: '1px solid var(--c-border2)', padding: 24, overflowY: 'auto', display: 'flex', flexDirection: 'column', height: '100%', gap: 12 }}>
        <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--c-text3)', marginBottom: 8 }}>
          SESSION HISTORY
        </div>
        <div style={{ fontSize: 13, color: 'var(--c-text2)' }}>
          Select a session from the sidebar to view its context and analysis.
        </div>
      </div>
    );
  }

  const ragGrade = r.confidence ? Math.round(r.confidence * 5) : 0;
  const confBarColor = CONF_COLOR[ragGrade] || 'var(--c-medium)';
  const confText = CONF_TEXT[ragGrade] || '';
  const sid = activeSession ? activeSession.slice(0, 12).toUpperCase() : null;

  return (
    <div style={{ width: 280, minWidth: 280, background: 'var(--c-bg2)', borderLeft: '1px solid var(--c-border2)', padding: 24, overflowY: 'auto', display: 'flex', flexDirection: 'column', opacity: show ? 1 : 0, transition: `opacity ${show ? '200ms' : '100ms'} ease`, height: '100%' }}>
      {r.intent && r.intent !== 'document_analysis' && ragGrade > 0 && (
        <div style={{ marginBottom: 24 }}>
          <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--c-text3)' }}>ANSWER CONFIDENCE</div>
          <div style={{ width: '100%', height: 6, borderRadius: 99, background: 'var(--c-border)', overflow: 'hidden', margin: '8px 0' }}>
            <div style={{ height: '100%', borderRadius: 99, width: show ? `${(ragGrade / 5) * 100}%` : '0%', background: confBarColor, transition: 'width 600ms ease-out' }} />
          </div>
          <div style={{ fontSize: 13, color: 'var(--c-text2)' }}>{confText}</div>
        </div>
      )}

      {r.citations?.length > 0 && (
        <div style={{ marginBottom: 24 }}>
          <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--c-text3)', marginBottom: 8 }}>SOURCES USED</div>
          {r.citations.map((c, i) => (
            <div key={i} style={{ padding: '8px 0', borderBottom: '1px solid var(--c-border2)', animation: `fadeIn 200ms ease forwards ${i * 60}ms`, opacity: 0 }}>
              <div style={{ fontFamily: 'var(--f-mono)', fontSize: 12, color: 'var(--c-gold)' }}>{c.source}</div>
              <div style={{ color: 'var(--c-text2)', fontSize: 13 }}>{c.section || c.sectionTitle}</div>
            </div>
          ))}
        </div>
      )}

      {r.intent === 'document_analysis' && r.riskLevel && (
        <div style={{ marginBottom: 24 }}>
          <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--c-text3)', marginBottom: 8 }}>DOCUMENT RISK</div>
          <div style={{ fontSize: 48, fontWeight: 700, color: `var(--c-${r.riskLevel === 'medium' ? 'medium' : r.riskLevel === 'high' || r.riskLevel === 'critical' ? 'high' : 'low'})` }}>
            {Math.round(r.riskScore * 100)}
          </div>
          <div style={{ fontSize: 14, color: 'var(--c-text2)', marginTop: 4, textTransform: 'capitalize' }}>{r.riskLevel} Level</div>
          <div style={{ fontSize: 13, color: 'var(--c-text2)', marginTop: 4 }}>
            {r.riskLevel === 'low' && 'This document appears standard with no major risk clauses.'}
            {r.riskLevel === 'medium' && 'Some clauses require attention. Review with a lawyer.'}
            {r.riskLevel === 'high' && 'Significant risk clauses detected. Do not sign without legal advice.'}
            {r.riskLevel === 'critical' && 'Critical violations detected. Seek immediate legal counsel.'}
          </div>
        </div>
      )}

      <div style={{ marginTop: 'auto', paddingTop: 24 }}>
        <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--c-text3)' }}>SESSION</div>
        <div style={{ fontFamily: 'var(--f-mono)', fontSize: 12, color: 'var(--c-text3)' }}>{sid}</div>
      </div>
    </div>
  );
}
