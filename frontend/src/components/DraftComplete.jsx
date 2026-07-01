import React, { useState, useEffect } from 'react';
import { exportPdf } from '../api';

export function parseText(text) {
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

export function DraftComplete({ msg, sessionId, toast }) {
  const [pdfLoading, setPdfLoading] = useState(false);
  const [pdfError, setPdfError] = useState('');

  useEffect(() => {
    console.log('DraftComplete rendered, sessionId:', sessionId);
    if (!sessionId) console.error('[DraftComplete] sessionId is undefined — PDF button will fail silently');
  }, [sessionId]);

  const handlePdfDownload = async () => {
    setPdfLoading(true);
    setPdfError('');
    try {
      const blob = await exportPdf(sessionId);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `legal_draft_${(sessionId || '').slice(0, 8)}.pdf`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setPdfError(err.message || 'PDF download failed');
    } finally {
      setPdfLoading(false);
    }
  };

  return (
    <div className="msg-enter" style={{ background: 'var(--c-surface)', border: '1px solid var(--c-border)', borderRadius: 'var(--r-md)', padding: 24, alignSelf: 'center', width: '100%', maxWidth: 760 }}>
      <div style={{ fontFamily: 'var(--f-head)', fontSize: 20, color: 'var(--c-gold)', marginBottom: 16 }}>Your Legal Draft is Ready</div>
      <div style={{ maxHeight: 400, overflowY: 'auto', background: 'var(--c-elevated)', borderRadius: 'var(--r-sm)', padding: 16, border: '1px solid var(--c-border)' }}>
        <pre style={{ fontFamily: 'var(--f-mono)', fontSize: 13, lineHeight: 1.7, color: 'var(--c-text2)', whiteSpace: 'pre-wrap' }}>{msg.draft}</pre>
      </div>
      <div style={{ display: 'flex', gap: 12, marginTop: 16, flexWrap: 'wrap', alignItems: 'center' }}>
        <button className="btn-ghost" onClick={() => { navigator.clipboard.writeText(msg.draft); toast('Copied Draft to Clipboard'); }}>Copy Draft</button>
        <button className="btn-gold pulse-btn" onClick={() => { const b = new Blob([msg.draft], { type: 'text/plain' }); const a = document.createElement('a'); a.href = URL.createObjectURL(b); a.download = 'legal_draft.txt'; a.click(); }}>
          Download .txt
        </button>
        <button
          id="download-pdf-btn"
          className="btn-gold"
          onClick={handlePdfDownload}
          disabled={pdfLoading}
          style={{ display: 'flex', alignItems: 'center', gap: 8, opacity: pdfLoading ? 0.7 : 1 }}
        >
          {pdfLoading ? (
            <>
              <span className="pdf-spinner" />
              Generating PDF…
            </>
          ) : (
            'Download as PDF'
          )}
        </button>
      </div>
      <div style={{ fontSize: 12, color: 'var(--c-text3)', marginTop: 6 }}>
        ↑ Use the buttons above to save your draft
      </div>
      {pdfError && (
        <div style={{ marginTop: 8, fontSize: 13, color: '#e74c3c', background: 'rgba(231,76,60,0.08)', padding: '6px 12px', borderRadius: 'var(--r-sm)' }}>
          ⚠ {pdfError}
        </div>
      )}
      <style>{`
        @keyframes singlePulse { 0% { box-shadow: 0 0 0 12px var(--c-gold-dim); } 100% { box-shadow: 0 0 0 0 var(--c-gold-dim); } }
        .pulse-btn { animation: singlePulse 600ms ease-out forwards; }
        @keyframes pdfSpin { to { transform: rotate(360deg); } }
        .pdf-spinner {
          width: 14px; height: 14px; border: 2px solid rgba(255,255,255,0.3);
          border-top-color: #fff; border-radius: 50%;
          animation: pdfSpin 0.7s linear infinite; display: inline-block;
        }
      `}</style>
      {msg.supportingDocuments?.length > 0 && (
        <div style={{ marginTop: 20 }}>
          <div style={{ fontSize: 12, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--c-text3)', marginBottom: 8 }}>SUPPORTING DOCUMENTS NEEDED</div>
          <ul style={{ margin: 0, paddingLeft: 20, color: 'var(--c-text2)', fontSize: 13, lineHeight: 1.6 }}>
            {msg.supportingDocuments.map((d, j) => <li key={j}>{d}</li>)}
          </ul>
        </div>
      )}
      {msg.filingAuthority && (
        <div style={{ marginTop: 16, padding: 12, background: 'var(--c-elevated)', borderLeft: '2px solid var(--c-gold)', borderRadius: '0 var(--r-sm) var(--r-sm) 0' }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--c-text)' }}>Filing Authority</div>
          <div style={{ fontSize: 13, color: 'var(--c-text2)', marginTop: 4 }}>{msg.filingAuthority}</div>
        </div>
      )}
    </div>
  );
}
