import React from 'react';

// Matches [IPC §302], [BNS §138], [Section 12 PWDVA], Section 138 of NI Act, etc.
const CITATION_RE = /(\[(?:IPC|BNS|CrPC|BNSS|CPC|NI Act|IT Act|IEA|TPA|CPA|PWDVA|SARFAESI|IDA|SRA)\s*[§S]?\s*[\dA-Za-z]+[A-Za-z]?\]|\[Section\s+[\dA-Za-z]+[A-Za-z]?\]|Section\s+\d+[A-Za-z]?(?:\([^)]+\))?\s+of\s+[A-Z][^,.\n]{2,40})/g;
const BOLD_RE = /\*\*([^*\n]+)\*\*/g;

function parseLine(line, keyPrefix) {
  const result = [];
  const combined = new RegExp(`${CITATION_RE.source}|\\*\\*([^*\\n]+)\\*\\*`, 'g');
  let last = 0;
  let idx = 0;
  let m;
  combined.lastIndex = 0;
  while ((m = combined.exec(line)) !== null) {
    if (m.index > last) result.push(line.slice(last, m.index));
    if (m[1]) {
      // citation match
      result.push(<span key={`${keyPrefix}-${idx++}`} className="citation-badge">{m[1]}</span>);
    } else if (m[2]) {
      // bold match
      result.push(<strong key={`${keyPrefix}-${idx++}`}>{m[2]}</strong>);
    }
    last = m.index + m[0].length;
  }
  if (last < line.length) result.push(line.slice(last));
  return result.length ? result : [line];
}

export default function parseResponse(text) {
  if (!text || typeof text !== 'string') return null;
  const blocks = text.split(/\n\n+/);
  const elements = [];

  blocks.forEach((block, bi) => {
    const lines = block.split('\n').map(l => l.trim()).filter(Boolean);
    if (!lines.length) return;

    const allBullet  = lines.every(l => /^[•\-]\s+/.test(l));
    const allOrdered = lines.every(l => /^\d+\.\s+/.test(l));

    if (allBullet) {
      elements.push(
        <ul key={`b${bi}`} className="parsed-list">
          {lines.map((l, li) => <li key={li}>{parseLine(l.replace(/^[•\-]\s+/, ''), `b${bi}l${li}`)}</li>)}
        </ul>
      );
    } else if (allOrdered) {
      elements.push(
        <ol key={`b${bi}`} className="parsed-list">
          {lines.map((l, li) => <li key={li}>{parseLine(l.replace(/^\d+\.\s+/, ''), `b${bi}l${li}`)}</li>)}
        </ol>
      );
    } else {
      const children = [];
      lines.forEach((l, li) => {
        if (/^[•\-]\s+/.test(l)) {
          children.push(<ul key={`il${li}`} className="parsed-list"><li>{parseLine(l.replace(/^[•\-]\s+/, ''), `b${bi}il${li}`)}</li></ul>);
        } else if (/^\d+\.\s+/.test(l)) {
          children.push(<ol key={`il${li}`} className="parsed-list"><li>{parseLine(l.replace(/^\d+\.\s+/, ''), `b${bi}il${li}`)}</li></ol>);
        } else {
          children.push(<span key={`il${li}`}>{parseLine(l, `b${bi}il${li}`)}{li < lines.length - 1 ? ' ' : ''}</span>);
        }
      });
      elements.push(<p key={`b${bi}`} className="parsed-para">{children}</p>);
    }
  });

  return elements.length ? elements : <p className="parsed-para">{text}</p>;
}
