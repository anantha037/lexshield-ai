const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const TOKEN_KEY = 'lexshield_token';

export const getToken = () => localStorage.getItem(TOKEN_KEY);
export const setToken = (t) => localStorage.setItem(TOKEN_KEY, t);
export const clearToken = () => localStorage.removeItem(TOKEN_KEY);

/* BUG4: abort with reason; BUG5: null-response validation */
export async function request(path, opts = {}) {
  const token = getToken();
  const headers = { ...(opts.headers || {}) };
  if (token) headers['Authorization'] = `Bearer ${token}`;
  if (!(opts.body instanceof FormData) && !headers['Content-Type'])
    headers['Content-Type'] = 'application/json';

  const isAiCall = path.includes('/query') || path.includes('/analyze');
  const timeout = opts.timeout || (isAiCall ? 60000 : 10000);
  const ctrl = opts.signal ? null : new AbortController();
  const signal = opts.signal || ctrl?.signal;
  const timer = ctrl ? setTimeout(() => ctrl.abort(new Error('Request timed out')), timeout) : null;

  try {
    const res = await fetch(`${BASE}${path}`, { ...opts, headers, signal });
    if (timer) clearTimeout(timer);
    if (res.status === 401) { clearToken(); window.location.href = '/'; throw new Error('Unauthorized'); }
    if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.detail || `HTTP ${res.status}`); }
    return res.json();
  } catch (err) {
    if (timer) clearTimeout(timer);
    if (err.name === 'AbortError') return null; /* BUG4: silently swallow */
    throw err;
  }
}

// Auth
export const authRegister = (email, password, full_name) =>
  request('/api/v1/auth/register', { method: 'POST', body: JSON.stringify({ email, password, full_name }) });
export const authLogin = (email, password) =>
  request('/api/v1/auth/login', { method: 'POST', body: JSON.stringify({ email, password }) });
export const authMe = () => request('/api/v1/auth/me');

// Sessions
export const getSessions = (type = 'all') => request(`/api/v1/master/sessions?type=${type}`);
export const getSessionHistory = (sid) => request(`/api/v1/master/session/${sid}/history`);
export const deleteSession = (sid) => request(`/api/v1/master/session/${sid}`, { method: 'DELETE' });

// Query — BUG3: accepts external AbortSignal; BUG5: validate non-null answer
export async function sendQuery(query, session_id, language, signal) {
  const raw = await request('/api/v1/master/query', {
    method: 'POST',
    body: JSON.stringify({ query, session_id, language }),
    signal,
  });
  if (!raw) return null; /* aborted */
  if (raw.scope_status !== 'out_of_scope' && !raw.answer_text && !raw.answer && !raw.draft && !raw.summary)
    throw new Error('Empty response from legal engine. Please try again.');
  return raw;
}

// Document
export const analyzeDocument = (file) => {
  const fd = new FormData(); fd.append('file', file);
  return request('/api/v1/document/analyze', { method: 'POST', body: fd });
};

/* BUG6: always JSON.stringify + explicit Content-Type */
export const queryDocument = (doc_text, question, session_id, language) =>
  request('/api/v1/document/query', {
    method: 'POST',
    body: JSON.stringify({ doc_text, question, session_id, language }),
    headers: { 'Content-Type': 'application/json' },
  });

/* BUG7: persist doc analysis session without LLM call */
export const saveDocumentSession = (payload) =>
  request('/api/v1/document/save-session', {
    method: 'POST',
    body: JSON.stringify(payload),
    headers: { 'Content-Type': 'application/json' },
  });

export const checkHealth = () => request('/health');

// ── Response adapters ────────────────────────────────────────────────────────

export function adaptQueryResponse(raw) {
  if (!raw) return null;

  // Extract structured case law results from backend
  const caseLawResults = (
    raw.case_law_results ||
    raw.case_law_result?.results ||
    []
  ).map(c => ({
    title:    c.title    || '',
    court:    c.court    || '',
    date:     c.date     || '',
    citation: c.citation || '',
    headline: c.headline || '',
    url:      c.url      || '',
    summary:  c.summary  || '',
  }));

  return {
    answer: raw.answer_text || raw.answer || '',
    summary: raw.summary || '',
    intent: raw.intent || 'legal_query',
    riskLevel: raw.risk?.level || raw.risk_level || 'low',
    riskScore: raw.risk?.score || raw.risk_score || 0,
    riskFactors: raw.risk?.factors || raw.risk_factors || [],
    citations: (raw.citations || []).map((c, i) => ({
      source: c.source || '', section: c.section || '',
      sectionTitle: c.section_title || '', preview: c.preview || '',
      relevanceScore: c.relevance_score, sourceNumber: c.source_number || i + 1, era: c.era || '',
    })),
    sessionId: raw.session_id || '',
    confidence: raw.confidence || 0,
    mode: raw.mode || 'simple',
    sourcesConsulted: raw.sources_consulted || 0,
    synthesisNote: raw.synthesis_note || '',
    groundingWarning: raw.grounding_warning || '',
    citationStatus: raw.citation_status || 'unverified',
    scopeStatus: raw.scope_status || 'in_scope',
    scopeMessage: raw.scope_message || '',
    rewrittenQueries: raw.rewritten_queries || [],
    rerankerUsed: raw.reranker_used || false,
    keyClauses: raw.key_clauses || [],
    suggestions: raw.suggestions || [],
    draft: raw.draft || '',
    stage: raw.stage || null,
    outline: raw.outline || null,
    supportingDocuments: raw.supporting_documents || [],
    filingAuthority: raw.filing_authority || '',
    nextSteps: raw.next_steps || '',
    caseLawResults,
  };
}

export function adaptDocAnalysis(raw) {
  if (!raw) return null;
  return {
    filename: raw.filename || '',
    text: raw.text || '',
    wordCount: raw.word_count || 0,
    ocrUsed: raw.ocr_used || false,
    pageCount: raw.page_count || 0,
    docType: raw.classification?.label_name || raw.doc_type || 'unknown',
    confidence: raw.classification?.confidence || raw.confidence || 0,
    riskScore: raw.risk?.overall_score || raw.risk_score || 0,
    riskLevel: raw.risk?.risk_level || raw.risk_level || 'low',
    riskSummary: raw.risk?.summary || '',
    clauseRisks: raw.risk?.clause_risks || [],
    entities: raw.entities || {},
    rightsAlerts: raw.rights_alerts || [],
    warning: raw.warning || null,
  };
}
