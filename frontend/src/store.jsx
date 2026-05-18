import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { getToken, setToken, clearToken, authMe, getSessions } from './api';

const Ctx = createContext(null);
export const useStore = () => useContext(Ctx);

export function StoreProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeView, setActiveView] = useState('chat');
  const [activeSession, setActiveSession] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [currentDoc, setCurrentDoc] = useState(null);
  const [lastResponse, setLastResponse] = useState(null);
  const [toasts, setToasts] = useState([]);
  // BUG1: chatMessages in store so Sidebar can inject history
  const [chatMessages, setChatMessages] = useState([]);
  // BUG9: prefillInput lets RightsView trigger a ChatView auto-submit
  const [prefillInput, setPrefillInput] = useState('');
  // Global language (shared across views)
  const [language, setLanguage] = useState('en');
  // FEATURE2: draft category selection
  const [draftCategory, setDraftCategory] = useState(null);

  useEffect(() => {
    const token = getToken();
    if (token) {
      authMe().then(u => setUser(u)).catch(() => clearToken()).finally(() => setLoading(false));
    } else { setLoading(false); }
  }, []);

  const login = useCallback((token, userData) => { setToken(token); setUser(userData); }, []);
  const logout = useCallback(() => {
    clearToken(); setUser(null); setActiveSession(null); setSessions([]);
    setChatMessages([]); setDraftCategory(null);
  }, []);

  const refreshSessions = useCallback(async (type = 'all') => {
    if (!getToken()) return;
    try { const s = await getSessions(type); setSessions(s); } catch {}
  }, []);

  const toast = useCallback((msg, type = 'success') => {
    const id = Date.now();
    setToasts(t => [...t, { id, msg, type }]);
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), 4000);
  }, []);

  return (
    <Ctx.Provider value={{
      user, loading, login, logout,
      activeView, setActiveView,
      activeSession, setActiveSession,
      sessions, setSessions, refreshSessions,
      currentDoc, setCurrentDoc,
      lastResponse, setLastResponse,
      toasts, toast,
      chatMessages, setChatMessages,
      prefillInput, setPrefillInput,
      language, setLanguage,
      draftCategory, setDraftCategory,
    }}>{children}</Ctx.Provider>
  );
}
