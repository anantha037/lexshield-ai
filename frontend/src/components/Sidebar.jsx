import { useState, useEffect } from 'react';
import { useStore } from '../store';
import { deleteSession, getSessionHistory } from '../api';
import { IconChat, IconDocument, IconDraft, IconShield, IconScale, IconPlus, IconTrash, IconLogout, IconGavel } from '../icons';

function timeAgo(ts) {
  if (!ts) return '';
  const s = Math.floor(Date.now() / 1000 - ts);
  if (s < 60) return 'now';
  if (s < 3600) return Math.floor(s / 60) + 'm';
  if (s < 86400) return Math.floor(s / 3600) + 'h';
  return Math.floor(s / 86400) + 'd';
}

const NAV = [
  { id: 'chat',     Icon: IconChat,     label: 'Legal Q&A' },
  { id: 'document', Icon: IconDocument, label: 'Document Analysis' },
  { id: 'draft',    Icon: IconDraft,    label: 'Draft Complaint' },
  { id: 'caselaw',  Icon: IconGavel,    label: 'Case Law' },
  { id: 'rights',   Icon: IconShield,   label: 'Know Your Rights' },
];

function SessionSection({ label, sessions, activeSession, onSelect, onDelete }) {
  if (!sessions || sessions.length === 0) return null;
  
  return (
    <div>
      <div className="history-label">{label}</div>
      {sessions.map((s, i) => (
        <div
          key={s.session_id}
          className={`session-item ${activeSession === s.session_id ? 'active' : ''}`}
          onClick={() => onSelect(s.session_id)}
          style={{ animationDelay: `${i * 40}ms` }}
        >
          <div style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden', flex: 1 }}>
            <span style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
              {s.first_message || 'New Session'}
            </span>
            <span style={{ fontSize: 10, color: 'var(--c-text3)' }}>
              {timeAgo(s.last_active || s.created_at)}
            </span>
          </div>
          <button className="session-del" onClick={(e) => { e.stopPropagation(); onDelete(s.session_id); }}>
            <IconTrash />
          </button>
        </div>
      ))}
    </div>
  );
}

export default function Sidebar() {
  const { 
    user, logout, activeView, setActiveView, 
    sessions, activeSession, setActiveSession, 
    refreshSessions, toast, setChatMessages, 
    setDraftCategory, setCurrentDoc,
    caseLawMode, setCaseLawMode,
  } = useStore();

  useEffect(() => {
    refreshSessions(activeView);
  }, [activeView, refreshSessions]);

  const handleLogoClick = () => {
    if (activeView === 'chat') {
      setChatMessages([]);
      setActiveSession(null);
    } else if (activeView === 'document') {
      setCurrentDoc(null);
    } else if (activeView === 'draft') {
      setDraftCategory(null);
    } else if (activeView === 'rights') {
      // Assuming there's a deselect category mechanism, but not explicitly mapped in store.
      // Might just do nothing special or clear category if added later.
    }
  };

  const handleNavClick = (id) => {
    if (id === 'caselaw') {
      // Case Law is rendered inside ChatView with case law mode on
      setCaseLawMode(true);
      setChatMessages([]);
      setActiveSession(null);
      setActiveView('chat');
      return;
    }
    // Clicking any non-caselaw nav disables case law mode
    if (caseLawMode) setCaseLawMode(false);
    if (activeView === id) {
      if (id === 'chat') {
        setChatMessages([]);
        setActiveSession(null);
      } else if (id === 'document') {
        setCurrentDoc(null);
      } else if (id === 'draft') {
        setDraftCategory(null);
      }
    } else {
      setActiveView(id);
    }
  };

  const handleNewChat = () => {
    setActiveSession(null);
    setChatMessages([]);
    setDraftCategory(null);
    setCaseLawMode(false);
    setActiveView('chat');
  };

  const handleDelete = async (sid) => {
    try {
      await deleteSession(sid);
      toast('Session deleted');
      refreshSessions();
      if (activeSession === sid) {
        setActiveSession(null);
        setChatMessages([]);
      }
    } catch {
      toast('Failed to delete', 'error');
    }
  };

  const handleSelect = async (sid) => {
    if (sid === activeSession) return;
    try {
      const data = await getSessionHistory(sid);
      const hist = data?.history || data;
      if (Array.isArray(hist)) {
        setChatMessages(hist.map(h => ({ role: h.role, content: h.content, intent: h.intent, ts: h.ts })));
      }
      setActiveSession(sid);
      setActiveView('chat');
    } catch {
      toast('Failed to load history', 'error');
    }
  };

  const getFilteredSessions = () => {
    if (activeView === 'chat') {
      return {
        label: 'RECENT QUERIES',
        sessions: sessions.filter(s => {
          const msg = (s.first_message || '').toLowerCase();
          return !msg.startsWith('document:') && !msg.startsWith('draft:') && !msg.includes('drafting') && !msg.includes('i need help drafting');
        })
      };
    }
    if (activeView === 'document') {
      return {
        label: 'RECENT DOCUMENTS',
        sessions: sessions.filter(s => (s.first_message || '').startsWith('document:'))
      };
    }
    if (activeView === 'draft') {
      return {
        label: 'RECENT DRAFTS',
        sessions: sessions.filter(s => {
          const msg = (s.first_message || '').toLowerCase();
          return msg.startsWith('draft:') || msg.includes('drafting') || msg.startsWith('i need help drafting') || msg.includes('complaint');
        })
      };
    }
    if (activeView === 'rights') {
      return {
        label: 'RECENT QUERIES',
        sessions: sessions.slice(0, 5) // last 5 sessions of any type
      };
    }
    // caselaw shares sessions with chat (both use master/query)
    return { label: 'RECENT QUERIES', sessions: sessions.slice(0, 5) };
  };

  const { label, sessions: filteredSessions } = getFilteredSessions();
  const initials = user?.full_name ? user.full_name.split(' ').map(w => w[0]).join('').toUpperCase().slice(0, 2) : '?';

  return (
    <div className="sidebar">
      <div className="sidebar-top">
        <div className="sidebar-logo" onClick={handleLogoClick}>
          <IconScale />
          <div>
            <div className="sidebar-logo-text">Lex<span>Shield</span></div>
            <div className="sidebar-subtitle">Legal Intelligence</div>
          </div>
        </div>
        
        {user && (
          <div className="sidebar-user">
            <div className="sidebar-avatar">{initials}</div>
            <div style={{ minWidth: 0, overflow: 'hidden' }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--c-text)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {user.full_name}
              </div>
              <div style={{ fontSize: 11, color: 'var(--c-text3)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {user.email}
              </div>
            </div>
          </div>
        )}

        <div style={{ marginTop: 16 }}>
          <button className="btn-gold" style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }} onClick={handleNewChat}>
            <IconPlus /> New Chat
          </button>
        </div>
      </div>

      <div className="sidebar-nav">
        {NAV.map(({ id, Icon, label }) => (
          <div 
            key={id} 
            className={`nav-item ${activeView === id ? 'active' : ''} ${id === 'caselaw' && caseLawMode && activeView === 'chat' ? 'active' : ''}`}
            onClick={() => handleNavClick(id)}
          >
            <Icon />
            {label}
          </div>
        ))}
        
        <hr className="nav-divider" />
        
        <SessionSection 
          label={label} 
          sessions={filteredSessions} 
          activeSession={activeSession} 
          onSelect={handleSelect} 
          onDelete={handleDelete} 
        />
      </div>

      {user && (
        <div className="sidebar-bottom">
          <button className="signout-btn" onClick={logout}>
            <IconLogout /> Sign Out
          </button>
        </div>
      )}
    </div>
  );
}
