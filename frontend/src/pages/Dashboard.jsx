import { useEffect, useState, useRef, useCallback } from 'react';
import { useStore } from '../store';
import Sidebar from '../components/Sidebar';
import ContextPanel from '../components/ContextPanel';
import ChatView from '../components/ChatView';
import DocumentView from '../components/DocumentView';
import DraftView from '../components/DraftView';
import RightsView from '../components/RightsView';

const VIEWS = {
  chat: ChatView,
  document: DocumentView,
  draft: DraftView,
  rights: RightsView,
};

const LS_SW = 'lx_sidebar_w';
const LS_CW = 'lx_context_w';
const SIDEBAR_MIN = 180, SIDEBAR_MAX = 400, SIDEBAR_DEFAULT = 248;
const CONTEXT_MIN = 200, CONTEXT_MAX = 480, CONTEXT_DEFAULT = 280;

function DragHandle({ onDragStart }) {
  const [dragging, setDragging] = useState(false);
  return (
    <div
      style={{
        width: 4, cursor: 'col-resize', background: dragging ? 'var(--c-gold)' : 'var(--c-border2)',
        flexShrink: 0, transition: 'background 150ms'
      }}
      onMouseEnter={(e) => { if (!dragging) e.currentTarget.style.background = 'var(--c-gold)'; }}
      onMouseLeave={(e) => { if (!dragging) e.currentTarget.style.background = 'var(--c-border2)'; }}
      onMouseDown={(e) => { setDragging(true); onDragStart(e); }}
      onMouseUp={() => setDragging(false)}
    />
  );
}

export default function Dashboard() {
  const { activeView, refreshSessions, user } = useStore();
  
  useEffect(() => { if (user) refreshSessions(); }, [user, refreshSessions]);

  const [sidebarWidth, setSidebarWidth] = useState(() => {
    const saved = localStorage.getItem(LS_SW);
    return saved ? parseInt(saved, 10) : SIDEBAR_DEFAULT;
  });
  const [contextWidth, setContextWidth] = useState(() => {
    const saved = localStorage.getItem(LS_CW);
    return saved ? parseInt(saved, 10) : CONTEXT_DEFAULT;
  });

  const [windowWidth, setWindowWidth] = useState(window.innerWidth);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    const handleResize = () => setWindowWidth(window.innerWidth);
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const dragState = useRef(null);

  const startSidebarDrag = useCallback((e) => {
    e.preventDefault();
    dragState.current = { type: 'sidebar', startX: e.clientX, startW: sidebarWidth };
    const onMove = (ev) => {
      if (!dragState.current) return;
      const delta = ev.clientX - dragState.current.startX;
      const newW = Math.min(SIDEBAR_MAX, Math.max(SIDEBAR_MIN, dragState.current.startW + delta));
      setSidebarWidth(newW);
      localStorage.setItem(LS_SW, String(newW));
    };
    const onUp = () => { dragState.current = null; document.removeEventListener('mousemove', onMove); document.removeEventListener('mouseup', onUp); };
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  }, [sidebarWidth]);

  const startContextDrag = useCallback((e) => {
    e.preventDefault();
    dragState.current = { type: 'context', startX: e.clientX, startW: contextWidth };
    const onMove = (ev) => {
      if (!dragState.current) return;
      const delta = dragState.current.startX - ev.clientX; 
      const newW = Math.min(CONTEXT_MAX, Math.max(CONTEXT_MIN, dragState.current.startW + delta));
      setContextWidth(newW);
      localStorage.setItem(LS_CW, String(newW));
    };
    const onUp = () => { dragState.current = null; document.removeEventListener('mousemove', onMove); document.removeEventListener('mouseup', onUp); };
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  }, [contextWidth]);

  // Handle activeView change on mobile to auto-close sidebar
  useEffect(() => {
    setMobileMenuOpen(false);
  }, [activeView]);

  const View = VIEWS[activeView] || ChatView;
  
  const showSidebar = windowWidth >= 768;
  const showContext = windowWidth >= 1100;

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden', background: 'var(--c-bg)' }}>
      {/* Mobile Overlay */}
      {!showSidebar && (
        <div 
          style={{
            position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)',
            opacity: mobileMenuOpen ? 1 : 0, pointerEvents: mobileMenuOpen ? 'auto' : 'none',
            transition: 'opacity 280ms ease', zIndex: 40
          }}
          onClick={() => setMobileMenuOpen(false)}
        />
      )}

      {/* Sidebar */}
      <div 
        style={{
          width: showSidebar ? sidebarWidth : 260,
          minWidth: showSidebar ? sidebarWidth : 260,
          background: 'var(--c-bg2)',
          position: showSidebar ? 'relative' : 'fixed',
          top: 0, bottom: 0, left: 0,
          transform: showSidebar ? 'none' : `translateX(${mobileMenuOpen ? '0%' : '-100%'})`,
          transition: showSidebar ? 'none' : 'transform 280ms ease',
          zIndex: 50,
          borderRight: '1px solid var(--c-border2)'
        }}
      >
        <Sidebar />
      </div>

      {showSidebar && <DragHandle onDragStart={startSidebarDrag} />}

      {/* Main Content */}
      <div className="view-container" style={{ position: 'relative' }}>
        {!showSidebar && (
          <button 
            style={{ position: 'absolute', top: 24, left: 16, zIndex: 30, background: 'var(--c-surface)', border: '1px solid var(--c-border)', borderRadius: 'var(--r-sm)', padding: 6, color: 'var(--c-text)', cursor: 'pointer' }}
            onClick={() => setMobileMenuOpen(true)}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="3" y1="12" x2="21" y2="12"></line>
              <line x1="3" y1="6" x2="21" y2="6"></line>
              <line x1="3" y1="18" x2="21" y2="18"></line>
            </svg>
          </button>
        )}
        <View />
      </div>

      {/* Context Panel */}
      {showContext && <DragHandle onDragStart={startContextDrag} />}
      {showContext && (
        <div style={{ width: contextWidth, minWidth: contextWidth }}>
          <ContextPanel />
        </div>
      )}
    </div>
  );
}
