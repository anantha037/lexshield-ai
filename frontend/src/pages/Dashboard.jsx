import { useEffect, useState, useRef, useCallback } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { useStore } from '../store';
import { requestVerification } from '../api';
import Sidebar from '../components/Sidebar';
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
const SIDEBAR_MIN = 180, SIDEBAR_MAX = 400, SIDEBAR_DEFAULT = 248;

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
  const { activeView, refreshSessions, user, toast } = useStore();
  
  useEffect(() => { 
    if (user) refreshSessions(); 
    const welcome = sessionStorage.getItem('lexshield_welcome');
    if (welcome) {
      toast(`Welcome, ${welcome}!`);
      sessionStorage.removeItem('lexshield_welcome');
    }
  }, [user, refreshSessions, toast]);

  const [sidebarWidth, setSidebarWidth] = useState(() => {
    const saved = localStorage.getItem(LS_SW);
    return saved ? parseInt(saved, 10) : SIDEBAR_DEFAULT;
  });

  const [windowWidth, setWindowWidth] = useState(window.innerWidth);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    const handleResize = () => setWindowWidth(window.innerWidth);
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const [bannerDismissed, setBannerDismissed] = useState(false);
  const [resending, setResending] = useState(false);

  const handleResendVerification = async () => {
    setResending(true);
    try {
      const res = await requestVerification();
      toast(res.message);
    } catch (err) {
      toast(err.message, 'error');
    } finally {
      setResending(false);
    }
  };

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

  // Handle activeView change on mobile to auto-close sidebar
  useEffect(() => {
    setMobileMenuOpen(false);
  }, [activeView]);

  const View = VIEWS[activeView] || ChatView;
  
  const showSidebar = windowWidth >= 768;

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
      <div className="view-container" style={{ flex: 1, position: 'relative', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {!showSidebar && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            padding: '12px 16px',
            borderBottom: '1px solid var(--c-border2)',
            background: 'var(--c-bg2)',
            flexShrink: 0
          }}>
            <button
              style={{
                background: 'var(--c-surface)',
                border: '1px solid var(--c-border)',
                borderRadius: 'var(--r-sm)',
                padding: 6,
                color: 'var(--c-text)',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}
              onClick={() => setMobileMenuOpen(true)}
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="3" y1="12" x2="21" y2="12"></line>
                <line x1="3" y1="6" x2="21" y2="6"></line>
                <line x1="3" y1="18" x2="21" y2="18"></line>
              </svg>
            </button>
            <span style={{ fontFamily: 'var(--f-head)', fontSize: 15, fontWeight: 700, color: 'var(--c-text)' }}>
              Lex<span style={{ color: 'var(--c-gold)' }}>Shield</span> AI
            </span>
          </div>
        )}

        {user && !user.is_email_verified && !bannerDismissed && (
          <div style={{ background: 'rgba(196,149,42,0.1)', borderBottom: '1px solid rgba(196,149,42,0.2)', padding: '12px 16px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 16 }}>
            <div style={{ fontSize: 13, color: 'var(--c-text)', display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ color: 'var(--c-gold)' }}>⚠️</span>
              Please verify your email address.
            </div>
            <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
              <button 
                onClick={handleResendVerification} 
                disabled={resending}
                style={{ background: 'none', border: 'none', color: 'var(--c-gold)', fontSize: 13, fontWeight: 600, cursor: 'pointer' }}
              >
                {resending ? 'Sending...' : 'Resend link'}
              </button>
              <button 
                onClick={() => setBannerDismissed(true)} 
                style={{ background: 'none', border: 'none', color: 'var(--c-text3)', fontSize: 16, cursor: 'pointer' }}
              >
                &times;
              </button>
            </div>
          </div>
        )}

        <AnimatePresence mode="wait" initial={false}>
          <motion.div
            key={activeView}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
            style={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}
          >
            <View />
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}
