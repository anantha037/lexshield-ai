import { useEffect, useState } from 'react';
import { useStore } from '../store';

const BORDER_COLORS = {
  success: 'var(--c-low)',
  error: 'var(--c-high)',
  warning: 'var(--c-medium)'
};

function ToastItem({ toast }) {
  const [width, setWidth] = useState(100);
  const [exiting, setExiting] = useState(false);
  const duration = toast.type === 'error' ? 6000 : 4000;
  
  useEffect(() => {
    const frame = requestAnimationFrame(() => {
      setWidth(0);
    });

    const timer = setTimeout(() => {
      setExiting(true);
    }, duration - 200);

    return () => {
      cancelAnimationFrame(frame);
      clearTimeout(timer);
    };
  }, [toast.type, duration]);

  const color = BORDER_COLORS[toast.type] || 'var(--c-low)';

  return (
    <div 
      className={`toast ${exiting ? 'toast-exit' : 'toast-enter'}`} 
      style={{ 
        padding: '12px 20px', 
        background: 'var(--c-elevated)', 
        border: '1px solid var(--c-border)', 
        borderRadius: 'var(--r-md)', 
        fontSize: 14, 
        minWidth: 280, 
        maxWidth: 380, 
        position: 'relative', 
        overflow: 'hidden',
        borderLeft: `3px solid ${color}` 
      }}
    >
      {toast.msg}
      <div 
        style={{
          position: 'absolute',
          bottom: 0,
          left: 0,
          height: 2,
          background: color,
          width: `${width}%`,
          transition: `width ${duration}ms linear`
        }}
      />
    </div>
  );
}

export default function Toast() {
  const { toasts } = useStore();
  if (!toasts.length) return null;
  return (
    <div className="toast-container" style={{ position: 'fixed', top: 24, right: 24, zIndex: 9999, display: 'flex', flexDirection: 'column', gap: 8 }}>
      {toasts.map(t => (
        <ToastItem key={t.id} toast={t} />
      ))}
    </div>
  );
}
