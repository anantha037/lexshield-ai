import { useState } from 'react';
import { motion } from 'framer-motion';
import { useStore } from '../store';
import { authRegister, authLogin } from '../api';
import { IconScale } from '../icons';

export default function Landing() {
  const { login, toast } = useStore();
  const [tab, setTab] = useState('login');
  const [form, setForm] = useState({ email: '', password: '', full_name: '', confirm: '' });
  const [errors, setErrors] = useState({});
  const [submitting, setSubmitting] = useState(false);

  const set = (k, v) => { setForm(f => ({ ...f, [k]: v })); setErrors(e => ({ ...e, [k]: '' })); };

  const validate = () => {
    const e = {};
    if (!form.email.includes('@')) e.email = 'Valid email required';
    if (form.password.length < 6) e.password = 'Min 6 characters';
    if (tab === 'register') {
      if (form.full_name.trim().length < 2) e.full_name = 'Name required';
      if (form.password !== form.confirm) e.confirm = 'Passwords do not match';
    }
    setErrors(e);
    return !Object.keys(e).length;
  };

  const handleSubmit = async (ev) => {
    ev.preventDefault();
    if (!validate()) return;
    setSubmitting(true);
    try {
      const res = tab === 'login'
        ? await authLogin(form.email, form.password)
        : await authRegister(form.email, form.password, form.full_name);
      login(res.access_token, res.user);
      sessionStorage.setItem('lexshield_welcome', res.user.full_name || 'User');
      window.location.reload();
    } catch (err) {
      toast(err.message || 'Authentication failed', 'error');
    } finally { setSubmitting(false); }
  };

  const goAnon = () => {
    sessionStorage.setItem('lexshield_anon', '1');
    window.location.reload();
  };

  return (
    <div className="landing-root">
      <div className="landing-hero-panel" style={{ background: 'var(--c-bg)', position: 'relative', overflow: 'hidden' }}>
        <motion.div
          animate={{
            background: [
              'radial-gradient(circle at 18% 25%, rgba(196,149,42,0.08), transparent 45%), radial-gradient(circle at 80% 70%, rgba(196,149,42,0.04), transparent 50%)',
              'radial-gradient(circle at 25% 60%, rgba(196,149,42,0.06), transparent 45%), radial-gradient(circle at 75% 20%, rgba(196,149,42,0.05), transparent 50%)',
              'radial-gradient(circle at 18% 25%, rgba(196,149,42,0.08), transparent 45%), radial-gradient(circle at 80% 70%, rgba(196,149,42,0.04), transparent 50%)',
            ]
          }}
          transition={{ duration: 22, repeat: Infinity, ease: 'easeInOut' }}
          style={{
            position: 'absolute', inset: '-10%',
            filter: 'blur(30px)',
            pointerEvents: 'none',
            zIndex: 0,
          }}
        />
        <div style={{
          position: 'absolute', inset: 0,
          pointerEvents: 'none', zIndex: 0, overflow: 'hidden'
        }}>
          {['⚖️', '📜', '🏛️', '⚖️', '📋'].map((icon, i) => (
            <motion.div
              key={i}
              style={{
                position: 'absolute',
                fontSize: 32,
                opacity: 0.04,
                left: `${[10, 25, 70, 80, 50][i]}%`,
                top: `${[15, 60, 25, 70, 85][i]}%`,
              }}
              animate={{ y: [0, -15, 0], rotate: [0, 3, 0] }}
              transition={{
                duration: [8, 10, 12, 9, 11][i],
                repeat: Infinity,
                ease: 'easeInOut',
                delay: i * 1.5,
              }}
            >
              {icon}
            </motion.div>
          ))}
        </div>
        <div style={{ position: 'relative', zIndex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 20 }}>
            <div style={{ width: 28, height: 1, background: 'var(--c-gold)' }} />
            <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.12em', color: 'var(--c-gold)', textTransform: 'uppercase' }}>INDIAN LEGAL INTELLIGENCE PLATFORM</div>
          </div>
          <h1 style={{ fontFamily: 'var(--f-head)', fontWeight: 700, lineHeight: 1.05, letterSpacing: '-0.02em', color: 'var(--c-text)' }}>
            <div style={{ display: 'flex', flexWrap: 'nowrap', gap: '16px' }}>
              {['Legal', 'Intelligence'].map((w, i) => (
                <motion.span
                  key={w}
                  style={{ display: 'inline-block' }}
                  initial={{ opacity: 0, y: 24 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{
                    duration: 0.7,
                    delay: 0.2 + i * 0.12,
                    ease: [0.22, 1, 0.36, 1]
                  }}
                >
                  {w}
                </motion.span>
              ))}
            </div>
            <div style={{ display: 'flex', flexWrap: 'nowrap', gap: '16px', fontStyle: 'italic', color: 'var(--c-gold)' }}>
              {['for', 'Every', 'Indian'].map((w, i) => (
                <motion.span
                  key={w}
                  initial={{ opacity: 0, y: 24 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{
                    duration: 0.7,
                    delay: 0.5 + i * 0.12,
                    ease: [0.22, 1, 0.36, 1]
                  }}
                >
                  {w}
                </motion.span>
              ))}
            </div>
          </h1>
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1.0, duration: 0.6 }}
            style={{ fontSize: 16, color: 'var(--c-text2)', lineHeight: 1.7, maxWidth: 480, marginTop: 20 }}
          >
            Understand your rights. Analyze any legal document. Draft complaints.
            In English, Malayalam, and Hindi.
          </motion.p>
          <motion.div
            initial="hidden"
            animate="visible"
            variants={{ visible: { transition: { staggerChildren: 0.1, delayChildren: 0.8 } } }}
            style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginTop: 32 }}
          >
            {['Grounded in Indian Law', 'Document Analysis', 'Multilingual'].map((f, i) => (
                <motion.div
                  key={f}
                  variants={{ hidden: { opacity: 0, y: 8 }, visible: { opacity: 1, y: 0 } }}
                  className="feature-pill"
                >
                {f}
              </motion.div>
            ))}
          </motion.div>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1.5 }}
            style={{ fontSize: 12, color: 'var(--c-text3)', marginTop: 48 }}
          >
            Not a substitute for professional legal advice
          </motion.div>
        </div>
      </div>

      <div className="landing-auth-panel">
        <motion.div
          className="landing-auth-card"
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2, ease: [0.22, 1, 0.36, 1] }}
          style={{ background: 'var(--c-surface)', border: '1px solid var(--c-border)', borderRadius: 'var(--r-lg)', padding: 36, transition: 'border-color 300ms, box-shadow 300ms' }}
          onHoverStart={e => {
            e.currentTarget.style.borderColor = 'rgba(196,149,42,0.35)';
            e.currentTarget.style.boxShadow = '0 0 40px rgba(196,149,42,0.08)';
          }}
          onHoverEnd={e => {
            e.currentTarget.style.borderColor = 'var(--c-border)';
            e.currentTarget.style.boxShadow = 'none';
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, justifyContent: 'center', marginBottom: 24 }}>
            <IconScale color="var(--c-gold)" size={20} />
            <div style={{ fontFamily: 'var(--f-head)', fontWeight: 700, fontSize: 20, color: 'var(--c-text)' }}>LexShield<span style={{ color: 'var(--c-gold)' }}>AI</span></div>
          </div>

          <div style={{ display: 'flex', marginBottom: 28, borderBottom: '1px solid var(--c-border)', position: 'relative' }}>
            <div
              style={{ 
                position: 'absolute', bottom: -1, height: 2, background: 'var(--c-gold)',
                width: '50%',
                left: tab === 'login' ? '0%' : '50%',
                transition: 'left 300ms cubic-bezier(0.4, 0, 0.2, 1)'
              }}
            />
            {[
              { key: 'login', label: 'Sign In' },
              { key: 'register', label: 'Create Account' }
            ].map((t) => (
              <div 
                key={t.key}
                style={{ flex: 1, padding: 10, textAlign: 'center', fontSize: 13, fontWeight: 600, color: tab === t.key ? 'var(--c-gold)' : 'var(--c-text3)', cursor: 'pointer', transition: 'color 150ms', position: 'relative', zIndex: 1 }} 
                onClick={() => setTab(t.key)}
                onMouseEnter={e => { if (tab !== t.key) e.currentTarget.style.color = 'var(--c-text2)'; }}
                onMouseLeave={e => { if (tab !== t.key) e.currentTarget.style.color = 'var(--c-text3)'; }}
              >
                {t.label}
              </div>
            ))}
          </div>

          <form onSubmit={handleSubmit}>
            {tab === 'register' && (
              <div style={{ marginBottom: 16 }}>
                <label style={{ display: 'block', fontSize: 12, fontWeight: 500, color: 'var(--c-text2)', marginBottom: 6, letterSpacing: '0.02em' }}>Full Name</label>
                <input className={`input${errors.full_name ? ' error' : ''}`} value={form.full_name} onChange={e => set('full_name', e.target.value)} placeholder="Anantha Krishnan K" />
                {errors.full_name && <div style={{ fontSize: 12, color: 'var(--c-high)', marginTop: 4 }}>{errors.full_name}</div>}
              </div>
            )}
            <div style={{ marginBottom: 16 }}>
              <label style={{ display: 'block', fontSize: 12, fontWeight: 500, color: 'var(--c-text2)', marginBottom: 6, letterSpacing: '0.02em' }}>Email</label>
              <input className={`input${errors.email ? ' error' : ''}`} type="email" value={form.email} onChange={e => set('email', e.target.value)} placeholder="you@example.com" />
              {errors.email && <div style={{ fontSize: 12, color: 'var(--c-high)', marginTop: 4 }}>{errors.email}</div>}
            </div>
            <div style={{ marginBottom: 16 }}>
              <label style={{ display: 'block', fontSize: 12, fontWeight: 500, color: 'var(--c-text2)', marginBottom: 6, letterSpacing: '0.02em' }}>Password</label>
              <input className={`input${errors.password ? ' error' : ''}`} type="password" value={form.password} onChange={e => set('password', e.target.value)} placeholder="••••••••" />
              {errors.password && <div style={{ fontSize: 12, color: 'var(--c-high)', marginTop: 4 }}>{errors.password}</div>}
            </div>
            {tab === 'register' && (
              <div style={{ marginBottom: 16 }}>
                <label style={{ display: 'block', fontSize: 12, fontWeight: 500, color: 'var(--c-text2)', marginBottom: 6, letterSpacing: '0.02em' }}>Confirm Password</label>
                <input className={`input${errors.confirm ? ' error' : ''}`} type="password" value={form.confirm} onChange={e => set('confirm', e.target.value)} placeholder="••••••••" />
                {errors.confirm && <div style={{ fontSize: 12, color: 'var(--c-high)', marginTop: 4 }}>{errors.confirm}</div>}
              </div>
            )}
            <motion.button className="btn-gold" type="submit" disabled={submitting} whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.97 }} style={{ width: '100%', height: 44, marginTop: 8 }}>
              {submitting ? 'Please wait...' : (tab === 'login' ? 'Sign In' : 'Create Account')}
            </motion.button>
          </form>

          <div style={{ textAlign: 'center', marginTop: 16 }}>
            <span style={{ fontSize: 12, color: 'var(--c-text3)', cursor: 'pointer', display: 'inline-block', transition: 'color 150ms, letter-spacing 150ms' }} onClick={goAnon} onMouseEnter={e => {
              e.target.style.color = 'var(--c-gold)';
              e.target.style.letterSpacing = '0.02em';
            }} onMouseLeave={e => {
              e.target.style.color = 'var(--c-text3)';
              e.target.style.letterSpacing = 'normal';
            }}>
              Continue without account →
            </span>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
