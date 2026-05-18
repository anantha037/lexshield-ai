import { useState } from 'react';
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
      toast(`Welcome${res.user.full_name ? ', ' + res.user.full_name : ''}!`);
    } catch (err) {
      toast(err.message || 'Authentication failed', 'error');
    } finally { setSubmitting(false); }
  };

  const goAnon = () => {
    sessionStorage.setItem('lexshield_anon', '1');
    window.location.reload();
  };

  return (
    <div style={{ display: 'flex', height: '100vh' }}>
      <div style={{ flex: '0 0 55%', background: 'var(--c-bg)', display: 'flex', flexDirection: 'column', justifyContent: 'center', padding: '0 80px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 20 }}>
          <div style={{ width: 28, height: 1, background: 'var(--c-gold)' }} />
          <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.12em', color: 'var(--c-gold)', textTransform: 'uppercase' }}>INDIAN LEGAL INTELLIGENCE PLATFORM</div>
        </div>
        <h1 style={{ fontFamily: 'var(--f-head)', fontSize: 58, fontWeight: 700, lineHeight: 1.05, letterSpacing: '-0.02em', color: 'var(--c-text)' }}>
          Legal Intelligence<br />
          <span style={{ color: 'var(--c-gold)' }}>for Every Indian</span>
        </h1>
        <p style={{ fontSize: 16, color: 'var(--c-text2)', lineHeight: 1.7, maxWidth: 480, marginTop: 20 }}>
          Understand your rights. Analyze any legal document. Draft complaints.
          In English, Malayalam, and Hindi.
        </p>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginTop: 32 }}>
          {['Grounded in Indian Law', 'Document Analysis', 'Multilingual'].map((f, i) => (
            <div key={f} style={{ padding: '7px 16px', borderRadius: 99, border: '1px solid var(--c-border)', fontSize: 13, color: 'var(--c-text2)', background: 'var(--c-surface)', animation: `fadeIn 200ms ease forwards ${i * 60}ms`, opacity: 0 }}>
              {f}
            </div>
          ))}
        </div>
        <div style={{ fontSize: 12, color: 'var(--c-text3)', marginTop: 48 }}>
          Not a substitute for professional legal advice
        </div>
      </div>

      <div style={{ flex: '0 0 45%', background: 'var(--c-bg2)', borderLeft: '1px solid var(--c-border2)', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 48 }}>
        <div style={{ width: '100%', maxWidth: 380, background: 'var(--c-surface)', border: '1px solid var(--c-border)', borderRadius: 'var(--r-lg)', padding: 36 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, justifyContent: 'center', marginBottom: 24 }}>
            <IconScale color="var(--c-gold)" size={20} />
            <div style={{ fontFamily: 'var(--f-head)', fontWeight: 700, fontSize: 20, color: 'var(--c-text)' }}>LexShield<span style={{ color: 'var(--c-gold)' }}>AI</span></div>
          </div>

          <div style={{ display: 'flex', marginBottom: 28, borderBottom: '1px solid var(--c-border)', position: 'relative' }}>
            <div style={{ flex: 1, padding: 10, textAlign: 'center', fontSize: 13, fontWeight: 600, color: tab === 'login' ? 'var(--c-gold)' : 'var(--c-text3)', cursor: 'pointer', transition: 'color 150ms' }} onClick={() => setTab('login')}>Sign In</div>
            <div style={{ flex: 1, padding: 10, textAlign: 'center', fontSize: 13, fontWeight: 600, color: tab === 'register' ? 'var(--c-gold)' : 'var(--c-text3)', cursor: 'pointer', transition: 'color 150ms' }} onClick={() => setTab('register')}>Create Account</div>
            <div style={{ position: 'absolute', bottom: -1, left: tab === 'login' ? '0%' : '50%', width: '50%', height: 2, background: 'var(--c-gold)', transition: 'left 200ms ease' }} />
          </div>

          <form onSubmit={handleSubmit}>
            {tab === 'register' && (
              <div style={{ marginBottom: 16 }}>
                <label style={{ display: 'block', fontSize: 12, fontWeight: 500, color: 'var(--c-text2)', marginBottom: 6, letterSpacing: '0.02em' }}>Full Name</label>
                <input className="input" value={form.full_name} onChange={e => set('full_name', e.target.value)} placeholder="Anantha Krishnan K" />
                {errors.full_name && <div style={{ fontSize: 12, color: 'var(--c-high)', marginTop: 4 }}>{errors.full_name}</div>}
              </div>
            )}
            <div style={{ marginBottom: 16 }}>
              <label style={{ display: 'block', fontSize: 12, fontWeight: 500, color: 'var(--c-text2)', marginBottom: 6, letterSpacing: '0.02em' }}>Email</label>
              <input className="input" type="email" value={form.email} onChange={e => set('email', e.target.value)} placeholder="you@example.com" />
              {errors.email && <div style={{ fontSize: 12, color: 'var(--c-high)', marginTop: 4 }}>{errors.email}</div>}
            </div>
            <div style={{ marginBottom: 16 }}>
              <label style={{ display: 'block', fontSize: 12, fontWeight: 500, color: 'var(--c-text2)', marginBottom: 6, letterSpacing: '0.02em' }}>Password</label>
              <input className="input" type="password" value={form.password} onChange={e => set('password', e.target.value)} placeholder="••••••••" />
              {errors.password && <div style={{ fontSize: 12, color: 'var(--c-high)', marginTop: 4 }}>{errors.password}</div>}
            </div>
            {tab === 'register' && (
              <div style={{ marginBottom: 16 }}>
                <label style={{ display: 'block', fontSize: 12, fontWeight: 500, color: 'var(--c-text2)', marginBottom: 6, letterSpacing: '0.02em' }}>Confirm Password</label>
                <input className="input" type="password" value={form.confirm} onChange={e => set('confirm', e.target.value)} placeholder="••••••••" />
                {errors.confirm && <div style={{ fontSize: 12, color: 'var(--c-high)', marginTop: 4 }}>{errors.confirm}</div>}
              </div>
            )}
            <button className="btn-gold" type="submit" disabled={submitting} style={{ width: '100%', height: 44, marginTop: 8 }}>
              {submitting ? 'Please wait...' : (tab === 'login' ? 'Sign In' : 'Create Account')}
            </button>
          </form>

          <div style={{ textAlign: 'center', marginTop: 16 }}>
            <span style={{ fontSize: 12, color: 'var(--c-text3)', cursor: 'pointer', display: 'inline-block', transition: 'color 150ms' }} onClick={goAnon} onMouseEnter={e => e.target.style.color = 'var(--c-text2)'} onMouseLeave={e => e.target.style.color = 'var(--c-text3)'}>
              Continue without account →
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
