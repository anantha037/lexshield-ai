import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useStore } from '../store';
import { resetPassword } from '../api';
import { IconScale } from '../icons';

export default function ResetPassword() {
  const { toast } = useStore();
  const [token, setToken] = useState('');
  const [form, setForm] = useState({ password: '', confirm: '' });
  const [errors, setErrors] = useState({});
  const [submitting, setSubmitting] = useState(false);
  const [success, setSuccess] = useState(false);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const t = params.get('token');
    if (t) setToken(t);
  }, []);

  const set = (k, v) => { setForm(f => ({ ...f, [k]: v })); setErrors(e => ({ ...e, [k]: '' })); };

  const validate = () => {
    const e = {};
    if (form.password.length < 6) e.password = 'Min 6 characters';
    if (form.password !== form.confirm) e.confirm = 'Passwords do not match';
    setErrors(e);
    return !Object.keys(e).length;
  };

  const handleSubmit = async (ev) => {
    ev.preventDefault();
    if (!token) {
      toast('No reset token found in URL.', 'error');
      return;
    }
    if (!validate()) return;
    setSubmitting(true);
    try {
      const res = await resetPassword(token, form.password);
      toast(res.message);
      setSuccess(true);
    } catch (err) {
      toast(err.message || 'Failed to reset password', 'error');
    } finally { setSubmitting(false); }
  };

  return (
    <div className="landing-root" style={{ justifyContent: 'center', background: 'var(--c-bg)' }}>
      <div style={{ width: '100%', maxWidth: 440, padding: 20 }}>
        <motion.div
          className="landing-auth-card"
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
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

          <h2 style={{ fontSize: 18, color: 'var(--c-text)', textAlign: 'center', marginBottom: 24 }}>Reset Password</h2>

          {success ? (
            <div style={{ textAlign: 'center' }}>
              <p style={{ fontSize: 14, color: 'var(--c-text2)', marginBottom: 24 }}>
                Your password has been reset successfully.
              </p>
              <button className="btn-gold" onClick={() => window.location.href = '/'} style={{ width: '100%', height: 44, border: 'none', borderRadius: 'var(--r-md)', cursor: 'pointer', fontWeight: 600 }}>
                Return to Sign In
              </button>
            </div>
          ) : (
            <form onSubmit={handleSubmit}>
              <div style={{ marginBottom: 16 }}>
                <label style={{ display: 'block', fontSize: 12, fontWeight: 500, color: 'var(--c-text2)', marginBottom: 6, letterSpacing: '0.02em' }}>New Password</label>
                <input className={`input${errors.password ? ' error' : ''}`} type="password" value={form.password} onChange={e => set('password', e.target.value)} placeholder="••••••••" />
                {errors.password && <div style={{ fontSize: 12, color: 'var(--c-high)', marginTop: 4 }}>{errors.password}</div>}
              </div>
              <div style={{ marginBottom: 24 }}>
                <label style={{ display: 'block', fontSize: 12, fontWeight: 500, color: 'var(--c-text2)', marginBottom: 6, letterSpacing: '0.02em' }}>Confirm Password</label>
                <input className={`input${errors.confirm ? ' error' : ''}`} type="password" value={form.confirm} onChange={e => set('confirm', e.target.value)} placeholder="••••••••" />
                {errors.confirm && <div style={{ fontSize: 12, color: 'var(--c-high)', marginTop: 4 }}>{errors.confirm}</div>}
              </div>
              <motion.button className="btn-gold" type="submit" disabled={submitting} whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.97 }} style={{ width: '100%', height: 44, border: 'none', borderRadius: 'var(--r-md)', cursor: 'pointer', fontWeight: 600 }}>
                {submitting ? 'Resetting...' : 'Reset Password'}
              </motion.button>
              <div style={{ textAlign: 'center', marginTop: 16 }}>
                <span style={{ fontSize: 12, color: 'var(--c-text3)', cursor: 'pointer', display: 'inline-block', transition: 'color 150ms' }} onClick={() => window.location.href = '/'} onMouseEnter={e => e.target.style.color = 'var(--c-text2)'} onMouseLeave={e => e.target.style.color = 'var(--c-text3)'}>
                  Cancel
                </span>
              </div>
            </form>
          )}
        </motion.div>
      </div>
    </div>
  );
}
