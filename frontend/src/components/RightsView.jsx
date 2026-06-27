import { useState } from 'react';
import { motion } from 'framer-motion';
import { useStore } from '../store';
import { IconShield, IconHome, IconBriefcase, IconCart, IconPhone } from '../icons';

const CATEGORIES = [
  { id: 'tenant',   Icon: IconHome,      label: 'Tenant Rights',  desc: 'Rent, eviction, deposits',   law: 'TPA 1882' },
  { id: 'employee', Icon: IconBriefcase, label: 'Employee Rights', desc: 'Wages, PF, termination',     law: 'ID Act 1947' },
  { id: 'consumer', Icon: IconCart,      label: 'Consumer Rights', desc: 'Refunds, defects, services', law: 'CPA 2019' },
  { id: 'women',    Icon: IconShield,    label: "Women's Rights",  desc: 'DV, dowry, maintenance',     law: 'PWDVA 2005' },
  { id: 'bail',     Icon: IconPhone,     label: 'Bail & Arrest',   desc: 'Arrest rights, bail',        law: 'BNSS 2023' },
];

const RIGHTS_DATA = {
  tenant: [
    { law: 'Sec 106, TPA 1882',      title: 'Protection from Arbitrary Eviction',  body: 'A landlord cannot arbitrarily evict a tenant without serving a valid notice to quit. Typically 15 days for monthly tenancies or 6 months for agricultural/manufacturing leases.' },
    { law: 'Art 21, Constitution',   title: 'Right to Essential Services',          body: 'Landlords are prohibited from cutting off water and electricity to force vacation. This violates the fundamental right to life under Article 21.' },
    { law: 'Rent Control Acts',      title: 'Right to Fair Rent',                   body: 'Tenants may challenge exorbitant rent hikes under state Rent Control Acts. Rent above the registered agreement or statutory limits is legally contestable.' },
    { law: 'Contract Act 1872',      title: 'Return of Security Deposit',           body: 'Upon vacating, tenants are entitled to a full refund of security deposit minus valid deductions for unpaid rent or documented damage.' },
  ],
  employee: [
    { law: 'Payment of Wages Act',   title: 'Right to Timely Wages',               body: 'Wages must be paid within 7–10 days of the wage period ending. Withholding wages without valid statutory reasons is a criminal offence.' },
    { law: 'EPF Act 1952',           title: 'Provident Fund Rights',                body: 'Establishments with 20+ employees must contribute to PF. You can file a complaint with EPFO if contributions are withheld.' },
    { law: 'ID Act 1947',            title: 'Protection from Wrongful Termination', body: 'Workers in 100+ employee establishments need government approval for termination. All workers are entitled to retrenchment notice and compensation.' },
    { law: 'POSH Act 2013',          title: 'Protection from Sexual Harassment',    body: 'Every workplace with 10+ employees must have an Internal Complaints Committee. Complaints must be filed within 3 months of the incident.' },
  ],
  consumer: [
    { law: 'CPA 2019 Sec 2(9)',      title: 'Right to Information',                body: 'Consumers have the right to know the quality, quantity, price and standard of goods or services before purchase.' },
    { law: 'CPA 2019 Sec 35',        title: 'Right to File Complaint',              body: 'File in District Forum (up to ₹1Cr), State Commission (up to ₹10Cr), or National Commission for larger disputes.' },
    { law: 'CPA 2019 Sec 84',        title: 'Product Liability',                    body: 'Manufacturers, sellers and service providers are liable for harm from defective products or deficient services without proof of negligence.' },
    { law: 'CPA 2019 Sec 47',        title: 'Right to Mediation',                  body: 'Consumers may resolve disputes via mediation before court proceedings — faster, cheaper, and binding when both parties agree.' },
  ],
  women: [
    { law: 'PWDVA 2005 Sec 12',      title: 'Protection Orders under DV Act',       body: 'Magistrates can grant Protection, Residence, Monetary Relief, Custody, and Compensation Orders to women facing domestic violence.' },
    { law: 'BNS Sec 85 / IPC 498A',  title: 'Matrimonial Cruelty',                  body: 'Cruelty by husband or his relatives — including dowry harassment — is cognizable and non-bailable, punishable up to 3 years.' },
    { law: 'HMA Sec 24',             title: 'Right to Maintenance',                 body: 'Either spouse with insufficient independent income may apply for interim maintenance during pendency of matrimonial proceedings.' },
    { law: 'Maternity Benefit Act',  title: 'Maternity Leave Rights',               body: 'Women in 10+ employee establishments get 26 weeks paid maternity leave for first two children, 12 weeks thereafter.' },
  ],
  bail: [
    { law: 'BNSS 2023 Sec 35',       title: 'Right to Know Grounds of Arrest',     body: 'Every arrested person must be immediately informed of grounds of arrest and has the right to consult a legal practitioner from the moment of arrest.' },
    { law: 'BNSS 2023 Sec 47',       title: 'Right to Bail in Bailable Offences',  body: 'Bail is a matter of right for bailable offences. For non-bailable offences, bail is at the court\'s discretion.' },
    { law: 'Art 22(2), Constitution', title: 'Right to be Produced Before Magistrate', body: 'Any arrested person must be produced before the nearest Magistrate within 24 hours of arrest, excluding travel time.' },
    { law: 'BNSS 2023 Sec 187',      title: 'Default Bail',                         body: 'If police fail to file a chargesheet within 60 or 90 days (depending on offence), the accused is entitled to statutory default bail.' },
  ],
};

export default function RightsView() {
  const { setActiveView, setPrefillInput, setActiveSession, setChatMessages } = useStore();
  const [selected, setSelected] = useState(null);
  const [question, setQuestion] = useState('');

  const askAbout = async (questionText) => {
    // BUG2 fix: get the new session_id BEFORE setting prefillInput so ChatView
    // always sends the query under the correct (fresh) session, not the stale one.
    let newSessionId = 'LX-' + Math.random().toString(36).substr(2,8).toUpperCase();
    try {
      const res = await fetch('http://localhost:8000/api/v1/master/session/new');
      const data = await res.json();
      newSessionId = data.session_id;
    } catch (e) { /* keep fallback id */ }

    // Clear messages and set session BEFORE navigating / setting prefill
    if (setChatMessages) setChatMessages([]);
    setActiveSession(newSessionId);
    setPrefillInput(questionText);
    setActiveView('chat');
  };

  const selectedCat = CATEGORIES.find(c => c.id === selected);
  const cards = selected ? (RIGHTS_DATA[selected] || []) : [];

  return (
    <div style={{ overflowY: 'auto', height: '100%' }} className="view-enter">
      <div className="view-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
          <div style={{ width: 28, height: 1, background: 'var(--c-gold)' }} />
          <span style={{ fontFamily: 'var(--f-mono)', fontSize: 10, fontWeight: 700, letterSpacing: '0.2em', color: 'var(--c-gold)', textTransform: 'uppercase' }}>
            LexShield · Legal Intelligence
          </span>
        </div>
        <h1 style={{ fontFamily: 'var(--f-head)', fontSize: 36, fontWeight: 700, color: 'var(--c-text)', letterSpacing: '-0.01em', margin: 0 }}>Know Your Rights</h1>
        <p style={{ fontSize: 15, color: 'var(--c-text2)', marginTop: 8, margin: 0 }}>Explore statutory protections and constitutional guarantees across key legal domains.</p>
      </div>

      <div style={{ padding: '32px 40px' }}>
        <div style={{ display: 'flex', gap: 10, overflowX: 'auto', paddingBottom: 4 }}>
          {CATEGORIES.map(({ id, Icon, label }) => (
            <div key={id} id={`rights-cat-${id}`}
              style={{
                display: 'flex', alignItems: 'center', gap: 8, padding: '12px 20px', background: selected === id ? 'var(--c-gold-dim)' : 'var(--c-surface)',
                border: `1px solid ${selected === id ? 'var(--c-gold)' : 'var(--c-border)'}`, borderRadius: 'var(--r-md)', cursor: 'pointer',
                whiteSpace: 'nowrap', fontSize: 13, fontWeight: 600, color: selected === id ? 'var(--c-gold)' : 'var(--c-text2)', transition: 'all 150ms',
                position: 'relative'
              }}
              onMouseEnter={(e) => {
                if (selected !== id) { e.currentTarget.style.borderColor = 'rgba(196,149,42,0.3)'; e.currentTarget.style.color = 'var(--c-text)'; }
              }}
              onMouseLeave={(e) => {
                if (selected !== id) { e.currentTarget.style.borderColor = 'var(--c-border)'; e.currentTarget.style.color = 'var(--c-text2)'; }
              }}
              onClick={(e) => {
                setSelected(id);
                e.currentTarget.style.animation = 'none';
                void e.currentTarget.offsetWidth;
                e.currentTarget.style.animation = 'rightsPulse 400ms ease-out forwards';
              }}
            >
              {selected === id && (
                <motion.span
                  layoutId="rights-tab-indicator"
                  style={{
                    position: 'absolute',
                    bottom: -1,
                    left: 0,
                    right: 0,
                    height: 2,
                    background: 'var(--c-gold)',
                    borderRadius: 1,
                  }}
                  transition={{ type: 'spring', stiffness: 400, damping: 32 }}
                />
              )}
              <Icon size={16} />
              {label}
            </div>
          ))}
        </div>

        {selected && cards.length > 0 && (
          <div className="fade-in">
            <motion.div
              key={selected}
              className="rights-grid"
              initial="hidden"
              animate="visible"
              variants={{ visible: { transition: { staggerChildren: 0.08 } } }}
              style={{ marginTop: 28 }}
            >
              {cards.map((r, i) => (
                <motion.div
                  key={i}
                  variants={{ hidden: { opacity: 0, y: 16 }, visible: { opacity: 1, y: 0 } }}
                  whileHover={{ y: -3 }}
                  className="right-card"
                  style={{ background: 'var(--c-surface)', border: '1px solid var(--c-border)', borderRadius: 'var(--r-md)', padding: '28px', transition: 'border-color 150ms', position: 'relative', overflow: 'hidden', borderLeft: '3px solid var(--c-gold)' }}
                >
                  <style>{`
                    .right-card:hover { border-color: rgba(196,149,42,0.25); }
                    .right-card::before { content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 2px; background: var(--c-gold); transform: scaleY(0); transform-origin: top; transition: transform 150ms ease; }
                    .right-card:hover::before { transform: scaleY(1); }
                    .right-card-ask { font-size: 13px; font-weight: 600; color: var(--c-gold); background: none; border: none; cursor: pointer; padding: 0; transition: color 150ms; }
                    .right-card-ask:hover { color: var(--c-gold2); text-decoration: underline; }
                  `}</style>
                  <span className="citation-badge" style={{ marginBottom: 12, display: 'inline-block' }}>{r.law}</span>
                  <div style={{ fontFamily: 'var(--f-head)', fontSize: 17, fontWeight: 600, color: 'var(--c-text)', marginBottom: 8, lineHeight: 1.3 }}>{r.title}</div>
                  <div style={{ fontSize: 13, color: 'var(--c-text2)', lineHeight: 1.65, marginBottom: 16 }}>{r.body}</div>
                  <button className="right-card-ask" onClick={() => askAbout(`Explain the law: ${r.title} — ${r.body.slice(0, 100)}`)}>
                    Ask about this →
                  </button>
                </motion.div>
              ))}
            </motion.div>

            <div style={{ marginTop: 32, paddingTop: 24, borderTop: '1px solid var(--c-border2)' }}>
              <label style={{ fontSize: 13, color: 'var(--c-text2)', marginBottom: 8, display: 'block' }}>
                Have a specific question about {selectedCat?.label}?
              </label>
              <div style={{ display: 'flex', gap: 10 }}>
                <input className="input" style={{ flex: 1 }} value={question} onChange={e => setQuestion(e.target.value)}
                  placeholder={`e.g. Can my landlord evict me without notice in Kerala?`}
                  onKeyDown={e => { if (e.key === 'Enter' && question.trim()) askAbout(question.trim()); }} />
                <button className="btn-gold" style={{ padding: '10px 20px' }} onClick={() => question.trim() && askAbout(question.trim())}>
                  Ask
                </button>
              </div>
            </div>
          </div>
        )}

        {!selected && (
          <div style={{ height: 300, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 12 }}>
            <div style={{ opacity: 0.15 }}><IconShield color="var(--c-gold)" size={48} /></div>
            <div style={{ fontFamily: 'var(--f-head)', fontSize: 24, fontWeight: 600, color: 'var(--c-text)' }}>Select a category above</div>
            <div style={{ fontSize: 14, color: 'var(--c-text2)' }}>Choose a legal domain to explore your rights and protections.</div>
          </div>
        )}
      </div>
      <style>{`
        @keyframes rightsPulse {
          0% { box-shadow: 0 0 0 8px var(--c-gold-dim); }
          100% { box-shadow: 0 0 0 0 transparent; }
        }
      `}</style>
    </div>
  );
}
