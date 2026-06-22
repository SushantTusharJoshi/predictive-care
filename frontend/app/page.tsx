'use client';
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { api, login as doLogin } from '../lib/api';

export default function LoginPage() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  async function handleLogin(e?: React.FormEvent) {
    if (e) e.preventDefault();
    setLoading(true);
    setError('');
    try {
      const res = await api('/auth/login', {
        method: 'POST',
        body: JSON.stringify({ username, password }),
      });
      doLogin(res.token, res.role, res.name);
      router.push('/dashboard');
    } catch (err: any) {
      setError('Invalid credentials');
    } finally {
      setLoading(false);
    }
  }

  const quickLogins = [
    { u: 'admin', label: 'Admin' },
    { u: 'physician', label: 'Physician' },
    { u: 'nurse', label: 'Nurse' },
    { u: 'coordinator', label: 'Coordinator' },
  ];

  return (
    <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <div className="card" style={{ width: 400, textAlign: 'center' }}>
        <div style={{ fontSize: 28, fontWeight: 700, marginBottom: 4 }}>
          <span style={{ color: 'var(--accent)' }}>Predictive</span>Care
        </div>
        <div style={{ color: 'var(--tx3)', fontSize: 14, marginBottom: 4 }}>
          Intelligent Patient Health Forecasting
        </div>
        <div style={{ color: 'var(--accent2)', fontSize: 11, marginBottom: 24, fontWeight: 600 }}>
          HIPAA-Compliant · XGBoost/LightGBM · SHAP Explainability
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <input placeholder="Username" value={username} onChange={e => setUsername(e.target.value)} />
          <input placeholder="Password" type="password" value={password} onChange={e => setPassword(e.target.value)}
                 onKeyDown={e => e.key === 'Enter' && handleLogin()} />
          {error && <div style={{ color: 'var(--red)', fontSize: 13 }}>{error}</div>}
          <button className="btn btn-primary" onClick={handleLogin} disabled={loading}>
            {loading ? 'Signing in...' : 'Sign In'}
          </button>
        </div>

        <div style={{ marginTop: 20, borderTop: '1px solid var(--border)', paddingTop: 16 }}>
          <div style={{ fontSize: 12, color: 'var(--tx3)', marginBottom: 8 }}>Quick Login (Demo)</div>
          <div style={{ display: 'flex', gap: 8, justifyContent: 'center', flexWrap: 'wrap' }}>
            {quickLogins.map(q => (
              <button key={q.u} className="btn btn-secondary" style={{ fontSize: 12, padding: '4px 12px' }}
                onClick={async () => {
                  setUsername(q.u); setPassword(q.u);
                  try {
                    const res = await api('/auth/login', {
                      method: 'POST', body: JSON.stringify({ username: q.u, password: q.u }),
                    });
                    doLogin(res.token, res.role, res.name);
                    router.push('/dashboard');
                  } catch { setError('Failed'); }
                }}>
                {q.label}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
