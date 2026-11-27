'use client';
import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Nav from '../../components/Nav';
import { api, getRole } from '../../lib/api';

// PC-F5: clinician approval queue. Nothing is scheduled until a real clinician acts here.
// Demo accounts see the queue; the backend refuses their actions and the buttons are disabled.
export default function ApprovalsPage() {
  const [mounted, setMounted] = useState(false);
  const [recs, setRecs] = useState<any[]>([]);
  const [canApprove, setCanApprove] = useState(false);
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState('');
  const router = useRouter();

  useEffect(() => {
    setMounted(true);
    if (!localStorage.getItem('token')) { router.push('/'); return; }
    if (!['admin', 'physician', 'coordinator'].includes(getRole())) { router.push('/dashboard'); return; }
    load();
  }, []);

  const load = async () => {
    setLoading(true);
    try {
      const res = await api('/schedule/pending');
      setRecs(res.recommendations || []);
      setCanApprove(!!res.can_approve);
    } catch { setMessage('Could not load the approval queue.'); }
    setLoading(false);
  };

  const act = async (id: string, action: 'approved' | 'rejected') => {
    setMessage('');
    try {
      await api(`/schedule/${id}/action`, { method: 'POST', body: JSON.stringify({ action }) });
      setRecs(r => r.filter(x => x.recommendation_id !== id));
      setMessage(`Recommendation ${action}.`);
    } catch { setMessage('Action refused.'); }
  };

  if (!mounted) return null;

  return (
    <div>
      <Nav />
      <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
        <h1 style={{ fontSize: 24, fontWeight: 700, marginBottom: 8 }}>Approval queue</h1>
        <p style={{ color: 'var(--tx3)', fontSize: 14, marginBottom: 16 }}>
          Model-suggested follow-ups wait here until a clinician approves or rejects them. Nothing is booked automatically.
        </p>
        {!canApprove && !loading && (
          <div className="card" style={{ marginBottom: 16, borderLeft: '3px solid var(--accent2)', fontSize: 14 }}>
            Demo account: you can view the queue, but approving and rejecting need a real clinician account.
          </div>
        )}
        {message && <div className="card" style={{ marginBottom: 16, fontSize: 14 }}>{message}</div>}
        {loading ? (
          <div style={{ padding: 40, textAlign: 'center', color: 'var(--tx3)' }}>Loading...</div>
        ) : recs.length === 0 ? (
          <div className="card" style={{ color: 'var(--tx3)' }}>No pending recommendations.</div>
        ) : (
          <div className="card" style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 14 }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border)' }}>
                  {['Patient', 'Prediction', 'Probability', 'Visit', 'Window', 'Created', ''].map(h => (
                    <th key={h} style={{ padding: '8px 12px', textAlign: 'left', color: 'var(--tx3)', fontSize: 12, fontWeight: 600 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {recs.map(r => (
                  <tr key={r.recommendation_id} style={{ borderBottom: '1px solid var(--border)' }}>
                    <td style={{ padding: '8px 12px' }}>
                      <a href={`/patient/${r.patient_id}`} style={{ color: 'var(--accent)' }}>{r.patient_name}</a>
                    </td>
                    <td style={{ padding: '8px 12px' }}>{String(r.prediction_type).replace(/_/g, ' ')}</td>
                    <td style={{ padding: '8px 12px' }}>{(Number(r.predicted_probability) * 100).toFixed(1)}%</td>
                    <td style={{ padding: '8px 12px' }}>{String(r.visit_type).replace(/_/g, ' ')}</td>
                    <td style={{ padding: '8px 12px' }}>{r.window_start} to {r.window_end}</td>
                    <td style={{ padding: '8px 12px', color: 'var(--tx3)' }}>{String(r.created_at).slice(0, 16)}</td>
                    <td style={{ padding: '8px 12px', whiteSpace: 'nowrap' }}>
                      <button className="btn btn-primary" disabled={!canApprove} onClick={() => act(r.recommendation_id, 'approved')}
                        style={{ fontSize: 12, padding: '4px 10px', marginRight: 6, opacity: canApprove ? 1 : 0.5 }}>Approve</button>
                      <button className="btn btn-secondary" disabled={!canApprove} onClick={() => act(r.recommendation_id, 'rejected')}
                        style={{ fontSize: 12, padding: '4px 10px', opacity: canApprove ? 1 : 0.5 }}>Reject</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
