'use client';
import { useState, useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import Nav from '../../components/Nav';
import { api } from '../../lib/api';

export default function AlertsPage() {
  const [mounted, setMounted] = useState(false);
  const [alerts, setAlerts] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<'all' | 'high' | 'moderate'>('all');
  const router = useRouter();

  useEffect(() => {
    setMounted(true);
    if (!localStorage.getItem('token')) { router.push('/'); return; }
    fetchAlerts();
  }, []);

  const fetchAlerts = async () => {
    setLoading(true);
    try {
      // Get high-risk patients
      const res = await api('/patients?page_size=100&sort_by=last_name&sort_dir=asc');
      const patients = res.patients || [];
      
      const alertList: any[] = [];
      patients.forEach((p: any) => {
        if (!p.risk_scores) return;
        Object.entries(p.risk_scores).forEach(([model, score]: [string, any]) => {
          const prob = Number(score);
          if (prob > 0.4) {
            alertList.push({
              patient_id: p.patient_id,
              name: `${p.first_name} ${p.last_name}`,
              age: p.age,
              archetype: p.adherence_archetype,
              model: model.replace(/_/g, ' '),
              probability: prob,
              risk_level: prob > 0.7 ? 'high' : 'moderate',
            });
          }
        });
      });

      alertList.sort((a, b) => b.probability - a.probability);
      setAlerts(alertList);
    } catch (e) { console.error(e); }
    setLoading(false);
  };

  if (!mounted) return null;

  const filtered = filter === 'all' ? alerts : alerts.filter(a => a.risk_level === filter);

  return (
    <div>
      <Nav />
      <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
          <h1 style={{ fontSize: 24, fontWeight: 700 }}>
            Risk Alerts <span style={{ fontSize: 14, color: 'var(--tx3)', fontWeight: 400 }}>
              ({filtered.length} alerts)
            </span>
          </h1>
          <div style={{ display: 'flex', gap: 8 }}>
            {(['all', 'high', 'moderate'] as const).map(f => (
              <button key={f} className={`btn ${filter === f ? 'btn-primary' : 'btn-secondary'}`}
                style={{ fontSize: 13, padding: '6px 14px', textTransform: 'capitalize' }}
                onClick={() => setFilter(f)}>
                {f === 'all' ? 'All' : f === 'high' ? '🔴 High' : '🟡 Moderate'}
              </button>
            ))}
          </div>
        </div>

        {loading ? (
          <div style={{ padding: 40, textAlign: 'center', color: 'var(--tx3)' }}>Loading alerts...</div>
        ) : filtered.length === 0 ? (
          <div className="card" style={{ padding: 40, textAlign: 'center', color: 'var(--tx3)' }}>
            No alerts match the current filter.
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {filtered.map((a, i) => (
              <div key={`${a.patient_id}-${a.model}-${i}`} className="card"
                style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer', padding: '14px 20px' }}
                onClick={() => router.push(`/patient/${a.patient_id}`)}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
                  <div style={{
                    width: 10, height: 10, borderRadius: '50%',
                    background: a.risk_level === 'high' ? 'var(--red)' : 'var(--yellow)',
                    boxShadow: a.risk_level === 'high' ? '0 0 8px rgba(239,68,68,0.5)' : 'none',
                  }} />
                  <div>
                    <div style={{ fontWeight: 600 }}>{a.name}</div>
                    <div style={{ fontSize: 12, color: 'var(--tx3)' }}>
                      {a.age}y · <span style={{ textTransform: 'capitalize' }}>{a.archetype}</span> adherence
                    </div>
                  </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: 13, color: 'var(--tx3)', textTransform: 'capitalize' }}>{a.model}</div>
                  <div style={{
                    fontSize: 18, fontWeight: 700,
                    color: a.risk_level === 'high' ? 'var(--red)' : 'var(--yellow)',
                  }}>
                    {(a.probability * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
