'use client';
import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Nav from '../../components/Nav';
import { api, getRole } from '../../lib/api';

export default function DashboardPage() {
  const [mounted, setMounted] = useState(false);
  const [stats, setStats] = useState<any>(null);
  const router = useRouter();

  useEffect(() => {
    setMounted(true);
    if (!localStorage.getItem('token')) { router.push('/'); return; }
    api('/stats').then(setStats).catch(() => {});
  }, []);

  if (!mounted) return null;

  return (
    <div>
      <Nav />
      <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
        <h1 style={{ fontSize: 24, fontWeight: 700, marginBottom: 20 }}>Dashboard</h1>

        {stats && (
          <>
            {/* Stat cards */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16, marginBottom: 24 }}>
              <StatCard label="Total Patients" value={stats.total_patients?.toLocaleString()} color="var(--accent)" />
              <StatCard label="Avg Age" value={stats.avg_age} color="var(--accent2)" />
              <StatCard label="ER Visits (30d)" value={stats.er_visits_30d} color="var(--red)" />
              <StatCard label="Adherence Groups" value={Object.keys(stats.by_archetype || {}).length} color="var(--green)" />
            </div>

            {/* Archetype distribution */}
            <div className="card" style={{ marginBottom: 24 }}>
              <h2 style={{ fontSize: 16, fontWeight: 600, marginBottom: 12 }}>Adherence Archetypes</h2>
              <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
                {Object.entries(stats.by_archetype || {}).map(([arch, count]: [string, any]) => (
                  <div key={arch} style={{ padding: '8px 16px', background: 'var(--bg3)', borderRadius: 8 }}>
                    <div style={{ fontSize: 13, color: 'var(--tx3)', textTransform: 'capitalize' }}>{arch}</div>
                    <div style={{ fontSize: 20, fontWeight: 700 }}>{count.toLocaleString()}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Model metrics */}
            {stats.model_metrics && (
              <div className="card">
                <h2 style={{ fontSize: 16, fontWeight: 600, marginBottom: 12 }}>Model Performance</h2>
                <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
                  {Object.entries(stats.model_metrics).map(([name, info]: [string, any]) => (
                    <div key={name} style={{ padding: '8px 16px', background: 'var(--bg3)', borderRadius: 8 }}>
                      <div style={{ fontSize: 13, color: 'var(--tx3)' }}>{name.replace(/_/g, ' ')}</div>
                      <div style={{ fontSize: 20, fontWeight: 700, color: info.auc > 0.75 ? 'var(--green)' : 'var(--yellow)' }}>
                        AUC {info.auc.toFixed(3)}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function StatCard({ label, value, color }: { label: string; value: any; color: string }) {
  return (
    <div className="card" style={{ textAlign: 'center' }}>
      <div style={{ fontSize: 13, color: 'var(--tx3)', marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 28, fontWeight: 700, color }}>{value}</div>
    </div>
  );
}
