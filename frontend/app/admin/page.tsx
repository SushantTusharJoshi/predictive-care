'use client';
import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Nav from '../../components/Nav';
import { api, getRole } from '../../lib/api';
import { BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, Legend } from 'recharts';

export default function AdminPage() {
  const [mounted, setMounted] = useState(false);
  const [tab, setTab] = useState('bias');
  const [auditLog, setAuditLog] = useState<any[]>([]);
  const [metrics, setMetrics] = useState<any>(null);
  const [biasData, setBiasData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  useEffect(() => {
    setMounted(true);
    if (!localStorage.getItem('token')) { router.push('/'); return; }
    if (getRole() !== 'admin') { router.push('/dashboard'); return; }
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const [auditRes, metricsRes, statsRes] = await Promise.all([
        api('/audit').catch(() => []),
        api('/models/metrics').catch(() => null),
        api('/stats').catch(() => null),
      ]);
      setAuditLog(Array.isArray(auditRes) ? auditRes : auditRes?.events || []);
      setMetrics(metricsRes);

      // Build bias audit data from stats
      if (statsRes?.bias_audit) {
        setBiasData(statsRes.bias_audit);
      } else if (statsRes?.by_archetype) {
        // Simulate bias audit from available data
        const archetypes = statsRes.by_archetype || {};
        const total = Object.values(archetypes).reduce((s: number, v: any) => s + Number(v), 0);
        const biasRows = Object.entries(archetypes).map(([arch, count]: [string, any]) => ({
          group: arch,
          count: Number(count),
          pct: ((Number(count) / total) * 100).toFixed(1),
        }));
        setBiasData({ archetype_distribution: biasRows, total_patients: total });
      }
    } catch (e) { console.error(e); }
    setLoading(false);
  };

  if (!mounted) return null;

  return (
    <div>
      <Nav />
      <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
        <h1 style={{ fontSize: 24, fontWeight: 700, marginBottom: 20 }}>Admin Panel</h1>

        <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid var(--border)', paddingBottom: 4 }}>
          {[
            { key: 'bias', label: 'Bias Audit' },
            { key: 'models', label: 'Model Performance' },
            { key: 'audit', label: 'Audit Log' },
          ].map(t => (
            <button key={t.key} className={`nav-link ${tab === t.key ? 'active' : ''}`}
              onClick={() => setTab(t.key)}>
              {t.label}
            </button>
          ))}
        </div>

        {loading ? (
          <div style={{ padding: 40, textAlign: 'center', color: 'var(--tx3)' }}>Loading...</div>
        ) : (
          <>
            {tab === 'bias' && <BiasAuditTab data={biasData} />}
            {tab === 'models' && <ModelMetricsTab metrics={metrics} />}
            {tab === 'audit' && <AuditLogTab events={auditLog} />}
          </>
        )}
      </div>
    </div>
  );
}

function BiasAuditTab({ data }: { data: any }) {
  if (!data) return <div className="card" style={{ color: 'var(--tx3)' }}>No bias audit data available.</div>;

  const archDist = data.archetype_distribution || [];
  const chartData = archDist.map((r: any) => ({ name: r.group, patients: r.count, pct: parseFloat(r.pct) }));

  return (
    <div>
      <div className="card" style={{ marginBottom: 16, borderLeft: '3px solid var(--accent2)' }}>
        <h3 style={{ fontSize: 14, fontWeight: 600, color: 'var(--accent2)', marginBottom: 8 }}>Fairness Overview</h3>
        <p style={{ color: 'var(--tx2)', lineHeight: 1.6, fontSize: 14 }}>
          This audit examines the distribution of adherence archetypes across the patient population
          ({data.total_patients?.toLocaleString()} patients). Significant imbalances may indicate
          systemic biases in data collection, model predictions, or care delivery patterns.
        </p>
      </div>

      {/* Archetype Distribution Chart */}
      {chartData.length > 0 && (
        <div className="card" style={{ marginBottom: 16 }}>
          <h3 style={{ fontSize: 14, fontWeight: 600, color: 'var(--tx3)', marginBottom: 12 }}>Adherence Archetype Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="name" tick={{ fontSize: 12, fill: 'var(--tx3)' }} />
              <YAxis tick={{ fontSize: 11, fill: 'var(--tx3)' }} />
              <Tooltip contentStyle={{ background: 'var(--bg2)', border: '1px solid var(--border)', borderRadius: 8 }} />
              <Legend />
              <Bar dataKey="patients" fill="var(--accent)" name="Patient Count" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Table */}
      <div className="card">
        <h3 style={{ fontSize: 14, fontWeight: 600, color: 'var(--tx3)', marginBottom: 12 }}>Breakdown</h3>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 14 }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)' }}>
              {['Archetype', 'Count', 'Percentage', 'Representation'].map(h => (
                <th key={h} style={{ padding: '10px 16px', textAlign: 'left', color: 'var(--tx3)', fontSize: 12, fontWeight: 600 }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {archDist.map((r: any) => {
              const pct = parseFloat(r.pct);
              const expected = 20; // 5 archetypes = 20% each if uniform
              const deviation = pct - expected;
              return (
                <tr key={r.group} style={{ borderBottom: '1px solid var(--border)' }}>
                  <td style={{ padding: '10px 16px', fontWeight: 500, textTransform: 'capitalize' }}>{r.group}</td>
                  <td style={{ padding: '10px 16px' }}>{Number(r.count).toLocaleString()}</td>
                  <td style={{ padding: '10px 16px' }}>{r.pct}%</td>
                  <td style={{ padding: '10px 16px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <div style={{ width: 80, height: 8, background: 'var(--bg3)', borderRadius: 4, overflow: 'hidden' }}>
                        <div style={{ width: `${Math.min(pct * 2.5, 100)}%`, height: '100%', borderRadius: 4,
                          background: Math.abs(deviation) > 10 ? 'var(--red)' : Math.abs(deviation) > 5 ? 'var(--yellow)' : 'var(--green)'
                        }} />
                      </div>
                      <span style={{ fontSize: 12, color: deviation > 0 ? 'var(--green)' : 'var(--red)' }}>
                        {deviation > 0 ? '+' : ''}{deviation.toFixed(1)}%
                      </span>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ModelMetricsTab({ metrics }: { metrics: any }) {
  if (!metrics) return <div className="card" style={{ color: 'var(--tx3)' }}>No model metrics available.</div>;

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: 16 }}>
      {Object.entries(metrics).map(([name, info]: [string, any]) => (
        <div key={name} className="card">
          <h3 style={{ fontSize: 16, fontWeight: 600, marginBottom: 16, textTransform: 'capitalize' }}>
            {name.replace(/_/g, ' ')}
          </h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            <MetricCard label="AUC-ROC" value={info.auc?.toFixed(4)} good={info.auc > 0.75} />
            <MetricCard label="F1 Score" value={info.f1?.toFixed(4)} good={info.f1 > 0.5} />
            <MetricCard label="Precision" value={info.precision?.toFixed(4)} good={info.precision > 0.5} />
            <MetricCard label="Recall" value={info.recall?.toFixed(4)} good={info.recall > 0.5} />
          </div>
          {info.feature_importances && (
            <div style={{ marginTop: 16 }}>
              <h4 style={{ fontSize: 13, color: 'var(--tx3)', marginBottom: 8 }}>Top Features</h4>
              {info.feature_importances.slice(0, 5).map((f: any, i: number) => (
                <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0', fontSize: 13 }}>
                  <span style={{ color: 'var(--tx2)' }}>{f.feature?.replace(/_/g, ' ') || f[0]}</span>
                  <span style={{ fontWeight: 600, color: 'var(--accent)' }}>
                    {(f.importance || f[1])?.toFixed(4)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function MetricCard({ label, value, good }: { label: string; value: any; good: boolean }) {
  return (
    <div style={{ padding: '8px 12px', background: 'var(--bg3)', borderRadius: 8, textAlign: 'center' }}>
      <div style={{ fontSize: 12, color: 'var(--tx3)' }}>{label}</div>
      <div style={{ fontSize: 20, fontWeight: 700, color: good ? 'var(--green)' : 'var(--yellow)' }}>
        {value || '—'}
      </div>
    </div>
  );
}

function AuditLogTab({ events }: { events: any[] }) {
  const [search, setSearch] = useState('');
  const filtered = events.filter(e =>
    !search || JSON.stringify(e).toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div>
      <div style={{ marginBottom: 16 }}>
        <input placeholder="Search audit log..." value={search}
          onChange={e => setSearch(e.target.value)}
          style={{ width: '100%', maxWidth: 400 }} />
      </div>

      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)' }}>
              {['Timestamp', 'User', 'Role', 'Action', 'Resource', 'Details'].map(h => (
                <th key={h} style={{ padding: '10px 12px', textAlign: 'left', color: 'var(--tx3)', fontSize: 11, fontWeight: 600 }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 ? (
              <tr><td colSpan={6} style={{ padding: 30, textAlign: 'center', color: 'var(--tx3)' }}>No audit events found.</td></tr>
            ) : filtered.slice(0, 100).map((e: any, i: number) => (
              <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                <td style={{ padding: '8px 12px', fontSize: 12, color: 'var(--tx3)' }}>{e.timestamp?.slice(0, 19)}</td>
                <td style={{ padding: '8px 12px', fontWeight: 500 }}>{e.user || e.username}</td>
                <td style={{ padding: '8px 12px' }}>{e.role}</td>
                <td style={{ padding: '8px 12px' }}>
                  <span style={{ padding: '2px 8px', borderRadius: 4, fontSize: 12, fontWeight: 600,
                    background: e.action?.includes('view') ? 'rgba(59,130,246,0.15)' : 'rgba(168,85,247,0.15)',
                    color: e.action?.includes('view') ? 'var(--accent)' : 'var(--accent2)',
                  }}>{e.action}</span>
                </td>
                <td style={{ padding: '8px 12px', fontSize: 12 }}>{e.resource}</td>
                <td style={{ padding: '8px 12px', fontSize: 12, color: 'var(--tx3)', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {typeof e.details === 'object' ? JSON.stringify(e.details) : e.details || '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
