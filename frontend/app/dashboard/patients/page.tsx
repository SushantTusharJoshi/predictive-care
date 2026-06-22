'use client';
import { useState, useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import Nav from '../../../components/Nav';
import { api } from '../../../lib/api';

export default function PatientsPage() {
  const [mounted, setMounted] = useState(false);
  const [data, setData] = useState<any>({ patients: [], total: 0, page: 1, total_pages: 1 });
  const [query, setQuery] = useState('');
  const [page, setPage] = useState(1);
  const [sortBy, setSortBy] = useState('last_name');
  const [sortDir, setSortDir] = useState('asc');
  const [archetype, setArchetype] = useState('');
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  useEffect(() => { setMounted(true); }, []);

  const fetchPatients = useCallback(async () => {
    if (!mounted) return;
    setLoading(true);
    try {
      const params = new URLSearchParams({
        q: query, page: String(page), page_size: '50',
        sort_by: sortBy, sort_dir: sortDir,
      });
      if (archetype) params.set('archetype', archetype);
      const res = await api(`/patients?${params}`);
      setData(res);
    } catch (e) { console.error(e); }
    setLoading(false);
  }, [mounted, query, page, sortBy, sortDir, archetype]);

  useEffect(() => { fetchPatients(); }, [fetchPatients]);

  if (!mounted) return null;

  const riskColor = (scores: any) => {
    if (!scores) return 'var(--tx3)';
    const max = Math.max(...Object.values(scores).map(Number));
    if (max > 0.7) return 'var(--red)';
    if (max > 0.4) return 'var(--yellow)';
    return 'var(--green)';
  };

  return (
    <div>
      <Nav />
      <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <h1 style={{ fontSize: 24, fontWeight: 700 }}>
            Patients <span style={{ fontSize: 14, color: 'var(--tx3)', fontWeight: 400 }}>
              ({data.total.toLocaleString()} total)
            </span>
          </h1>
        </div>

        {/* Search + Filters */}
        <div style={{ display: 'flex', gap: 12, marginBottom: 16, flexWrap: 'wrap' }}>
          <input style={{ flex: 1, minWidth: 250 }}
            placeholder="Search by name or patient ID..."
            value={query}
            onChange={e => { setQuery(e.target.value); setPage(1); }}
          />
          <select value={archetype} onChange={e => { setArchetype(e.target.value); setPage(1); }}>
            <option value="">All Archetypes</option>
            {['excellent', 'good', 'moderate', 'poor', 'erratic'].map(a => (
              <option key={a} value={a}>{a}</option>
            ))}
          </select>
          <select value={sortBy} onChange={e => setSortBy(e.target.value)}>
            <option value="last_name">Name</option>
            <option value="age">Age</option>
            <option value="bmi">BMI</option>
            <option value="n_sdoh_risks">SDOH Risks</option>
          </select>
          <button className="btn btn-secondary" onClick={() => setSortDir(d => d === 'asc' ? 'desc' : 'asc')}>
            {sortDir === 'asc' ? '↑' : '↓'}
          </button>
        </div>

        {/* Table */}
        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 14 }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border)' }}>
                {['Name', 'Age', 'Gender', 'Race', 'BMI', 'Insurance', 'Archetype', 'Risk', 'Actions'].map(h => (
                  <th key={h} style={{ padding: '12px 16px', textAlign: 'left', color: 'var(--tx3)', fontWeight: 600, fontSize: 12, textTransform: 'uppercase' }}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr><td colSpan={9} style={{ padding: 40, textAlign: 'center', color: 'var(--tx3)' }}>Loading...</td></tr>
              ) : data.patients.map((p: any) => (
                <tr key={p.patient_id} style={{ borderBottom: '1px solid var(--border)', cursor: 'pointer' }}
                    onClick={() => router.push(`/patient/${p.patient_id}`)}>
                  <td style={{ padding: '10px 16px', fontWeight: 500 }}>{p.first_name} {p.last_name}</td>
                  <td style={{ padding: '10px 16px' }}>{p.age}</td>
                  <td style={{ padding: '10px 16px' }}>{p.gender}</td>
                  <td style={{ padding: '10px 16px', textTransform: 'capitalize' }}>{p.race}</td>
                  <td style={{ padding: '10px 16px' }}>{p.bmi}</td>
                  <td style={{ padding: '10px 16px', textTransform: 'capitalize' }}>{p.insurance_type?.replace('_', ' ')}</td>
                  <td style={{ padding: '10px 16px' }}>
                    <span className={`badge badge-${p.adherence_archetype === 'poor' || p.adherence_archetype === 'erratic' ? 'high' : p.adherence_archetype === 'moderate' ? 'moderate' : 'low'}`}>
                      {p.adherence_archetype}
                    </span>
                  </td>
                  <td style={{ padding: '10px 16px' }}>
                    <span style={{ color: riskColor(p.risk_scores), fontWeight: 600 }}>
                      {p.risk_scores ? Math.max(...Object.values(p.risk_scores).map(Number)).toFixed(0) + '%' : '—'}
                    </span>
                  </td>
                  <td style={{ padding: '10px 16px' }}>
                    <button className="btn btn-secondary" style={{ fontSize: 12, padding: '4px 10px' }}
                      onClick={e => { e.stopPropagation(); router.push(`/patient/${p.patient_id}`); }}>
                      View
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: 12, marginTop: 16 }}>
          <button className="btn btn-secondary" disabled={page <= 1} onClick={() => setPage(p => p - 1)}>
            ← Prev
          </button>
          <span style={{ fontSize: 14, color: 'var(--tx3)' }}>
            Page {data.page} of {data.total_pages}
          </span>
          <button className="btn btn-secondary" disabled={page >= data.total_pages} onClick={() => setPage(p => p + 1)}>
            Next →
          </button>
        </div>
      </div>
    </div>
  );
}
