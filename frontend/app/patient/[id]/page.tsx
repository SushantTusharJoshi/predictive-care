'use client';
import { useState, useEffect, use } from 'react';
import { useRouter } from 'next/navigation';
import Nav from '../../../components/Nav';
import { api } from '../../../lib/api';
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, BarChart, Bar, Legend, AreaChart, Area } from 'recharts';

export default function PatientDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [mounted, setMounted] = useState(false);
  const [patient, setPatient] = useState<any>(null);
  const [tab, setTab] = useState('overview');
  const [similar, setSimilar] = useState<any>(null);
  const [reminders, setReminders] = useState<any>(null);
  const [longitudinal, setLongitudinal] = useState<any>(null);
  const [narrative, setNarrative] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  useEffect(() => {
    setMounted(true);
    if (!localStorage.getItem('token')) { router.push('/'); return; }
    api(`/patients/${id}`).then(p => { setPatient(p); setLoading(false); }).catch(() => setLoading(false));
  }, [id]);

  if (!mounted || loading) return <div><Nav /><div style={{ padding: 40, textAlign: 'center', color: 'var(--tx3)' }}>Loading...</div></div>;
  if (!patient) return <div><Nav /><div style={{ padding: 40, textAlign: 'center' }}>Patient not found</div></div>;

  const loadSimilar = () => { if (!similar) api(`/patients/${id}/similar`).then(setSimilar).catch(() => {}); };
  const loadReminders = () => { if (!reminders) api(`/patients/${id}/reminders`).then(setReminders).catch(() => {}); };
  const loadLongitudinal = () => { if (!longitudinal) api(`/patients/${id}/longitudinal`).then(setLongitudinal).catch(() => {}); };
  const loadNarrative = (predType: string) => {
    setNarrative({ loading: true });
    api(`/patients/${id}/shap-narrative/${predType}`).then(setNarrative).catch(() => setNarrative({ error: 'Failed' }));
  };

  const tabs = [
    { key: 'overview', label: 'Overview' },
    { key: 'predictions', label: 'Risk Predictions' },
    { key: 'adherence', label: 'Adherence' },
    { key: 'reminders', label: 'Med Reminders', onSelect: loadReminders },
    { key: 'longitudinal', label: '5-Year Analysis', onSelect: loadLongitudinal },
    { key: 'similar', label: 'Similar Patients', onSelect: loadSimilar },
    { key: 'labs', label: 'Labs & Vitals' },
  ];

  return (
    <div>
      <Nav />
      <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 20 }}>
          <div>
            <button className="btn btn-secondary" style={{ fontSize: 12, marginBottom: 8 }}
              onClick={() => router.push('/dashboard/patients')}>← Back</button>
            <h1 style={{ fontSize: 24, fontWeight: 700 }}>{patient.first_name} {patient.last_name}</h1>
            <div style={{ color: 'var(--tx3)', fontSize: 14 }}>
              {patient.age}y {patient.gender} · {patient.race} · BMI {patient.bmi} · {patient.insurance_type?.replace('_', ' ')}
            </div>
            <div style={{ fontSize: 12, color: 'var(--tx3)', marginTop: 4 }}>ID: {patient.patient_id}</div>
          </div>
          <div style={{ textAlign: 'right' }}>
            <span className={`badge badge-${patient.adherence_archetype === 'poor' || patient.adherence_archetype === 'erratic' ? 'high' : patient.adherence_archetype === 'moderate' ? 'moderate' : 'low'}`}
              style={{ fontSize: 14, padding: '4px 14px' }}>
              {patient.adherence_archetype} adherence
            </span>
            {patient.sdoh_risk_factors?.length > 0 && (
              <div style={{ marginTop: 8, fontSize: 12, color: 'var(--orange)' }}>
                SDOH: {patient.sdoh_risk_factors.join(', ').replace(/_/g, ' ')}
              </div>
            )}
          </div>
        </div>

        {/* Tabs */}
        <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid var(--border)', paddingBottom: 4, overflowX: 'auto' }}>
          {tabs.map(t => (
            <button key={t.key}
              className={`nav-link ${tab === t.key ? 'active' : ''}`}
              onClick={() => { setTab(t.key); t.onSelect?.(); }}>
              {t.label}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        {tab === 'overview' && <OverviewTab patient={patient} />}
        {tab === 'predictions' && <PredictionsTab patient={patient} narrative={narrative} onNarrative={loadNarrative} />}
        {tab === 'adherence' && <AdherenceTab patient={patient} />}
        {tab === 'reminders' && <RemindersTab data={reminders} />}
        {tab === 'longitudinal' && <LongitudinalTab data={longitudinal} />}
        {tab === 'similar' && <SimilarTab data={similar} router={router} />}
        {tab === 'labs' && <LabsTab patient={patient} />}
      </div>
    </div>
  );
}

function OverviewTab({ patient }: { patient: any }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
      <div className="card">
        <h3 style={{ fontSize: 14, fontWeight: 600, color: 'var(--tx3)', marginBottom: 12 }}>Diagnoses</h3>
        {(patient.diagnoses || []).map((d: any) => (
          <div key={d.diagnosis_id} style={{ padding: '6px 0', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between' }}>
            <span style={{ textTransform: 'capitalize' }}>{d.condition?.replace(/_/g, ' ')}</span>
            <div>
              <span className={`badge badge-${d.severity === 'severe' ? 'high' : d.severity === 'moderate' ? 'moderate' : 'low'}`}>{d.severity}</span>
              <span style={{ marginLeft: 8, fontSize: 12, color: 'var(--tx3)' }}>{d.status}</span>
            </div>
          </div>
        ))}
        {(!patient.diagnoses || patient.diagnoses.length === 0) && <div style={{ color: 'var(--tx3)' }}>No diagnoses</div>}
      </div>

      <div className="card">
        <h3 style={{ fontSize: 14, fontWeight: 600, color: 'var(--tx3)', marginBottom: 12 }}>Medications</h3>
        {(patient.medications || []).slice(0, 10).map((m: any) => (
          <div key={m.medication_id} style={{ padding: '6px 0', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between' }}>
            <span style={{ textTransform: 'capitalize' }}>{m.medication_name?.replace(/_/g, ' ')}</span>
            <span style={{ fontSize: 13, color: 'var(--tx3)' }}>{m.dosage} · {m.frequency?.replace('_', ' ')}</span>
          </div>
        ))}
      </div>

      <div className="card">
        <h3 style={{ fontSize: 14, fontWeight: 600, color: 'var(--tx3)', marginBottom: 12 }}>Adherence Summary</h3>
        {(patient.adherence_summary || []).map((a: any) => (
          <div key={a.medication_name} style={{ padding: '6px 0', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between' }}>
            <span style={{ textTransform: 'capitalize' }}>{a.medication_name?.replace(/_/g, ' ')}</span>
            <span style={{ fontWeight: 600, color: a.adherence_pct > 80 ? 'var(--green)' : a.adherence_pct > 60 ? 'var(--yellow)' : 'var(--red)' }}>
              {a.adherence_pct}%
            </span>
          </div>
        ))}
      </div>

      <div className="card">
        <h3 style={{ fontSize: 14, fontWeight: 600, color: 'var(--tx3)', marginBottom: 12 }}>Recent Encounters</h3>
        {(patient.encounters || []).slice(0, 8).map((e: any) => (
          <div key={e.encounter_id} style={{ padding: '6px 0', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between' }}>
            <span>{e.encounter_date} — <span style={{ textTransform: 'capitalize', color: e.encounter_type === 'er' ? 'var(--red)' : 'var(--tx2)' }}>{e.encounter_type}</span></span>
            <span style={{ fontSize: 12, color: 'var(--tx3)' }}>{e.chief_complaint?.replace(/_/g, ' ')}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function PredictionsTab({ patient, narrative, onNarrative }: { patient: any; narrative: any; onNarrative: (t: string) => void }) {
  const preds = patient.predictions || {};
  if (preds.error) return <div className="card"><div style={{ color: 'var(--red)' }}>Error: {preds.error}</div></div>;

  return (
    <div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: 16 }}>
        {Object.entries(preds).map(([name, info]: [string, any]) => (
          <div key={name} className="card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
              <h3 style={{ fontSize: 16, fontWeight: 600, textTransform: 'capitalize' }}>{name.replace(/_/g, ' ')}</h3>
              <span className={`badge badge-${info.risk_level}`} style={{ fontSize: 14, padding: '4px 14px' }}>
                {(info.probability * 100).toFixed(1)}%
              </span>
            </div>

            {/* SHAP bar chart */}
            <div style={{ marginBottom: 12 }}>
              {(info.top_features || []).slice(0, 5).map((f: any, i: number) => (
                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                  <span style={{ fontSize: 12, color: 'var(--tx3)', width: 140, textAlign: 'right', flexShrink: 0 }}>
                    {f.feature.replace(/_/g, ' ')}
                  </span>
                  <div style={{ flex: 1, height: 16, background: 'var(--bg3)', borderRadius: 4, position: 'relative', overflow: 'hidden' }}>
                    <div style={{
                      position: 'absolute',
                      left: f.shap_value > 0 ? '50%' : `${50 + Math.max(f.shap_value * 200, -50)}%`,
                      width: `${Math.min(Math.abs(f.shap_value) * 200, 50)}%`,
                      height: '100%',
                      background: f.shap_value > 0 ? 'var(--red)' : 'var(--green)',
                      borderRadius: 4,
                    }} />
                  </div>
                  <span style={{ fontSize: 11, color: 'var(--tx3)', width: 50 }}>
                    {f.shap_value > 0 ? '+' : ''}{f.shap_value.toFixed(3)}
                  </span>
                </div>
              ))}
            </div>

            <button className="btn btn-primary" style={{ width: '100%', fontSize: 13 }}
              onClick={() => onNarrative(name)}>
              Generate AI Explanation
            </button>
          </div>
        ))}
      </div>

      {/* Narrative */}
      {narrative && (
        <div className="card" style={{ marginTop: 16 }}>
          <h3 style={{ fontSize: 14, fontWeight: 600, color: 'var(--accent)', marginBottom: 8 }}>AI-Generated Explanation (Groq)</h3>
          {narrative.loading ? (
            <div style={{ color: 'var(--tx3)' }}>Generating narrative...</div>
          ) : narrative.error ? (
            <div style={{ color: 'var(--red)' }}>{narrative.error}</div>
          ) : (
            <div style={{ color: 'var(--tx2)', lineHeight: 1.6 }}>{narrative.narrative}</div>
          )}
        </div>
      )}
    </div>
  );
}

function AdherenceTab({ patient }: { patient: any }) {
  const summary = patient.adherence_summary || [];
  const recent = patient.recent_adherence || [];

  // Aggregate daily rates from recent events
  const dailyMap: Record<string, { total: number; taken: number }> = {};
  recent.forEach((e: any) => {
    if (!dailyMap[e.event_date]) dailyMap[e.event_date] = { total: 0, taken: 0 };
    dailyMap[e.event_date].total++;
    if (e.taken) dailyMap[e.event_date].taken++;
  });
  const chartData = Object.entries(dailyMap)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([date, v]) => ({ date: date.slice(5), rate: Math.round(v.taken / v.total * 100) }));

  return (
    <div>
      {chartData.length > 0 && (
        <div className="card" style={{ marginBottom: 16 }}>
          <h3 style={{ fontSize: 14, fontWeight: 600, color: 'var(--tx3)', marginBottom: 12 }}>30-Day Adherence Trend</h3>
          <ResponsiveContainer width="100%" height={250}>
            <AreaChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="date" tick={{ fontSize: 11, fill: 'var(--tx3)' }} />
              <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: 'var(--tx3)' }} />
              <Tooltip contentStyle={{ background: 'var(--bg2)', border: '1px solid var(--border)', borderRadius: 8 }} />
              <Area type="monotone" dataKey="rate" stroke="var(--accent)" fill="rgba(59,130,246,0.2)" name="Rate %" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      <div className="card">
        <h3 style={{ fontSize: 14, fontWeight: 600, color: 'var(--tx3)', marginBottom: 12 }}>Per-Medication Adherence</h3>
        {summary.map((a: any) => (
          <div key={a.medication_name} style={{ padding: '10px 0', borderBottom: '1px solid var(--border)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
              <span style={{ fontWeight: 500, textTransform: 'capitalize' }}>{a.medication_name?.replace(/_/g, ' ')}</span>
              <span style={{ fontWeight: 700, color: a.adherence_pct > 80 ? 'var(--green)' : a.adherence_pct > 60 ? 'var(--yellow)' : 'var(--red)' }}>
                {a.adherence_pct}%
              </span>
            </div>
            <div style={{ display: 'flex', gap: 16, fontSize: 12, color: 'var(--tx3)' }}>
              <span>{a.taken_count}/{a.total_events} taken</span>
              {a.avg_latency_min != null && <span>Avg response: {a.avg_latency_min}min</span>}
              <span>{a.first_event} → {a.last_event}</span>
            </div>
            <div style={{ marginTop: 4, height: 6, background: 'var(--bg3)', borderRadius: 3 }}>
              <div style={{ height: '100%', width: `${a.adherence_pct}%`, borderRadius: 3,
                background: a.adherence_pct > 80 ? 'var(--green)' : a.adherence_pct > 60 ? 'var(--yellow)' : 'var(--red)' }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function RemindersTab({ data }: { data: any }) {
  if (!data) return <div style={{ color: 'var(--tx3)', padding: 20 }}>Loading reminders...</div>;

  return (
    <div>
      <div className="card" style={{ marginBottom: 16 }}>
        <h3 style={{ fontSize: 14, fontWeight: 600, color: 'var(--tx3)', marginBottom: 4 }}>Medication Reminder Schedule</h3>
        <p style={{ fontSize: 13, color: 'var(--tx3)', marginBottom: 16 }}>
          Simulated 15-min and 5-min warnings before each scheduled dose
        </p>
      </div>

      {(data.medications || []).map((med: any) => (
        <div key={med.medication_id} className="card" style={{ marginBottom: 12 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
            <div>
              <h4 style={{ fontSize: 16, fontWeight: 600, textTransform: 'capitalize' }}>
                {med.medication_name?.replace(/_/g, ' ')}
              </h4>
              <span style={{ fontSize: 13, color: 'var(--tx3)' }}>{med.dosage} · {med.frequency?.replace('_', ' ')}</span>
            </div>
            <span style={{ fontWeight: 700, fontSize: 18, color: med.recent_rate > 80 ? 'var(--green)' : med.recent_rate > 60 ? 'var(--yellow)' : 'var(--red)' }}>
              {med.recent_rate?.toFixed(0)}%
            </span>
          </div>

          {/* Reminder timeline */}
          {med.reminder_timeline && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '12px 16px', background: 'var(--bg3)', borderRadius: 8, marginBottom: 12 }}>
              <TimeBlock label="15-min warning" time={med.reminder_timeline.warning_15min} color="var(--yellow)" icon="⚠️" />
              <div style={{ width: 30, borderTop: '2px dashed var(--border)' }} />
              <TimeBlock label="5-min warning" time={med.reminder_timeline.warning_5min} color="var(--orange)" icon="🔔" />
              <div style={{ width: 30, borderTop: '2px dashed var(--border)' }} />
              <TimeBlock label="Dose due" time={med.reminder_timeline.due_time} color="var(--accent)" icon="💊" />
            </div>
          )}

          {/* Recent adherence */}
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {(med.recent_adherence || []).map((e: any, i: number) => (
              <div key={i} style={{
                width: 32, height: 32, borderRadius: 6, display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 11, fontWeight: 600,
                background: e.taken ? 'rgba(34,197,94,0.2)' : 'rgba(239,68,68,0.2)',
                color: e.taken ? 'var(--green)' : 'var(--red)',
                border: `1px solid ${e.taken ? 'var(--green)' : 'var(--red)'}`,
              }} title={`${e.event_date}: ${e.taken ? `Taken at ${e.taken_time}` : 'Missed'}`}>
                {e.taken ? '✓' : '✗'}
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function TimeBlock({ label, time, color, icon }: { label: string; time: string; color: string; icon: string }) {
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{ fontSize: 18 }}>{icon}</div>
      <div style={{ fontSize: 16, fontWeight: 700, color }}>{time}</div>
      <div style={{ fontSize: 11, color: 'var(--tx3)' }}>{label}</div>
    </div>
  );
}

function LongitudinalTab({ data }: { data: any }) {
  if (!data) return <div style={{ color: 'var(--tx3)', padding: 20 }}>Loading 5-year analysis...</div>;

  const adhData = (data.trend_data?.adherence_quarterly || []).map((q: any) => ({
    quarter: q.quarter,
    rate: q.rate,
    latency: q.avg_latency || 0,
  }));

  const vitalData = data.trend_data?.vital_trends || [];

  return (
    <div>
      {/* AI Narrative */}
      {data.narrative && (
        <div className="card" style={{ marginBottom: 16, borderLeft: '3px solid var(--accent)' }}>
          <h3 style={{ fontSize: 14, fontWeight: 600, color: 'var(--accent)', marginBottom: 8 }}>AI Behavior Analysis</h3>
          <p style={{ color: 'var(--tx2)', lineHeight: 1.6 }}>{data.narrative}</p>
        </div>
      )}

      {/* Adherence trend */}
      {adhData.length > 0 && (
        <div className="card" style={{ marginBottom: 16 }}>
          <h3 style={{ fontSize: 14, fontWeight: 600, color: 'var(--tx3)', marginBottom: 12 }}>Quarterly Adherence Rate (5 Years)</h3>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={adhData}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="quarter" tick={{ fontSize: 10, fill: 'var(--tx3)' }} angle={-45} textAnchor="end" height={60} />
              <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: 'var(--tx3)' }} />
              <Tooltip contentStyle={{ background: 'var(--bg2)', border: '1px solid var(--border)', borderRadius: 8 }} />
              <Legend />
              <Line type="monotone" dataKey="rate" stroke="var(--accent)" name="Adherence %" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="latency" stroke="var(--orange)" name="Avg Latency (min)" strokeWidth={1} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Vital trends */}
      {vitalData.length > 0 && (
        <div className="card" style={{ marginBottom: 16 }}>
          <h3 style={{ fontSize: 14, fontWeight: 600, color: 'var(--tx3)', marginBottom: 12 }}>Vital Sign Trends</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={vitalData}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="quarter" tick={{ fontSize: 10, fill: 'var(--tx3)' }} angle={-45} textAnchor="end" height={60} />
              <YAxis tick={{ fontSize: 11, fill: 'var(--tx3)' }} />
              <Tooltip contentStyle={{ background: 'var(--bg2)', border: '1px solid var(--border)', borderRadius: 8 }} />
              <Legend />
              <Line type="monotone" dataKey="avg_sbp" stroke="var(--red)" name="SBP" dot={false} />
              <Line type="monotone" dataKey="avg_dbp" stroke="var(--orange)" name="DBP" dot={false} />
              <Line type="monotone" dataKey="avg_hr" stroke="var(--accent)" name="HR" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Encounter pattern */}
      {data.trend_data?.encounter_yearly?.length > 0 && (
        <div className="card">
          <h3 style={{ fontSize: 14, fontWeight: 600, color: 'var(--tx3)', marginBottom: 12 }}>Encounter Frequency by Year</h3>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {data.trend_data.encounter_yearly.map((e: any, i: number) => (
              <div key={i} style={{ padding: '6px 12px', background: 'var(--bg3)', borderRadius: 6, fontSize: 13 }}>
                <span style={{ color: 'var(--tx3)' }}>{e.year}</span>{' '}
                <span style={{ color: e.encounter_type === 'er' ? 'var(--red)' : 'var(--tx2)', fontWeight: 600 }}>
                  {e.encounter_type}
                </span>{' '}
                ×{e.count}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function SimilarTab({ data, router }: { data: any; router: any }) {
  if (!data) return <div style={{ color: 'var(--tx3)', padding: 20 }}>Loading similar patients...</div>;

  const prediction = data.predicted_behavior || {};

  return (
    <div>
      {/* Prediction from similar patients */}
      {prediction.predicted_adherence_rate != null && (
        <div className="card" style={{ marginBottom: 16, borderLeft: '3px solid var(--accent2)' }}>
          <h3 style={{ fontSize: 14, fontWeight: 600, color: 'var(--accent2)', marginBottom: 8 }}>Predicted Behavior (from similar patients)</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12 }}>
            <div style={{ padding: '8px 16px', background: 'var(--bg3)', borderRadius: 8 }}>
              <div style={{ fontSize: 12, color: 'var(--tx3)' }}>Predicted Adherence</div>
              <div style={{ fontSize: 24, fontWeight: 700, color: 'var(--accent)' }}>{prediction.predicted_adherence_rate}%</div>
            </div>
            <div style={{ padding: '8px 16px', background: 'var(--bg3)', borderRadius: 8 }}>
              <div style={{ fontSize: 12, color: 'var(--tx3)' }}>Likely Archetype</div>
              <div style={{ fontSize: 18, fontWeight: 700, textTransform: 'capitalize' }}>{prediction.likely_archetype}</div>
            </div>
            <div style={{ padding: '8px 16px', background: 'var(--bg3)', borderRadius: 8 }}>
              <div style={{ fontSize: 12, color: 'var(--tx3)' }}>Confidence</div>
              <div style={{ fontSize: 18, fontWeight: 700, textTransform: 'capitalize',
                color: prediction.confidence === 'high' ? 'var(--green)' : prediction.confidence === 'medium' ? 'var(--yellow)' : 'var(--red)' }}>
                {prediction.confidence}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Similar patients list */}
      <div className="card">
        <h3 style={{ fontSize: 14, fontWeight: 600, color: 'var(--tx3)', marginBottom: 12 }}>
          Top Similar Patients ({data.similar_patients?.length || 0})
        </h3>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 14 }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)' }}>
              {['Name', 'Age', 'Gender', 'Race', 'BMI', 'Insurance', 'Archetype', 'Adherence', 'Similarity'].map(h => (
                <th key={h} style={{ padding: '8px 12px', textAlign: 'left', color: 'var(--tx3)', fontSize: 12, fontWeight: 600 }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {(data.similar_patients || []).map((s: any) => (
              <tr key={s.patient_id} style={{ borderBottom: '1px solid var(--border)', cursor: 'pointer' }}
                  onClick={() => router.push(`/patient/${s.patient_id}`)}>
                <td style={{ padding: '8px 12px', fontWeight: 500 }}>{s.name}</td>
                <td style={{ padding: '8px 12px' }}>{s.age}</td>
                <td style={{ padding: '8px 12px' }}>{s.gender}</td>
                <td style={{ padding: '8px 12px', textTransform: 'capitalize' }}>{s.race}</td>
                <td style={{ padding: '8px 12px' }}>{s.bmi}</td>
                <td style={{ padding: '8px 12px', textTransform: 'capitalize' }}>{s.insurance_type?.replace('_', ' ')}</td>
                <td style={{ padding: '8px 12px', textTransform: 'capitalize' }}>{s.adherence_archetype}</td>
                <td style={{ padding: '8px 12px', fontWeight: 600,
                  color: s.avg_adherence_pct > 80 ? 'var(--green)' : s.avg_adherence_pct > 60 ? 'var(--yellow)' : 'var(--red)' }}>
                  {s.avg_adherence_pct}%
                </td>
                <td style={{ padding: '8px 12px' }}>
                  <span style={{ color: 'var(--accent)', fontWeight: 600 }}>{s.similarity_score}%</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function LabsTab({ patient }: { patient: any }) {
  const labs = patient.labs || [];
  const vitals = patient.vitals || [];

  // Group labs by test name for trend charts
  const labsByTest: Record<string, any[]> = {};
  labs.forEach((l: any) => {
    if (!labsByTest[l.test_name]) labsByTest[l.test_name] = [];
    labsByTest[l.test_name].push({ date: l.lab_date?.slice(0, 7), value: l.value, flag: l.flag });
  });

  return (
    <div>
      {/* Lab trend charts */}
      {Object.entries(labsByTest).slice(0, 4).map(([test, data]) => (
        <div key={test} className="card" style={{ marginBottom: 16 }}>
          <h3 style={{ fontSize: 14, fontWeight: 600, color: 'var(--tx3)', marginBottom: 8, textTransform: 'capitalize' }}>
            {test.replace(/_/g, ' ')}
          </h3>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="date" tick={{ fontSize: 10, fill: 'var(--tx3)' }} />
              <YAxis tick={{ fontSize: 11, fill: 'var(--tx3)' }} />
              <Tooltip contentStyle={{ background: 'var(--bg2)', border: '1px solid var(--border)', borderRadius: 8 }} />
              <Line type="monotone" dataKey="value" stroke="var(--accent)" strokeWidth={2} dot={{ r: 2 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      ))}

      {/* Vitals table */}
      {vitals.length > 0 && (
        <div className="card">
          <h3 style={{ fontSize: 14, fontWeight: 600, color: 'var(--tx3)', marginBottom: 12 }}>Recent Vitals</h3>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border)' }}>
                  {['Date', 'SBP', 'DBP', 'HR', 'Temp', 'SpO2', 'Weight'].map(h => (
                    <th key={h} style={{ padding: '8px', textAlign: 'left', color: 'var(--tx3)', fontSize: 11 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {vitals.slice(-12).map((v: any) => (
                  <tr key={v.vital_id} style={{ borderBottom: '1px solid var(--border)' }}>
                    <td style={{ padding: '6px 8px' }}>{v.vital_date}</td>
                    <td style={{ padding: '6px 8px', color: v.systolic_bp > 140 ? 'var(--red)' : 'var(--tx2)' }}>{v.systolic_bp}</td>
                    <td style={{ padding: '6px 8px' }}>{v.diastolic_bp}</td>
                    <td style={{ padding: '6px 8px' }}>{v.heart_rate}</td>
                    <td style={{ padding: '6px 8px' }}>{v.temperature}</td>
                    <td style={{ padding: '6px 8px', color: v.spo2 < 94 ? 'var(--red)' : 'var(--tx2)' }}>{v.spo2}</td>
                    <td style={{ padding: '6px 8px' }}>{v.weight_lbs}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
