'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { getName, getRole, logout } from '../lib/api';

export default function Nav() {
  const path = usePathname();
  const [mounted, setMounted] = useState(false);
  const [role, setRole] = useState('');
  const [name, setName] = useState('');

  useEffect(() => {
    setMounted(true);
    setRole(getRole());
    setName(getName());
  }, []);

  const links = [
    { href: '/dashboard', label: 'Dashboard' },
    { href: '/dashboard/patients', label: 'Patients' },
    { href: '/alerts', label: 'Alerts' },
    ...(role === 'admin' ? [{ href: '/admin', label: 'Admin' }] : []),
  ];

  return (
    <nav style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                  padding: '12px 24px', background: 'var(--bg2)', borderBottom: '1px solid var(--border)' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 24 }}>
        <Link href="/dashboard" style={{ fontSize: 18, fontWeight: 700, textDecoration: 'none', color: 'var(--tx1)' }}>
          <span style={{ color: 'var(--accent)' }}>Predictive</span>Care
        </Link>
        {links.map(l => (
          <Link key={l.href} href={l.href}
            className={`nav-link ${path === l.href ? 'active' : ''}`}>
            {l.label}
          </Link>
        ))}
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        {mounted && (
          <>
            <span style={{ fontSize: 13, color: 'var(--tx3)' }}>{name} ({role})</span>
            <button className="btn btn-secondary" style={{ fontSize: 12, padding: '4px 12px' }} onClick={logout}>
              Logout
            </button>
          </>
        )}
      </div>
    </nav>
  );
}
