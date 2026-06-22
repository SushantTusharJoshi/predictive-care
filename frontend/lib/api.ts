const API = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';

export function apiBase() { return API; }

export async function api(path: string, opts?: RequestInit) {
  const token = typeof window !== 'undefined' ? localStorage.getItem('token') : null;
  const headers: Record<string, string> = { 'Content-Type': 'application/json', ...opts?.headers as Record<string, string> };
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const r = await fetch(`${API}${path}`, { ...opts, headers, cache: 'no-store' });
  if (r.status === 401) {
    if (typeof window !== 'undefined') {
      localStorage.clear();
      window.location.href = '/';
    }
    throw new Error('Unauthorized');
  }
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export function getToken() { return typeof window !== 'undefined' ? localStorage.getItem('token') : null; }
export function getRole() { return typeof window !== 'undefined' ? localStorage.getItem('role') || '' : ''; }
export function getName() { return typeof window !== 'undefined' ? localStorage.getItem('name') || '' : ''; }

export function login(token: string, role: string, name: string) {
  localStorage.setItem('token', token);
  localStorage.setItem('role', role);
  localStorage.setItem('name', name);
}

export function logout() {
  localStorage.clear();
  window.location.href = '/';
}
