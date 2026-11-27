# Security

PredictiveCare is a portfolio demo on synthetic data. It holds no real patient data and is not a covered system.

## Reporting a problem

Please do not open a public issue. Use GitHub's private vulnerability reporting on this repository (Security tab,
"Report a vulnerability"), or contact the maintainer through GitHub.

## Audit of 2026-11-04 and status

| Finding | Status |
|---|---|
| Accounts with password equal to username could run in production | Fixed in [#11](https://github.com/SushantTusharJoshi/predictive-care/pull/11): off in production unless `DEMO_LOGINS=true`; real accounts are bcrypt hashes in `AUTH_USERS` |
| Five routes read patient data without an audit-log row | Fixed in [#12](https://github.com/SushantTusharJoshi/predictive-care/pull/12), with a static test that fails if a new PHI route forgets `log_audit` |
| `sort_by` reached `getattr` on the ORM model unchecked | Fixed in [#14](https://github.com/SushantTusharJoshi/predictive-care/pull/14): allowlist, 400 otherwise |
| Free-text `insurance_type` could reach an LLM prompt | Fixed in [#14](https://github.com/SushantTusharJoshi/predictive-care/pull/14): fixed list on create, mapped to that list before any prompt |
| Error bodies echoed exception text | Fixed in [#14](https://github.com/SushantTusharJoshi/predictive-care/pull/14): generic body, details logged server-side |
| CORS allowed any `*.vercel.app` origin with credentials | Fixed in [#14](https://github.com/SushantTusharJoshi/predictive-care/pull/14): only `FRONTEND_URL` |

## Hardening added in the same series

- Demo accounts are view-only for actions: no approvals, overrides, patient creation or audit-log reads ([#14](https://github.com/SushantTusharJoshi/predictive-care/pull/14)).
- Model pickles are loaded only when their sha256 matches the manifest ([#16](https://github.com/SushantTusharJoshi/predictive-care/pull/16)).
- LLM narratives pass a SHAP check and local rules, and Bedrock Guardrails when configured; otherwise the template is shown ([#18](https://github.com/SushantTusharJoshi/predictive-care/pull/18)).
- A least-privilege AWS policy template for the runtime ([#20](https://github.com/SushantTusharJoshi/predictive-care/pull/20)).

## Known open items

- The JWT is kept in `localStorage`, so an XSS bug would expose it. No MFA, no SSO.
- Rate limits and the narrative cache and caps are in-process, per worker.
- No encryption at rest is configured by this repo; that is the database host's setting.
- The model hash check guards against corruption and swapped files, not against someone who can rewrite the
  manifest too. Signing the manifest would close that.
- The privacy controls are HIPAA-aligned design choices, not a compliance claim.
