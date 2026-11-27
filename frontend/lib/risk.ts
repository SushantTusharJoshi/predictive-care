// Risk display for the patient list and Alerts page. Scores are 0-1 model probabilities.
// A target whose calibrated flag is explicitly false has a score that only orders
// patients, so it is shown as a rank, not a percent. A missing flag means calibrated.

export type RiskScores = Record<string, number> | null | undefined;
export type RiskCalibrated = Record<string, boolean | null> | null | undefined;

export const NO_RISK = '-';

export function formatPercent(prob: number | null | undefined): string {
  if (prob === null || prob === undefined || !Number.isFinite(Number(prob))) return NO_RISK;
  return `${(Number(prob) * 100).toFixed(1)}%`;
}

export function isUncalibrated(calibrated: RiskCalibrated, target: string): boolean {
  return !!calibrated && calibrated[target] === false;
}

// Highest-scoring target for one patient, or null when there are no scores.
export function topRisk(scores: RiskScores): { target: string; prob: number } | null {
  if (!scores) return null;
  const entries = Object.entries(scores).filter(([, v]) => Number.isFinite(Number(v)));
  if (entries.length === 0) return null;
  const [target, prob] = entries.reduce((a, b) => (Number(b[1]) > Number(a[1]) ? b : a));
  return { target, prob: Number(prob) };
}

// 1-based rank of prob among all values (highest = 1).
export function rankOf(prob: number, all: number[]): number {
  return all.filter(v => v > prob).length + 1;
}

export function rankLabel(rank: number, total: number): string {
  return `Rank ${rank} of ${total}`;
}
