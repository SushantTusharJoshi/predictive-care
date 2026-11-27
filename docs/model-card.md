# Model card: PredictiveCare risk models

> Demo on synthetic data. Not clinically validated. Do not use for care decisions.
> SHAP attributions describe what moved the model's score, not what causes the outcome.

## What the models predict

| Target | Label |
|---|---|
| `er_visit_30d` | an ER visit in the 30 days after the index date |
| `care_need_30d` | an ER, inpatient or urgent-care encounter in the 30 days after the index date |
| `care_need_90d` | the same, over 90 days |

Intended use: a portfolio demo of a time-honest training pipeline, calibration, explanations and a clinician gate. Not intended for any real patient.

## Data

- Source: Postgres seeded with `python -m app.data.ingest_synthea --fallback` (synthetic patients, no real person). The fallback run ends with `app.data.rearrange_outcomes`, which plants the outcome signal described below; on an existing database it can be run on its own.
- Index date = latest encounter in the database minus 90 days (`as_of` and `index_date` are in metrics.json). Every feature uses events strictly before the index date; labels use only the window after it.
- 17 features: age, sex, BMI, smoker, SDOH risk count, insurance type (one-hot), diagnosis and active-medication counts, adherence rate over the last 90 days and a year earlier, adherence trend, response latency, average systolic BP and heart rate. Archetype labels and prior ER counts are excluded as leaky (the script asserts this).
- Serving computes the same features with the same defaults in `get_patient_features`.

## Planted outcome signal (documented, synthetic)

The generator gives every encounter a uniformly random type, independent of the patient, so on its own the labels
carry no signal. `app.data.rearrange_outcomes` creates no data: it keeps every encounter row, its patient, date and
chief complaint, and the overall count of each encounter type. It only changes which rows hold an acute type
(inpatient, er, urgent_care), so that acute visits concentrate in patients with a higher planted risk:

| Input (all from before the index date) | Weight on the logit scale |
|---|---|
| Age | 0.04 per year above 45 |
| Diagnoses with onset before the index date | heart failure 1.0, COPD 0.9, CKD 0.8, type 2 diabetes 0.6, atrial fibrillation 0.4, asthma 0.3, hypertension 0.25, depression 0.2, anxiety 0.15, osteoarthritis 0.1, any other 0.2 |
| Smoker | 0.6 |
| SDOH risk factors | 0.35 each |
| BMI below 18.5 or 35 and above | 0.5 |
| Poor adherence: 1 minus the adherence rate in the 90 days before the index date | x 2.0 |
| Erratic adherence: absolute change from the same 90 days a year earlier | x 2.0 |

Adherence rates come from the adherence events with the same windows as training, not from the archetype label,
which is not a model input. Each encounter row gets the key 1.5 x risk + Gumbel noise (seeded, drawn in
encounter_id order); the rows with the highest keys take the acute types, as many as there were, and the most
serious types go to the highest keys. This is a weighted draw without replacement with weight exp(1.5 x risk).
Dates are not changed, so the index date and the time split are the same before and after. A second run changes
nothing. On the 1,000-patient run the riskiest fifth of patients have 83% of their encounters acute, the lowest
fifth 5%.

The direction of every weight is clinically plausible; the sizes are chosen to make the signal recoverable at
demo scale, not taken from any study. The model does not see the kind of diagnosis (only the count), so part of
the planted risk is noise to it. The numbers below therefore show that the pipeline recovers a known, planted
signal, not that it would work on real patients.

## Training (one script: `python -m app.ml.train`)

1. Order patients by patient time (first recorded encounter) and cut 70/10/20 into train, calibration and test. No patient is in two slices.
2. Fit four candidates on train: a logistic regression baseline (standardised, class-balanced), XGBoost, LightGBM, and the ensemble that averages the two tree models' log-odds.
3. On the calibration slice only: pick the candidate with the best ROC-AUC, choose isotonic or sigmoid calibration by 5-fold CV Brier, and pick the decision threshold that maximises F1. Calibration is skipped, and says why, when the slice has fewer than 10 positives or 10 negatives.
4. Report once on the untouched test slice: ROC-AUC, PR-AUC, Brier, precision, recall and F1 at the chosen threshold, reliability bins and ECE before and after calibration. Calibrated metrics are only claimed when test ECE did not get worse.
5. Save `models-<version>.pkl`, its `.sha256`, `metrics.json` and a `current.json` manifest. The API refuses a file whose hash does not match.

## Results on a 1,000-patient synthetic database

Commands, run on 2026-11-27 against a throwaway Postgres: `python -m app.data.ingest_synthea --fallback --max-patients 1000` (17,615 encounters), `python -m app.ml.train` for the "before" run, then `python -m app.data.rearrange_outcomes` (8,787 encounter types changed; a second run changed 0) and `python -m app.ml.train` again. Full output: [metrics/postgres-1000-planted.json](metrics/postgres-1000-planted.json) (version `20261127T205414Z-e025414`) and [metrics/postgres-1000-before.json](metrics/postgres-1000-before.json). Tables printed by `python -m app.ml.report docs/metrics/postgres-1000-planted.json`.

Split: train 700 patients (first encounter 2021-11-28 to 2022-04-12), calibration 100 (2022-04-12 to 2022-06-02), test 200 (2022-06-03 to 2024-07-15). Index date 2026-08-29. Precision, recall and F1 are at the threshold chosen on the calibration slice (max F1), recorded in metrics.json as `targets.<target>.threshold`; `f1_at_0_5` is reported next to it.

After rearranging:

| Target | Selected | Test ROC-AUC | PR-AUC | Brier | Threshold | Precision | Recall | F1 | Baseline ROC-AUC | Positives train/cal/test |
|---|---|---|---|---|---|---|---|---|---|---|
| er_visit_30d | logreg, no calibration | 0.829 | 0.155 | 0.190 | 0.961 | 0.125 | 0.111 | 0.118 | 0.829 | 28/5/9 |
| care_need_90d | XGBoost + sigmoid | 0.681 | 0.281 | 0.133 | 0.163 | 0.227 | 0.625 | 0.333 | 0.723 | 156/17/32 |
| care_need_30d | XGBoost, no calibration | 0.640 | 0.151 | 0.106 | 0.247 | 0.140 | 0.375 | 0.203 | 0.671 | 62/8/16 |

| Target | Calibration | ECE before | ECE after | Claimed |
|---|---|---|---|---|
| er_visit_30d | skipped: 5 positives in the calibration slice | 0.282 | 0.282 | no |
| care_need_90d | sigmoid | 0.181 | 0.063 | yes |
| care_need_30d | skipped: 8 positives in the calibration slice | 0.104 | 0.104 | no |

Before rearranging (uniform encounter types):

| Target | Selected | Test ROC-AUC | Threshold | F1 | F1 at 0.5 | Positives train/cal/test |
|---|---|---|---|---|---|---|
| er_visit_30d | LightGBM | 0.417 | 0.051 | 0.071 | 0.154 | 38/4/9 |
| care_need_90d | XGBoost + sigmoid | 0.555 | 0.170 | 0.252 | 0.000 | 178/17/32 |
| care_need_30d | logreg | 0.408 | 0.496 | 0.088 | 0.092 | 70/8/15 |

How to read this: before, the scores are at chance, as they should be with no feature-outcome link. After, every
target beats chance, so the pipeline finds the planted signal. Model selection uses a calibration slice with 5 to
17 positives, so it is noisy: on care_need_90d and care_need_30d the logistic baseline scored higher on test than
the selected model. The test slice holds 9 to 32 positives per target, so every number has a wide margin of error.
Over eight other draws of a 1,000-patient in-memory cohort with the same rule, care_need_90d test ROC-AUC ranged
from 0.64 to 0.77. An earlier 600-patient run without the planted signal is kept in
[metrics/postgres-demo-600.json](metrics/postgres-demo-600.json).

## Sanity check: the pipeline recovers a known signal

Command: `python -m app.ml.train --source memory --n 20000 --label "demo: in-memory synthetic sanity check"`. Full output: [metrics/memory-sanity-20k.json](metrics/memory-sanity-20k.json). The in-memory cohort draws outcomes from a logistic function of the features, so a working pipeline must beat chance.

| Target | Selected | Test ROC-AUC | PR-AUC | Brier (before -> after calibration) | ECE (before -> after) | F1 at chosen threshold |
|---|---|---|---|---|---|---|
| er_visit_30d | logreg + sigmoid | 0.689 | 0.436 | 0.220 -> 0.169 | 0.222 -> 0.014 | 0.455 |
| care_need_90d | logreg + sigmoid | 0.691 | 0.626 | 0.222 -> 0.218 | 0.066 -> 0.021 | 0.629 |
| care_need_30d | logreg + sigmoid | 0.695 | 0.378 | 0.220 -> 0.147 | 0.266 -> 0.017 | 0.426 |

The logistic baseline wins here because the simulated outcome is itself logistic; XGBoost and LightGBM land 0.004 to 0.012 ROC-AUC below it (see the json). This is a check of the code, not evidence about real patients.

## Fairness by adherence archetype (test slice)

The 1,000-patient run after rearranging reports, per archetype, the observed and mean predicted rate, the share flagged at the chosen
threshold, recall, false positive rate and ROC-AUC (`targets.<target>.fairness` in the json; shown on the admin
page). A group needs 5 positives and 5 negatives before its rates are compared.

| Target | Groups with enough cases | What they show |
|---|---|---|
| care_need_90d | excellent (31 patients, 5 positives), good (56, 8), moderate (51, 7), poor (44, 11) | recall 0.40 / 0.38 / 0.71 / 0.82, false positive rate 0.19 / 0.21 / 0.48 / 0.70 |
| er_visit_30d | none (9 positives in the whole test slice) | nothing to compare |
| care_need_30d | poor only (5 positives) | nothing to compare |

The planted rule makes poor adherence a risk factor, so the poor-adherence group is flagged more often by design;
with 5 to 11 positives per group the recall gaps are within noise. Archetype is not a model input (it is excluded
as leaky) and the planted rule does not read it; it is only the grouping. Race and sex groups are not reported yet.

## Explanations

- Each prediction returns its top 8 features by absolute SHAP value. Trees use exact TreeSHAP (the libraries' native contribution output, checked against `shap.TreeExplainer` in the tests); the baseline uses linear SHAP with the training mean as background; the ensemble averages its members.
- SHAP explains the uncalibrated log-odds. Calibration is a monotone map applied afterwards, so it does not flip the direction of any feature.
- An LLM narrative is shown only if every feature it names is in that patient's top SHAP features, in the same direction and size order; otherwise the template is shown (see the README).

## Limits

- Synthetic data only; no external or prospective validation; not clinically validated.
- The feature-outcome link is planted by the documented rule above; recovering it checks the code, not clinical validity.
- 1,000 patients is too small for stable metrics: 9 to 32 test positives per target.
- Fairness is reported by archetype only, and the demo has too few cases per group to compare.
- Attribution is not causation: a feature with a large SHAP value is not a treatment target.
- Probabilities from targets without calibration (marked "no" above) should not be read as calibrated risks.
