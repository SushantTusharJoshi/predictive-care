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

- Live demo model: version `20261128T015035Z-nogit`, trained on 10,000 synthetic patients copied from a local synthetic 100K cohort, sampled only from patients that have visits. History length is blended: 5,000 patients with 2 years of events, 3,000 with 5 years, 2,000 with 1 year. Visit types were rearranged by the planted-risk rule below (`rearrange_outcomes.py`). Source Postgres, `as_of` 2026-08-21, index date 2026-05-23.
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
nothing. On the earlier 1,000-patient run the riskiest fifth of patients have 83% of their encounters acute, the lowest
fifth 5%.

The direction of every weight is clinically plausible; the sizes are chosen to make the signal recoverable at
demo scale, not taken from any study. The model does not see the kind of diagnosis (only the count), so part of
the planted risk is noise to it. The numbers below therefore show that the pipeline recovers a known, planted
signal, not that it would work on real patients.

## Training (one script: `python -m app.ml.train`)

1. Cohort split by enrolment date (the test set is the most recently enrolled patients over the same label window): patients sorted by enrolment date (first recorded encounter), cut 70/10/20 into train, calibration and test. No patient is in two slices.
2. Fit four candidates on train: a logistic regression baseline (standardised, class-balanced), XGBoost, LightGBM, and the ensemble that averages the two tree models' log-odds.
3. On the calibration slice only: pick the candidate with the best ROC-AUC, choose isotonic or sigmoid calibration by 5-fold CV Brier, and pick the decision threshold that maximises F1. Calibration is skipped, and says why, when the slice has fewer than 10 positives or 10 negatives.
4. Report once on the untouched test slice: ROC-AUC, PR-AUC, Brier, precision, recall and F1 at the chosen threshold, reliability bins and ECE before and after calibration. Calibrated metrics are only claimed when test ECE did not get worse.
5. Save `models-<version>.pkl`, its `.sha256`, `metrics.json` and a `current.json` manifest. The API refuses a file whose hash does not match.

## Results: live demo model (10,000 synthetic patients)

Version `20261128T015035Z-nogit`, read from `metrics-20261128T015035Z-nogit.json` on the Railway models volume. Source Postgres, `as_of` 2026-08-21, index date 2026-05-23; every feature uses events strictly before the index date. Serving and the drift check anchor features the same way as each other. The calibration slice chose the model, the calibration and the decision threshold (max F1); the test slice was read once. F1 is at that threshold, not at 0.5.

| Target | Selected | Calibrated | Test ROC-AUC | F1 | Threshold | Brier |
|---|---|---|---|---|---|---|
| er_visit_30d | LightGBM | yes | 0.657 | 0.212 | 0.061 | 0.082 |
| care_need_90d | logistic regression | yes | 0.659 | 0.453 | 0.178 | 0.208 |
| care_need_30d | LightGBM | yes | 0.624 | 0.261 | 0.067 | 0.112 |

How to read this: every target beats chance, so the pipeline recovers the planted signal on a larger and more mixed cohort than before. All three targets are calibrated, so the app shows a percent; an uncalibrated target would show a rank, not a percent.

### History: 1,000-patient run

The earlier 1,000-patient database (version `20261127T205414Z-e025414`, [metrics/postgres-1000-planted.json](metrics/postgres-1000-planted.json)) gave test ROC-AUC 0.829 / 0.681 / 0.640 for er_visit_30d / care_need_90d / care_need_30d, with only 9 to 32 test positives per target and calibration skipped for two targets. Before the outcome rearrangement the same database scored at chance (0.417 / 0.555 / 0.408, [metrics/postgres-1000-before.json](metrics/postgres-1000-before.json)). An earlier 600-patient run without the planted signal is kept in [metrics/postgres-demo-600.json](metrics/postgres-demo-600.json).

## Sanity check: the pipeline recovers a known signal

Command: `python -m app.ml.train --source memory --n 20000 --label "demo: in-memory synthetic sanity check"`. Full output: [metrics/memory-sanity-20k.json](metrics/memory-sanity-20k.json). The in-memory cohort draws outcomes from a logistic function of the features, so a working pipeline must beat chance.

| Target | Selected | Test ROC-AUC | PR-AUC | Brier (before -> after calibration) | ECE (before -> after) | F1 at chosen threshold |
|---|---|---|---|---|---|---|
| er_visit_30d | logreg + sigmoid | 0.689 | 0.436 | 0.220 -> 0.169 | 0.222 -> 0.014 | 0.455 |
| care_need_90d | logreg + sigmoid | 0.691 | 0.626 | 0.222 -> 0.218 | 0.066 -> 0.021 | 0.629 |
| care_need_30d | logreg + sigmoid | 0.695 | 0.378 | 0.220 -> 0.147 | 0.266 -> 0.017 | 0.426 |

The logistic baseline wins here because the simulated outcome is itself logistic; XGBoost and LightGBM land 0.004 to 0.012 ROC-AUC below it (see the json). This is a check of the code, not evidence about real patients.

## Fairness by adherence archetype (test slice)

Each run reports, per archetype, the observed and mean predicted rate, the share flagged at the chosen threshold, recall, false positive rate and ROC-AUC (`targets.<target>.fairness` in the json; shown on the admin page). A group needs 5 positives and 5 negatives before its rates are compared. On the 1,000-patient run only care_need_90d had enough cases per group, and its gaps were within noise.

The planted rule makes poor adherence a risk factor, so the poor-adherence group is flagged more often by design. Archetype is not a model input (it is excluded as leaky) and the planted rule does not read it; it is only the grouping. Race and sex groups are not reported yet.

## Lifecycle

- Deploy: automatic after CI (lint, test, narrative-eval) passes on `main`.
- Drift: nightly check, PSI on every feature and every target's score, against a baseline built from the same population the live check scores. Last run: status `ok`.
- Retrain: weekly, and also when the drift check reports `drift`.
- Clinician overrides: an evaluation feedback set only, never training labels, until there are at least 200.

## Explanations

- Each prediction returns its top 8 features by absolute SHAP value. Trees use exact TreeSHAP (the libraries' native contribution output, checked against `shap.TreeExplainer` in the tests); the baseline uses linear SHAP with the training mean as background; the ensemble averages its members.
- SHAP explains the uncalibrated log-odds. Calibration is a monotone map applied afterwards, so it does not flip the direction of any feature.
- An LLM narrative is shown only if every feature it names is in that patient's top SHAP features, in the same direction and size order; otherwise the template is shown (see the README).

## Limits

- Synthetic data only; no external or prospective validation; not clinically validated.
- The feature-outcome link is planted by the documented rule above; recovering it checks the code, not clinical validity.
- 10,000 synthetic patients with a planted signal; test ROC-AUC 0.62 to 0.66, and no external validation.
- Fairness is reported by archetype only; race and sex are not reported.
- Attribution is not causation: a feature with a large SHAP value is not a treatment target.
- Scores from a target without calibration should not be read as calibrated risks; the app shows them as a rank, not a percent.
