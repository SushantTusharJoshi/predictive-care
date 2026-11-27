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

- Source for the deployed demo model: Postgres seeded with `python -m app.data.ingest_synthea --fallback --max-patients 600` (synthetic patients, no real person).
- Index date = latest encounter in the database minus 90 days (`as_of` and `index_date` are in metrics.json). Every feature uses events strictly before the index date; labels use only the window after it.
- 17 features: age, sex, BMI, smoker, SDOH risk count, insurance type (one-hot), diagnosis and active-medication counts, adherence rate over the last 90 days and a year earlier, adherence trend, response latency, average systolic BP and heart rate. Archetype labels and prior ER counts are excluded as leaky (the script asserts this).
- Serving computes the same features with the same defaults in `get_patient_features`.

## Training (one script: `python -m app.ml.train`)

1. Order patients by patient time (first recorded encounter) and cut 70/10/20 into train, calibration and test. No patient is in two slices.
2. Fit four candidates on train: a logistic regression baseline (standardised, class-balanced), XGBoost, LightGBM, and the ensemble that averages the two tree models' log-odds.
3. On the calibration slice only: pick the candidate with the best ROC-AUC, choose isotonic or sigmoid calibration by 5-fold CV Brier, and pick the decision threshold that maximises F1. Calibration is skipped, and says why, when the slice has fewer than 10 positives or 10 negatives.
4. Report once on the untouched test slice: ROC-AUC, PR-AUC, Brier, precision, recall and F1 at the chosen threshold, reliability bins and ECE before and after calibration. Calibrated metrics are only claimed when test ECE did not get worse.
5. Save `models-<version>.pkl`, its `.sha256`, `metrics.json` and a `current.json` manifest. The API refuses a file whose hash does not match.

## Results of the deployed demo run

Command: `DATABASE_URL_SYNC=postgresql://pc_user@127.0.0.1:55433/predictive_care python -m app.ml.train --label "demo: 600 synthetic patients (ingest_synthea --fallback)"`, run on 2026-11-27 against a throwaway 600-patient database. Full output: [metrics/postgres-demo-600.json](metrics/postgres-demo-600.json). Tables printed by `python -m app.ml.report docs/metrics/postgres-demo-600.json`.

Split: train 420 patients (first encounter 2021-11-28 to 2022-04-10), calibration 60 (2022-04-12 to 2022-05-24), test 120 (2022-05-25 to 2024-08-30). Index date 2026-08-29.

| Target | Selected | Test ROC-AUC | PR-AUC | Brier | Threshold | Precision | Recall | F1 | Baseline ROC-AUC | Positives train/cal/test |
|---|---|---|---|---|---|---|---|---|---|---|
| er_visit_30d | logreg, no calibration | 0.462 | 0.047 | 0.227 | 0.500 | 0.021 | 0.500 | 0.040 | 0.462 | 32/0/2 |
| care_need_90d | logreg + sigmoid | 0.523 | 0.180 | 0.123 | 0.168 | 0.157 | 0.471 | 0.235 | 0.523 | 124/10/17 |
| care_need_30d | logreg, no calibration | 0.383 | 0.042 | 0.243 | 0.446 | 0.028 | 0.400 | 0.052 | 0.383 | 52/3/5 |

| Target | Calibration | ECE before | ECE after | Claimed |
|---|---|---|---|---|
| er_visit_30d | skipped: 0 positives in the calibration slice | 0.424 | 0.424 | no |
| care_need_90d | sigmoid (CV Brier 0.136 vs isotonic 0.142) | 0.360 | 0.051 | yes |
| care_need_30d | skipped: 3 positives in the calibration slice | 0.430 | 0.430 | no |

How to read this: the scores are at chance level, and that is the expected, honest result. The fallback generator draws encounters at random, independently of every patient feature, so there is no signal to learn. A model that scored well here would point to leakage. The test slice holds 2 to 17 positives per target, so every number has a wide margin of error. The XGBoost and LightGBM rows for every target are in metrics.json.

## Sanity check: the pipeline recovers a known signal

Command: `python -m app.ml.train --source memory --n 20000 --label "demo: in-memory synthetic sanity check"`. Full output: [metrics/memory-sanity-20k.json](metrics/memory-sanity-20k.json). The in-memory cohort draws outcomes from a logistic function of the features, so a working pipeline must beat chance.

| Target | Selected | Test ROC-AUC | PR-AUC | Brier (before -> after calibration) | ECE (before -> after) | F1 at chosen threshold |
|---|---|---|---|---|---|---|
| er_visit_30d | logreg + sigmoid | 0.689 | 0.436 | 0.220 -> 0.169 | 0.222 -> 0.014 | 0.455 |
| care_need_90d | logreg + sigmoid | 0.691 | 0.626 | 0.222 -> 0.218 | 0.066 -> 0.021 | 0.629 |
| care_need_30d | logreg + sigmoid | 0.695 | 0.378 | 0.220 -> 0.147 | 0.266 -> 0.017 | 0.426 |

The logistic baseline wins here because the simulated outcome is itself logistic; XGBoost and LightGBM land 0.004 to 0.012 ROC-AUC below it (see the json). This is a check of the code, not evidence about real patients.

## Explanations

- Each prediction returns its top 8 features by absolute SHAP value. Trees use exact TreeSHAP (the libraries' native contribution output, checked against `shap.TreeExplainer` in the tests); the baseline uses linear SHAP with the training mean as background; the ensemble averages its members.
- SHAP explains the uncalibrated log-odds. Calibration is a monotone map applied afterwards, so it does not flip the direction of any feature.
- An LLM narrative is shown only if every feature it names is in that patient's top SHAP features, in the same direction and size order; otherwise the template is shown (see the README).

## Limits

- Synthetic data only; no external or prospective validation; not clinically validated.
- The 600-patient demo is too small for stable metrics, and its generator has no feature-outcome link.
- Fairness across groups has not been evaluated in this run.
- Attribution is not causation: a feature with a large SHAP value is not a treatment target.
- Probabilities from targets without calibration (marked "no" above) should not be read as calibrated risks.
