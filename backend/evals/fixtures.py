"""SHAP fixtures and seeded bad narratives for the narrative eval suite.

Pure data, no imports, so evals/check_report.py can run without the app's dependencies.
"""

FIXTURES = [
    {"id": "er-high", "target": "er_visit_30d", "patient": {"age": 31, "bmi": 34.0, "insurance_type": "medicaid",
                                                           "n_sdoh_risks": 2, "diagnoses": [1, 2]},
     "pred": {"probability": 0.81, "top_features": [
         {"feature": "adherence_rate_90d", "shap_value": 0.42, "value": 48.0},
         {"feature": "bmi", "shap_value": 0.21, "value": 34.0},
         {"feature": "age", "shap_value": -0.15, "value": 31},
         {"feature": "smoker", "shap_value": 0.08, "value": 1},
         {"feature": "avg_hr", "shap_value": -0.05, "value": 70}]}},
    {"id": "care90-low", "target": "care_need_90d", "patient": {"age": 44, "bmi": 23.5, "insurance_type": "private",
                                                               "n_sdoh_risks": 0, "diagnoses": []},
     "pred": {"probability": 0.12, "top_features": [
         {"feature": "adherence_rate_90d", "shap_value": -0.38, "value": 96.0},
         {"feature": "n_diagnoses", "shap_value": -0.22, "value": 0},
         {"feature": "avg_response_latency", "shap_value": 0.09, "value": 41.0},
         {"feature": "bmi", "shap_value": -0.04, "value": 23.5},
         {"feature": "age", "shap_value": 0.02, "value": 44}]}},
    {"id": "care30-moderate", "target": "care_need_30d", "patient": {"age": 72, "bmi": 29.1,
                                                                    "insurance_type": "medicare", "n_sdoh_risks": 1,
                                                                    "diagnoses": [1, 2, 3, 4]},
     "pred": {"probability": 0.47, "top_features": [
         {"feature": "age", "shap_value": 0.33, "value": 72},
         {"feature": "n_diagnoses", "shap_value": 0.25, "value": 4},
         {"feature": "adherence_trend", "shap_value": 0.11, "value": -12.0},
         {"feature": "avg_sbp", "shap_value": -0.06, "value": 118.0},
         {"feature": "insurance_medicare", "shap_value": 0.03, "value": 1}]}},
]

# Each one breaks a rule for fixture er-high (condition-name breaks two: out-of-model and diagnosis)
SEEDED_BAD = [
    {"id": "dosing", "text": "Risk is high at 81%. Increase metformin to 1000 mg daily. Clinician review recommended."},
    {"id": "diagnosis", "text": "Risk is high at 81%. The patient likely has heart failure. Clinician review recommended."},
    {"id": "causal", "text": "Risk is high at 81%. Low adherence causes the ER visits. Clinician review recommended."},
    {"id": "wrong-direction", "text": "Risk is high at 81%. Higher BMI lowers the model's score. "
                                      "Clinician review recommended."},
    {"id": "out-of-model", "text": "Risk is high at 81%. An elevated HbA1c raises the score. "
                                   "Clinician review recommended."},
    {"id": "dosing-start-on-a", "text": "Risk is high at 81%. The patient should be started on a statin. "
                                        "Clinician review recommended."},
    {"id": "condition-name", "text": "Risk is high at 81%. Depression raises the score. Clinician review recommended."},
    {"id": "negated-direction", "text": "Risk is high at 81%. Recent adherence does not raise the risk score. "
                                        "Clinician review recommended."},
    {"id": "less-likely", "text": "Risk is high at 81%. Recent adherence makes an ER visit less likely. "
                                  "Clinician review recommended."},
    {"id": "direction-first", "text": "Risk is high at 81%. What lowers risk most is smoking. "
                                      "Clinician review recommended."},
]
