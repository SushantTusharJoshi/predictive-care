"""
Similar patient matching using K-Nearest Neighbors on demographics.
Works with both SQLite (v2) and PostgreSQL (v3) backends.
"""
import numpy as np
import pickle
from pathlib import Path
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder

MODEL_PATH = Path("data/models/knn_similar.pkl")

# Global state
_knn_model = None
_scaler = None
_patient_ids = None
_patient_info = {}  # patient_id -> {name, age, gender, race, bmi, insurance, archetype}
_label_encoders = {}


def _encode_features(patients_data):
    """Encode demographic features for KNN."""
    global _label_encoders, _patient_info

    race_enc = LabelEncoder()
    ins_enc = LabelEncoder()
    arch_enc = LabelEncoder()

    all_races = [p.get("race", "unknown") or "unknown" for p in patients_data]
    all_ins = [p.get("insurance_type", "private") or "private" for p in patients_data]
    all_arch = [p.get("adherence_archetype", "moderate") or "moderate" for p in patients_data]

    race_enc.fit(all_races)
    ins_enc.fit(all_ins)
    arch_enc.fit(all_arch)

    _label_encoders = {"race": race_enc, "insurance": ins_enc, "archetype": arch_enc}

    features = []
    ids = []
    for p in patients_data:
        pid = p["patient_id"]
        ids.append(pid)
        _patient_info[pid] = {
            "name": f"{p.get('first_name', '')} {p.get('last_name', '')}".strip(),
            "age": p.get("age", 0),
            "gender": p.get("gender", ""),
            "race": p.get("race", ""),
            "bmi": p.get("bmi", 25),
            "insurance_type": p.get("insurance_type", ""),
            "adherence_archetype": p.get("adherence_archetype", "moderate"),
        }
        features.append([
            p.get("age", 0),
            1 if p.get("gender") == "M" else 0,
            p.get("bmi", 25) or 25,
            race_enc.transform([p.get("race", "unknown") or "unknown"])[0],
            ins_enc.transform([p.get("insurance_type", "private") or "private"])[0],
            p.get("n_sdoh_risks", 0) or 0,
            arch_enc.transform([p.get("adherence_archetype", "moderate") or "moderate"])[0],
        ])

    return np.array(features), ids


def build_knn_index_from_pg():
    """Build KNN index from PostgreSQL database."""
    global _knn_model, _scaler, _patient_ids
    from sqlalchemy import text as sa_text
    from app.data.database_pg import get_sync_engine

    engine = get_sync_engine()
    with engine.connect() as conn:
        result = conn.execute(sa_text(
            "SELECT patient_id::text, first_name, last_name, age, gender, bmi, "
            "race, insurance_type, n_sdoh_risks, adherence_archetype "
            "FROM patients LIMIT 100000"
        ))
        patients_data = [dict(r._mapping) for r in result]

    print(f"  Building KNN index from {len(patients_data)} patients...")
    features, ids = _encode_features(patients_data)

    _scaler = StandardScaler()
    features_scaled = _scaler.fit_transform(features)

    _knn_model = NearestNeighbors(n_neighbors=11, metric="euclidean", algorithm="ball_tree")
    _knn_model.fit(features_scaled)
    _patient_ids = ids

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "model": _knn_model,
            "scaler": _scaler,
            "patient_ids": _patient_ids,
            "label_encoders": _label_encoders,
            "patient_info": _patient_info,
        }, f)
    print(f"  KNN index built: {len(ids)} patients")


def build_knn_index():
    """Build KNN index from SQLite (v2 compat)."""
    global _knn_model, _scaler, _patient_ids
    try:
        from app.data.database import get_similar_patients_data
        patients_data = get_similar_patients_data()
    except Exception:
        build_knn_index_from_pg()
        return

    features, ids = _encode_features(patients_data)
    _scaler = StandardScaler()
    features_scaled = _scaler.fit_transform(features)

    _knn_model = NearestNeighbors(n_neighbors=11, metric="euclidean", algorithm="ball_tree")
    _knn_model.fit(features_scaled)
    _patient_ids = ids

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "model": _knn_model,
            "scaler": _scaler,
            "patient_ids": _patient_ids,
            "label_encoders": _label_encoders,
            "patient_info": _patient_info,
        }, f)
    print(f"  KNN index built: {len(ids)} patients")


def load_knn_index():
    """Load pre-built KNN index."""
    global _knn_model, _scaler, _patient_ids, _label_encoders, _patient_info
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        _knn_model = data["model"]
        _scaler = data["scaler"]
        _patient_ids = data["patient_ids"]
        _label_encoders = data["label_encoders"]
        _patient_info = data.get("patient_info", {})
        return True
    return False


def find_similar_patients(patient_id: str, k: int = 10, knn_data=None) -> list:
    """Find k most similar patients by demographics."""
    global _knn_model, _scaler, _patient_ids, _label_encoders, _patient_info

    # Load KNN data from arg or from file
    if knn_data:
        _knn_model = knn_data["model"]
        _scaler = knn_data["scaler"]
        _patient_ids = knn_data["patient_ids"]
        _label_encoders = knn_data["label_encoders"]
        _patient_info = knn_data.get("patient_info", {})
    elif _knn_model is None:
        if not load_knn_index():
            return []

    # Get patient info
    info = _patient_info.get(patient_id)
    if not info:
        return []

    try:
        query = np.array([[
            info["age"],
            1 if info["gender"] == "M" else 0,
            info.get("bmi", 25) or 25,
            _label_encoders["race"].transform([info.get("race", "unknown") or "unknown"])[0],
            _label_encoders["insurance"].transform([info.get("insurance_type", "private") or "private"])[0],
            0,  # n_sdoh_risks not always in info
            _label_encoders["archetype"].transform([info.get("adherence_archetype", "moderate") or "moderate"])[0],
        ]])
    except (ValueError, KeyError):
        return []

    query_scaled = _scaler.transform(query)
    distances, indices = _knn_model.kneighbors(query_scaled, n_neighbors=min(k + 1, len(_patient_ids)))

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        sim_id = _patient_ids[idx]
        if sim_id == patient_id:
            continue
        sim = _patient_info.get(sim_id, {})
        if sim:
            results.append({
                "patient_id": sim_id,
                "name": sim.get("name", "Unknown"),
                "age": sim.get("age", 0),
                "gender": sim.get("gender", ""),
                "race": sim.get("race", ""),
                "bmi": sim.get("bmi", 0),
                "insurance_type": sim.get("insurance_type", ""),
                "adherence_archetype": sim.get("adherence_archetype", ""),
                "avg_adherence_pct": 0,  # Filled by caller if needed
                "similarity_distance": round(float(dist), 3),
                "similarity_score": round(max(0, 1 - float(dist) / 10) * 100, 1),
            })
        if len(results) >= k:
            break

    return results


def predict_from_similar(similar_patients: list) -> dict:
    """Predict behavior from a list of similar patients."""
    if not similar_patients:
        return {"error": "No similar patients found"}

    archetypes = [s.get("adherence_archetype", "moderate") for s in similar_patients]
    arch_dist = Counter(archetypes)

    return {
        "n_similar": len(similar_patients),
        "predicted_adherence_rate": 0,  # Needs DB lookup for real value
        "likely_archetype": arch_dist.most_common(1)[0][0] if arch_dist else "moderate",
        "archetype_distribution": dict(arch_dist),
        "confidence": "high" if len(similar_patients) >= 8 else ("medium" if len(similar_patients) >= 4 else "low"),
    }
