"""Patient creation + auto-classification (to be added to database_pg.py)."""
import uuid as uuid_mod
from datetime import date
from app.data.database_pg import get_session
from app.data.models_pg import Patient


async def create_patient(data: dict) -> dict:
    """Create a new patient and auto-assign adherence archetype based on demographics."""
    pid = uuid_mod.uuid4()

    # Auto-classify initial archetype based on SDOH risk + insurance
    # (real model would use more data; this is the cold-start heuristic)
    n_sdoh = len(data.get("sdoh_risk_factors", []))
    insurance = data.get("insurance_type", "private")
    age = data.get("age", 50)

    if n_sdoh == 0 and insurance in ("private", "employer"):
        archetype = "good"
    elif n_sdoh >= 2 or insurance == "uninsured":
        archetype = "poor"
    elif age > 75:
        archetype = "moderate"
    else:
        archetype = "moderate"

    async with get_session() as session:
        patient = Patient(
            patient_id=pid,
            first_name=data["first_name"],
            last_name=data["last_name"],
            age=age,
            gender=data.get("gender", "M"),
            race=data.get("race", ""),
            ethnicity=data.get("ethnicity", ""),
            address_city=data.get("city", ""),
            address_state=data.get("state", ""),
            address_zip=data.get("zip", ""),
            insurance_type=insurance,
            bmi=data.get("bmi", 25.0),
            smoker=data.get("smoker", False),
            n_sdoh_risks=n_sdoh,
            sdoh_risk_factors=data.get("sdoh_risk_factors", []),
            adherence_archetype=archetype,
        )
        session.add(patient)

    return {
        "patient_id": str(pid),
        "first_name": data["first_name"],
        "last_name": data["last_name"],
        "adherence_archetype": archetype,
        "classification_reason": f"Initial: {n_sdoh} SDOH risks, {insurance} insurance, age {age}",
    }
