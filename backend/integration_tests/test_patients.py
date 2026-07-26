from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from app.main import app
from app.services.auth import create_token
from unittest.mock import AsyncMock, patch

client = TestClient(app)


def get_headers():
    token = create_token({
        "username": "admin",
        "role": "admin",
        "name": "System Admin"
    })
    return {"Authorization": f"Bearer {token}"}


@patch("app.main_pg.search_patients", new_callable=AsyncMock)
def test_list_patients(mock_search):

    mock_search.return_value = {
        "patients": [
            {
                "patient_id": "P001",
                "first_name": "John",
                "last_name": "Doe",
                "age": 60,
                "adherence_archetype": "good",
                "n_sdoh_risks": 1
            }
        ],
        "total": 1
    }

    response = client.get(
        "/patients",
        headers=get_headers()
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total"] == 1
    assert data["patients"][0]["patient_id"] == "P001"



from unittest.mock import AsyncMock, patch


@patch("app.main_pg.log_audit", new_callable=AsyncMock)
@patch("app.main_pg.create_patient", new_callable=AsyncMock)
def test_create_patient(mock_create_patient, mock_log_audit):

    mock_create_patient.return_value = {
        "patient_id": "P1001",
        "message": "Patient created"
    }

    mock_log_audit.return_value = None

    patient = {
        "first_name": "John",
        "last_name": "Doe",
        "age": 45,
        "gender": "Male"
    }

    response = client.post(
        "/patients",
        json=patient,
        headers=get_headers()
    )

    assert response.status_code == 200

    data = response.json()

    assert data["patient_id"] == "P1001"

    mock_create_patient.assert_called_once()
    mock_log_audit.assert_called_once()