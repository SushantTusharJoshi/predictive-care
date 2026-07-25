"""Tests for HIPAA middleware and de-identification."""
from app.middleware.hipaa import de_identify_for_llm


class TestDeIdentify:
    def test_strips_pii_fields(self):
        data = {
            "first_name": "John",
            "last_name": "Doe",
            "patient_id": "123",
            "age": 45,
            "risk_score": 0.72,
            "diagnosis": "hypertension",
        }
        safe = de_identify_for_llm(data)
        assert "first_name" not in safe
        assert "last_name" not in safe
        assert "patient_id" not in safe
        assert safe["age"] == 45
        assert safe["risk_score"] == 0.72
        assert safe["diagnosis"] == "hypertension"

    def test_strips_email_and_ssn(self):
        data = {"email": "test@example.com", "ssn": "123-45-6789", "bmi": 28.5}
        safe = de_identify_for_llm(data)
        assert "email" not in safe
        assert "ssn" not in safe
        assert safe["bmi"] == 28.5

    def test_strips_address_fields(self):
        data = {
            "address_city": "Boston",
            "address_state": "MA",
            "address_zip": "02115",
            "smoker": False,
        }
        safe = de_identify_for_llm(data)
        assert "address_city" not in safe
        assert "address_state" not in safe
        assert "address_zip" not in safe
        assert safe["smoker"] is False

    def test_empty_dict(self):
        assert de_identify_for_llm({}) == {}

    def test_no_pii_passthrough(self):
        data = {"diagnosis": "diabetes", "lab_glucose": 126}
        safe = de_identify_for_llm(data)
        assert safe == data

    def test_case_insensitive_matching(self):
        data = {"Date_Of_Birth": "1980-01-01", "Phone": "555-1234", "lab_a1c": 7.2}
        safe = de_identify_for_llm(data)
        assert "Date_Of_Birth" not in safe
        assert "Phone" not in safe
        assert safe["lab_a1c"] == 7.2
