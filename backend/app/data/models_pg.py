"""SQLAlchemy ORM models for PredictiveCare v3.1 PostgreSQL schema.
Includes: PharmacyRefill, TrustScore, ReminderLog, SchedulingRecommendation,
HitlAction — all required by the requirements spec.
"""
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, Date, DateTime, Text,
    ForeignKey, Index, JSON, func
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import declarative_base, relationship
import uuid

Base = declarative_base()


class Patient(Base):
    __tablename__ = "patients"
    patient_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    date_of_birth = Column(Date)
    age = Column(Integer)
    gender = Column(String(20))
    race = Column(String(50))
    ethnicity = Column(String(50))
    address_city = Column(String(100))
    address_state = Column(String(50))
    address_zip = Column(String(20))
    insurance_type = Column(String(50))
    bmi = Column(Float)
    smoker = Column(Boolean, default=False)
    n_sdoh_risks = Column(Integer, default=0)
    sdoh_risk_factors = Column(ARRAY(String), default=[])
    adherence_archetype = Column(String(30))
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    diagnoses = relationship("Diagnosis", back_populates="patient", lazy="selectin")
    medications = relationship("Medication", back_populates="patient", lazy="selectin")
    adherence_events = relationship("AdherenceEvent", back_populates="patient", lazy="noload")
    lab_results = relationship("LabResult", back_populates="patient", lazy="noload")
    vitals = relationship("Vital", back_populates="patient", lazy="noload")
    encounters = relationship("Encounter", back_populates="patient", lazy="noload")
    pharmacy_refills = relationship("PharmacyRefill", back_populates="patient", lazy="noload")
    trust_scores = relationship("TrustScore", back_populates="patient", lazy="noload")
    reminder_logs = relationship("ReminderLog", back_populates="patient", lazy="noload")

    __table_args__ = (
        Index("ix_patients_name", "last_name", "first_name"),
        Index("ix_patients_archetype", "adherence_archetype"),
        Index("ix_patients_age", "age"),
    )


class Diagnosis(Base):
    __tablename__ = "diagnoses"
    diagnosis_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.patient_id", ondelete="CASCADE"), nullable=False)
    condition = Column(String(200), nullable=False)
    icd10_code = Column(String(20))
    severity = Column(String(20))
    status = Column(String(20), default="active")
    onset_date = Column(Date)
    resolved_date = Column(Date)
    patient = relationship("Patient", back_populates="diagnoses")
    __table_args__ = (Index("ix_diagnoses_patient", "patient_id"), Index("ix_diagnoses_condition", "condition"))


class Medication(Base):
    __tablename__ = "medications"
    medication_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.patient_id", ondelete="CASCADE"), nullable=False)
    medication_name = Column(String(200), nullable=False)
    rxnorm_code = Column(String(20))
    dosage = Column(String(100))
    frequency = Column(String(50))
    route = Column(String(50))
    start_date = Column(Date)
    end_date = Column(Date)
    active = Column(Boolean, default=True)
    reminder_time_1 = Column(String(10))
    reminder_time_2 = Column(String(10))
    patient = relationship("Patient", back_populates="medications")
    __table_args__ = (Index("ix_medications_patient", "patient_id"), Index("ix_medications_active", "active"))


class AdherenceEvent(Base):
    __tablename__ = "adherence_events"
    event_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.patient_id", ondelete="CASCADE"), nullable=False)
    medication_id = Column(UUID(as_uuid=True), ForeignKey("medications.medication_id", ondelete="CASCADE"))
    medication_name = Column(String(200))
    event_date = Column(Date, nullable=False)
    taken = Column(Boolean, nullable=False)
    taken_time = Column(String(10))
    reminder_15min_sent = Column(Boolean, default=False)
    reminder_5min_sent = Column(Boolean, default=False)
    missed_alert_sent = Column(Boolean, default=False)
    patient_response = Column(String(20))  # YES/NO/SNOOZE/NO_RESPONSE
    response_latency_min = Column(Float)
    source = Column(String(50))
    system_confidence = Column(Float)
    clinician_verified = Column(Boolean, default=False)
    patient = relationship("Patient", back_populates="adherence_events")
    __table_args__ = (Index("ix_adherence_patient_date", "patient_id", "event_date"), Index("ix_adherence_medication", "medication_name"))


class PharmacyRefill(Base):
    __tablename__ = "pharmacy_refills"
    refill_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.patient_id", ondelete="CASCADE"), nullable=False)
    medication_id = Column(UUID(as_uuid=True), ForeignKey("medications.medication_id", ondelete="CASCADE"))
    medication_name = Column(String(200))
    refill_date = Column(Date, nullable=False)
    supply_days = Column(Integer)
    expected_gap_days = Column(Integer)
    actual_gap_days = Column(Integer)
    pharmacy_name = Column(String(200))
    patient = relationship("Patient", back_populates="pharmacy_refills")
    __table_args__ = (Index("ix_refills_patient", "patient_id"), Index("ix_refills_date", "refill_date"))


class TrustScore(Base):
    __tablename__ = "trust_scores"
    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.patient_id", ondelete="CASCADE"), nullable=False)
    medication_name = Column(String(200), nullable=False)
    score = Column(Float, nullable=False)
    classification = Column(String(30))
    components = Column(JSON)
    computed_at = Column(DateTime, server_default=func.now())
    patient = relationship("Patient", back_populates="trust_scores")
    __table_args__ = (Index("ix_trust_patient_med", "patient_id", "medication_name"),)


class ReminderLog(Base):
    __tablename__ = "reminder_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.patient_id", ondelete="CASCADE"), nullable=False)
    medication_name = Column(String(200))
    reminder_type = Column(String(30))
    scheduled_time = Column(String(10))
    sent_at = Column(DateTime, server_default=func.now())
    channel = Column(String(30))
    acknowledged = Column(Boolean, default=False)
    acknowledged_at = Column(DateTime)
    patient = relationship("Patient", back_populates="reminder_logs")
    __table_args__ = (Index("ix_reminder_patient", "patient_id"),)


class LabResult(Base):
    __tablename__ = "lab_results"
    lab_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.patient_id", ondelete="CASCADE"), nullable=False)
    test_name = Column(String(200), nullable=False)
    loinc_code = Column(String(20))
    value = Column(Float)
    unit = Column(String(50))
    reference_low = Column(Float)
    reference_high = Column(Float)
    flag = Column(String(20))
    lab_date = Column(Date, nullable=False)
    patient = relationship("Patient", back_populates="lab_results")
    __table_args__ = (Index("ix_labs_patient_date", "patient_id", "lab_date"), Index("ix_labs_test", "test_name"))


class Vital(Base):
    __tablename__ = "vitals"
    vital_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.patient_id", ondelete="CASCADE"), nullable=False)
    vital_date = Column(Date, nullable=False)
    systolic_bp = Column(Integer)
    diastolic_bp = Column(Integer)
    heart_rate = Column(Integer)
    temperature = Column(Float)
    spo2 = Column(Float)
    weight_lbs = Column(Float)
    height_inches = Column(Float)
    patient = relationship("Patient", back_populates="vitals")
    __table_args__ = (Index("ix_vitals_patient_date", "patient_id", "vital_date"),)


class Encounter(Base):
    __tablename__ = "encounters"
    encounter_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.patient_id", ondelete="CASCADE"), nullable=False)
    encounter_date = Column(Date, nullable=False)
    encounter_type = Column(String(50))
    chief_complaint = Column(String(200))
    provider = Column(String(200))
    discharge_disposition = Column(String(100))
    patient = relationship("Patient", back_populates="encounters")
    __table_args__ = (Index("ix_encounters_patient_date", "patient_id", "encounter_date"), Index("ix_encounters_type", "encounter_type"))


class SchedulingRecommendation(Base):
    __tablename__ = "scheduling_recommendations"
    recommendation_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.patient_id", ondelete="CASCADE"), nullable=False)
    prediction_type = Column(String(50))
    predicted_probability = Column(Float)
    recommended_visit_type = Column(String(50))
    recommended_window_start = Column(Date)
    recommended_window_end = Column(Date)
    status = Column(String(20), default="pending")
    actioned_by = Column(String(100))
    actioned_at = Column(DateTime)
    modification_reason = Column(Text)
    created_at = Column(DateTime, server_default=func.now())
    __table_args__ = (Index("ix_sched_patient", "patient_id"), Index("ix_sched_status", "status"))


class HitlAction(Base):
    __tablename__ = "hitl_actions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.patient_id", ondelete="CASCADE"))
    clinician_username = Column(String(100), nullable=False)
    action_type = Column(String(50))
    prediction_type = Column(String(50))
    original_value = Column(Float)
    overridden_value = Column(Float)
    reason = Column(Text)
    feedback_rating = Column(String(20))
    created_at = Column(DateTime, server_default=func.now())
    __table_args__ = (Index("ix_hitl_clinician", "clinician_username"), Index("ix_hitl_patient", "patient_id"))


class AuditLog(Base):
    __tablename__ = "audit_log"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, server_default=func.now(), nullable=False)
    username = Column(String(100))
    role = Column(String(50))
    action = Column(String(100), nullable=False)
    resource = Column(String(200))
    resource_type = Column(String(50))
    patient_id_accessed = Column(String(50))
    details = Column(JSON)
    ip_address = Column(String(50))
    user_agent = Column(String(500))
    __table_args__ = (Index("ix_audit_timestamp", "timestamp"), Index("ix_audit_user", "username"), Index("ix_audit_patient", "patient_id_accessed"))


class ModelMetadata(Base):
    __tablename__ = "model_metadata"
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    version = Column(String(50), nullable=False)
    trained_at = Column(DateTime, server_default=func.now())
    metrics = Column(JSON)
    feature_columns = Column(JSON)
    n_training_samples = Column(Integer)
    file_path = Column(String(500))
    is_active = Column(Boolean, default=True)
    __table_args__ = (Index("ix_model_active", "model_name", "is_active"),)
