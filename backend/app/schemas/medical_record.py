from typing import Any, Literal
from pydantic import BaseModel, Field


class MedicalRecordCreate(BaseModel):
    record_type: Literal["clinical_note"] = "clinical_note"
    symptoms: list[str] = Field(default_factory=list)
    diagnoses: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    procedures: list[str] = Field(default_factory=list)
    vitals: dict[str, Any] = Field(default_factory=dict)
    visit_date: str | None = None
    notes: str
    source_text: str


class MedicalRecordStored(MedicalRecordCreate):
    record_id: str
    user_id: str
    source_session_id: str | None = None
    created_at: str