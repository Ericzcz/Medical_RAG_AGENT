from app.services.medical_record_service import save_medical_record
from app.skills.base import BaseSkill, SkillContext, SkillResult
from app.schemas import MedicalRecordCreate

class MedicalRecordInsertSkill(BaseSkill):
    name = "insert_medical_record"
    description = (
        "Save user-provided medical record information such as symptoms, "
        "allergies, medications, diagnoses, procedures, vitals, and notes. "
        "Use this only when the user explicitly asks to record, save, or store medical information."
    )

    parameters = {
        "type": "object",
        "properties": {
            "record_type": {
                "type": "string",
                "enum": ["clinical_note"],
                "description": "The type of medical record.",
            },
            "symptoms": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Symptoms explicitly mentioned by the user.",
            },
            "diagnoses": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Diagnoses explicitly mentioned by the user.",
            },
            "medications": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Medications explicitly mentioned by the user.",
            },
            "allergies": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Allergies explicitly mentioned by the user.",
            },
            "procedures": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Medical procedures explicitly mentioned by the user.",
            },
            "vitals": {
                "type": "object",
                "properties": {
                    "blood_pressure": {"type": ["string", "null"]},
                    "heart_rate": {"type": ["string", "null"]},
                    "temperature": {"type": ["string", "null"]},
                    "respiratory_rate": {"type": ["string", "null"]},
                    "oxygen_saturation": {"type": ["string", "null"]},
                    "height": {"type": ["string", "null"]},
                    "weight": {"type": ["string", "null"]},
                },
                "required": [
                    "blood_pressure",
                    "heart_rate",
                    "temperature",
                    "respiratory_rate",
                    "oxygen_saturation",
                    "height",
                    "weight",
                ],
                "additionalProperties": False,
                "description": "Vital signs explicitly mentioned by the user.",
            },
            "visit_date": {
                "type": ["string", "null"],
                "description": "Visit date if explicitly provided.",
            },
            "notes": {
                "type": "string",
                "description": "A concise medical note based only on user-provided information.",
            },
            "source_text": {
                "type": "string",
                "description": "The original user text that this record was extracted from.",
            },
        },
        "required": [
            "record_type",
            "symptoms",
            "diagnoses",
            "medications",
            "allergies",
            "procedures",
            "vitals",
            "visit_date",
            "notes",
            "source_text",
        ],
        "additionalProperties": False,
    }

    async def execute(self, arguments: dict, context: SkillContext) -> SkillResult:
        if not context.user_id:
            return SkillResult(content="Unable to save medical record because user_id is missing.")

        record = MedicalRecordCreate(**arguments)

        stored_record = await save_medical_record(
            user_id=context.user_id,
            session_id=context.session_id,
            record=record,
        )

        return SkillResult(
            content=(
                "Medical record saved successfully. "
                f"record_id={stored_record.record_id}"
            )
        )
