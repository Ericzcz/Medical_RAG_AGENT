from app.services import get_medical_records
from app.skills.base import BaseSkill, SkillContext, SkillResult
import json

class MedicalRecordQuerySkill(BaseSkill):
    name = "query_medical_records"
    description = (
        "Query the user's stored medical records, including allergies, symptoms, "
        "medications, diagnoses, procedures, vitals, and clinical notes."
    )

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Question about the user's stored medical records.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of recent medical records to retrieve.",
            },
            "field": {
                "type": "string",
                "enum": [
                    "allergies",
                    "symptoms",
                    "medications",
                    "diagnoses",
                    "procedures",
                    "vitals",
                    "notes",
                    "all",
                ],
                "description": "The medical record field to query.",
            }
        },
        "required": ["query", "limit", "field"],
        "additionalProperties": False,
    }

    async def execute(self, arguments: dict, context: SkillContext) -> SkillResult:
        if not context.user_id:
            return SkillResult(content="Unable to get medical record because user_id is missing.")
        
        limit = arguments.get("limit", 20)
        field = arguments["field"]

        records = await get_medical_records(
            user_id=context.user_id,
            field=field,
            limit=limit,
            )
        
        if not records:
            return SkillResult(content="No medical records found for this user.")
        
        return SkillResult(
            content=json.dumps(records, ensure_ascii=False)
        )

        

    


