from app.skills.local_rag_skill import LocalRagSkill
from app.skills.web_search_skill import WebSearchSkill
from app.skills.medical_record_insert_skill import MedicalRecordInsertSkill


def get_default_skills():
    return [
        LocalRagSkill(),
        WebSearchSkill(),
        MedicalRecordInsertSkill(),
    ]


def get_tool_schemas(skills):
    return [skill.tool_schema() for skill in skills]


def get_skill_map(skills):
    return {skill.name: skill for skill in skills}
