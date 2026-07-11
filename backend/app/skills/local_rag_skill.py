from app.services.local_rag_service import search_local_knowledge
from app.skills.base import BaseSkill, SkillContext, SkillResult


class LocalRagSkill(BaseSkill):
    name = "search_local_knowledge"
    description = "Search the local medical knowledge base and return relevant evidence."

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A medical question to search in the local knowledge base.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    async def execute(self, arguments: dict, context: SkillContext) -> SkillResult:
        answer = await search_local_knowledge(
            arguments["query"],
            context.model,
            chat_history=context.chat_history,
        )
        return SkillResult(content=answer)