import asyncio

from app.services.web_search_service import search_web
from app.skills.base import BaseSkill, SkillContext, SkillResult


class WebSearchSkill(BaseSkill):
    name = "search_web"
    description = "Search the web for recent or external medical information."

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A standalone search query.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    async def execute(self, arguments: dict, context: SkillContext) -> SkillResult:
        result = await asyncio.to_thread(search_web, arguments["query"])
        return SkillResult(content=result)
