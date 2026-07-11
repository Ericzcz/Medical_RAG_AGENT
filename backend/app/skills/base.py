from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any


@dataclass
class SkillContext:
    model: str
    chat_history: list[dict] | None = None
    session_id: str | None = None
    user_id: str | None = None
    redis_client: Any | None = None


@dataclass
class SkillResult:
    content: str
    confidence: float | None = None
    metadata: dict[str, Any] | None = None


class BaseSkill(ABC):
    name: str
    description: str
    parameters: dict

    def tool_schema(self) -> dict:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "strict": True,
        }

    @abstractmethod
    async def execute(self, arguments: dict, context: SkillContext) -> SkillResult:
        pass
