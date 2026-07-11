import json
import asyncio

from langsmith import traceable
from langsmith.wrappers import wrap_openai

from openai import AsyncOpenAI

from app.skills.base import SkillContext
from app.skills.registry import get_default_skills, get_skill_map, get_tool_schemas


@traceable(
        name="Week4 Agent",
        run_type="chain",
        process_inputs=lambda inputs: {
            "query": inputs["user_query"],
        },
    )
async def run_agent(
    user_query: str,
    *,
    model: str = "gpt-5.5",
    tool_response_model: str | None = None,
    instructions: str | None = None,
    chat_history: list[dict] | None = None,
) -> str:
    client = wrap_openai(AsyncOpenAI())

    skills = get_default_skills()
    tools = get_tool_schemas(skills)
    skill_map = get_skill_map(skills)

    skill_context = SkillContext(
        model=model,
        chat_history=chat_history,
    )

    default_instructions = """
        You are a ReAct-style medical agent. Choose between local knowledge-base retrieval
        and web search depending on the question.

        When calling the search_web tool, first use the conversation history to rewrite
        pronouns, omitted references, and context-dependent questions into standalone,
        explicit search queries. Do not pass vague references such as "it", "this",
        "the above", or "what we just discussed" directly to search_web.

        When calling the search_local_knowledge tool, you may keep the user's original
        question because the local RAG chain rewrites the question using chat_history.
        """
    long_term_memory_instructions = instructions or ""
    final_instructions = f"""
        {default_instructions}

        {long_term_memory_instructions}
        """

    input_items = [
        *format_chat_history_for_responses(chat_history),
        {"role": "user", "content": user_query}
        ]
    
    response = await client.responses.create(
        model=model,
        tools=tools,
        input=input_items,
        instructions=final_instructions,
    )

    while True:
        input_items += response.output
        function_calls = [item for item in response.output if item.type == "function_call"]
        if not function_calls:
            return response.output_text

        for item in function_calls:
            arguments = json.loads(item.arguments)

            skill = skill_map.get(item.name)
            if skill is None:
                raise RuntimeError(f"Unknown skill: {item.name}")

            result = await skill.execute(arguments, skill_context)
            output = result.content

            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": output,
                }
            )

        response = await client.responses.create(
            model=tool_response_model or model,
            tools=tools,
            input=input_items,
            instructions=final_instructions,
        )


async def run_one_agent(query: str, model: str, sem: asyncio.Semaphore) -> dict[str, str | None]:
    async with sem:
        try:
            answer = await run_agent(query, model=model)
            return {
                "query": query,
                "answer": answer,
                "error": None,
            }
        except Exception as e:
            return {
                "query": query,
                "answer": None,
                "error": str(e),
            }

@traceable(name="Week4 Agent Batch", run_type="chain")
async def run_agent_batch(
    queries: list[str],
    model: str,
    max_concurrency: int = 3,
) -> list[dict[str, str | None]]:
    sem = asyncio.Semaphore(max_concurrency)
    tasks = [
        run_one_agent(query, model, sem)
        for query in queries
    ]
    return await asyncio.gather(*tasks)


def format_chat_history_for_responses(chat_history: list[dict] | None) -> list[dict]:
    if not chat_history:
        return []

    role_map = {
        "human": "user",
        "ai": "assistant",
        "user": "user",
        "assistant": "assistant",
        "system": "system",
    }

    input_items = []
    for message in chat_history:
        role = role_map.get(message.get("role"))
        content = message.get("content")
        if role and content:
            input_items.append({"role": role, "content": content})

    return input_items
