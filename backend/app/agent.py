import json
import asyncio
from typing import Awaitable, Callable, Dict

from langsmith import traceable
from langsmith.wrappers import wrap_openai

from openai import AsyncOpenAI
from tavily import TavilyClient

from .rag_chain import search_local_knowledge


def search_web(query: str) -> str:
    tavily_client = TavilyClient()
    response = tavily_client.search(
        query=query,
        topic="general",
        search_depth="advanced",
        max_results=5,
        include_answer=False,
        include_raw_content=False,
    )

    results = response.get("results", [])
    if not results:
        return "No relevant web results found."

    blocks = []
    for idx, item in enumerate(results, 1):
        blocks.append(
            "\n".join(
                [
                    f"[{idx}]",
                    f"title: {item.get('title', '')}",
                    f"url: {item.get('url', '')}",
                    f"content: {item.get('content', '')}",
                    f"score: {item.get('score', '')}",
                ]
            )
        )

    return "\n\n".join(blocks)



def get_tools():
    return [
        {
            "type": "function",
            "name": "search_local_knowledge",
            "description": (
                "Search the local machine learning knowledge base and return "
                "relevant evidence snippets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A machine learning question to search in the local "
                            "knowledge base."
                        ),
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "search_web",
            "description": "Search the web for recent or external information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A query to search on the web.",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    ]

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
) -> str:
    client = wrap_openai(AsyncOpenAI())
    tools = get_tools()

    @traceable(run_type="tool", name="Local Knowledge Search")
    async def local_tool(query: str) -> str:
        return await search_local_knowledge(query, model)
    
    @traceable(run_type="tool", name="Web Search")
    async def search_web_async(query: str) -> str:
        return await asyncio.to_thread(search_web, query)
    
    tool_handlers: Dict[str, Callable[[str], Awaitable[str]]] = {
        "search_local_knowledge": local_tool,
        "search_web": search_web_async,
    }

    input_items = [{"role": "user", "content": user_query}]
    response = await client.responses.create(
        model=model,
        tools=tools,
        input=input_items,
        instructions=instructions,
    )

    while True:
        input_items += response.output
        function_calls = [item for item in response.output if item.type == "function_call"]
        if not function_calls:
            return response.output_text

        for item in function_calls:
            query = json.loads(item.arguments)["query"]
            output = await tool_handlers[item.name](query)
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
            instructions=instructions,
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