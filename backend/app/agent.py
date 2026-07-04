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
    chat_history: list[dict] | None = None,
) -> str:
    client = wrap_openai(AsyncOpenAI())
    tools = get_tools()

    default_instructions = """
        你是一个 ReAct 风格的 医疗Agent，可以根据问题选择本地知识库检索或网络搜索工具。

        当调用 search_web 工具时，必须先结合对话历史，将代词、省略表达和上下文相关问题改写成独立、明确的搜索查询。
        不要把“它”“这个”“上面提到的”“刚才说的”等模糊表达直接传给 search_web。

        当调用 search_local_knowledge 工具时，可以保留用户原始问题，因为本地 RAG 链会结合 chat_history 进行问题改写。
        """
    long_term_memory_instructions = instructions or ""
    final_instructions = f"""
        {default_instructions}

        {long_term_memory_instructions}
        """

    @traceable(run_type="tool", name="Local Knowledge Search")
    async def local_tool(query: str) -> str:
        return await search_local_knowledge(
            query, 
            model,
            chat_history=chat_history,
            )
    
    @traceable(run_type="tool", name="Web Search")
    async def search_web_async(query: str) -> str:
        return await asyncio.to_thread(search_web, query)
    
    tool_handlers: Dict[str, Callable[[str], Awaitable[str]]] = {
        "search_local_knowledge": local_tool,
        "search_web": search_web_async,
    }

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