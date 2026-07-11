from tavily import TavilyClient

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