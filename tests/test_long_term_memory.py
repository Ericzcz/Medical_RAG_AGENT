import asyncio

import redis.asyncio as redis

from app.long_term_memory import (
    extract_long_term_memory,
    save_long_term_memory,
    get_long_term_context,
)


async def main():
    redis_client = redis.Redis(
        host="localhost",
        port=6383,
        db=2,
        decode_responses=True,
    )

    memories = await extract_long_term_memory(
        user_query="以后你不要直接改我代码，先一步一步教我。",
        assistant_answer="好的，我之后会先解释步骤，再让你自己改。",
        model="gpt-4o-mini",
    )

    await save_long_term_memory(
        redis_client=redis_client,
        user_id="eric",
        session_id="test-session",
        memories=memories,
        model="gpt-4o-mini"
    )

    # for memory in memories:
    #     print(memory.model_dump())


    context = await get_long_term_context(
        redis_client=redis_client,
        user_id="eric",
    )

    print(context)

    await redis_client.aclose()

if __name__ == "__main__":
    asyncio.run(main())