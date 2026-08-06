import asyncio

from app.services.intent_router import classify_intent

async def main():
    model = "gpt-5.5"

    queries = [
        "What is Aarskog-Scott syndrome?",
        "What are the latest FDA updates today?",
        "Please record that I am allergic to aspirin.",
        "What allergies do I have in my medical record?",
        "Explain this Python function step by step.",
    ]

    for query in queries:
        result = await classify_intent(query, model)
        print(query)
        print(result.model_dump())
        print()


if __name__ == "__main__":
    asyncio.run(main())