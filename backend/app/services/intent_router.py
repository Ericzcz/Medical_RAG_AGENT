import json

from langchain_openai import ChatOpenAI

from app.schemas import IntentDecision

INTENT_ROUTER_PROMPT = """
    You are an intent router for a Medical RAG Agent.

    Classify the user's query into exactly one intent:

    1. local_medical_qa
    Use when the user asks medical knowledge questions that can be answered from the local medical knowledge base.

    2. web_search
    Use when the user asks for recent, external, time-sensitive, or internet-dependent information.

    3. medical_record_insert
    Use when the user asks to record, save, store, add, or remember their medical information.

    4. medical_record_query
    Use when the user asks about their stored medical records, such as allergies, medications, symptoms, diagnoses, procedures, vitals, or past notes.

    5. general_chat
    Use for greetings, project discussion, coding questions, or anything that does not require medical RAG, web search, or medical records.

    Return only JSON:
        {
        "intent": "...",
        "confidence": 0.0,
        "reason": "..."
        }
    """

async def classify_intent(
    query: str,
    model: str,
) -> IntentDecision:
    llm = ChatOpenAI(model=model, temperature=0)

    response = await llm.ainvoke(
        [
            ("system", INTENT_ROUTER_PROMPT),
            ("human", query),
        ]
    )

    try:
        data = json.loads(response.content)
        return IntentDecision(**data)
    except Exception:
        return IntentDecision(
            intent="general_chat",
            confidence=0.0,
            reason="Failed to parse intent router output.",
        )
    
def get_confidence_policy(confidence: float) -> str:
    if confidence >= 0.85:
        return "strong"
    if confidence >= 0.55:
        return "weak"
    return "uncertain"