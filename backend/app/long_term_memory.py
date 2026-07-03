import json
from langchain_openai import ChatOpenAI
from app.schemas import ExtractedMemory

MEMORY_EXTRACT_PROMPT = """
    你是一个长期记忆提取器。

    你的任务是判断当前这一轮用户问题和助手回答中，是否包含未来对话仍然有价值的信息。

    只保存以下类型的信息：
    1. preference: 用户长期偏好，例如语言、解释风格、代码风格
    2. project: 用户正在做的项目背景、技术栈、目标
    3. fact: 稳定事实，例如用户使用的工具、项目名称
    4. correction: 用户对助手行为的纠正，例如不要直接改代码、要一步一步教

    不要保存：
    1. 临时问题
    2. 一次性的测试数据
    3. session_id、临时命令结果
    4. 普通技术解释
    5. 敏感医疗个人信息，除非用户明确要求长期保存

    请只返回 JSON 数组，不要返回 markdown，不要解释。

    如果没有值得保存的信息，返回空数组 []。

    格式：
    [
    {
        "memory_type": "preference",
        "content": "用户希望用中文解释技术问题。",
        "importance": 4
    }
    ]
    """


async def extract_long_term_memory(
    user_query: str,
    assistant_answer: str,
    model: str,
) -> list[ExtractedMemory]:
    llm = ChatOpenAI(model=model, temperature=0)

    message = f"""
        用户问题：
        {user_query}

        助手回答：
        {assistant_answer}
        """

    response = await llm.ainvoke(
        [
            ("system", MEMORY_EXTRACT_PROMPT),
            ("human", message),
        ]
    )

    try:
        raw_memories = json.loads(response.content)
    except json.JSONDecodeError:
        return []

    memories: list[ExtractedMemory] = []

    for item in raw_memories:
        try:
            memories.append(ExtractedMemory(**item))
        except Exception:
            continue

    return memories