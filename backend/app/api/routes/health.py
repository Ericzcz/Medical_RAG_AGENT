from fastapi import APIRouter


router = APIRouter()


@router.get("/")
async def root():
    return {
        "status": "ok",
        "service": "Medical_RAG_Agent_API",
        "docs": "/docs",
        "health": "/health",
        "endpoints": [
            "/health",
            "/local_query",
            "/batch_local_query",
            "/agent_query",
            "/batch_agent_query",
            "/cache",
            "/index",
            "/tasks/{task_id}",
        ],
    }


@router.get("/health")
async def health_check():
    return {"status": "ok"}
