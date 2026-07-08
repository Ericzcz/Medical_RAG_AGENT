import logging

from fastapi import APIRouter

from app.Redis_Celery.celery_app import celery_app
from app.Redis_Celery.tasks import index_document
from app.schemas import TaskResponse


router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/index")
def index(force_rebuild: bool = False):
    try:
        task = index_document.delay(force_rebuild=force_rebuild)
    except Exception:
        logger.exception("index task submit failed")
        raise

    logger.info("index task submitted", extra={"task_id": task.id})

    return {
        "task_id": task.id,
        "status": "indexing_submitted",
    }


@router.get("/tasks/{task_id}", response_model=TaskResponse)
def get_task(task_id : str):
    task = celery_app.AsyncResult(task_id)

    response = {
        "task_id": task_id,
        "status": task.status,
        "result": None,
        "progress": None,
    }

    if task.status == "PROGRESS":
        response["progress"] = task.info
    elif task.ready():
        response["result"] = task.result
    return response
