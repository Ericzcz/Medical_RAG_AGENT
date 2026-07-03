import logging

from app.Redis_Celery.celery_app import celery_app
from app.logging_config import setup_logging
from app.rag_chain import build_milvus_collection
from langsmith import traceable

setup_logging()
logger = logging.getLogger(__name__)

@celery_app.task(bind=True)
@traceable(name="Medical RAG Build Milvus Index", run_type="chain")
def index_document(
    self, 
    collection_name: str = "RAG_collection", 
    force_rebuild: bool = False,
    embedding_batch_size: int = 64,
    embedding_max_concurrency: int = 3,
    ):
    logger.info(
        "index document task started",
        extra={
            "collection_name": collection_name,
            "force_rebuild": force_rebuild,
            "embedding_batch_size": embedding_batch_size,
            "embedding_max_concurrency": embedding_max_concurrency,
        },
    )

    try:
        self.update_state(
            state="PROGRESS",
            meta={"step": "building_index"},
        )

        build_milvus_collection(
            batch_size=embedding_batch_size,
            max_concurrency=embedding_max_concurrency,
            collection_name=collection_name,
            force_rebuild=force_rebuild,
        )
    except Exception:
        logger.exception(
            "index document task failed",
            extra={
                "collection_name": collection_name,
                "force_rebuild": force_rebuild,
            },
        )
        raise

    logger.info(
        "index document task completed",
        extra={"collection_name": collection_name},
    )

    return {
        "status": "index",
        "collection_name": collection_name,
    }
