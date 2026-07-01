from celery import Celery

celery_app = Celery(
    "worker",
    broker="redis://redis:6383/0",
    backend="redis://redis:6383/1",
    include=["app.Redis_Celery.tasks"],
)

celery_app.conf.task_track_started = True

