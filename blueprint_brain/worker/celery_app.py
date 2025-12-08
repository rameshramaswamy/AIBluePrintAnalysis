from celery import Celery
from blueprint_brain.config.settings import settings

celery_app = Celery(
    "blueprint_worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["blueprint_brain.worker.tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_routes={
        "blueprint_brain.worker.tasks.process_blueprint": {"queue": settings.TASK_QUEUE_NAME}
    },
    # GPU Optimization:
    # Only fetch 1 task at a time per worker process.
    # This prevents the worker from hoarding tasks it can't process yet.
    worker_prefetch_multiplier=1,
    
    # If the task is acknowledged late, another worker can pick it up if this one dies
    task_acks_late=True,
    
    # Clean up backend results after 1 day
    result_expires=86400
)