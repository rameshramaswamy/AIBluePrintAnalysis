import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from blueprint_brain.api.dependencies import get_api_key
from blueprint_brain.worker.tasks import process_blueprint
from blueprint_brain.services.storage import StorageService
from blueprint_brain.api.schemas import JobResponse, JobStatus
# Database Deps
from blueprint_brain.src.db.session import get_db
from blueprint_brain.src.db import crud
from blueprint_brain.api.schemas import JobResponse, ProcessingRequest

from fastapi.middleware.gzip import GZipMiddleware


from blueprint_brain.api.dependencies import get_current_client
from blueprint_brain.src.hitl.routes import router as hitl_router
from blueprint_brain.src.monitoring.metrics import HTTP_REQUESTS_TOTAL

# Setup Rate Limiter (Redis backend recommended for Prod)
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Blueprint AI Enterprise")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
import hashlib
from pydantic import HttpUrl
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram


# Define Custom Metrics
JOB_COUNTER = Counter(
    "blueprint_jobs_total", 
    "Total number of blueprint jobs submitted", 
    ["status"] # label by status (queued, failed)
)
INFERENCE_LATENCY = Histogram(
    "blueprint_inference_seconds",
    "Time spent in AI inference pipeline",
    buckets=[1, 5, 10, 30, 60, 120] # buckets in seconds
)

# ... app definition ...
app = FastAPI(title="Blueprint AI Enterprise")
app.add_middleware(GZipMiddleware, minimum_size=1000)
# Initialize Prometheus
# 2. Prometheus Instrumentation
@app.on_event("startup")
async def startup():
    Instrumentator().instrument(app).expose(app) # Exposes /metrics

storage = StorageService()
app.include_router(hitl_router)

@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    """Add unique ID to every request for tracing logs."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

# Add Webhook URL to input schema
class ProcessingRequest(BaseModel):
    file_key: str  # The file is already in S3 (via presigned upload)
    webhook_url: Optional[str] = None

@app.post("/upload/direct", dependencies=[Depends(get_api_key)])
async def get_upload_url(filename: str):
    """
    Step 1: Client asks for permission to upload.
    Returns: A direct S3 URL and a Job ID.
    """
    job_id = str(uuid.uuid4())
    ext = filename.split('.')[-1]
    key = f"uploads/{job_id}.{ext}"
    
    presigned_data = storage.generate_presigned_upload(key)
    
    return {
        "job_id": job_id,
        "file_key": key,
        "upload_data": presigned_data, # Client posts form-data here
        "message": "Upload file to 'upload_data.url' using 'upload_data.fields', then call POST /process"
    }

@app.post("/process", response_model=JobResponse, dependencies=[Depends(get_api_key)])
async def start_processing(req: ProcessingRequest):
    """
    Step 2: Client tells us the upload is done. We verify and start working.
    """
    # 1. Verify file exists in S3 (Lightweight check)
    try:
        storage.s3.head_object(Bucket=storage.bucket, Key=req.file_key)
    except Exception:
        raise HTTPException(404, "File not found in storage. Did you upload it?")


    # DB: Create Document
    # Ideally, we calculate hash here, but that requires downloading the file (slow).
    # We defer hashing to the worker, then update the DB later.
    filename = req.file_key.split('/')[-1]
    doc = crud.create_document(db, filename=filename, s3_key=req.file_key)
    
    # DB: Create Job
    job = crud.create_job(db, document_id=doc.id)

    # Dispatch (Pass job.id, NOT the random UUID from upload)
    process_blueprint.apply_async(
        args=[req.file_key, req.webhook_url, job.id], # Pass DB Job ID
        task_id=job.id
    )

    return JobResponse(job_id=job.id, status="queued", message="Job persisted and started")

@app.get("/jobs/{job_id}", dependencies=[Depends(get_api_key)])
async def get_status(job_id: str, db: Session = Depends(get_db)):
    """
    Get status from DB (Truth) + Redis (Real-time Progress).
    """
    # 1. Check DB first
    job = crud.get_job(db, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
        
    response = {
        "job_id": job.id,
        "status": job.status.lower(),
        "created_at": job.created_at,
        "completed_at": job.completed_at,
        "result": None,
        "error": job.error_message
    }
    
    # 2. If Processing, check Celery for % progress
    if job.status in ["QUEUED", "PROCESSING"]:
        from celery.result import AsyncResult
        from blueprint_brain.worker.celery_app import celery_app
        res = AsyncResult(job_id, app=celery_app)
        if res.state == 'PROCESSING' and isinstance(res.info, dict):
            response['progress'] = res.info.get('progress', 0)
            response['message'] = res.info.get('status', '')

    # 3. If Completed, fetch result
    if job.status == "COMPLETED" and job.result_s3_key:
        import json, io
        try:
            stream = io.BytesIO()
            storage.s3.download_fileobj(storage.bucket, job.result_s3_key, stream)
            stream.seek(0)
            data = json.load(stream)
            
            # Sign URLs
            for page in data.get('results', []):
                page['image_url'] = storage.generate_presigned_url(page['image_key'])
            
            response['result'] = data
        except Exception:
            response['status'] = 'completed_but_data_missing'

    return response

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "blueprint-brain"}