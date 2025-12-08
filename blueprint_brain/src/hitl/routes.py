from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session

from blueprint_brain.src.db.session import get_db
from blueprint_brain.src.db.models.job import Job
from blueprint_brain.services.storage import StorageService
import json

router = APIRouter(prefix="/admin", tags=["HITL"])
storage = StorageService()

# In a real app, protect this with a specific "Admin" role or internal VPN check
# For now, we reuse the API Key check but you'd likely want OAuth2/OIDC here.
from blueprint_brain.api.dependencies import get_current_client

@router.get("/jobs/failed")
async def list_failed_jobs(limit: int = 50, db: Session = Depends(get_db)):
    """List recent failed jobs for review."""
    jobs = db.query(Job).filter(Job.status == "FAILED")\
        .order_by(Job.created_at.desc()).limit(limit).all()
    return jobs

@router.get("/jobs/{job_id}/debug")
async def get_job_debug_data(job_id: str, db: Session = Depends(get_db)):
    """Get metadata and error logs for a job."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job: raise HTTPException(404)
    return {
        "error": job.error_message,
        "meta": job.meta_data,
        "doc_id": job.document_id
    }

@router.patch("/jobs/{job_id}/fix")
async def fix_job_result(
    job_id: str, 
    corrected_data: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db)
):
    """
    HITL Action: Manually upload corrected JSON and mark job as COMPLETED.
    Used when the AI missed a room or got the scale wrong.
    """
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job: raise HTTPException(404)

    # 1. Overwrite result in S3
    result_key = job.result_s3_key or f"results/{job_id}/manual_fix.json"
    
    # Create temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp:
        json.dump(corrected_data, tmp)
        tmp_path = tmp.name
    
    try:
        storage.upload_file(open(tmp_path, 'rb'), result_key, content_type="application/json")
    finally:
        import os
        os.remove(tmp_path)

    # 2. Update DB
    job.status = "COMPLETED"
    job.result_s3_key = result_key
    job.error_message = None # Clear error
    job.meta_data = {"source": "human_correction"} # Audit trail
    
    db.commit()
    
    return {"status": "fixed", "message": "Job marked as completed with manual data."}