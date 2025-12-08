import uuid
from datetime import datetime
from sqlalchemy.orm import Session
from blueprint_brain.src.db.models.project import Document
from blueprint_brain.src.db.models.job import Job

def create_document(db: Session, filename: str, s3_key: str, file_hash: str = None) -> Document:
    doc = Document(
        id=str(uuid.uuid4()),
        filename=filename,
        s3_key=s3_key,
        file_hash=file_hash,
        upload_date=datetime.utcnow()
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc

def create_job(db: Session, document_id: str) -> Job:
    job = Job(
        id=str(uuid.uuid4()),
        document_id=document_id,
        status="QUEUED",
        created_at=datetime.utcnow()
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job

def get_job(db: Session, job_id: str) -> Job:
    return db.query(Job).filter(Job.id == job_id).first()

def get_document_by_hash(db: Session, file_hash: str) -> Document:
    return db.query(Document).filter(Document.file_hash == file_hash).first()

def update_job_status(db: Session, job_id: str, status: str, result_key: str = None, meta: dict = None, error: str = None):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        return
    
    job.status = status
    if status == "PROCESSING":
        job.started_at = datetime.utcnow()
    
    if status in ["COMPLETED", "FAILED"]:
        job.completed_at = datetime.utcnow()
        if job.started_at:
            job.processing_duration = (job.completed_at - job.started_at).total_seconds()
    
    if result_key:
        job.result_s3_key = result_key
    if meta:
        job.meta_data = meta
    if error:
        job.error_message = error
        
    db.commit()
    return job