import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, ForeignKey, JSON, Float
from sqlalchemy.orm import relationship
from blueprint_brain.src.db.base import Base

class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("documents.id"))
    status = Column(String, default="QUEUED", index=True) # QUEUED, PROCESSING, COMPLETED, FAILED
    
    # Performance Metrics
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    processing_duration = Column(Float, nullable=True)
    
    # Results
    result_s3_key = Column(String, nullable=True)
    meta_data = Column(JSON, nullable=True) # Store sqft totals here for quick querying
    error_message = Column(String, nullable=True)

    # Relationships
    document = relationship("Document", back_populates="jobs")