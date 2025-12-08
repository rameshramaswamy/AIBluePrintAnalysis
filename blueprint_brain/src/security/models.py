import uuid
import secrets
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Boolean, Integer
from blueprint_brain.src.db.base import Base

class ApiKey(Base):
    """
    Identity Management for B2B Clients.
    Stores hashed keys and usage quotas.
    """
    __tablename__ = "api_keys"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    client_name = Column(String, nullable=False)
    
    # Security: We store a prefix (for display) and a hash (for validation)
    key_prefix = Column(String(8), nullable=False) 
    key_hash = Column(String, nullable=False) 
    
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Rate Limiting Policies
    rate_limit_per_minute = Column(Integer, default=50) # Requests per minute
    
    # Usage Tracking (Optional, for billing)
    last_used_at = Column(DateTime, nullable=True)