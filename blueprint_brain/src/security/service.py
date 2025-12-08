import hashlib
import secrets
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from blueprint_brain.src.security.models import ApiKey

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class SecurityService:
    
    @staticmethod
    def create_api_key(db: Session, client_name: str, limit: int = 50) -> str:
        """
        Generates a new API Key, hashes it, and saves to DB.
        Returns: The raw API Key (Show this once to user!)
        Format: bp_live_randomstring
        """
        # Generate generic 32-char secret
        raw_secret = secrets.token_urlsafe(32)
        raw_key = f"bp_live_{raw_secret}"
        
        # Hash it
        key_hash = pwd_context.hash(raw_key)
        
        # Save Metadata
        db_key = ApiKey(
            client_name=client_name,
            key_prefix=raw_key[:12], # Store "bp_live_abcd"
            key_hash=key_hash,
            rate_limit_per_minute=limit
        )
        db.add(db_key)
        db.commit()
        
        return raw_key

    @staticmethod
    def validate_key(db: Session, raw_key: str) -> ApiKey:
        """
        Validates an API Key.
        Returns the ApiKey object if valid, None otherwise.
        """
        if not raw_key or not raw_key.startswith("bp_live_"):
            return None
            
        # Optimization: We can't lookup by hash. 
        # But we can assume the client sends a Client-ID header OR 
        # we scan keys (slow) OR we use the prefix to narrow down (Collision risk low).
        # For high performance, we usually cache active keys in Redis.
        # Here, we do a naive DB lookup for MVP (searching by prefix is safer if unique).
        
        prefix = raw_key[:12]
        candidate = db.query(ApiKey).filter(
            ApiKey.key_prefix == prefix, 
            ApiKey.is_active == True
        ).first()
        
        if candidate and pwd_context.verify(raw_key, candidate.key_hash):
            return candidate
            
        return None