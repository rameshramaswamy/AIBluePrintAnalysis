from fastapi import Security, HTTPException, status, Depends
from fastapi.security.api_key import APIKeyHeader
from sqlalchemy.orm import Session

from blueprint_brain.src.db.session import get_db
from blueprint_brain.src.security.service import SecurityService
from blueprint_brain.src.security.limiter import RateLimiter
from blueprint_brain.src.security.models import ApiKey

# Define Header
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)

async def get_current_client(
    key_header: str = Security(api_key_header),
    db: Session = Depends(get_db)
) -> ApiKey:
    """
    1. Checks if API Key exists in Header.
    2. Validates Key Hash against DB.
    3. Checks Redis Rate Limit for this specific client.
    """
    if not key_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-KEY header"
        )

    # 1. Validate Credential
    client_api_key = SecurityService.validate_key(db, key_header)
    if not client_api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key"
        )

    # 2. Rate Limiting (Redis)
    # We use the UUID of the key as the identifier
    RateLimiter.check_limit(client_api_key.id, client_api_key.rate_limit_per_minute)

    return client_api_key