import time
import redis
from fastapi import HTTPException
from blueprint_brain.config.settings import settings

# Initialize Redis
r_conn = redis.from_url(settings.REDIS_URL)

class RateLimiter:
    """
    Sliding Window Rate Limiter using Redis.
    """
    
    @staticmethod
    def check_limit(key_id: str, limit_per_minute: int):
        """
        Raises HTTPException if limit exceeded.
        """
        current_minute = int(time.time() // 60)
        redis_key = f"rate_limit:{key_id}:{current_minute}"
        
        # Atomic Increment
        pipe = r_conn.pipeline()
        pipe.incr(redis_key)
        pipe.expire(redis_key, 90) # TTL 90s just to be safe
        result, _ = pipe.execute()
        
        request_count = result
        
        if request_count > limit_per_minute:
            raise HTTPException(
                status_code=429, 
                detail=f"Rate limit exceeded. Quota: {limit_per_minute} req/min."
            )