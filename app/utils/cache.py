import json
import logging
import os
from typing import Optional, Dict, Any

# Try to import Redis with fallback
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

logger = logging.getLogger(__name__)

class CacheManager:
    """Manages Redis cache for OpenAPI specs and other data"""
    
    def __init__(self):
        self.redis_client = None
        self.redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        self.in_memory_cache = {}  # Fallback cache
        
    async def init(self):
        """Initialize Redis connection"""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available. Using in-memory cache.")
            return
            
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                encoding='utf-8',
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory cache.")
            self.redis_client = None
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached data"""
        try:
            if self.redis_client:
                data = await self.redis_client.get(key)
                return json.loads(data) if data else None
            else:
                # Use in-memory cache
                return self.in_memory_cache.get(key)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    async def set(self, key: str, value: Dict[str, Any], ttl: int = 3600):
        """Set cached data with TTL"""
        try:
            if self.redis_client:
                await self.redis_client.setex(
                    key, 
                    ttl, 
                    json.dumps(value, default=str)
                )
            else:
                # Use in-memory cache
                self.in_memory_cache[key] = value
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    async def delete(self, key: str):
        """Delete cached data"""
        try:
            if self.redis_client:
                await self.redis_client.delete(key)
            else:
                self.in_memory_cache.pop(key, None)
        except Exception as e:
            logger.error(f"Cache delete error: {e}")