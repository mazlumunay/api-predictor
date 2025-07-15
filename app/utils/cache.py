import json
import logging
import os
import time
import hashlib
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
    """Optimized cache manager with aggressive caching for performance"""
    
    def __init__(self):
        self.redis_client = None
        self.redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        self.in_memory_cache = {}  # Fallback cache
        self.prediction_cache = {}  # Fast in-memory prediction cache
        
    async def init(self):
        """Initialize Redis connection"""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available. Using in-memory cache.")
            return
            
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                encoding='utf-8',
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2
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
        """Get cached data with performance optimization"""
        try:
            # Check fast in-memory cache first
            if key in self.in_memory_cache:
                cached_item = self.in_memory_cache[key]
                if isinstance(cached_item, dict) and 'expires' in cached_item:
                    if time.time() < cached_item['expires']:
                        return cached_item['data']
                    else:
                        # Remove expired item
                        del self.in_memory_cache[key]
                else:
                    # Old format, return as-is
                    return cached_item
            
            # Check Redis if available
            if self.redis_client:
                data = await self.redis_client.get(key)
                if data:
                    parsed_data = json.loads(data)
                    # Cache in memory for faster access
                    self.in_memory_cache[key] = {
                        'data': parsed_data,
                        'expires': time.time() + 3600  # 1 hour in-memory cache
                    }
                    return parsed_data
            
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    async def set(self, key: str, value: Dict[str, Any], ttl: int = 7200):
        """Set cached data with extended TTL for performance"""
        try:
            # Always cache in memory for fast access
            self.in_memory_cache[key] = {
                'data': value,
                'expires': time.time() + min(ttl, 3600)  # Max 1 hour in-memory
            }
            
            # Also cache in Redis for persistence
            if self.redis_client:
                await self.redis_client.setex(
                    key, 
                    ttl, 
                    json.dumps(value, default=str)
                )
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    async def delete(self, key: str):
        """Delete cached data"""
        try:
            # Remove from in-memory cache
            self.in_memory_cache.pop(key, None)
            
            # Remove from Redis
            if self.redis_client:
                await self.redis_client.delete(key)
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
    
    async def cache_prediction(self, user_id: str, events_hash: str, prompt_hash: str, prediction: Dict[str, Any]):
        """Cache prediction results for identical requests"""
        cache_key = f"prediction:{user_id}:{events_hash}:{prompt_hash}"
        await self.set(cache_key, prediction, ttl=600)  # 10 minutes for predictions
    
    async def get_cached_prediction(self, user_id: str, events_hash: str, prompt_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached prediction if available"""
        cache_key = f"prediction:{user_id}:{events_hash}:{prompt_hash}"
        return await self.get(cache_key)
    
    def generate_events_hash(self, events: list) -> str:
        """Generate hash for events list"""
        events_str = json.dumps([{
            'endpoint': e.endpoint if hasattr(e, 'endpoint') else e.get('endpoint', ''),
            'ts': e.ts if hasattr(e, 'ts') else e.get('ts', '')
        } for e in events], sort_keys=True)
        return hashlib.md5(events_str.encode()).hexdigest()[:12]
    
    def generate_prompt_hash(self, prompt: Optional[str]) -> str:
        """Generate hash for prompt"""
        if not prompt:
            return "no_prompt"
        return hashlib.md5(prompt.encode()).hexdigest()[:8]
    
    async def cleanup_expired(self):
        """Clean up expired in-memory cache entries"""
        current_time = time.time()
        expired_keys = []
        
        for key, value in self.in_memory_cache.items():
            if isinstance(value, dict) and 'expires' in value:
                if current_time >= value['expires']:
                    expired_keys.append(key)
        
        for key in expired_keys:
            del self.in_memory_cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")