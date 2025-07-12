import time
import logging
from typing import Dict, List
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
import statistics
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Single performance measurement"""
    timestamp: datetime
    endpoint: str
    duration_ms: float
    status_code: int
    user_id: str
    cache_hit: bool = False
    ml_used: bool = False

class PerformanceMonitor:
    """Monitor and track API performance metrics"""
    
    def __init__(self, max_metrics: int = 1000):
        self.metrics: deque[PerformanceMetric] = deque(maxlen=max_metrics)
        self.start_time = datetime.utcnow()
        
    def record_metric(self, metric: PerformanceMetric):
        """Record a performance metric"""
        self.metrics.append(metric)
        
        # Log slow requests
        if metric.duration_ms > 2000:  # > 2 seconds
            logger.warning(
                f"Slow request: {metric.endpoint} took {metric.duration_ms:.0f}ms "
                f"for user {metric.user_id}"
            )
    
    def get_stats(self, minutes: int = 60) -> Dict:
        """Get performance statistics for the last N minutes"""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        recent_metrics = [m for m in self.metrics if m.timestamp > cutoff]
        
        if not recent_metrics:
            return {
                "period_minutes": minutes,
                "total_requests": 0,
                "median_ms": 0,
                "p95_ms": 0,
                "p99_ms": 0,
                "cache_hit_rate": 0,
                "ml_usage_rate": 0,
                "error_rate": 0
            }
        
        durations = [m.duration_ms for m in recent_metrics]
        error_count = sum(1 for m in recent_metrics if m.status_code >= 400)
        cache_hits = sum(1 for m in recent_metrics if m.cache_hit)
        ml_used = sum(1 for m in recent_metrics if m.ml_used)
        
        return {
            "period_minutes": minutes,
            "total_requests": len(recent_metrics),
            "median_ms": statistics.median(durations),
            "p95_ms": statistics.quantiles(durations, n=20)[18] if len(durations) > 1 else durations[0],  # 95th percentile
            "p99_ms": statistics.quantiles(durations, n=100)[98] if len(durations) > 1 else durations[0],  # 99th percentile
            "min_ms": min(durations),
            "max_ms": max(durations),
            "avg_ms": statistics.mean(durations),
            "cache_hit_rate": cache_hits / len(recent_metrics) if recent_metrics else 0,
            "ml_usage_rate": ml_used / len(recent_metrics) if recent_metrics else 0,
            "error_rate": error_count / len(recent_metrics) if recent_metrics else 0,
            "uptime_minutes": (datetime.utcnow() - self.start_time).total_seconds() / 60
        }
    
    def get_endpoint_stats(self, endpoint: str, minutes: int = 60) -> Dict:
        """Get stats for a specific endpoint"""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        endpoint_metrics = [
            m for m in self.metrics 
            if m.timestamp > cutoff and m.endpoint == endpoint
        ]
        
        if not endpoint_metrics:
            return {"endpoint": endpoint, "requests": 0}
        
        durations = [m.duration_ms for m in endpoint_metrics]
        
        return {
            "endpoint": endpoint,
            "requests": len(endpoint_metrics),
            "median_ms": statistics.median(durations),
            "avg_ms": statistics.mean(durations),
            "max_ms": max(durations),
            "cache_hit_rate": sum(1 for m in endpoint_metrics if m.cache_hit) / len(endpoint_metrics)
        }
    
    def is_healthy(self) -> bool:
        """Check if performance is within acceptable bounds"""
        stats = self.get_stats(minutes=10)  # Last 10 minutes
        
        if stats["total_requests"] == 0:
            return True  # No requests to evaluate
        
        # Check performance requirements
        median_ok = stats["median_ms"] < 1000  # < 1 second
        p95_ok = stats["p95_ms"] < 3000  # < 3 seconds
        error_rate_ok = stats["error_rate"] < 0.05  # < 5% errors
        
        return median_ok and p95_ok and error_rate_ok

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

class PerformanceTracker:
    """Context manager for tracking request performance"""
    
    def __init__(self, endpoint: str, user_id: str = "unknown"):
        self.endpoint = endpoint
        self.user_id = user_id
        self.start_time = None
        self.cache_hit = False
        self.ml_used = False
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000
        status_code = 500 if exc_type else 200
        
        metric = PerformanceMetric(
            timestamp=datetime.utcnow(),
            endpoint=self.endpoint,
            duration_ms=duration_ms,
            status_code=status_code,
            user_id=self.user_id,
            cache_hit=self.cache_hit,
            ml_used=self.ml_used
        )
        
        performance_monitor.record_metric(metric)
        
        # Log performance info
        logger.info(
            f"Request completed: {self.endpoint} - {duration_ms:.0f}ms "
            f"(cache: {self.cache_hit}, ml: {self.ml_used})"
        )

class AsyncPerformanceTracker:
    """Async context manager for tracking request performance"""
    
    def __init__(self, endpoint: str, user_id: str = "unknown"):
        self.endpoint = endpoint
        self.user_id = user_id
        self.start_time = None
        self.cache_hit = False
        self.ml_used = False
        
    async def __aenter__(self):
        self.start_time = time.time()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000
        status_code = 500 if exc_type else 200
        
        metric = PerformanceMetric(
            timestamp=datetime.utcnow(),
            endpoint=self.endpoint,
            duration_ms=duration_ms,
            status_code=status_code,
            user_id=self.user_id,
            cache_hit=self.cache_hit,
            ml_used=self.ml_used
        )
        
        performance_monitor.record_metric(metric)

def optimize_for_performance():
    """Apply performance optimizations"""
    
    # Set event loop policy for better async performance
    if hasattr(asyncio, 'WindowsProactorEventLoopPolicy'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Configure logging for performance
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    logger.info("Performance optimizations applied")

# Performance utilities
async def with_timeout(coro, timeout_seconds: float = 5.0, default=None):
    """Execute coroutine with timeout"""
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.warning(f"Operation timed out after {timeout_seconds}s")
        return default

def measure_time(func):
    """Decorator to measure function execution time"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = (time.time() - start) * 1000
        logger.debug(f"{func.__name__} took {duration:.2f}ms")
        return result
    return wrapper

def measure_async_time(func):
    """Decorator to measure async function execution time"""
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        duration = (time.time() - start) * 1000
        logger.debug(f"{func.__name__} took {duration:.2f}ms")
        return result
    return wrapper