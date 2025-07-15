import os
import json
import logging
import time
import asyncio
from typing import List, Dict, Any, Optional

# Try to import OpenAI with fallback
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None

logger = logging.getLogger(__name__)

class CandidateGenerator:
    """
    Optimized AI layer that generates candidate API calls using LLM
    """
    
    def __init__(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI package not available. Using fallback candidates.")
            self.client = None
        elif not api_key or api_key in ['your_openai_api_key_here', 'disabled_for_testing', '']:
            logger.warning("OpenAI API key not set. Using fallback candidates.")
            self.client = None
        else:
            try:
                self.client = AsyncOpenAI(
                    api_key=api_key,
                    timeout=10.0,  # Global timeout for all requests
                    max_retries=1   # Reduce retries for faster failure
                )
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}. Using fallback.")
                self.client = None
        
        # Cache for similar requests
        self.request_cache = {}
        self.cache_max_age = 300  # 5 minutes
        
    async def generate_candidates(
        self, 
        events: List[Any],  # These are APIEvent Pydantic models
        prompt: Optional[str], 
        spec_data: Dict[str, Any], 
        k: int
    ) -> List[Dict[str, Any]]:
        """
        Generate candidate API calls using optimized LLM approach
        """
        try:
            # Check cache first for similar requests
            cache_key = self._generate_cache_key(events, prompt, spec_data, k)
            cached_candidates = self._get_cached_candidates(cache_key)
            if cached_candidates:
                logger.info("Using cached AI candidates")
                return cached_candidates[:k * 2]
            
            candidates = []
            
            if self.client:
                # Use optimized OpenAI generation
                candidates = await self._generate_with_openai_optimized(events, prompt, spec_data, k)
            
            # Always add fallback candidates for robustness
            if len(candidates) < k:
                fallback = self._generate_fallback_candidates(events, spec_data, k * 2 - len(candidates))
                candidates.extend(fallback)
            
            # Cache successful results
            if candidates:
                self._cache_candidates(cache_key, candidates)
            
            return candidates[:k * 2]  # Return extra for ML layer to rank
            
        except Exception as e:
            logger.error(f"Error generating candidates: {e}")
            # Fallback to heuristic candidates
            return self._generate_fallback_candidates(events, spec_data, k * 2)
    
    async def _generate_with_openai_optimized(
        self, 
        events: List[Any], 
        prompt: Optional[str], 
        spec_data: Dict[str, Any], 
        k: int
    ) -> List[Dict[str, Any]]:
        """Generate candidates using optimized OpenAI calls"""
        
        # Build minimal context for faster processing
        context = self._build_optimized_context(events, prompt, spec_data, k)
        
        try:
            # Optimized OpenAI call with reduced latency
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Generate API predictions as concise JSON array. Be fast and accurate."},
                        {"role": "user", "content": context}
                    ],
                    temperature=0.1,  # Lower temperature for faster, more deterministic responses
                    max_tokens=600,   # Reduced from 1500 for faster processing
                    top_p=0.9,       # Reduce sampling space for speed
                    frequency_penalty=0,  # No penalties for speed
                    presence_penalty=0
                ),
                timeout=8.0  # 8 second timeout
            )
            
            response_text = response.choices[0].message.content
            candidates = self._parse_openai_response(response_text, spec_data)
            
            logger.info(f"OpenAI generated {len(candidates)} candidates")
            return candidates
            
        except asyncio.TimeoutError:
            logger.warning("OpenAI request timed out, using fallback")
            return []
        except Exception as e:
            logger.warning(f"OpenAI API error: {e}")
            return []
    
    def _build_optimized_context(
        self, 
        events: List[Any], 
        prompt: Optional[str], 
        spec_data: Dict[str, Any], 
        k: int
    ) -> str:
        """Build minimal context prompt for faster LLM processing"""
        
        # Only last 3 events for speed
        events_summary = "\n".join([
            f"- {event.endpoint}"
            for event in events[-3:]  # Reduced from 5
        ])
        
        # Only top 8 endpoints for speed
        endpoints_sample = "\n".join([
            f"- {ep['endpoint']}"
            for ep in spec_data.get('endpoints', [])[:8]  # Reduced from 15
        ])
        
        # Simplified, shorter context
        context = f"""Recent actions:
{events_summary}

Intent: {prompt or 'Continue workflow'}

Available API endpoints:
{endpoints_sample}

Generate {k} likely next API calls as JSON array:
[{{"endpoint": "METHOD /path", "params": {{}}, "reasoning": "brief reason"}}]"""

        return context
    
    def _parse_openai_response(self, response: str, spec_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse and validate OpenAI response with error tolerance"""
        try:
            # Clean up response
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:-3]
            elif response.startswith('```'):
                response = response[3:-3]
            
            # Try to parse JSON
            try:
                candidates_raw = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from text
                import re
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    candidates_raw = json.loads(json_match.group())
                else:
                    logger.warning("Could not parse OpenAI JSON response")
                    return []
            
            # Validate and structure candidates
            candidates = []
            available_endpoints = [ep['endpoint'] for ep in spec_data.get('endpoints', [])]
            
            for candidate in candidates_raw:
                if self._validate_candidate(candidate, available_endpoints):
                    candidates.append({
                        'endpoint': candidate['endpoint'],
                        'params': candidate.get('params', {}),
                        'reasoning': candidate.get('reasoning', 'AI-generated prediction'),
                        'source': 'openai'
                    })
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error parsing OpenAI response: {e}")
            return []
    
    def _validate_candidate(self, candidate: Dict[str, Any], available_endpoints: List[str]) -> bool:
        """Fast validation of candidate prediction"""
        endpoint = candidate.get('endpoint', '')
        
        # Basic validation
        if not endpoint or ' ' not in endpoint:
            return False
            
        method = endpoint.split()[0].upper()
        if method not in ['GET', 'POST', 'PUT', 'PATCH', 'DELETE']:
            return False
        
        return True
    
    def _generate_fallback_candidates(
        self, 
        events: List[Any], 
        spec_data: Dict[str, Any], 
        count: int
    ) -> List[Dict[str, Any]]:
        """Generate fast fallback candidates using simple heuristics"""
        candidates = []
        endpoints = spec_data.get('endpoints', [])
        
        # Prefer safe endpoints
        safe_endpoints = [ep for ep in endpoints if not ep.get('is_destructive', False)]
        
        # Simple heuristics based on last action
        if events:
            last_endpoint = events[-1].endpoint
            last_method = last_endpoint.split()[0]
            
            # Fast pattern matching
            if last_method == 'GET':
                preferred_methods = ['POST', 'PUT']
            elif last_method == 'POST':
                preferred_methods = ['GET', 'PUT']
            else:
                preferred_methods = ['GET']
        else:
            preferred_methods = ['GET']
        
        # Generate candidates quickly
        for method in preferred_methods:
            method_endpoints = [ep for ep in safe_endpoints if ep['endpoint'].startswith(method)]
            for ep in method_endpoints[:count]:
                candidates.append({
                    'endpoint': ep['endpoint'],
                    'params': {},
                    'reasoning': f'Fallback prediction - common {method} operation',
                    'source': 'fallback'
                })
                if len(candidates) >= count:
                    break
            if len(candidates) >= count:
                break
        
        # Fill with GET endpoints if needed
        if len(candidates) < count:
            get_endpoints = [ep for ep in safe_endpoints if ep['endpoint'].startswith('GET')]
            for ep in get_endpoints[:count - len(candidates)]:
                candidates.append({
                    'endpoint': ep['endpoint'],
                    'params': {},
                    'reasoning': 'Safe GET operation',
                    'source': 'fallback'
                })
        
        return candidates[:count]
    
    def _generate_cache_key(self, events: List[Any], prompt: Optional[str], spec_data: Dict[str, Any], k: int) -> str:
        """Generate cache key for similar requests"""
        events_key = '-'.join([event.endpoint.split()[0] for event in events[-3:]])
        prompt_key = (prompt or 'no_prompt')[:20]
        spec_key = spec_data.get('title', 'unknown')[:10]
        return f"ai_candidates:{events_key}:{prompt_key}:{spec_key}:{k}"
    
    def _get_cached_candidates(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """Get candidates from cache if recent enough"""
        if cache_key in self.request_cache:
            cached_item = self.request_cache[cache_key]
            if time.time() - cached_item['timestamp'] < self.cache_max_age:
                return cached_item['candidates']
            else:
                # Remove expired cache
                del self.request_cache[cache_key]
        return None
    
    def _cache_candidates(self, cache_key: str, candidates: List[Dict[str, Any]]):
        """Cache candidates for reuse"""
        self.request_cache[cache_key] = {
            'candidates': candidates,
            'timestamp': time.time()
        }
        
        # Keep cache size reasonable
        if len(self.request_cache) > 100:
            # Remove oldest entries
            oldest_keys = sorted(self.request_cache.keys(), 
                               key=lambda k: self.request_cache[k]['timestamp'])[:50]
            for key in oldest_keys:
                del self.request_cache[key]