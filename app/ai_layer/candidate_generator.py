import os
import json
import logging
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
    AI layer that generates candidate API calls using LLM
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
                self.client = AsyncOpenAI(api_key=api_key)
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}. Using fallback.")
                self.client = None
        
    async def generate_candidates(
        self, 
        events: List[Any],  # These are APIEvent Pydantic models
        prompt: Optional[str], 
        spec_data: Dict[str, Any], 
        k: int
    ) -> List[Dict[str, Any]]:
        """
        Generate candidate API calls using LLM
        """
        try:
            if self.client:
                # Use OpenAI for generation
                candidates = await self._generate_with_openai(events, prompt, spec_data, k)
            else:
                # Fallback to heuristic generation
                candidates = self._generate_fallback_candidates(events, spec_data, k)
            
            # Ensure we have at least k candidates
            if len(candidates) < k:
                fallback = self._generate_fallback_candidates(events, spec_data, k - len(candidates))
                candidates.extend(fallback)
            
            return candidates[:k * 2]  # Return extra for ML layer to rank
            
        except Exception as e:
            logger.error(f"Error generating candidates: {e}")
            # Fallback to heuristic candidates
            return self._generate_fallback_candidates(events, spec_data, k * 2)
    
    async def _generate_with_openai(
        self, 
        events: List[Any], 
        prompt: Optional[str], 
        spec_data: Dict[str, Any], 
        k: int
    ) -> List[Dict[str, Any]]:
        """Generate candidates using OpenAI"""
        
        # Build context for LLM
        context = self._build_context(events, prompt, spec_data, k)
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert API prediction system. Always respond with valid JSON arrays."},
                    {"role": "user", "content": context}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            response_text = response.choices[0].message.content
            candidates = self._parse_openai_response(response_text, spec_data)
            
            return candidates
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return []
    
    def _build_context(
        self, 
        events: List[Any], 
        prompt: Optional[str], 
        spec_data: Dict[str, Any], 
        k: int
    ) -> str:
        """Build context prompt for LLM"""
        
        # Recent events summary - fix the Pydantic model access
        events_summary = "\n".join([
            f"- {event.endpoint} with params: {event.params}"
            for event in events[-5:]  # Last 5 events
        ])
        
        # Available endpoints (sample)
        endpoints_sample = "\n".join([
            f"- {ep['endpoint']}: {ep.get('summary', 'No description')}"
            for ep in spec_data.get('endpoints', [])[:15]  # First 15 endpoints
        ])
        
        context = f"""You are predicting the next API call for a user of {spec_data.get('title', 'this API')}.

RECENT USER ACTIVITY:
{events_summary}

USER INTENT: {prompt or 'No specific intent provided'}

AVAILABLE ENDPOINTS (sample):
{endpoints_sample}

TASK: Generate {k} most likely next API calls.

RULES:
1. Never suggest DELETE operations unless explicitly requested
2. Use realistic parameter values based on recent activity  
3. Consider typical API workflows
4. Provide clear reasoning

Return as JSON array:
[
  {{
    "endpoint": "POST /invoices",
    "params": {{"customer_id": "cus_123", "amount": 5000}},
    "reasoning": "User recently updated invoice status, creating new invoice is logical next step"
  }}
]

Generate exactly {k} predictions:"""

        return context
    
    def _parse_openai_response(self, response: str, spec_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse and validate OpenAI response"""
        try:
            # Clean up response
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:-3]
            elif response.startswith('```'):
                response = response[3:-3]
            
            candidates_raw = json.loads(response)
            
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
        """Validate a candidate prediction"""
        endpoint = candidate.get('endpoint', '')
        
        # Basic validation
        if not endpoint:
            return False
            
        # Check if endpoint format is correct
        if ' ' not in endpoint:
            return False
            
        method = endpoint.split()[0].upper()
        if method not in ['GET', 'POST', 'PUT', 'PATCH', 'DELETE']:
            return False
        
        # For now, allow any endpoint (we'll improve validation later)
        return True
    
    def _generate_fallback_candidates(
        self, 
        events: List[Any], 
        spec_data: Dict[str, Any], 
        count: int
    ) -> List[Dict[str, Any]]:
        """Generate fallback candidates using simple heuristics"""
        candidates = []
        endpoints = spec_data.get('endpoints', [])
        
        # Prefer safe endpoints
        safe_endpoints = [ep for ep in endpoints if not ep.get('is_destructive', False)]
        
        # Simple heuristics based on last action
        if events:
            last_endpoint = events[-1].endpoint  # Fixed: use .endpoint instead of ['endpoint']
            last_method = last_endpoint.split()[0]
            
            # Common patterns
            if last_method == 'GET':
                preferred_methods = ['POST', 'PUT']
            elif last_method == 'POST':
                preferred_methods = ['GET', 'PUT']
            else:
                preferred_methods = ['GET']
        else:
            preferred_methods = ['GET']
        
        # Generate candidates
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
        
        # Fill remaining with GET endpoints if needed
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