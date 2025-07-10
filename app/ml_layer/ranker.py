import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class PredictionRanker:
    """
    ML layer that ranks candidate predictions
    """
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        
    async def load_model(self):
        """Load pre-trained model or initialize heuristic ranking"""
        try:
            # For now, we'll use heuristic ranking
            # Later we can add proper ML model loading
            logger.info("Prediction ranker initialized with heuristic ranking")
            self.is_trained = True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.is_trained = False
    
    async def rank_candidates(
        self, 
        candidates: List[Dict[str, Any]], 
        events: List[Any],  # APIEvent Pydantic models
        prompt: Optional[str], 
        user_id: str
    ) -> List[Dict[str, Any]]:
        """
        Rank candidate predictions and return scored results
        """
        if not candidates:
            return []
        
        try:
            # Use heuristic ranking for now
            scored_candidates = self._heuristic_rank(candidates, events, prompt)
            
            # Sort by score (descending)
            scored_candidates.sort(key=lambda x: x['score'], reverse=True)
            
            return scored_candidates
            
        except Exception as e:
            logger.error(f"Error ranking candidates: {e}")
            # Fallback to original order with default scores
            return [
                {
                    'endpoint': c['endpoint'],
                    'params': c.get('params', {}),
                    'score': 0.5,
                    'why': c.get('reasoning', 'Default ranking')
                }
                for c in candidates
            ]
    
    def _heuristic_rank(
        self, 
        candidates: List[Dict[str, Any]], 
        events: List[Any], 
        prompt: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Rank using simple heuristics"""
        scored_candidates = []
        
        for candidate in candidates:
            score = self._calculate_heuristic_score(candidate, events, prompt)
            scored_candidates.append({
                'endpoint': candidate['endpoint'],
                'params': candidate.get('params', {}),
                'score': score,
                'why': self._generate_heuristic_explanation(candidate, events, prompt, score)
            })
        
        return scored_candidates
    
    def _calculate_heuristic_score(
        self, 
        candidate: Dict[str, Any], 
        events: List[Any], 
        prompt: Optional[str]
    ) -> float:
        """Calculate score using simple heuristics"""
        score = 0.5  # Base score
        
        endpoint = candidate['endpoint']
        method = endpoint.split()[0]
        path = endpoint.split(' ', 1)[1] if ' ' in endpoint else ''
        
        # Higher score for AI-generated vs fallback
        if candidate.get('source') == 'openai':
            score += 0.3
        elif candidate.get('source') == 'ai':
            score += 0.2
        
        # Recent endpoint pattern matching
        if events:
            last_endpoint = events[-1].endpoint  # Fixed: use .endpoint
            last_method = last_endpoint.split()[0]
            
            # Bonus for logical progression (GET -> POST, etc.)
            if last_method == 'GET' and method == 'POST':
                score += 0.15
            elif last_method == 'POST' and method in ['PUT', 'PATCH']:
                score += 0.1
            elif last_method == 'GET' and method == 'GET':
                score += 0.05  # Same method continuation
        
        # Prompt keyword matching
        if prompt:
            prompt_lower = prompt.lower()
            path_lower = path.lower()
            
            # Common keywords and their path matches
            keyword_matches = [
                ('create', 'post'),
                ('new', 'post'),
                ('add', 'post'),
                ('update', 'put'),
                ('edit', 'put'),
                ('modify', 'patch'),
                ('get', 'get'),
                ('list', 'get'),
                ('view', 'get'),
                ('invoice', 'invoice'),
                ('payment', 'payment'),
                ('customer', 'customer')
            ]
            
            for keyword, path_keyword in keyword_matches:
                if keyword in prompt_lower:
                    if path_keyword in path_lower or path_keyword == method.lower():
                        score += 0.1
        
        # Safety penalty for risky operations
        if method == 'DELETE':
            score *= 0.3
        elif 'delete' in path.lower():
            score *= 0.5
        
        # Bonus for common safe operations
        if method == 'GET':
            score += 0.05
        elif method == 'POST' and any(word in path.lower() for word in ['create', 'new']):
            score += 0.1
        
        return min(score, 1.0)
    
    def _generate_heuristic_explanation(
        self, 
        candidate: Dict[str, Any], 
        events: List[Any], 
        prompt: Optional[str], 
        score: float
    ) -> str:
        """Generate explanation for heuristic scoring"""
        explanations = []
        
        # Source explanation
        source = candidate.get('source', 'unknown')
        if source == 'openai':
            explanations.append("AI-generated with OpenAI")
        elif source == 'ai':
            explanations.append("AI-generated prediction")
        else:
            explanations.append("Heuristic fallback")
        
        # Pattern explanation
        if events:
            last_endpoint = events[-1].endpoint
            last_method = last_endpoint.split()[0]
            current_method = candidate['endpoint'].split()[0]
            
            if last_method == 'GET' and current_method == 'POST':
                explanations.append("logical GET→POST progression")
            elif last_method == current_method:
                explanations.append(f"continues {current_method} pattern")
        
        # Intent explanation
        if prompt:
            explanations.append(f"matches intent: '{prompt[:30]}{'...' if len(prompt) > 30 else ''}'")
        
        # Confidence explanation
        if score > 0.8:
            confidence = "high confidence"
        elif score > 0.6:
            confidence = "good confidence"
        elif score > 0.4:
            confidence = "moderate confidence"
        else:
            confidence = "low confidence"
        
        explanations.append(f"{confidence} ({score:.2f})")
        
        return "; ".join(explanations)