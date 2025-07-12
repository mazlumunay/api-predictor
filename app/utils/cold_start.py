import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ColdStartHandler:
    """
    Handles predictions for users with insufficient history or when ML models are unavailable
    
    Cold-start conditions:
    - Users with < 3 events
    - Models with < 100 training samples
    - ML model unavailable or failed to load
    """
    
    def __init__(self):
        self.safe_endpoints_cache = {}
        
    def is_cold_start(self, events: List[Any], model_trained: bool = False, model_samples: int = 0) -> tuple[bool, str]:
        """
        Determine if this is a cold-start scenario and why
        
        Returns:
            (is_cold_start, reason)
        """
        reasons = []
        
        # Check user history
        if len(events) < 3:
            reasons.append(f"user has only {len(events)} events (need ≥3)")
        
        # Check model availability
        if not model_trained:
            reasons.append("ML model not trained or unavailable")
        elif model_samples < 100:
            reasons.append(f"ML model trained on only {model_samples} samples (need ≥100)")
        
        is_cold_start = len(reasons) > 0
        reason = "; ".join(reasons) if reasons else "sufficient data available"
        
        if is_cold_start:
            logger.info(f"Cold-start scenario detected: {reason}")
        
        return is_cold_start, reason
    
    def generate_cold_start_predictions(
        self, 
        events: List[Any], 
        prompt: Optional[str], 
        spec_data: Dict[str, Any], 
        k: int,
        reason: str = "insufficient user history"
    ) -> List[Dict[str, Any]]:
        """
        Generate predictions for cold-start scenarios using documented heuristics
        """
        logger.info(f"Generating cold-start predictions: {reason}")
        
        # Extract safe endpoints from spec
        safe_endpoints = self._get_safe_endpoints(spec_data)
        
        # Generate predictions using different strategies
        predictions = []
        
        # Strategy 1: Prompt-based predictions (highest priority)
        if prompt:
            prompt_predictions = self._prompt_based_predictions(prompt, safe_endpoints, k)
            predictions.extend(prompt_predictions)
        
        # Strategy 2: Last action continuation (if any events)
        if events:
            continuation_predictions = self._last_action_continuation(events, safe_endpoints, k)
            predictions.extend(continuation_predictions)
        
        # Strategy 3: Common workflow patterns
        workflow_predictions = self._common_workflow_predictions(events, safe_endpoints, k)
        predictions.extend(workflow_predictions)
        
        # Strategy 4: Popular safe endpoints (fallback)
        popular_predictions = self._popular_safe_endpoints(safe_endpoints, k)
        predictions.extend(popular_predictions)
        
        # Remove duplicates while preserving order
        seen_endpoints = set()
        unique_predictions = []
        for pred in predictions:
            endpoint = pred['endpoint']
            if endpoint not in seen_endpoints:
                seen_endpoints.add(endpoint)
                unique_predictions.append(pred)
                if len(unique_predictions) >= k:
                    break
        
        # Ensure we have enough predictions
        while len(unique_predictions) < k and len(unique_predictions) < len(safe_endpoints):
            remaining_endpoints = [ep for ep in safe_endpoints 
                                 if ep['endpoint'] not in seen_endpoints]
            if remaining_endpoints:
                ep = remaining_endpoints[0]
                unique_predictions.append({
                    'endpoint': ep['endpoint'],
                    'params': {},
                    'score': 0.3,
                    'why': f"Cold-start fallback: safe {ep['endpoint'].split()[0]} operation"
                })
                seen_endpoints.add(ep['endpoint'])
            else:
                break
        
        # Adjust scores to indicate cold-start uncertainty
        for pred in unique_predictions:
            pred['score'] *= 0.8  # Reduce confidence for cold-start
            pred['why'] = f"[COLD-START: {reason}] {pred['why']}"
        
        logger.info(f"Generated {len(unique_predictions)} cold-start predictions")
        return unique_predictions[:k]
    
    def _get_safe_endpoints(self, spec_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract and cache safe endpoints from OpenAPI spec"""
        spec_title = spec_data.get('title', 'unknown')
        
        if spec_title in self.safe_endpoints_cache:
            return self.safe_endpoints_cache[spec_title]
        
        all_endpoints = spec_data.get('endpoints', [])
        safe_endpoints = []
        
        for endpoint in all_endpoints:
            if not endpoint.get('is_destructive', False):
                # Calculate safety score
                method = endpoint['endpoint'].split()[0]
                path = endpoint['endpoint'].split(' ', 1)[1] if ' ' in endpoint['endpoint'] else ''
                
                safety_score = self._calculate_endpoint_safety_score(method, path, endpoint)
                
                safe_endpoints.append({
                    **endpoint,
                    'safety_score': safety_score
                })
        
        # Sort by safety score (higher is safer)
        safe_endpoints.sort(key=lambda x: x['safety_score'], reverse=True)
        
        # Cache for future use
        self.safe_endpoints_cache[spec_title] = safe_endpoints
        
        return safe_endpoints
    
    def _calculate_endpoint_safety_score(self, method: str, path: str, endpoint_data: Dict[str, Any]) -> float:
        """Calculate safety score for an endpoint"""
        score = 0.5  # Base score
        
        # Method-based scoring
        method_scores = {
            'GET': 1.0,
            'POST': 0.7,
            'PUT': 0.5,
            'PATCH': 0.4,
            'DELETE': 0.1
        }
        score += method_scores.get(method, 0.3)
        
        # Path-based scoring
        safe_path_keywords = ['list', 'get', 'view', 'read', 'show', 'search', 'find']
        risky_path_keywords = ['delete', 'remove', 'destroy', 'clear', 'reset', 'purge']
        
        path_lower = path.lower()
        
        for keyword in safe_path_keywords:
            if keyword in path_lower:
                score += 0.2
                break
        
        for keyword in risky_path_keywords:
            if keyword in path_lower:
                score -= 0.5
                break
        
        # Summary-based scoring
        summary = endpoint_data.get('summary', '').lower()
        if any(word in summary for word in ['get', 'list', 'view', 'read']):
            score += 0.1
        if any(word in summary for word in ['delete', 'remove', 'destroy']):
            score -= 0.3
        
        # Tags-based scoring (assume certain tags are safer)
        tags = endpoint_data.get('tags', [])
        safe_tags = ['users', 'products', 'info', 'health', 'status']
        for tag in tags:
            if tag.lower() in safe_tags:
                score += 0.1
                break
        
        return max(0.0, min(2.0, score))  # Clamp between 0 and 2
    
    def _prompt_based_predictions(self, prompt: str, safe_endpoints: List[Dict], k: int) -> List[Dict[str, Any]]:
        """Generate predictions based on natural language prompt"""
        predictions = []
        prompt_lower = prompt.lower()
        
        # Intent keyword mapping
        intent_mappings = {
            'create': ['POST'],
            'new': ['POST'],
            'add': ['POST'],
            'make': ['POST'],
            'get': ['GET'],
            'list': ['GET'],
            'view': ['GET'],
            'show': ['GET'],
            'find': ['GET'],
            'search': ['GET'],
            'update': ['PUT', 'PATCH'],
            'edit': ['PUT', 'PATCH'],
            'modify': ['PUT', 'PATCH'],
            'change': ['PUT', 'PATCH']
        }
        
        # Entity keyword mapping
        entity_keywords = ['user', 'customer', 'product', 'order', 'invoice', 'payment', 
                          'item', 'cart', 'account', 'profile', 'setting']
        
        # Find matching methods
        preferred_methods = []
        for keyword, methods in intent_mappings.items():
            if keyword in prompt_lower:
                preferred_methods.extend(methods)
        
        # Find matching entities
        relevant_entities = [entity for entity in entity_keywords if entity in prompt_lower]
        
        # Generate predictions
        for endpoint in safe_endpoints[:k*2]:  # Check more endpoints
            method = endpoint['endpoint'].split()[0]
            path = endpoint['endpoint'].split(' ', 1)[1] if ' ' in endpoint['endpoint'] else ''
            
            score = 0.5
            reasons = []
            
            # Method matching
            if method in preferred_methods:
                score += 0.4
                reasons.append(f"method {method} matches intent")
            
            # Entity matching
            for entity in relevant_entities:
                if entity in path.lower() or entity in endpoint.get('summary', '').lower():
                    score += 0.3
                    reasons.append(f"matches entity '{entity}'")
                    break
            
            # Summary matching
            summary = endpoint.get('summary', '').lower()
            prompt_words = prompt_lower.split()
            common_words = set(prompt_words) & set(summary.split())
            if len(common_words) > 0:
                score += 0.2 * len(common_words)
                reasons.append(f"summary matches prompt keywords")
            
            if score > 0.6:  # Only include good matches
                predictions.append({
                    'endpoint': endpoint['endpoint'],
                    'params': {},
                    'score': min(score, 1.0),
                    'why': f"Prompt match: {'; '.join(reasons)}"
                })
            
            if len(predictions) >= k:
                break
        
        return predictions
    
    def _last_action_continuation(self, events: List[Any], safe_endpoints: List[Dict], k: int) -> List[Dict[str, Any]]:
        """Generate predictions based on the last user action"""
        if not events:
            return []
        
        predictions = []
        last_event = events[-1]
        last_method = last_event.endpoint.split()[0]
        last_path = last_event.endpoint.split(' ', 1)[1] if ' ' in last_event.endpoint else ''
        
        # Common continuation patterns
        continuation_patterns = {
            'GET': [
                ('POST', 0.8, 'create after viewing'),
                ('PUT', 0.6, 'update after viewing'),
                ('GET', 0.4, 'continue browsing')
            ],
            'POST': [
                ('GET', 0.9, 'view after creating'),
                ('PUT', 0.5, 'update after creating')
            ],
            'PUT': [
                ('GET', 0.8, 'view after updating')
            ]
        }
        
        patterns = continuation_patterns.get(last_method, [])
        
        for next_method, base_score, reason in patterns:
            # Find endpoints matching the pattern
            matching_endpoints = [ep for ep in safe_endpoints 
                                if ep['endpoint'].split()[0] == next_method]
            
            # Prefer endpoints with similar paths
            for endpoint in matching_endpoints[:3]:  # Top 3 per pattern
                ep_path = endpoint['endpoint'].split(' ', 1)[1] if ' ' in endpoint['endpoint'] else ''
                
                # Calculate path similarity
                path_similarity = self._calculate_path_similarity(last_path, ep_path)
                final_score = base_score * (0.5 + 0.5 * path_similarity)
                
                predictions.append({
                    'endpoint': endpoint['endpoint'],
                    'params': {},
                    'score': final_score,
                    'why': f"Continuation: {reason} (last: {last_method})"
                })
                
                if len(predictions) >= k:
                    break
            
            if len(predictions) >= k:
                break
        
        return predictions
    
    def _common_workflow_predictions(self, events: List[Any], safe_endpoints: List[Dict], k: int) -> List[Dict[str, Any]]:
        """Generate predictions based on common API workflow patterns"""
        predictions = []
        
        # Common workflow starting points
        workflow_starters = [
            ('GET', 'list', 0.9, 'common workflow start'),
            ('GET', 'user', 0.8, 'user management workflow'),
            ('GET', 'product', 0.7, 'product browsing workflow'),
            ('POST', 'auth', 0.6, 'authentication workflow')
        ]
        
        for method, path_keyword, score, reason in workflow_starters:
            matching_endpoints = [
                ep for ep in safe_endpoints 
                if (ep['endpoint'].split()[0] == method and 
                    path_keyword in ep['endpoint'].lower())
            ]
            
            for endpoint in matching_endpoints[:2]:  # Top 2 per workflow
                predictions.append({
                    'endpoint': endpoint['endpoint'],
                    'params': {},
                    'score': score,
                    'why': reason
                })
                
                if len(predictions) >= k:
                    break
            
            if len(predictions) >= k:
                break
        
        return predictions
    
    def _popular_safe_endpoints(self, safe_endpoints: List[Dict], k: int) -> List[Dict[str, Any]]:
        """Generate predictions based on popular safe endpoints"""
        predictions = []
        
        # Prioritize GET endpoints as they're always safe
        get_endpoints = [ep for ep in safe_endpoints if ep['endpoint'].split()[0] == 'GET']
        
        for endpoint in get_endpoints[:k]:
            predictions.append({
                'endpoint': endpoint['endpoint'],
                'params': {},
                'score': 0.4 + endpoint.get('safety_score', 0.5) * 0.2,
                'why': f"Safe {endpoint['endpoint'].split()[0]} operation"
            })
        
        return predictions
    
    def _calculate_path_similarity(self, path1: str, path2: str) -> float:
        """Calculate similarity between two API paths"""
        if path1 == path2:
            return 1.0
        
        # Simple similarity based on common segments
        segments1 = set(path1.split('/'))
        segments2 = set(path2.split('/'))
        
        if not segments1 or not segments2:
            return 0.0
        
        intersection = segments1 & segments2
        union = segments1 | segments2
        
        return len(intersection) / len(union) if union else 0.0