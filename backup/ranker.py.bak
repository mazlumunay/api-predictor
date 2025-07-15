import os
import logging
import time
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from .trainer import MLModelTrainer
from ..utils.cold_start import ColdStartHandler

logger = logging.getLogger(__name__)

class PredictionRanker:
    """
    Optimized ML layer that ranks candidate predictions with performance improvements
    """
    
    def __init__(self):
        self.ml_trainer = MLModelTrainer()
        self.cold_start_handler = ColdStartHandler()
        self.is_trained = False
        self.training_samples = 0
        
        # Performance optimizations
        self.feature_cache = {}  # Cache computed features
        self.heuristic_cache = {}  # Cache heuristic scores
        self.cache_max_size = 1000
        self.cache_ttl = 300  # 5 minutes
        
    async def load_model(self):
        """Load pre-trained model or train if not available"""
        try:
            # Try to load existing model
            if self.ml_trainer.load_model():
                logger.info("ML model loaded successfully")
                self.is_trained = True
                self.training_samples = 1000  # Assume default training size
            else:
                # Train a new model if none exists
                logger.info("No ML model found, training new model...")
                metrics = self.ml_trainer.train_model(n_samples=1000)
                self.training_samples = metrics.get('n_samples', 1000)
                logger.info(f"New model trained with {self.training_samples} samples, metrics: {metrics}")
                self.is_trained = True
        except Exception as e:
            logger.error(f"Error with ML model: {e}. Falling back to heuristics.")
            self.is_trained = False
            self.training_samples = 0
    
    async def rank_candidates(
        self, 
        candidates: List[Dict[str, Any]], 
        events: List[Any],  # APIEvent Pydantic models
        prompt: Optional[str], 
        user_id: str,
        spec_data: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Optimized ranking with caching and performance improvements
        """
        if not candidates:
            return []
        
        # Quick heuristic ranking for small candidate sets (performance optimization)
        if len(candidates) <= 3:
            return self._quick_heuristic_rank(candidates, events, prompt)
        
        # Check for cold-start conditions
        is_cold_start, cold_start_reason = self.cold_start_handler.is_cold_start(
            events=events,
            model_trained=self.is_trained,
            model_samples=self.training_samples
        )
        
        try:
            if is_cold_start and spec_data:
                # Use cold-start predictions instead of ranking candidates
                logger.info(f"Using cold-start predictions: {cold_start_reason}")
                scored_candidates = self.cold_start_handler.generate_cold_start_predictions(
                    events=events,
                    prompt=prompt,
                    spec_data=spec_data,
                    k=len(candidates),
                    reason=cold_start_reason
                )
            elif self.is_trained and self.ml_trainer.model is not None and not is_cold_start:
                # Use optimized ML model for ranking
                scored_candidates = self._ml_rank_optimized(candidates, events, prompt)
                logger.info(f"Used optimized ML model for ranking {len(candidates)} candidates")
            else:
                # Fallback to optimized heuristic ranking
                scored_candidates = self._heuristic_rank_optimized(candidates, events, prompt)
                logger.info(f"Used optimized heuristic ranking for {len(candidates)} candidates")
            
            # Sort by score (descending)
            scored_candidates.sort(key=lambda x: x['score'], reverse=True)
            
            return scored_candidates
            
        except Exception as e:
            logger.error(f"Error ranking candidates: {e}")
            # Fallback to simple scoring
            return self._simple_fallback_rank(candidates)
    
    def _ml_rank_optimized(
        self, 
        candidates: List[Dict[str, Any]], 
        events: List[Any], 
        prompt: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Optimized ML ranking with feature caching"""
        scored_candidates = []
        
        # Generate cache key for events
        events_key = self._generate_events_cache_key(events)
        
        # Get or compute base features
        if events_key in self.feature_cache:
            cached_item = self.feature_cache[events_key]
            if time.time() - cached_item['timestamp'] < self.cache_ttl:
                base_features = cached_item['features']
            else:
                base_features = self._extract_base_features(events)
                self._cache_features(events_key, base_features)
        else:
            base_features = self._extract_base_features(events)
            self._cache_features(events_key, base_features)
        
        # Process candidates efficiently
        for candidate in candidates:
            try:
                # Extract candidate-specific features
                candidate_features = self._extract_candidate_features(candidate, prompt)
                
                # Combine features
                features = {**base_features, **candidate_features}
                
                # Fast ML prediction
                ml_score = self._fast_ml_predict(features)
                
                # Combine with cached heuristic score
                heuristic_score = self._get_cached_heuristic_score(candidate, events, prompt)
                final_score = 0.7 * ml_score + 0.3 * heuristic_score
                final_score = max(0.0, min(1.0, final_score))  # Clamp to [0,1]
                
                scored_candidates.append({
                    'endpoint': candidate['endpoint'],
                    'params': candidate.get('params', {}),
                    'score': final_score,
                    'why': f"ML prediction ({ml_score:.3f}) + heuristics ({heuristic_score:.3f}) = {final_score:.3f}"
                })
                
            except Exception as e:
                logger.debug(f"Error in ML scoring for {candidate.get('endpoint', 'unknown')}: {e}")
                # Fast fallback for this candidate
                heuristic_score = self._calculate_heuristic_score(candidate, events, prompt)
                scored_candidates.append({
                    'endpoint': candidate['endpoint'],
                    'params': candidate.get('params', {}),
                    'score': heuristic_score,
                    'why': f"Heuristic fallback: {heuristic_score:.3f}"
                })
        
        return scored_candidates
    
    def _heuristic_rank_optimized(
        self, 
        candidates: List[Dict[str, Any]], 
        events: List[Any], 
        prompt: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Optimized heuristic ranking with caching"""
        scored_candidates = []
        
        for candidate in candidates:
            score = self._get_cached_heuristic_score(candidate, events, prompt)
            scored_candidates.append({
                'endpoint': candidate['endpoint'],
                'params': candidate.get('params', {}),
                'score': score,
                'why': self._generate_heuristic_explanation(candidate, events, prompt, score)
            })
        
        return scored_candidates
    
    def _quick_heuristic_rank(
        self, 
        candidates: List[Dict[str, Any]], 
        events: List[Any], 
        prompt: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Very fast ranking for small candidate sets"""
        scored_candidates = []
        
        for candidate in candidates:
            score = 0.6  # Base score
            
            # Quick prompt matching
            if prompt:
                endpoint_lower = candidate['endpoint'].lower()
                prompt_lower = prompt.lower()
                
                # Fast keyword matching
                if any(word in endpoint_lower for word in prompt_lower.split()[:3]):
                    score += 0.3
            
            # Quick source bonus
            if candidate.get('source') == 'openai':
                score += 0.2
            
            scored_candidates.append({
                'endpoint': candidate['endpoint'],
                'params': candidate.get('params', {}),
                'score': min(score, 1.0),
                'why': 'Quick heuristic ranking'
            })
        
        return sorted(scored_candidates, key=lambda x: x['score'], reverse=True)
    
    def _simple_fallback_rank(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simple fallback ranking when all else fails"""
        return [
            {
                'endpoint': c['endpoint'],
                'params': c.get('params', {}),
                'score': 0.5,
                'why': "Simple fallback ranking"
            }
            for c in candidates
        ]
    
    def _fast_ml_predict(self, features: Dict[str, Any]) -> float:
        """Optimized ML prediction"""
        try:
            import pandas as pd
            feature_vector = pd.DataFrame([features])
            
            # Ensure all required columns are present
            for col in self.ml_trainer.feature_columns:
                if col not in feature_vector.columns:
                    feature_vector[col] = 0
            
            # Reorder columns to match training
            feature_vector = feature_vector[self.ml_trainer.feature_columns]
            
            # Fast prediction
            prediction = self.ml_trainer.model.predict(feature_vector)[0]
            return max(0.0, min(1.0, prediction))
            
        except Exception as e:
            logger.debug(f"Fast ML prediction failed: {e}")
            return 0.5
    
    def _extract_base_features(self, events: List[Any]) -> Dict[str, Any]:
        """Extract cacheable base features from events"""
        last_event = events[-1] if events else None
        
        features = {
            # Basic sequence features
            'sequence_length': len(events),
            'time_since_last_min': 1.0,  # Simplified
            
            # Method distribution in history
            'get_count': sum(1 for e in events if e.endpoint.split()[0] == 'GET'),
            'post_count': sum(1 for e in events if e.endpoint.split()[0] == 'POST'),
            'put_count': sum(1 for e in events if e.endpoint.split()[0] == 'PUT'),
            'delete_count': sum(1 for e in events if e.endpoint.split()[0] == 'DELETE'),
            
            # Pattern features
            'same_method_streak': self._calculate_same_method_streak(events),
            'crud_pattern_score': self._calculate_crud_pattern_score(events),
        }
        
        # Last action features
        if last_event:
            last_method = last_event.endpoint.split()[0]
            last_path = ' '.join(last_event.endpoint.split()[1:]) if ' ' in last_event.endpoint else '/'
            
            # Safe encoding with fallback
            features['last_method_encoded'] = self._safe_encode_method(last_method)
            features['last_path_encoded'] = self._safe_encode_path(last_path)
        else:
            features['last_method_encoded'] = 0
            features['last_path_encoded'] = 0
        
        return features
    
    def _extract_candidate_features(self, candidate: Dict[str, Any], prompt: Optional[str]) -> Dict[str, Any]:
        """Extract candidate-specific features"""
        features = {
            'has_prompt': prompt is not None,
            'prompt_create_keywords': 0,
            'prompt_read_keywords': 0,
            'prompt_update_keywords': 0,
            'prompt_delete_keywords': 0,
        }
        
        # Prompt keyword analysis (optimized)
        if prompt:
            prompt_lower = prompt.lower()
            
            # Fast keyword matching
            create_keywords = ['create', 'new', 'add', 'make']
            read_keywords = ['get', 'list', 'view', 'show', 'find']
            update_keywords = ['update', 'edit', 'modify', 'change']
            delete_keywords = ['delete', 'remove', 'destroy']
            
            features['prompt_create_keywords'] = sum(1 for kw in create_keywords if kw in prompt_lower)
            features['prompt_read_keywords'] = sum(1 for kw in read_keywords if kw in prompt_lower)
            features['prompt_update_keywords'] = sum(1 for kw in update_keywords if kw in prompt_lower)
            features['prompt_delete_keywords'] = sum(1 for kw in delete_keywords if kw in prompt_lower)
        
        return features
    
    def _get_cached_heuristic_score(self, candidate: Dict[str, Any], events: List[Any], prompt: Optional[str]) -> float:
        """Get heuristic score with caching"""
        # Generate cache key
        cache_key = f"{candidate['endpoint']}:{len(events)}:{prompt or 'none'}"
        
        # Check cache
        if cache_key in self.heuristic_cache:
            cached_item = self.heuristic_cache[cache_key]
            if time.time() - cached_item['timestamp'] < self.cache_ttl:
                return cached_item['score']
        
        # Calculate and cache
        score = self._calculate_heuristic_score(candidate, events, prompt)
        
        # Cache with size limit
        if len(self.heuristic_cache) >= self.cache_max_size:
            # Remove oldest entries
            oldest_key = min(self.heuristic_cache.keys(), 
                           key=lambda k: self.heuristic_cache[k]['timestamp'])
            del self.heuristic_cache[oldest_key]
        
        self.heuristic_cache[cache_key] = {
            'score': score,
            'timestamp': time.time()
        }
        
        return score
    
    def _calculate_heuristic_score(
        self, 
        candidate: Dict[str, Any], 
        events: List[Any], 
        prompt: Optional[str]
    ) -> float:
        """Fast heuristic scoring"""
        score = 0.5  # Base score
        
        endpoint = candidate['endpoint']
        method = endpoint.split()[0]
        path = endpoint.split(' ', 1)[1] if ' ' in endpoint else ''
        
        # Source bonus
        if candidate.get('source') == 'openai':
            score += 0.3
        elif candidate.get('source') == 'ai':
            score += 0.2
        
        # Pattern matching (optimized)
        if events:
            last_endpoint = events[-1].endpoint
            last_method = last_endpoint.split()[0]
            
            # Fast transition scoring
            transition_bonus = {
                ('GET', 'POST'): 0.15,
                ('POST', 'PUT'): 0.1,
                ('POST', 'PATCH'): 0.1,
                ('GET', 'GET'): 0.05
            }
            score += transition_bonus.get((last_method, method), 0)
        
        # Prompt matching (optimized)
        if prompt:
            prompt_lower = prompt.lower()
            path_lower = path.lower()
            
            # Fast keyword matching
            quick_matches = [
                ('create', 'post'), ('new', 'post'), ('add', 'post'),
                ('update', 'put'), ('edit', 'put'), ('get', 'get'),
                ('list', 'get'), ('view', 'get')
            ]
            
            for keyword, target_method in quick_matches:
                if keyword in prompt_lower and target_method == method.lower():
                    score += 0.1
                    break
        
        # Safety penalty
        if method == 'DELETE':
            score *= 0.3
        elif 'delete' in path.lower():
            score *= 0.5
        
        return min(score, 1.0)
    
    def _generate_events_cache_key(self, events: List[Any]) -> str:
        """Generate cache key for events"""
        if not events:
            return "no_events"
        
        # Use last 3 events for cache key
        key_events = events[-3:]
        key_parts = [f"{e.endpoint.split()[0]}" for e in key_events]
        return ":".join(key_parts)
    
    def _cache_features(self, cache_key: str, features: Dict[str, Any]):
        """Cache features with size management"""
        if len(self.feature_cache) >= self.cache_max_size:
            # Remove oldest entry
            oldest_key = min(self.feature_cache.keys(), 
                           key=lambda k: self.feature_cache[k]['timestamp'])
            del self.feature_cache[oldest_key]
        
        self.feature_cache[cache_key] = {
            'features': features,
            'timestamp': time.time()
        }
    
    def _safe_encode_method(self, method: str) -> int:
        """Safe method encoding with fallback"""
        try:
            if hasattr(self.ml_trainer, 'label_encoders') and 'last_method' in self.ml_trainer.label_encoders:
                encoder = self.ml_trainer.label_encoders['last_method']
                if method in encoder.classes_:
                    return encoder.transform([method])[0]
            return 0
        except:
            return 0
    
    def _safe_encode_path(self, path: str) -> int:
        """Safe path encoding with fallback"""
        try:
            if hasattr(self.ml_trainer, 'label_encoders') and 'last_path' in self.ml_trainer.label_encoders:
                encoder = self.ml_trainer.label_encoders['last_path']
                if path in encoder.classes_:
                    return encoder.transform([path])[0]
            return 0
        except:
            return 0
    
    def _calculate_same_method_streak(self, events: List[Any]) -> int:
        """Fast same method streak calculation"""
        if not events:
            return 0
        
        last_method = events[-1].endpoint.split()[0]
        streak = 1
        
        for i in range(len(events) - 2, max(len(events) - 6, -1), -1):  # Only check last 5
            if events[i].endpoint.split()[0] == last_method:
                streak += 1
            else:
                break
        
        return streak
    
    def _calculate_crud_pattern_score(self, events: List[Any]) -> float:
        """Fast CRUD pattern scoring"""
        if len(events) < 2:
            return 0.5
        
        score = 0.0
        transitions = 0
        
        # Only check last few transitions for speed
        for i in range(max(1, len(events) - 4), len(events)):
            prev_method = events[i-1].endpoint.split()[0]
            curr_method = events[i].endpoint.split()[0]
            
            # Fast transition scoring
            if prev_method == 'GET' and curr_method in ['POST', 'PUT']:
                score += 1.0
            elif prev_method == 'POST' and curr_method == 'GET':
                score += 0.8
            elif prev_method == 'PUT' and curr_method == 'GET':
                score += 0.8
            elif prev_method == curr_method and curr_method == 'GET':
                score += 0.5
            
            transitions += 1
        
        return score / transitions if transitions > 0 else 0.5
    
    def _generate_heuristic_explanation(
        self, 
        candidate: Dict[str, Any], 
        events: List[Any], 
        prompt: Optional[str], 
        score: float
    ) -> str:
        """Fast explanation generation"""
        explanations = []
        
        # Source explanation
        source = candidate.get('source', 'unknown')
        if source == 'openai':
            explanations.append("AI-generated")
        elif source == 'ai':
            explanations.append("AI prediction")
        else:
            explanations.append("Heuristic")
        
        # Pattern explanation (simplified)
        if events:
            last_method = events[-1].endpoint.split()[0]
            current_method = candidate['endpoint'].split()[0]
            
            if last_method == 'GET' and current_method == 'POST':
                explanations.append("GET→POST flow")
            elif last_method == current_method:
                explanations.append(f"{current_method} continuation")
        
        # Confidence
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