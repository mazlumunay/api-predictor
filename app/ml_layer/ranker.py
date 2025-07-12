import os
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from .trainer import MLModelTrainer
from ..utils.cold_start import ColdStartHandler

logger = logging.getLogger(__name__)

class PredictionRanker:
    """
    ML layer that ranks candidate predictions
    """
    
    def __init__(self):
        self.ml_trainer = MLModelTrainer()
        self.cold_start_handler = ColdStartHandler()
        self.is_trained = False
        self.training_samples = 0
        
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
        Rank candidate predictions and return scored results
        """
        if not candidates:
            return []
        
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
                # Use ML model for ranking
                scored_candidates = self._ml_rank(candidates, events, prompt)
                logger.info(f"Used ML model for ranking {len(candidates)} candidates")
            else:
                # Fallback to heuristic ranking
                scored_candidates = self._heuristic_rank(candidates, events, prompt)
                logger.info(f"Used heuristic ranking for {len(candidates)} candidates")
            
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
                    'why': f"Default ranking due to error: {str(e)}"
                }
                for c in candidates
            ]
    
    def _ml_rank(
        self, 
        candidates: List[Dict[str, Any]], 
        events: List[Any], 
        prompt: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Rank using trained ML model"""
        scored_candidates = []
        
        for candidate in candidates:
            try:
                # Extract features for this candidate
                features = self._extract_ml_features(candidate, events, prompt)
                
                # Predict score using ML model
                feature_vector = pd.DataFrame([features])
                # Ensure all required columns are present
                for col in self.ml_trainer.feature_columns:
                    if col not in feature_vector.columns:
                        feature_vector[col] = 0
                
                # Reorder columns to match training
                feature_vector = feature_vector[self.ml_trainer.feature_columns]
                
                ml_score = self.ml_trainer.model.predict(feature_vector)[0]
                
                # Combine with heuristic score for robustness
                heuristic_score = self._calculate_heuristic_score(candidate, events, prompt)
                final_score = 0.7 * ml_score + 0.3 * heuristic_score
                final_score = max(0.0, min(1.0, final_score))  # Clamp to [0,1]
                
                scored_candidates.append({
                    'endpoint': candidate['endpoint'],
                    'params': candidate.get('params', {}),
                    'score': final_score,
                    'why': f"ML prediction ({ml_score:.3f}) + heuristics ({heuristic_score:.3f}) = {final_score:.3f}"
                })
                
            except Exception as e:
                logger.error(f"Error in ML scoring for {candidate.get('endpoint', 'unknown')}: {e}")
                # Fallback to heuristic for this candidate
                heuristic_score = self._calculate_heuristic_score(candidate, events, prompt)
                scored_candidates.append({
                    'endpoint': candidate['endpoint'],
                    'params': candidate.get('params', {}),
                    'score': heuristic_score,
                    'why': f"Heuristic fallback: {heuristic_score:.3f}"
                })
        
        return scored_candidates
    
    def _extract_ml_features(
        self, 
        candidate: Dict[str, Any], 
        events: List[Any], 
        prompt: Optional[str]
    ) -> Dict[str, Any]:
        """Extract features for ML model prediction"""
        
        last_event = events[-1] if events else None
        
        features = {
            # Basic sequence features
            'sequence_length': len(events),
            'time_since_last_min': 1.0,  # Simplified - could calculate actual time
            
            # Method distribution in history
            'get_count': sum(1 for e in events if e.endpoint.split()[0] == 'GET'),
            'post_count': sum(1 for e in events if e.endpoint.split()[0] == 'POST'),
            'put_count': sum(1 for e in events if e.endpoint.split()[0] == 'PUT'),
            'delete_count': sum(1 for e in events if e.endpoint.split()[0] == 'DELETE'),
            
            # Pattern features
            'same_method_streak': self._calculate_same_method_streak(events),
            'crud_pattern_score': self._calculate_crud_pattern_score(events),
            
            # Prompt features
            'has_prompt': prompt is not None,
            'prompt_create_keywords': 0,
            'prompt_read_keywords': 0,
            'prompt_update_keywords': 0,
            'prompt_delete_keywords': 0,
        }
        
        # Last action features (encoded)
        if last_event:
            last_method = last_event.endpoint.split()[0]
            last_path = ' '.join(last_event.endpoint.split()[1:]) if ' ' in last_event.endpoint else '/'
            
            # Encode using label encoders (with fallback for unseen values)
            try:
                if 'last_method' in self.ml_trainer.label_encoders:
                    encoder = self.ml_trainer.label_encoders['last_method']
                    if last_method in encoder.classes_:
                        features['last_method_encoded'] = encoder.transform([last_method])[0]
                    else:
                        features['last_method_encoded'] = 0  # Unknown method
                else:
                    features['last_method_encoded'] = 0
                
                if 'last_path' in self.ml_trainer.label_encoders:
                    encoder = self.ml_trainer.label_encoders['last_path']
                    if last_path in encoder.classes_:
                        features['last_path_encoded'] = encoder.transform([last_path])[0]
                    else:
                        features['last_path_encoded'] = 0  # Unknown path
                else:
                    features['last_path_encoded'] = 0
                    
            except Exception as e:
                logger.debug(f"Error encoding features: {e}")
                features['last_method_encoded'] = 0
                features['last_path_encoded'] = 0
        else:
            features['last_method_encoded'] = 0
            features['last_path_encoded'] = 0
        
        # Prompt keyword analysis
        if prompt:
            prompt_lower = prompt.lower()
            
            create_keywords = ['create', 'new', 'add', 'make']
            read_keywords = ['get', 'list', 'view', 'show', 'find']
            update_keywords = ['update', 'edit', 'modify', 'change']
            delete_keywords = ['delete', 'remove', 'destroy']
            
            features['prompt_create_keywords'] = sum(1 for kw in create_keywords if kw in prompt_lower)
            features['prompt_read_keywords'] = sum(1 for kw in read_keywords if kw in prompt_lower)
            features['prompt_update_keywords'] = sum(1 for kw in update_keywords if kw in prompt_lower)
            features['prompt_delete_keywords'] = sum(1 for kw in delete_keywords if kw in prompt_lower)
        
        return features
    
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
            last_endpoint = events[-1].endpoint  # Fixed: use .endpoint instead of ['endpoint']
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
    
    def _calculate_same_method_streak(self, events: List[Any]) -> int:
        """Calculate streak of same HTTP method"""
        if not events:
            return 0
        
        last_method = events[-1].endpoint.split()[0]
        streak = 1
        
        for i in range(len(events) - 2, -1, -1):
            if events[i].endpoint.split()[0] == last_method:
                streak += 1
            else:
                break
        
        return streak
    
    def _calculate_crud_pattern_score(self, events: List[Any]) -> float:
        """Calculate how well the sequence follows CRUD patterns"""
        if len(events) < 2:
            return 0.5
        
        score = 0.0
        transitions = 0
        
        for i in range(1, len(events)):
            prev_method = events[i-1].endpoint.split()[0]
            curr_method = events[i].endpoint.split()[0]
            
            # Score common transitions
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