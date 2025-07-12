import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import logging
from typing import List, Dict, Any, Tuple
import os
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    """Generate synthetic API usage data for training"""
    
    def __init__(self):
        # Common API patterns and workflows
        self.api_workflows = {
            'crud_basic': [
                ('GET', '/users', 0.9),
                ('POST', '/users', 0.7),
                ('GET', '/users/{id}', 0.8),
                ('PUT', '/users/{id}', 0.6),
                ('DELETE', '/users/{id}', 0.3)
            ],
            'e_commerce': [
                ('GET', '/products', 0.9),
                ('GET', '/products/{id}', 0.8),
                ('POST', '/cart/items', 0.7),
                ('GET', '/cart', 0.8),
                ('POST', '/orders', 0.6),
                ('GET', '/orders/{id}', 0.7)
            ],
            'billing': [
                ('GET', '/customers', 0.8),
                ('GET', '/invoices', 0.9),
                ('POST', '/invoices', 0.6),
                ('PUT', '/invoices/{id}', 0.5),
                ('GET', '/payments', 0.7),
                ('POST', '/payments', 0.5)
            ],
            'social': [
                ('GET', '/posts', 0.9),
                ('POST', '/posts', 0.6),
                ('GET', '/posts/{id}', 0.8),
                ('POST', '/posts/{id}/comments', 0.5),
                ('GET', '/users/{id}/followers', 0.7)
            ]
        }
        
        self.transition_patterns = {
            'GET': {'GET': 0.4, 'POST': 0.3, 'PUT': 0.2, 'DELETE': 0.1},
            'POST': {'GET': 0.5, 'PUT': 0.3, 'POST': 0.1, 'DELETE': 0.1},
            'PUT': {'GET': 0.6, 'POST': 0.2, 'PUT': 0.1, 'DELETE': 0.1},
            'DELETE': {'GET': 0.7, 'POST': 0.2, 'PUT': 0.05, 'DELETE': 0.05}
        }
        
        self.prompt_keywords = {
            'create': ['POST'],
            'new': ['POST'],
            'add': ['POST'],
            'update': ['PUT', 'PATCH'],
            'edit': ['PUT', 'PATCH'],
            'modify': ['PUT', 'PATCH'],
            'delete': ['DELETE'],
            'remove': ['DELETE'],
            'get': ['GET'],
            'list': ['GET'],
            'view': ['GET'],
            'show': ['GET']
        }
    
    def generate_training_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic training data"""
        logger.info(f"Generating {n_samples} synthetic training samples")
        
        data = []
        
        for _ in range(n_samples):
            # Choose a workflow
            workflow_name = random.choice(list(self.api_workflows.keys()))
            workflow = self.api_workflows[workflow_name]
            
            # Generate a sequence of 2-5 events
            sequence_length = random.randint(2, 5)
            events = []
            
            for i in range(sequence_length):
                if i == 0:
                    # First event - higher chance of GET
                    method, path, base_score = random.choice(
                        [w for w in workflow if w[0] == 'GET'] or [workflow[0]]
                    )
                else:
                    # Subsequent events follow transition patterns
                    last_method = events[-1]['method']
                    method_probs = self.transition_patterns.get(last_method, {})
                    method = np.random.choice(
                        list(method_probs.keys()),
                        p=list(method_probs.values())
                    )
                    # Find matching path from workflow
                    matching_paths = [w for w in workflow if w[0] == method]
                    if matching_paths:
                        _, path, base_score = random.choice(matching_paths)
                    else:
                        method, path, base_score = random.choice(workflow)
                
                events.append({
                    'method': method,
                    'path': path,
                    'timestamp': datetime.now() - timedelta(minutes=sequence_length-i)
                })
            
            # Generate next action (target)
            last_method = events[-1]['method']
            next_method_probs = self.transition_patterns.get(last_method, {})
            next_method = np.random.choice(
                list(next_method_probs.keys()),
                p=list(next_method_probs.values())
            )
            
            # Find target endpoint
            target_candidates = [w for w in workflow if w[0] == next_method]
            if target_candidates:
                target_method, target_path, target_score = random.choice(target_candidates)
            else:
                target_method, target_path, target_score = random.choice(workflow)
            
            # Generate prompt (sometimes)
            prompt = None
            if random.random() < 0.4:  # 40% chance of having a prompt
                prompt_words = random.choice([
                    'create new user', 'update profile', 'delete item',
                    'get details', 'list all', 'add to cart', 'checkout',
                    'view invoice', 'make payment'
                ])
                prompt = prompt_words
            
            # Create features
            sample = self._extract_features(events, prompt, target_method, target_path, target_score)
            data.append(sample)
        
        df = pd.DataFrame(data)
        logger.info(f"Generated dataset shape: {df.shape}")
        return df
    
    def _extract_features(self, events: List[Dict], prompt: str, 
                         target_method: str, target_path: str, target_score: float) -> Dict:
        """Extract features from event sequence"""
        
        # Basic sequence features
        last_event = events[-1]
        
        features = {
            # Last action features
            'last_method': last_event['method'],
            'last_path': last_event['path'],
            
            # Sequence features
            'sequence_length': len(events),
            'time_since_last_min': 1.0,  # Simplified for synthetic data
            
            # Method distribution in history
            'get_count': sum(1 for e in events if e['method'] == 'GET'),
            'post_count': sum(1 for e in events if e['method'] == 'POST'),
            'put_count': sum(1 for e in events if e['method'] == 'PUT'),
            'delete_count': sum(1 for e in events if e['method'] == 'DELETE'),
            
            # Pattern features
            'same_method_streak': self._calculate_same_method_streak(events),
            'crud_pattern_score': self._calculate_crud_pattern_score(events),
            
            # Prompt features
            'has_prompt': prompt is not None,
            'prompt_create_keywords': 0,
            'prompt_read_keywords': 0,
            'prompt_update_keywords': 0,
            'prompt_delete_keywords': 0,
            
            # Target (what we're predicting)
            'target_method': target_method,
            'target_path': target_path,
            'target_score': target_score
        }
        
        # Prompt keyword analysis
        if prompt:
            prompt_lower = prompt.lower()
            for keyword, methods in self.prompt_keywords.items():
                if keyword in prompt_lower:
                    if 'POST' in methods:
                        features['prompt_create_keywords'] += 1
                    if 'GET' in methods:
                        features['prompt_read_keywords'] += 1
                    if any(m in methods for m in ['PUT', 'PATCH']):
                        features['prompt_update_keywords'] += 1
                    if 'DELETE' in methods:
                        features['prompt_delete_keywords'] += 1
        
        return features
    
    def _calculate_same_method_streak(self, events: List[Dict]) -> int:
        """Calculate streak of same HTTP method"""
        if not events:
            return 0
        
        last_method = events[-1]['method']
        streak = 1
        
        for i in range(len(events) - 2, -1, -1):
            if events[i]['method'] == last_method:
                streak += 1
            else:
                break
        
        return streak
    
    def _calculate_crud_pattern_score(self, events: List[Dict]) -> float:
        """Calculate how well the sequence follows CRUD patterns"""
        if len(events) < 2:
            return 0.5
        
        score = 0.0
        transitions = 0
        
        for i in range(1, len(events)):
            prev_method = events[i-1]['method']
            curr_method = events[i]['method']
            
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


class MLModelTrainer:
    """Train and manage ML models for API prediction"""
    
    def __init__(self, model_dir: str = "data"):
        self.model_dir = model_dir
        self.model = None
        self.label_encoders = {}
        self.feature_columns = []
        self.model_path = os.path.join(model_dir, "prediction_model.joblib")
        self.encoders_path = os.path.join(model_dir, "label_encoders.joblib")
        
        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)
    
    def train_model(self, n_samples: int = 2000) -> Dict[str, float]:
        """Train the prediction model"""
        logger.info("Starting model training...")
        
        # Generate synthetic data
        generator = SyntheticDataGenerator()
        df = generator.generate_training_data(n_samples)
        
        # Prepare features and target
        features, target = self._prepare_training_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        # Train model
        logger.info("Training Random Forest model...")
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        metrics = {
            'train_mse': mean_squared_error(y_train, train_pred),
            'test_mse': mean_squared_error(y_test, test_pred),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'n_samples': len(df),
            'n_features': len(self.feature_columns)
        }
        
        logger.info(f"Model training completed. Test MSE: {metrics['test_mse']:.4f}")
        
        # Save model
        self._save_model()
        
        return metrics
    
    def _prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for training"""
        
        # Define categorical and numerical features
        categorical_features = ['last_method', 'last_path', 'target_method', 'target_path']
        numerical_features = [
            'sequence_length', 'time_since_last_min',
            'get_count', 'post_count', 'put_count', 'delete_count',
            'same_method_streak', 'crud_pattern_score',
            'has_prompt', 'prompt_create_keywords', 'prompt_read_keywords',
            'prompt_update_keywords', 'prompt_delete_keywords'
        ]
        
        # Encode categorical variables
        for col in categorical_features:
            if col not in ['target_method', 'target_path']:  # Don't encode targets
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
        
        # Prepare feature columns
        self.feature_columns = numerical_features + [f'{col}_encoded' for col in categorical_features if col not in ['target_method', 'target_path']]
        
        # Features and target
        features = df[self.feature_columns]
        target = df['target_score']
        
        return features, target
    
    def _save_model(self):
        """Save trained model and encoders"""
        logger.info(f"Saving model to {self.model_path}")
        joblib.dump({
            'model': self.model,
            'feature_columns': self.feature_columns
        }, self.model_path)
        
        joblib.dump(self.label_encoders, self.encoders_path)
        logger.info("Model and encoders saved successfully")
    
    def load_model(self) -> bool:
        """Load trained model"""
        try:
            if os.path.exists(self.model_path):
                logger.info(f"Loading model from {self.model_path}")
                model_data = joblib.load(self.model_path)
                self.model = model_data['model']
                self.feature_columns = model_data['feature_columns']
                
                if os.path.exists(self.encoders_path):
                    self.label_encoders = joblib.load(self.encoders_path)
                
                logger.info("Model loaded successfully")
                return True
            else:
                logger.info("No saved model found")
                return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False


if __name__ == "__main__":
    # Training script
    logging.basicConfig(level=logging.INFO)
    
    trainer = MLModelTrainer()
    metrics = trainer.train_model(n_samples=2000)
    
    print("Training completed!")
    print(f"Metrics: {metrics}")