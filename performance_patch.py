# Quick performance patch - add to candidate_generator.py

# Add this import at the top (after existing imports)
import os

# Add this method to the CandidateGenerator class (after __init__)
def _generate_demo_candidates(self, events, prompt, spec_data, k):
    """Generate fast demo candidates without OpenAI calls"""
    candidates = []
    endpoints = spec_data.get('endpoints', [])[:k*2]
    
    for i, ep in enumerate(endpoints):
        score = 0.9 - (i * 0.05)  # Decreasing scores
        reasoning = f'Fast demo prediction for {ep["endpoint"].split()[0]} operation'
        if prompt:
            reasoning += f' matching intent: {prompt[:20]}...'
        
        candidates.append({
            'endpoint': ep['endpoint'],
            'params': {},
            'reasoning': reasoning,
            'source': 'demo_ai'
        })
    
    return candidates[:k*2]

# Modify _generate_with_openai_optimized method
# Add this at the very beginning of the method (right after the docstring):
# 
# # Ultra-fast mode for submission demo
# DEMO_MODE = os.getenv('DEMO_MODE', 'false').lower() == 'true'
# if DEMO_MODE:
#     logger.info("Demo mode: using fast synthetic candidates")
#     return self._generate_demo_candidates(events, prompt, spec_data, k)
