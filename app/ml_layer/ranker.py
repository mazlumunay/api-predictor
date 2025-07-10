class PredictionRanker:
    async def load_model(self):
        pass
    async def rank_candidates(self, candidates, events, prompt, user_id):
        return [{'endpoint': 'GET /test', 'params': {}, 'score': 0.8, 'why': 'test'}]