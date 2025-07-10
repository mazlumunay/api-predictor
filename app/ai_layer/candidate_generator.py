class CandidateGenerator:
    async def generate_candidates(self, events, prompt, spec_data, k):
        return [{'endpoint': 'GET /test', 'params': {}, 'source': 'test'}]