# Assumptions.md

## Clarifying Questions I Would Have Asked

### Business Context
1. **Target APIs**: Are we primarily focusing on REST APIs, or should we handle GraphQL/RPC APIs as well?
2. **User Types**: Are users typically developers, end-users, or both? This affects prediction complexity.
3. **Usage Patterns**: What's the expected volume? (1 prediction/sec vs 1000/sec affects caching strategy)
4. **Accuracy vs Speed**: Would you prefer slightly slower but more accurate predictions, or faster with moderate accuracy?

### Technical Constraints
1. **OpenAI Dependencies**: Is OpenAI API availability acceptable, or do we need fully offline operation?
2. **Data Privacy**: Can we log user interactions for model improvement, or is this privacy-sensitive?
3. **Spec Diversity**: Should we handle broken/incomplete OpenAPI specs gracefully?
4. **Real-time Learning**: Is online learning from user feedback a priority for v1?

### Safety & Ethics
1. **Destructive Operations**: What constitutes "clearly implied" for DELETE operations?
2. **Rate Limiting**: Should we implement per-user rate limiting to prevent abuse?
3. **Audit Trail**: Do we need to log predictions for compliance/debugging?

## Key Assumptions Made

### Data & Training
- **No Real Usage Data**: Assumed no access to real SaaS API usage patterns, so implemented synthetic data generation
- **Cold Start Priority**: Assumed many users will have minimal history, so prioritized robust cold-start handling
- **OpenAPI Completeness**: Assumed specs may be incomplete/inconsistent, built fallback mechanisms
- **Temporal Patterns**: Assumed recent actions are more predictive than historical ones (5 most recent events)

### AI Layer Design
- **LLM Selection**: Chose GPT-4o-mini for cost-effectiveness over GPT-4 for accuracy
- **Prompt Engineering**: Assumed structured prompts with examples work better than few-shot learning
- **Fallback Strategy**: Assumed heuristic fallback is essential when AI is unavailable
- **Context Window**: Limited to recent events + spec sample to stay within token limits

### ML Layer Approach
- **Heuristic Start**: Started with heuristic scoring as baseline before ML model training
- **Feature Engineering**: Assumed sequence patterns and prompt keywords are most predictive
- **Model Complexity**: Chose Random Forest over deep learning for interpretability and speed
- **Training Data**: Generated synthetic data based on common API workflow patterns

### Performance & Scalability
- **Cache Strategy**: Assumed OpenAPI specs change infrequently (1-hour TTL)
- **Memory vs Speed**: Optimized for response time over memory usage within 4GB limit
- **Concurrent Users**: Designed for moderate concurrency (10-100 concurrent users)
- **Resource Limits**: Assumed 2 CPU / 4GB RAM is sufficient for target workload

### Safety & Security
- **Conservative Safety**: Assumed it's better to be overly cautious with destructive operations
- **Input Validation**: Assumed malicious input is possible, implemented comprehensive validation
- **Spec Validation**: Assumed some OpenAPI specs may be malicious or malformed
- **Error Handling**: Prioritized graceful degradation over failing fast

## Trade-offs Made

### AI vs ML Balance
**Decision**: 70% AI layer weight, 30% ML layer weight in final scoring
**Rationale**: AI provides better semantic understanding, ML provides pattern recognition
**Trade-off**: Slightly slower due to AI API calls, but better prediction quality

### Synthetic vs Real Data
**Decision**: Use synthetic training data with realistic API patterns
**Alternative**: Wait for real data collection
**Trade-off**: Less accurate initially, but faster time-to-market and privacy-friendly

### OpenAI vs Local Models
**Decision**: Use OpenAI with robust fallback to heuristics
**Alternative**: Local LLM deployment (e.g., Ollama)
**Trade-off**: External dependency vs infrastructure complexity

### Performance vs Accuracy
**Decision**: Optimize for meeting performance SLAs (median <1s, p95 <3s)
**Alternative**: More complex ML models with higher accuracy
**Trade-off**: Simpler models, faster responses, slightly lower accuracy

### Memory vs Disk Storage
**Decision**: In-memory caching with Redis for performance
**Alternative**: Persistent disk-based caching
**Trade-off**: Faster access, but data lost on restart

## Future Work Priorities

### Immediate (Next Sprint)
1. **Real Training Data**: Replace synthetic data with actual API usage patterns
2. **Advanced Prompting**: Implement few-shot learning and chain-of-thought reasoning
3. **Better Parameter Generation**: Use OpenAPI schemas for realistic parameter values
4. **Comprehensive Testing**: Unit tests, integration tests, and load testing

### Short Term (1-2 Months)
1. **Active Learning**: Capture user feedback (accept/reject predictions) for model improvement
2. **Multi-Model Ensemble**: Combine multiple prediction approaches for better accuracy
3. **Personalization**: User-specific behavior modeling and preferences
4. **Advanced Safety**: More sophisticated destructive operation detection

### Medium Term (3-6 Months)
1. **Online Learning**: Real-time model updates without service downtime
2. **Cross-API Transfer Learning**: Apply patterns learned from one API to others
3. **Advanced Analytics**: Success@K metrics, confidence calibration, A/B testing
4. **Cost Optimization**: Dynamic model routing based on complexity and budget

### Long Term (6+ Months)
1. **Intent Classification**: Better natural language understanding for complex prompts
2. **Workflow Prediction**: Predict entire sequences, not just next actions
3. **Multi-Modal Inputs**: Handle API documentation, code context, etc.
4. **Federated Learning**: Learn from multiple deployments while preserving privacy

## Risk Mitigation Strategies

### Technical Risks
- **OpenAI Outages**: Robust fallback to heuristic predictions
- **Memory Leaks**: Resource monitoring and automatic restarts
- **Malformed Specs**: Comprehensive parsing with fallback specs
- **Performance Degradation**: Real-time metrics and alerting

### Business Risks
- **Accuracy Issues**: Clear confidence indicators and explanations
- **Safety Concerns**: Conservative defaults and comprehensive testing
- **Scalability Limits**: Horizontal scaling and caching strategies
- **Cost Overruns**: OpenAI usage monitoring and rate limiting

### Operational Risks
- **Deployment Issues**: Docker containerization and health checks
- **Monitoring Blind Spots**: Comprehensive metrics and logging
- **Data Privacy**: Minimal data retention and anonymization
- **Security Vulnerabilities**: Input validation and security scanning

## Success Metrics

### Functional Metrics
- **Prediction Accuracy**: Success@1, Success@3, Success@5 rates
- **Response Quality**: User acceptance rate of predictions
- **Coverage**: Percentage of requests successfully handled
- **Safety**: Zero harmful predictions in production

### Performance Metrics
- **Response Time**: Median <1s, P95 <3s, P99 <5s
- **Throughput**: >100 requests/minute sustained
- **Cache Hit Rate**: >80% for OpenAPI specs
- **Resource Usage**: <80% of allocated CPU/memory

### Business Metrics
- **User Engagement**: Daily/weekly active users
- **API Coverage**: Number of different APIs successfully handled
- **User Satisfaction**: Net Promoter Score from developers
- **Cost Efficiency**: Prediction cost per API call

## Data Ethics Considerations

### Privacy
- Minimal data collection (only what's needed for predictions)
- No persistent storage of user API events
- Anonymized metrics and logging
- Clear data retention policies

### Fairness
- Equal service quality across different API types
- No bias toward popular APIs or large customers
- Transparent explanation of prediction reasoning
- Accessible to developers with varying experience levels

### Transparency
- Open source approach where possible
- Clear documentation of limitations
- Honest communication about prediction confidence
- Regular bias and fairness audits
