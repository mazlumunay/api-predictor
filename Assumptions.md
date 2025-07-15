# Assumptions.md

## Clarifying Questions I Would Have Asked

### Business Context
1. **Target APIs**: Are we primarily focusing on REST APIs, or should we handle GraphQL/RPC APIs as well?
   - **Assumption Made**: Focused on REST APIs with standard HTTP methods based on OpenAPI specs
   
2. **User Types**: Are users typically developers, end-users, or both? This affects prediction complexity.
   - **Assumption Made**: Primarily developers using APIs programmatically, not end-users clicking buttons

3. **Usage Patterns**: What's the expected volume? (1 prediction/sec vs 1000/sec affects caching strategy)
   - **Assumption Made**: Moderate load (10-100 concurrent users) based on take-home challenge scope

4. **Accuracy vs Speed**: Would you prefer slightly slower but more accurate predictions, or faster with moderate accuracy?
   - **Assumption Made**: Prioritized meeting performance SLAs (median <1s, p95 <3s) while maintaining good accuracy

### Technical Constraints
1. **OpenAI Dependencies**: Is OpenAI API dependency acceptable, or do we need fully offline operation?
   - **Assumption Made**: External API dependency acceptable with robust fallback to heuristics

2. **Data Privacy**: Can we log user interactions for model improvement, or is this privacy-sensitive?
   - **Assumption Made**: Conservative approach - minimal logging, no persistent storage of user events

3. **Spec Diversity**: Should we handle broken/incomplete OpenAPI specs gracefully?
   - **Assumption Made**: Yes, implemented fallback spec and graceful error handling

4. **Real-time Learning**: Is online learning from user feedback a priority for v1?
   - **Assumption Made**: Not required for v1, but designed architecture to support future active learning

### Safety & Ethics
1. **Destructive Operations**: What constitutes "clearly implied" for DELETE operations?
   - **Assumption Made**: Conservative approach - heavy penalty for DELETE unless explicit destructive keywords in prompt

2. **Rate Limiting**: Should we implement per-user rate limiting to prevent abuse?
   - **Assumption Made**: Not implemented for take-home, but would add in production

3. **Audit Trail**: Do we need to log predictions for compliance/debugging?
   - **Assumption Made**: Basic logging only, no persistent audit trail for take-home scope

## Key Assumptions Made

### Data & Training
- **No Real Usage Data**: Assumed no access to real SaaS API usage patterns, so implemented synthetic data generation based on common workflows
- **Cold Start Priority**: Assumed many users will have minimal history, so prioritized robust cold-start handling with specialized predictions
- **OpenAPI Completeness**: Assumed specs may be incomplete/inconsistent, built fallback mechanisms and error tolerance
- **Temporal Patterns**: Assumed recent actions (last 3-5 events) are more predictive than full historical analysis

### AI Layer Design
- **LLM Selection**: Chose GPT-4o-mini for cost-effectiveness over GPT-4 for maximum accuracy
- **Prompt Engineering**: Assumed structured prompts with reduced context work better than complex few-shot learning for speed
- **Fallback Strategy**: Assumed heuristic fallback is essential when AI is unavailable or slow
- **Context Window**: Limited to recent events + sample endpoints to stay within token limits and improve response time

### ML Layer Approach
- **Heuristic Start**: Started with heuristic scoring as baseline before implementing ML model training
- **Feature Engineering**: Assumed sequence patterns, method transitions, and prompt keywords are most predictive features
- **Model Complexity**: Chose RandomForest over deep learning for interpretability, speed, and lower resource requirements
- **Training Data**: Generated synthetic data based on common API workflow patterns (CRUD operations, GET→POST flows)

### Performance & Scalability
- **Cache Strategy**: Assumed OpenAPI specs change infrequently (2-hour TTL), prediction results can be cached briefly (10 minutes)
- **Memory vs Speed**: Optimized for response time over memory usage within 4GB Docker limit
- **Concurrent Users**: Designed for moderate concurrency (10-100 concurrent users) typical for take-home scope
- **Resource Limits**: Assumed 2 CPU / 4GB RAM is sufficient for demonstration workload

### Safety & Security
- **Conservative Safety**: Assumed it's better to be overly cautious with destructive operations than risk damage
- **Input Validation**: Assumed malicious input is possible, implemented comprehensive validation and sanitization
- **Spec Validation**: Assumed some OpenAPI specs may be malicious or malformed, added URL validation and timeouts
- **Error Handling**: Prioritized graceful degradation over failing fast to maintain service availability

## Trade-offs Made

### AI vs ML Balance
**Decision**: 70% AI layer weight, 30% ML layer weight in final scoring
**Rationale**: AI provides better semantic understanding of user intent, ML provides pattern recognition from usage history
**Trade-off**: Slightly slower due to AI API calls, but significantly better prediction quality

### Synthetic vs Real Data
**Decision**: Use synthetic training data with realistic API patterns
**Alternative**: Wait for real data collection or use public API logs
**Trade-off**: Less accurate initially, but faster time-to-market and privacy-friendly approach

### OpenAI vs Local Models
**Decision**: Use OpenAI with robust fallback to heuristics
**Alternative**: Local LLM deployment (e.g., Ollama, Hugging Face)
**Trade-off**: External dependency vs infrastructure complexity and resource requirements

### Performance vs Accuracy
**Decision**: Optimize for meeting performance SLAs (median <1s, p95 <3s)
**Alternative**: More complex ML models with higher accuracy but slower inference
**Trade-off**: Simpler models and caching for speed, slightly lower accuracy ceiling

### Memory vs Disk Storage
**Decision**: In-memory caching with Redis for performance
**Alternative**: Persistent disk-based caching
**Trade-off**: Faster access but data lost on restart, higher memory usage

### Cold-Start Strategy
**Decision**: Specialized cold-start predictions instead of trying to use ML with insufficient data
**Alternative**: Use ML anyway with padded features or default predictions
**Trade-off**: More complex logic but much better user experience for new users

## Future Work Priorities

### Immediate (Next Sprint)
1. **Real Training Data**: Replace synthetic data with actual API usage patterns from beta users
2. **Advanced Prompting**: Implement few-shot learning and chain-of-thought reasoning for better AI predictions
3. **Better Parameter Generation**: Use OpenAPI schemas more intelligently for realistic parameter values
4. **Comprehensive Testing**: Unit tests, integration tests, and extensive load testing

### Short Term (1-2 Months)
1. **Active Learning**: Capture user feedback (accept/reject predictions) for model improvement
2. **Multi-Model Ensemble**: Combine multiple prediction approaches (different LLMs, ML models) for better accuracy
3. **Personalization**: User-specific behavior modeling and API usage preferences
4. **Advanced Safety**: More sophisticated destructive operation detection and context-aware safety

### Medium Term (3-6 Months)
1. **Online Learning**: Real-time model updates without service downtime using streaming ML
2. **Cross-API Transfer Learning**: Apply patterns learned from one API to others for better cold-start
3. **Advanced Analytics**: Success@K metrics, confidence calibration, A/B testing infrastructure
4. **Cost Optimization**: Dynamic model routing based on complexity, accuracy requirements, and budget

### Long Term (6+ Months)
1. **Intent Classification**: Better natural language understanding for complex, multi-step prompts
2. **Workflow Prediction**: Predict entire sequences of API calls, not just next actions
3. **Multi-Modal Inputs**: Handle API documentation, code context, UI screenshots for richer predictions
4. **Federated Learning**: Learn from multiple deployments while preserving user privacy

## Risk Mitigation Strategies

### Technical Risks
- **OpenAI Outages**: Robust fallback to heuristic predictions ensures service availability
- **Memory Leaks**: Resource monitoring, cache size limits, and automatic cleanup processes
- **Malformed Specs**: Comprehensive parsing with fallback specs and graceful error handling
- **Performance Degradation**: Real-time metrics, alerting, and automatic scaling mechanisms

### Business Risks
- **Accuracy Issues**: Clear confidence indicators, explanations, and conservative defaults
- **Safety Concerns**: Conservative defaults, comprehensive testing, and monitoring for harmful predictions
- **Scalability Limits**: Horizontal scaling design, caching strategies, and performance optimization
- **Cost Overruns**: OpenAI usage monitoring, rate limiting, and fallback to free heuristics

### Operational Risks
- **Deployment Issues**: Docker containerization, health checks, and rollback capabilities
- **Monitoring Blind Spots**: Comprehensive metrics, logging, and alerting infrastructure
- **Data Privacy**: Minimal data retention, anonymization, and clear privacy policies
- **Security Vulnerabilities**: Input validation, secure coding practices, and regular security audits

## Success Metrics

### Functional Metrics
- **Prediction Accuracy**: Success@1, Success@3, Success@5 rates measured against user selections
- **Response Quality**: User acceptance rate and feedback scores for predictions
- **Coverage**: Percentage of requests successfully handled without errors
- **Safety**: Zero harmful predictions reaching production users

### Performance Metrics
- **Response Time**: Median <1s, P95 <3s, P99 <5s consistently maintained
- **Throughput**: >100 requests/minute sustained without degradation
- **Cache Hit Rate**: >80% for OpenAPI specs, >50% for prediction results
- **Resource Usage**: <80% of allocated CPU/memory under normal load

### Business Metrics
- **User Engagement**: Daily/weekly active users and session length
- **API Coverage**: Number of different APIs and organizations successfully using the service
- **User Satisfaction**: Net Promoter Score and qualitative feedback from developers
- **Cost Efficiency**: Total cost per prediction and optimization opportunities

## Data Ethics Considerations

### Privacy
- **Minimal Data Collection**: Only collect what's necessary for predictions (events, prompts)
- **No Persistent Storage**: User API events not stored beyond session duration
- **Anonymized Metrics**: All performance metrics anonymized and aggregated
- **Clear Data Policies**: Transparent about what data is used and how

### Fairness
- **Equal Service Quality**: Same performance across different API types and user segments
- **No Bias Toward Popular APIs**: Ensure quality predictions for niche/specialized APIs
- **Transparent Reasoning**: Clear explanations of prediction logic and confidence
- **Accessible Design**: Works well for developers with varying experience levels

### Transparency
- **Open Source Approach**: Core algorithms and approaches documented publicly where possible
- **Clear Documentation**: Honest communication about limitations and confidence levels
- **Regular Audits**: Periodic review of predictions for bias, accuracy, and safety
- **Community Feedback**: Mechanisms for users to report issues and suggest improvements

### Accountability
- **Monitoring and Alerting**: Continuous monitoring for harmful or biased predictions
- **Rapid Response**: Clear escalation procedures for safety or ethical concerns
- **Regular Review**: Quarterly assessment of ethical implications and improvements
- **External Oversight**: Consider third-party audits for critical safety decisions
