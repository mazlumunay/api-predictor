#!/bin/bash

echo "🚀 Performance Testing Script - OpenSesame Requirements"
echo "Target: Median <1000ms, P95 <3000ms"
echo "======================================"

BASE_URL="http://localhost:8000"
SPEC_URL="https://petstore.swagger.io/v2/swagger.json"

# Check if service is running
echo "🔍 Checking service health..."
if ! curl -s "$BASE_URL/health" > /dev/null 2>&1; then
    echo "❌ Service not running. Please start with: docker-compose up -d"
    exit 1
fi

echo "✅ Service is running"

# Warm up cache
echo -e "\n🔥 Warming up cache..."
curl -s -X POST $BASE_URL/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "warmup",
    "events": [{"ts": "2025-07-14T14:00:00Z", "endpoint": "GET /pet/1", "params": {}}],
    "spec_url": "'$SPEC_URL'",
    "k": 3
  }' > /dev/null

echo "✅ Cache warmed up"

# Performance test with optimizations
times=()
cached_times=()

echo -e "\n⚡ Running performance tests..."
echo "Testing 20 requests (mix of cache miss and cache hit scenarios)"

for i in {1..20}; do
  echo -n "Request $i: "
  
  # Vary the user_id to test different scenarios
  if [ $((i % 4)) -eq 0 ]; then
    # Every 4th request uses same user (cache hit scenario)
    user_id="cache-user"
    scenario="cached"
  else
    # Other requests use unique user (cache miss scenario)
    user_id="perf-test-$i"
    scenario="fresh"
  fi
  
  start_time=$(date +%s%N)
  response=$(curl -s -X POST $BASE_URL/predict \
    -H "Content-Type: application/json" \
    -d '{
      "user_id": "'$user_id'",
      "events": [
        {"ts": "2025-07-14T14:00:00Z", "endpoint": "GET /pet/findByStatus", "params": {"status": "available"}},
        {"ts": "2025-07-14T14:01:00Z", "endpoint": "GET /pet/'$((i+100))'", "params": {}}
      ],
      "spec_url": "'$SPEC_URL'",
      "k": 3
    }')
  end_time=$(date +%s%N)
  
  # Check if successful
  if echo "$response" | jq -e '.predictions' > /dev/null 2>&1; then
    processing_time=$(echo "$response" | jq -r '.processing_time_ms')
    total_time=$(( (end_time - start_time) / 1000000 ))
    is_cached=$(echo "$response" | jq -r '.cached // false')
    
    times+=($total_time)
    
    if [ "$is_cached" = "true" ] || [ "$scenario" = "cached" ]; then
      cached_times+=($total_time)
      echo "${processing_time}ms (total: ${total_time}ms) 🟢 cached"
    else
      echo "${processing_time}ms (total: ${total_time}ms) 🔵 fresh"
    fi
  else
    echo "FAILED ❌"
  fi
  
  # Small delay to avoid overwhelming
  sleep 0.05
done

# Calculate statistics
if [ ${#times[@]} -gt 0 ]; then
  echo -e "\n📊 Performance Results:"
  
  # Overall statistics
  IFS=$'\n' sorted=($(sort -n <<<"${times[*]}"))
  n=${#sorted[@]}
  median=${sorted[$((n/2))]}
  p95=${sorted[$((n*95/100))]}
  min=${sorted[0]}
  max=${sorted[$((n-1))]}
  
  # Calculate average
  sum=0
  for time in "${times[@]}"; do
    sum=$((sum + time))
  done
  avg=$((sum / n))
  
  echo "  📈 Overall Performance:"
  echo "    Min:     ${min}ms"
  echo "    Median:  ${median}ms (target: <1000ms) $([ $median -lt 1000 ] && echo "✅" || echo "❌")"
  echo "    Average: ${avg}ms"
  echo "    P95:     ${p95}ms (target: <3000ms) $([ $p95 -lt 3000 ] && echo "✅" || echo "❌")"
  echo "    Max:     ${max}ms"
  echo "    Success: ${#times[@]}/20 requests"
  
  # Cached request statistics
  if [ ${#cached_times[@]} -gt 0 ]; then
    IFS=$'\n' cached_sorted=($(sort -n <<<"${cached_times[*]}"))
    cached_n=${#cached_sorted[@]}
    cached_median=${cached_sorted[$((cached_n/2))]}
    cached_max=${cached_sorted[$((cached_n-1))]}
    
    echo -e "\n  🟢 Cached Request Performance:"
    echo "    Median:  ${cached_median}ms"
    echo "    Max:     ${cached_max}ms" 
    echo "    Count:   ${cached_n} requests"
  fi
  
  # Performance assessment
  echo -e "\n🎯 Assessment:"
  
  if [ $median -lt 1000 ] && [ $p95 -lt 3000 ]; then
    echo "🎉 PERFORMANCE REQUIREMENTS MET!"
    echo "✅ Median: ${median}ms < 1000ms target"
    echo "✅ P95: ${p95}ms < 3000ms target"
    
    if [ ${#cached_times[@]} -gt 0 ] && [ $cached_median -lt 500 ]; then
      echo "🚀 Excellent cache performance: ${cached_median}ms median for cached requests"
    fi
    
  elif [ $median -lt 1000 ]; then
    echo "⚠️  Median target met, but P95 needs improvement"
    echo "✅ Median: ${median}ms < 1000ms target"
    echo "❌ P95: ${p95}ms ≥ 3000ms target"
    echo "💡 Suggestion: Check for OpenAI API latency spikes"
    
  elif [ $p95 -lt 3000 ]; then
    echo "⚠️  P95 target met, but median needs improvement"
    echo "❌ Median: ${median}ms ≥ 1000ms target"
    echo "✅ P95: ${p95}ms < 3000ms target"
    echo "💡 Suggestion: Improve cache hit rate or optimize first-time requests"
    
  else:
    echo "❌ Performance targets not met"
    echo "❌ Median: ${median}ms ≥ 1000ms target"
    echo "❌ P95: ${p95}ms ≥ 3000ms target"
    echo "💡 Optimization suggestions:"
    echo "   - Check OpenAI API key and quota"
    echo "   - Verify Redis is running and connected"
    echo "   - Monitor OpenAPI spec parsing times"
    echo "   - Consider reducing timeout values"
  fi
  
  # Improvement suggestions
  echo -e "\n💡 Performance Optimization Tips:"
  
  if [ $median -gt 800 ]; then
    echo "  🔧 Consider enabling more aggressive caching"
    echo "  🔧 Reduce OpenAI context size for faster responses"
  fi
  
  if [ $p95 -gt 2000 ]; then
    echo "  🔧 Check for OpenAI API timeout issues"
    echo "  🔧 Implement request batching for similar queries"
  fi
  
  if [ ${#cached_times[@]} -eq 0 ]; then
    echo "  🔧 No cached requests detected - check cache implementation"
  fi
  
else
  echo "❌ No successful requests - check service health"
  exit 1
fi

# Additional diagnostics
echo -e "\n🔍 Diagnostic Information:"

# Check cache metrics
echo "📊 Cache Performance:"
cache_metrics=$(curl -s $BASE_URL/metrics | jq '.last_10_minutes')
if [ "$cache_metrics" != "null" ]; then
  cache_hit_rate=$(echo "$cache_metrics" | jq -r '.cache_hit_rate // 0')
  cache_percentage=$(echo "$cache_hit_rate * 100" | bc -l 2>/dev/null || echo "0")
  echo "  Cache hit rate: ${cache_percentage}%"
else
  echo "  Cache metrics unavailable"
fi

# Check OpenAI status
echo "🤖 AI Layer Status:"
if grep -q "OpenAI client initialized successfully" <<< "$(docker-compose logs api-predictor 2>/dev/null)" 2>/dev/null; then
  echo "  ✅ OpenAI client initialized"
else
  echo "  ⚠️  OpenAI client may not be initialized"
fi

if grep -q "insufficient_quota\|Incorrect API key" <<< "$(docker-compose logs api-predictor 2>/dev/null)" 2>/dev/null; then
  echo "  ❌ OpenAI API key issues detected"
else
  echo "  ✅ No OpenAI API errors in recent logs"
fi

# Check ML model status
echo "🧠 ML Layer Status:"
if grep -q "ML model loaded successfully\|New model trained" <<< "$(docker-compose logs api-predictor 2>/dev/null)" 2>/dev/null; then
  echo "  ✅ ML model operational"
else
  echo "  ⚠️  ML model status unclear"
fi

echo -e "\n📋 Test Summary:"
echo "  Total requests: 20"
echo "  Successful: ${#times[@]}"
echo "  Failed: $((20 - ${#times[@]}))"
echo "  Cached responses: ${#cached_times[@]}"
echo "  Fresh responses: $((${#times[@]} - ${#cached_times[@]}))"

# Final recommendation
echo -e "\n🎯 Final Recommendation:"
if [ $median -lt 1000 ] && [ $p95 -lt 3000 ]; then
  echo "🎉 Ready for OpenSesame submission!"
  echo "   Your implementation meets all performance requirements."
else
  echo "🔧 Apply optimizations and retest before submission."
  echo "   Focus on the suggestions above to meet performance targets."
fi

echo -e "\n✅ Performance test completed!"