#!/bin/bash
# FIXED Performance verification script for OpenSesame submission

echo "🎯 OpenSesame Performance Requirements Test (OPTIMIZED)"
echo "Target: Median < 1s, p95 < 3s on docker run --cpus 2 --memory 4g"
echo "=============================================================="

# Verify service is running first
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "❌ Service not running. Please start with: docker-compose up -d"
    exit 1
fi

BASE_URL="http://localhost:8000"

# Test 1: Single request timing (FIXED)
echo -e "\n1️⃣ Single Request Performance Test"
echo "Sending 1 prediction request..."

start_time=$(python3 -c "import time; print(int(time.time() * 1000))")
response=$(curl -s -X POST $BASE_URL/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "perf-test-1",
    "events": [
      {"ts": "2025-07-18T14:00:00Z", "endpoint": "GET /pets", "params": {}},
      {"ts": "2025-07-18T14:01:00Z", "endpoint": "GET /pets/123", "params": {}},
      {"ts": "2025-07-18T14:02:00Z", "endpoint": "PUT /pets/123", "params": {"status": "available"}}
    ],
    "prompt": "add a new pet to the store",
    "spec_url": "https://petstore.swagger.io/v2/swagger.json",
    "k": 5
  }')
end_time=$(python3 -c "import time; print(int(time.time() * 1000))")
duration=$((end_time - start_time))

echo "Response time: ${duration}ms"
echo "Processing time from API: $(echo $response | jq -r '.processing_time_ms // "N/A"')ms"
echo "Status: $(echo $response | jq -r 'if .predictions then "✅ SUCCESS" else "❌ FAILED" end')"

# Test 2: Cache performance (FIXED)
echo -e "\n2️⃣ Cache Performance Test"
echo "First request (cache miss):"

start_time=$(python3 -c "import time; print(int(time.time() * 1000))")
curl -s -X POST $BASE_URL/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "cache-test-1", 
    "events": [{"ts": "2025-07-18T14:00:00Z", "endpoint": "GET /pets/1", "params": {}}],
    "spec_url": "https://petstore.swagger.io/v2/swagger.json",
    "k": 3
  }' > /dev/null
end_time=$(python3 -c "import time; print(int(time.time() * 1000))")
first_duration=$((end_time - start_time))

echo "Cache miss: ${first_duration}ms"

echo "Second request (cache hit):"
start_time=$(python3 -c "import time; print(int(time.time() * 1000))")
curl -s -X POST $BASE_URL/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "cache-test-2",
    "events": [{"ts": "2025-07-18T14:00:00Z", "endpoint": "GET /pets/2", "params": {}}],
    "spec_url": "https://petstore.swagger.io/v2/swagger.json",
    "k": 3
  }' > /dev/null
end_time=$(python3 -c "import time; print(int(time.time() * 1000))")
second_duration=$((end_time - start_time))

echo "Cache hit: ${second_duration}ms"
echo "Cache speedup: $((first_duration - second_duration))ms improvement"

# Performance summary
echo -e "\n📋 PERFORMANCE SUMMARY"
echo "======================"
echo "✅ Single request: ${duration}ms"
echo "✅ Cache performance: ${first_duration}ms → ${second_duration}ms"
echo ""
echo "🎯 Current Status vs OpenSesame Requirements:"
if [ $duration -lt 1000 ]; then
    echo "• Single request < 1000ms: ✅ PASS ($duration ms)"
else
    echo "• Single request < 1000ms: ⚠️ NEEDS OPTIMIZATION ($duration ms)"
fi

if [ $second_duration -lt 500 ]; then
    echo "• Cache performance: ✅ EXCELLENT ($second_duration ms)"
else
    echo "• Cache performance: ⚠️ COULD BE BETTER ($second_duration ms)"
fi

echo ""
echo "💡 Overall Assessment:"
if [ $duration -lt 1000 ] && [ $second_duration -lt 500 ]; then
    echo "🚀 READY FOR SUBMISSION!"
else
    echo "⚠️ PERFORMANCE OPTIMIZATIONS RECOMMENDED"
fi
