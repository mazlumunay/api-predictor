#!/bin/bash

echo "🚀 Deploying Optimized API Predictor"
echo "===================================="

# Step 1: Backup current files (if they exist)
echo "📦 Backing up current implementation..."
if [ -d "backup" ]; then
  rm -rf backup
fi
mkdir -p backup

# Backup key files if they exist
for file in app/main.py app/utils/cache.py app/ai_layer/candidate_generator.py app/ml_layer/ranker.py requirements.txt; do
  if [ -f "$file" ]; then
    cp "$file" "backup/$(basename $file).bak"
    echo "  ✅ Backed up $file"
  fi
done

# Step 2: Update requirements.txt with performance optimizations
echo -e "\n📋 Updating requirements.txt..."
cat > requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
openai==1.51.0
httpx==0.25.2
redis==5.0.1
requests==2.31.0
pyyaml==6.0.1
lightgbm==4.1.0
scikit-learn==1.3.2
pandas==2.1.4
numpy==1.24.3
aiofiles==23.2.1
python-multipart==0.0.6

# Performance optimizations
orjson==3.9.10
asyncio-throttle==1.0.2
EOF

echo "✅ Requirements updated with performance packages"

# Step 3: Create performance test script
echo -e "\n🧪 Creating performance test script..."
chmod +x test_performance.sh
echo "✅ Performance test script ready"

# Step 4: Stop current services
echo -e "\n🛑 Stopping current services..."
docker-compose down

# Step 5: Rebuild with optimizations
echo -e "\n🔨 Building optimized containers..."
docker-compose build --no-cache

# Step 6: Start optimized services
echo -e "\n🚀 Starting optimized services..."
docker-compose up -d

# Step 7: Wait for services to be ready
echo -e "\n⏳ Waiting for services to initialize..."
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
  if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Services are ready!"
    break
  fi
  
  echo -n "."
  sleep 2
  attempt=$((attempt + 1))
done

if [ $attempt -eq $max_attempts ]; then
  echo "❌ Services failed to start within 60 seconds"
  echo "Check logs with: docker-compose logs"
  exit 1
fi

# Step 8: Run initial health check
echo -e "\n🔍 Performing health check..."
health_response=$(curl -s http://localhost:8000/health)

if echo "$health_response" | jq -e '.status' > /dev/null 2>&1; then
  status=$(echo "$health_response" | jq -r '.status')
  echo "✅ Health check passed: $status"
else
  echo "❌ Health check failed"
  echo "Response: $health_response"
  exit 1
fi

# Step 9: Test basic functionality
echo -e "\n🧪 Testing basic functionality..."
test_response=$(curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "deploy-test",
    "events": [{"ts": "2025-07-14T14:00:00Z", "endpoint": "GET /pet/1", "params": {}}],
    "spec_url": "https://petstore.swagger.io/v2/swagger.json",
    "k": 3
  }')

if echo "$test_response" | jq -e '.predictions' > /dev/null 2>&1; then
  processing_time=$(echo "$test_response" | jq -r '.processing_time_ms')
  predictions_count=$(echo "$test_response" | jq '.predictions | length')
  echo "✅ Basic functionality test passed"
  echo "   Processing time: ${processing_time}ms"
  echo "   Predictions: $predictions_count"
else
  echo "❌ Basic functionality test failed"
  echo "Response: $test_response"
  exit 1
fi

# Step 10: Quick performance test
echo -e "\n⚡ Quick performance test..."
echo "Running 5 requests to check optimization impact..."

quick_times=()
for i in {1..5}; do
  echo -n "Request $i: "
  
  start_time=$(date +%s%N)
  response=$(curl -s -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{
      "user_id": "quick-test-'$i'",
      "events": [{"ts": "2025-07-14T14:00:00Z", "endpoint": "GET /pet/'$i'", "params": {}}],
      "spec_url": "https://petstore.swagger.io/v2/swagger.json",
      "k": 3
    }')
  end_time=$(date +%s%N)
  
  if echo "$response" | jq -e '.predictions' > /dev/null 2>&1; then
    total_time=$(( (end_time - start_time) / 1000000 ))
    processing_time=$(echo "$response" | jq -r '.processing_time_ms')
    quick_times+=($total_time)
    echo "${processing_time}ms (total: ${total_time}ms) ✅"
  else
    echo "FAILED ❌"
  fi
  
  sleep 0.2
done

if [ ${#quick_times[@]} -gt 0 ]; then
  # Calculate median of quick test
  IFS=$'\n' sorted_quick=($(sort -n <<<"${quick_times[*]}"))
  quick_median=${sorted_quick[${#sorted_quick[@]}/2]}
  
  echo -e "\n📊 Quick Test Results:"
  echo "   Median response time: ${quick_median}ms"
  
  if [ $quick_median -lt 1000 ]; then
    echo "   🎉 Performance looks good! (under 1000ms target)"
  elif [ $quick_median -lt 2000 ]; then
    echo "   ⚠️  Performance needs improvement (target: <1000ms)"
  else
    echo "   ❌ Performance issues detected (target: <1000ms)"
  fi
fi

# Step 11: Display optimization summary
echo -e "\n📈 Optimization Summary:"
echo "✅ Aggressive caching implemented (2-hour TTL)"
echo "✅ OpenAI call optimization (reduced context, faster timeouts)"
echo "✅ Background processing for parameter enhancement"
echo "✅ FastAPI performance improvements (ORJson, compression)"
echo "✅ ML layer caching and feature optimization"
echo "✅ Prediction result caching"

# Step 12: Next steps
echo -e "\n🎯 Next Steps:"
echo "1. Run full performance test:"
echo "   ./test_performance.sh"
echo ""
echo "2. Check detailed metrics:"
echo "   curl http://localhost:8000/metrics | jq"
echo ""
echo "3. Monitor logs for issues:"
echo "   docker-compose logs -f api-predictor"
echo ""
echo "4. Create demo and assumptions files:"
echo "   - Create demo.sh with curl examples"
echo "   - Create Assumptions.md with design decisions"

# Step 13: Resource usage check
echo -e "\n💻 Resource Usage:"
docker stats --no-stream api-predictor-api-predictor-1 api-predictor-redis-1 2>/dev/null || docker stats --no-stream

echo -e "\n🎉 Optimized deployment completed successfully!"
echo ""
echo "📊 Performance targets:"
echo "   Target: Median <1000ms, P95 <3000ms"
echo "   Quick test median: ${quick_median:-"N/A"}ms"
echo ""
echo "🚀 Your optimized API Predictor is ready for OpenSesame submission!"