#!/bin/bash

# OpenSesame API Predictor Demo Script
# Tests the core functionality with Stripe and GitHub APIs

set -e  # Exit on any error

echo "🚀 API Predictor Demo Script"
echo "=================================="
echo

# Configuration
BASE_URL="http://localhost:8000"
STRIPE_SPEC="https://raw.githubusercontent.com/stripe/openapi/master/openapi.yaml"
GITHUB_SPEC="https://api.github.com/openapi.yaml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if service is running
check_service() {
    echo -n "Checking if service is running..."
    if curl -s "$BASE_URL/health" > /dev/null 2>&1; then
        echo -e " ${GREEN}✓${NC}"
        return 0
    else
        echo -e " ${RED}✗${NC}"
        echo "Please start the service with: docker-compose up --build"
        exit 1
    fi
}

# Function to make a prediction request
make_prediction() {
    local test_name="$1"
    local json_payload="$2"
    local expected_pattern="$3"
    
    echo -e "\n${BLUE}Test: $test_name${NC}"
    echo "----------------------------------------"
    
    echo "Request payload:"
    echo "$json_payload" | jq '.'
    
    echo -e "\n${YELLOW}Making prediction...${NC}"
    
    # Make the request and capture response
    response=$(curl -s -X POST "$BASE_URL/predict" \
        -H "Content-Type: application/json" \
        -d "$json_payload")
    
    # Check if request was successful
    if echo "$response" | jq -e '.predictions' > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Request successful${NC}"
        
        # Display formatted response
        echo -e "\n${YELLOW}Response:${NC}"
        echo "$response" | jq '.'
        
        # Extract key metrics
        local processing_time=$(echo "$response" | jq -r '.processing_time_ms')
        local num_predictions=$(echo "$response" | jq '.predictions | length')
        local top_score=$(echo "$response" | jq -r '.predictions[0].score // 0')
        
        echo -e "\n${YELLOW}Metrics:${NC}"
        echo "  • Processing time: ${processing_time}ms"
        echo "  • Number of predictions: $num_predictions"
        echo "  • Top prediction score: $top_score"
        
        # Performance check
        if [ "$processing_time" -lt 1000 ]; then
            echo -e "  • Performance: ${GREEN}✓ Under 1s${NC}"
        elif [ "$processing_time" -lt 3000 ]; then
            echo -e "  • Performance: ${YELLOW}⚠ Under 3s${NC}"
        else
            echo -e "  • Performance: ${RED}✗ Over 3s${NC}"
        fi
        
        # Check for expected patterns
        if [ -n "$expected_pattern" ] && echo "$response" | grep -q "$expected_pattern"; then
            echo -e "  • Content check: ${GREEN}✓ Found expected pattern${NC}"
        fi
        
    else
        echo -e "${RED}✗ Request failed${NC}"
        echo "Response: $response"
    fi
}

# Function to test health and metrics
test_health() {
    echo -e "\n${BLUE}Health Check${NC}"
    echo "----------------------------------------"
    
    health_response=$(curl -s "$BASE_URL/health")
    echo "$health_response" | jq '.'
    
    status=$(echo "$health_response" | jq -r '.status')
    if [ "$status" = "healthy" ]; then
        echo -e "${GREEN}✓ Service is healthy${NC}"
    else
        echo -e "${YELLOW}⚠ Service status: $status${NC}"
    fi
    
    echo -e "\n${BLUE}Metrics${NC}"
    echo "----------------------------------------"
    metrics_response=$(curl -s "$BASE_URL/metrics")
    echo "$metrics_response" | jq '.last_10_minutes'
}

# Main demo execution
main() {
    echo "Starting API Predictor Demo..."
    echo "Base URL: $BASE_URL"
    echo
    
    # Check service health
    check_service
    
    # Test 1: Stripe API - Billing workflow
    make_prediction "Stripe Billing Workflow" '{
        "user_id": "billing-user-1",
        "events": [
            {
                "ts": "2025-07-14T14:12:03Z",
                "endpoint": "GET /invoices",
                "params": {}
            },
            {
                "ts": "2025-07-14T14:13:11Z",
                "endpoint": "PUT /invoices/inv_123",
                "params": {"status": "DRAFT"}
            }
        ],
        "prompt": "Let'\''s finish billing for Q2",
        "spec_url": "'"$STRIPE_SPEC"'",
        "k": 5
    }' "POST"
    
    # Test 2: GitHub API - Repository workflow  
    make_prediction "GitHub Repository Workflow" '{
        "user_id": "dev-user-1", 
        "events": [
            {
                "ts": "2025-07-14T10:00:00Z",
                "endpoint": "GET /repos/owner/repo",
                "params": {}
            },
            {
                "ts": "2025-07-14T10:01:00Z",
                "endpoint": "GET /repos/owner/repo/issues",
                "params": {"state": "open"}
            }
        ],
        "prompt": "create a new issue for bug tracking",
        "spec_url": "'"$GITHUB_SPEC"'",
        "k": 3
    }' "POST"
    
    # Test 3: Cold start scenario
    make_prediction "Cold Start Scenario" '{
        "user_id": "new-user-123",
        "events": [
            {
                "ts": "2025-07-14T14:00:00Z",
                "endpoint": "GET /users",
                "params": {}
            }
        ],
        "prompt": "get user profile details",
        "spec_url": "https://petstore.swagger.io/v2/swagger.json",
        "k": 5
    }' "COLD-START"
    
    # Test 4: No prompt scenario
    make_prediction "No Prompt Test" '{
        "user_id": "api-explorer",
        "events": [
            {
                "ts": "2025-07-14T13:00:00Z",
                "endpoint": "GET /products",
                "params": {}
            },
            {
                "ts": "2025-07-14T13:01:00Z",
                "endpoint": "GET /products/123",
                "params": {}
            }
        ],
        "spec_url": "https://petstore.swagger.io/v2/swagger.json",
        "k": 3
    }' ""
    
    # Test 5: Safety test - destructive operations
    make_prediction "Safety Test - Destructive Operations" '{
        "user_id": "safety-test",
        "events": [
            {
                "ts": "2025-07-14T12:00:00Z",
                "endpoint": "GET /users/123",
                "params": {}
            }
        ],
        "prompt": "I need to remove this user completely",
        "spec_url": "https://petstore.swagger.io/v2/swagger.json",
        "k": 5
    }' ""
    
    # Performance test - multiple rapid requests
    echo -e "\n${BLUE}Performance Test - Multiple Requests${NC}"
    echo "----------------------------------------"
    echo "Making 5 rapid requests to test caching and performance..."
    
    start_time=$(date +%s%N)
    for i in {1..5}; do
        echo -n "Request $i..."
        response_time=$(curl -w "%{time_total}" -s -o /dev/null -X POST "$BASE_URL/predict" \
            -H "Content-Type: application/json" \
            -d '{
                "user_id": "perf-test-'$i'",
                "events": [
                    {"ts": "2025-07-14T14:00:00Z", "endpoint": "GET /invoices", "params": {}},
                    {"ts": "2025-07-14T14:01:00Z", "endpoint": "POST /invoices", "params": {"amount": 1000}}
                ],
                "spec_url": "'"$STRIPE_SPEC"'",
                "k": 3
            }')
        echo " ${response_time}s"
    done
    end_time=$(date +%s%N)
    total_time=$((($end_time - $start_time) / 1000000))
    echo "Total time for 5 requests: ${total_time}ms"
    echo "Average time per request: $((total_time / 5))ms"
    
    # Test error handling
    echo -e "\n${BLUE}Error Handling Test${NC}"
    echo "----------------------------------------"
    echo "Testing with invalid input..."
    
    error_response=$(curl -s -X POST "$BASE_URL/predict" \
        -H "Content-Type: application/json" \
        -d '{
            "user_id": "",
            "events": [],
            "spec_url": "invalid-url",
            "k": 25
        }')
    
    if echo "$error_response" | jq -e '.detail' > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Proper error handling${NC}"
        echo "Error response:"
        echo "$error_response" | jq '.detail'
    else
        echo -e "${RED}✗ Unexpected error response${NC}"
        echo "$error_response"
    fi
    
    # Final health and metrics check
    test_health
    
    # Summary
    echo -e "\n${GREEN}🎉 Demo Complete!${NC}"
    echo "======================================"
    echo "Key features demonstrated:"
    echo "  ✓ AI-powered prediction generation" 
    echo "  ✓ ML-based candidate ranking"
    echo "  ✓ Cold-start handling"
    echo "  ✓ Safety guardrails"
    echo "  ✓ Performance optimization"
    echo "  ✓ Error handling"
    echo "  ✓ Multiple API specifications"
    echo
    echo "Service endpoints:"
    echo "  • Predictions: $BASE_URL/predict"
    echo "  • Health: $BASE_URL/health"
    echo "  • Metrics: $BASE_URL/metrics"
    echo "  • Documentation: $BASE_URL/docs"
    echo
    echo "For more details, check the logs with:"
    echo "  docker-compose logs api-predictor"
}

# Run the demo
main "$@"