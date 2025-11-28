#!/bin/bash
# Vritti Load Testing Script
#
# Usage:
#   ./scripts/run_load_test.sh [scenario] [duration]
#
# Scenarios:
#   baseline  - Standard load test (50 users, 5 min)
#   stress    - High throughput stress test (200 users, 10 min)
#   soak      - Extended soak test (100 users, 1 hour)
#   spike     - Spike test (ramp to 500 users)
#   reflection - LLM reflection load test (20 users)
#
# Examples:
#   ./scripts/run_load_test.sh baseline
#   ./scripts/run_load_test.sh stress 15m
#   ./scripts/run_load_test.sh soak 2h

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="${PROJECT_ROOT}/results/load_tests"
LOCUSTFILE="${PROJECT_ROOT}/tests/load/locustfile.py"
HOST="${VRITTI_HOST:-http://localhost:8000}"
API_KEY="${VRITTI_API_KEY:-test-api-key-for-load-testing}"

# Default values
SCENARIO="${1:-baseline}"
DURATION="${2:-5m}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create results directory
mkdir -p "${RESULTS_DIR}"

echo "=============================================="
echo "Vritti Load Test"
echo "=============================================="
echo "Scenario: ${SCENARIO}"
echo "Duration: ${DURATION}"
echo "Host: ${HOST}"
echo "Results: ${RESULTS_DIR}/${SCENARIO}_${TIMESTAMP}"
echo "=============================================="

# Check dependencies
if ! command -v locust &> /dev/null; then
    echo "Error: locust not found. Install with: pip install locust"
    exit 1
fi

if ! command -v bc &> /dev/null; then
    echo "Error: bc not found. Install with: apt-get install bc (Linux) or brew install bc (macOS)"
    exit 1
fi

# Check service is running
echo "Checking service health..."
if ! curl -sf "${HOST}/health/liveness" > /dev/null 2>&1; then
    echo "Error: Service not responding at ${HOST}"
    echo "Make sure Vritti is running before starting load test."
    exit 1
fi
echo "Service is healthy."

# Export API key for locust
export VRITTI_API_KEY="${API_KEY}"

# Run appropriate scenario
case "${SCENARIO}" in
    baseline)
        echo "Running baseline load test..."
        locust -f "${LOCUSTFILE}" \
            --host "${HOST}" \
            --headless \
            --users 50 \
            --spawn-rate 5 \
            --run-time "${DURATION}" \
            --csv "${RESULTS_DIR}/baseline_${TIMESTAMP}" \
            --html "${RESULTS_DIR}/baseline_${TIMESTAMP}.html" \
            --only-summary \
            VrittiUser
        ;;
    
    stress)
        echo "Running stress test..."
        locust -f "${LOCUSTFILE}" \
            --host "${HOST}" \
            --headless \
            --users 200 \
            --spawn-rate 20 \
            --run-time "${DURATION}" \
            --csv "${RESULTS_DIR}/stress_${TIMESTAMP}" \
            --html "${RESULTS_DIR}/stress_${TIMESTAMP}.html" \
            --only-summary \
            --tags stress \
            HighThroughputUser
        ;;
    
    soak)
        echo "Running soak test (long-duration stability test)..."
        locust -f "${LOCUSTFILE}" \
            --host "${HOST}" \
            --headless \
            --users 100 \
            --spawn-rate 10 \
            --run-time "${DURATION}" \
            --csv "${RESULTS_DIR}/soak_${TIMESTAMP}" \
            --html "${RESULTS_DIR}/soak_${TIMESTAMP}.html" \
            --only-summary \
            VrittiUser
        ;;
    
    spike)
        echo "Running spike test..."
        # Ramp up quickly, hold, ramp down
        locust -f "${LOCUSTFILE}" \
            --host "${HOST}" \
            --headless \
            --users 500 \
            --spawn-rate 100 \
            --run-time "${DURATION}" \
            --csv "${RESULTS_DIR}/spike_${TIMESTAMP}" \
            --html "${RESULTS_DIR}/spike_${TIMESTAMP}.html" \
            --only-summary \
            VrittiUser
        ;;
    
    reflection)
        echo "Running reflection load test..."
        locust -f "${LOCUSTFILE}" \
            --host "${HOST}" \
            --headless \
            --users 20 \
            --spawn-rate 2 \
            --run-time "${DURATION}" \
            --csv "${RESULTS_DIR}/reflection_${TIMESTAMP}" \
            --html "${RESULTS_DIR}/reflection_${TIMESTAMP}.html" \
            --only-summary \
            --tags reflection \
            ReflectionLoadUser
        ;;
    
    search-only)
        echo "Running search-only test..."
        locust -f "${LOCUSTFILE}" \
            --host "${HOST}" \
            --headless \
            --users 100 \
            --spawn-rate 10 \
            --run-time "${DURATION}" \
            --csv "${RESULTS_DIR}/search_${TIMESTAMP}" \
            --html "${RESULTS_DIR}/search_${TIMESTAMP}.html" \
            --only-summary \
            --tags search \
            VrittiUser
        ;;
    
    *)
        echo "Unknown scenario: ${SCENARIO}"
        echo "Available scenarios: baseline, stress, soak, spike, reflection, search-only"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "Load Test Complete"
echo "=============================================="
echo "Results saved to:"
echo "  - ${RESULTS_DIR}/${SCENARIO}_${TIMESTAMP}_stats.csv"
echo "  - ${RESULTS_DIR}/${SCENARIO}_${TIMESTAMP}.html"
echo ""

# Print summary if stats file exists
STATS_FILE="${RESULTS_DIR}/${SCENARIO}_${TIMESTAMP}_stats.csv"
if [ -f "${STATS_FILE}" ]; then
    echo "Summary:"
    echo "--------"
    # Parse CSV for key metrics
    tail -1 "${STATS_FILE}" | awk -F',' '{
        printf "Total Requests: %s\n", $3
        printf "Failure Rate: %.2f%%\n", ($4/$3)*100
        printf "Avg Response Time: %s ms\n", $6
        printf "P95 Response Time: %s ms\n", $14
        printf "P99 Response Time: %s ms\n", $15
        printf "RPS: %s\n", $10
    }' 2>/dev/null || echo "(Could not parse stats)"
fi

# SLO Thresholds
SLO_P99_MS=50
SLO_ERROR_RATE_PERCENT=1.0

# Check SLO compliance
echo ""
echo "SLO Compliance Check:"
echo "---------------------"

if [ -f "${STATS_FILE}" ]; then
    # Use python to parse CSV by header name for robustness
    # Locust CSV headers: "Type","Name","Request Count","Failure Count","Median Response Time","Average Response Time","Minimum Response Time","Maximum Response Time","Average Content Size","Requests/s","Failures/s","50%","66%","75%","80%","90%","95%","98%","99%","99.9%","99.99%","100%"
    
    P99=$(python3 -c "import csv, sys; reader = csv.DictReader(sys.stdin); row = list(reader)[-1]; print(row['99%'])" < "${STATS_FILE}" 2>/dev/null || echo "0")
    FAIL_RATE=$(python3 -c "import csv, sys; reader = csv.DictReader(sys.stdin); row = list(reader)[-1]; reqs = float(row['Request Count']); fails = float(row['Failure Count']); print((fails/reqs)*100 if reqs > 0 else 0)" < "${STATS_FILE}" 2>/dev/null || echo "0")
    
    # P99 < SLO (use bash arithmetic for integer comparison)
    P99_NUM=$(printf "%.0f" "${P99}" 2>/dev/null || echo "999999")
    if (( P99_NUM < SLO_P99_MS )); then
        echo "[PASS] P99 Latency: ${P99}ms < ${SLO_P99_MS}ms SLO"
    else
        echo "[FAIL] P99 Latency: ${P99}ms >= ${SLO_P99_MS}ms SLO"
    fi

    # Error rate < SLO (requires bc for float comparison, fail explicitly if bc unavailable)
    if ! command -v bc &> /dev/null; then
        echo "[WARN] bc not available - skipping error rate SLO check"
    else
        if (( $(echo "${FAIL_RATE} < ${SLO_ERROR_RATE_PERCENT}" | bc -l) )); then
            echo "[PASS] Error Rate: ${FAIL_RATE}% < ${SLO_ERROR_RATE_PERCENT}% SLO"
        else
            echo "[FAIL] Error Rate: ${FAIL_RATE}% >= ${SLO_ERROR_RATE_PERCENT}% SLO"
        fi
    fi
fi

echo ""
echo "For detailed analysis, open the HTML report:"
echo "  open ${RESULTS_DIR}/${SCENARIO}_${TIMESTAMP}.html"
