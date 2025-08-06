from prometheus_client import Counter, Histogram

# Counter for number of prediction requests
prediction_counter = Counter(
    "prediction_requests_total",
    "Total number of prediction requests"
)

# Histogram to track prediction latency
prediction_latency = Histogram(
    "prediction_request_duration_seconds",
    "Duration (seconds) of prediction requests"
)
