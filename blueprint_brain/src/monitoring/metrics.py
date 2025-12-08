from prometheus_client import Counter, Histogram, Gauge

# 1. Throughput Metrics
HTTP_REQUESTS_TOTAL = Counter(
    "blueprint_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)

# 2. Latency Metrics
# Buckets optimized for AI workloads (1s to 120s)
INFERENCE_DURATION = Histogram(
    "blueprint_inference_duration_seconds",
    "Time spent in AI inference",
    ["model_type"] # 'vision', 'ocr', 'fusion'
)

# 3. Business Metrics
JOBS_PROCESSED = Counter(
    "blueprint_jobs_processed_total",
    "Total jobs processed",
    ["status"] # 'success', 'failed'
)

ROOMS_DETECTED = Counter(
    "blueprint_rooms_detected_total",
    "Total number of rooms detected across all jobs"
)

# 4. Infrastructure Metrics
GPU_MEMORY_USAGE = Gauge(
    "blueprint_gpu_memory_usage_mb",
    "Current GPU memory usage"
)