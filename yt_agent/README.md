# YouTube Shorts Analysis API 🎬

A Flask-based REST API for analyzing YouTube Shorts trends and generating comprehensive reports. This API provides asynchronous job processing, PDF report generation, and comprehensive trend analysis capabilities.

## Features ✨

- **Asynchronous Analysis**: Submit analysis jobs and check status without blocking
- **Synchronous Analysis**: Direct analysis for quick requests
- **PDF Reports**: Automatically generated downloadable trend reports
- **Job Management**: Full CRUD operations for analysis jobs
- **Health Monitoring**: Built-in health checks and logging
- **CORS Support**: Ready for frontend integration
- **Flexible Queries**: Support for various analysis types and filters

## Installation 🚀

### Prerequisites

- Python 3.7+
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd youtube-shorts-analysis
   ```

2. **Install dependencies**
   ```bash
   pip install flask flask-cors
   # Add other dependencies as needed for ytwatch module
   ```

3. **Ensure ytwatch module is available**
   Make sure your `ytwatch.py` file (containing `YouTubeTrendsWorkflow`) is in the same directory.

4. **Run the API**
   ```bash
   python ytwatch-api.py
   ```

The API will start on `http://localhost:8080`

## API Endpoints 📡

### Core Analysis Endpoints

#### `POST /analyze`
Start an asynchronous analysis job.

**Request:**
```json
{
  "request": "Get 5 trends from US"
}
```

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "pending",
  "message": "Analysis job started",
  "request": "Get 5 trends from US",
  "created_at": "2025-01-01T12:00:00"
}
```

#### `POST /analyze/sync`
Run analysis synchronously (for testing/small jobs).

**Request:**
```json
{
  "request": "Get 3 comedy shorts from Japan"
}
```

**Response:**
```json
{
  "status": "completed",
  "request": "Get 3 comedy shorts from Japan",
  "result": { /* analysis results */ },
  "completed_at": "2025-01-01T12:05:00"
}
```

### Job Management Endpoints

#### `GET /jobs/<job_id>`
Get job status and basic information.

**Response:**
```json
{
  "id": "job-uuid",
  "request": "Get 5 trends from US",
  "status": "completed",
  "created_at": "2025-01-01T12:00:00",
  "completed_at": "2025-01-01T12:05:00",
  "summary": {
    "analysis_mode": "trending",
    "region_code": "US",
    "videos_analyzed": 5,
    "pdf_available": true,
    "from_cache": false
  }
}
```

#### `GET /jobs/<job_id>/results`
Get detailed analysis results.

#### `GET /jobs/<job_id>/download`
Download PDF report for completed job.

#### `GET /jobs`
List all jobs with optional filtering.

**Query Parameters:**
- `status`: Filter by status (pending, running, completed, failed)
- `limit`: Limit number of results

#### `DELETE /jobs/<job_id>`
Delete job data and associated files.

#### `POST /jobs/<job_id>/cancel`
Cancel a running job.

### Utility Endpoints

#### `GET /`
API documentation and status.

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "workflow_initialized": true,
  "active_jobs": 2,
  "total_jobs": 10,
  "timestamp": "2025-01-01T12:00:00"
}
```

## Supported Analysis Types 🎯

The API supports various types of analysis requests:

- **Trending Analysis**: `"Get 5 trends from US"`
- **Specific Video Analysis**: `"Analyze this video: youtube.com/shorts/ABC123"`
- **Category-based**: `"Show me 3 comedy shorts from Japan"`
- **Time-based**: `"Find trending music videos last 2 hours"`
- **Hashtag Analysis**: `"Get 7 #dance videos from Germany"`

## Usage Examples 💡

### Basic Usage with curl

```bash
# Start an analysis job
curl -X POST http://localhost:8080/analyze \
  -H "Content-Type: application/json" \
  -d '{"request": "Get 5 trends from US"}'

# Check job status
curl http://localhost:8080/jobs/<job_id>

# Download PDF report
curl -O http://localhost:8080/jobs/<job_id>/download

# List all jobs
curl http://localhost:8080/jobs

# Health check
curl http://localhost:8080/health
```

### Python Client Example

```python
import requests
import time

API_BASE = "http://localhost:8080"

# Start analysis
response = requests.post(f"{API_BASE}/analyze", json={
    "request": "Get 5 trending videos from US"
})
job_data = response.json()
job_id = job_data["job_id"]

# Poll for completion
while True:
    status_response = requests.get(f"{API_BASE}/jobs/{job_id}")
    status = status_response.json()
    
    if status["status"] == "completed":
        print("Analysis completed!")
        break
    elif status["status"] == "failed":
        print("Analysis failed:", status.get("error"))
        break
    
    print("Still running...")
    time.sleep(5)

# Download PDF report
pdf_response = requests.get(f"{API_BASE}/jobs/{job_id}/download")
with open("report.pdf", "wb") as f:
    f.write(pdf_response.content)
```

// Usage
analyzeYouTubeShorts('Get 5 trends from US')
    .then(result => console.log('Analysis completed:', result))
    .catch(error => console.error('Analysis failed:', error));
```

## Job Status Flow 🔄

```
PENDING → RUNNING → COMPLETED ✅
    ↓        ↓
   FAILED   FAILED ❌
```

- **PENDING**: Job created, waiting to start
- **RUNNING**: Analysis in progress
- **COMPLETED**: Analysis finished successfully
- **FAILED**: Analysis failed or was cancelled


```

### Logging

The API uses Python's logging module. Configure logging level:

```python
logging.basicConfig(level=logging.DEBUG)  # For detailed logs
```

## Error Handling 🚨

The API returns consistent error responses:

```json
{
  "error": "Description of what went wrong",
  "status": "failed"
}
```

Common HTTP status codes:
- `200`: Success
- `202`: Accepted (async job started)
- `400`: Bad Request (invalid input)
- `404`: Not Found (job/endpoint doesn't exist)
- `500`: Internal Server Error


## Troubleshooting 🔧

### Common Issues

1. **Workflow initialization failed**
   - Check that `ytwatch.py` and `YouTubeTrendsWorkflow` are available
   - Verify all dependencies are installed

2. **PDF download fails**
   - Ensure write permissions in the working directory
   - Check if the analysis actually completed successfully

3. **Jobs stuck in "running" status**
   - Check logs for errors in the background thread
   - Restart the API to clear stuck jobs

4. **CORS issues**
   - CORS is enabled by default for all origins
   - Modify CORS settings in the Flask app if needed
