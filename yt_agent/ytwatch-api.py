from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import threading
import time
from datetime import datetime
import uuid
import logging
from typing import Dict, List, Any, Optional
import tempfile
import zipfile

# Import your workflow class
from ytwatch import YouTubeTrendsWorkflow

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# In-memory storage for job status and results
# In production, use Redis, database, or similar persistent storage
job_storage = {}
workflow_instance = None

class JobStatus:
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"

def initialize_workflow():
    """Initialize the workflow instance"""
    global workflow_instance
    try:
        workflow_instance = YouTubeTrendsWorkflow()
        logger.info("✅ YouTube Trends Workflow initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize workflow: {e}")
        return False

def run_analysis_async(job_id: str, user_request: str):
    """Run analysis in background thread"""
    global job_storage
    
    try:
        job_storage[job_id]["status"] = JobStatus.RUNNING
        job_storage[job_id]["started_at"] = datetime.now().isoformat()
        
        logger.info(f"🚀 Starting analysis job {job_id}: '{user_request}'")
        
        # Run the workflow
        result = workflow_instance.run_workflow(user_request)
        
        # Update job status
        if "error" in result:
            job_storage[job_id]["status"] = JobStatus.FAILED
            job_storage[job_id]["error"] = result["error"]
            logger.error(f"❌ Job {job_id} failed: {result['error']}")
        else:
            job_storage[job_id]["status"] = JobStatus.COMPLETED
            job_storage[job_id]["result"] = result
            logger.info(f"✅ Job {job_id} completed successfully")
        
        job_storage[job_id]["completed_at"] = datetime.now().isoformat()
        
    except Exception as e:
        job_storage[job_id]["status"] = JobStatus.FAILED
        job_storage[job_id]["error"] = str(e)
        job_storage[job_id]["completed_at"] = datetime.now().isoformat()
        logger.error(f"❌ Job {job_id} failed with exception: {e}")

# API Routes

@app.route("/", methods=["GET"])
def home():
    """API status and documentation"""
    return jsonify({
        "service": "YouTube Shorts Trend Analysis API",
        "version": "1.0.0",
        "status": "running" if workflow_instance else "initializing",
        "endpoints": {
            "POST /analyze": "Start new analysis job",
            "GET /jobs/<job_id>": "Get job status and results",
            "GET /jobs/<job_id>/download": "Download PDF report",
            "GET /jobs": "List all jobs",
            "DELETE /jobs/<job_id>": "Delete job data",
            "POST /analyze/sync": "Run analysis synchronously (not recommended for large jobs)"
        },
        "supported_requests": [
            "Get 5 trends from US",
            "Analyze this video: youtube.com/shorts/ABC123", 
            "Show me 3 comedy shorts from Japan",
            "Find trending music videos last 2 hours",
            "Get 7 #dance videos from Germany"
        ]
    })

@app.route("/analyze", methods=["POST"])
def analyze_async():
    """Start asynchronous analysis job"""
    try:
        data = request.get_json()
        
        if not data or "request" not in data:
            return jsonify({
                "error": "Missing 'request' field in JSON body",
                "example": {"request": "Get 5 trends from US"}
            }), 400
        
        user_request = data["request"].strip()
        if not user_request:
            return jsonify({"error": "Request cannot be empty"}), 400
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Initialize job data
        job_storage[job_id] = {
            "id": job_id,
            "request": user_request,
            "status": JobStatus.PENDING,
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "result": None,
            "error": None
        }
        
        # Start background thread
        thread = threading.Thread(target=run_analysis_async, args=(job_id, user_request))
        thread.daemon = True
        thread.start()
        
        logger.info(f"📝 Created analysis job {job_id}")
        
        return jsonify({
            "job_id": job_id,
            "status": JobStatus.PENDING,
            "message": "Analysis job started. Use GET /jobs/{job_id} to check status.",
            "request": user_request,
            "created_at": job_storage[job_id]["created_at"]
        }), 202
        
    except Exception as e:
        logger.error(f"❌ Error creating analysis job: {e}")
        return jsonify({"error": f"Failed to create analysis job: {str(e)}"}), 500

@app.route("/analyze/sync", methods=["POST"])
def analyze_sync():
    """Run analysis synchronously (for testing/small jobs)"""
    try:
        data = request.get_json()
        
        if not data or "request" not in data:
            return jsonify({
                "error": "Missing 'request' field in JSON body",
                "example": {"request": "Get 5 trends from US"}
            }), 400
        
        user_request = data["request"].strip()
        if not user_request:
            return jsonify({"error": "Request cannot be empty"}), 400
        
        logger.info(f"🚀 Running synchronous analysis: '{user_request}'")
        
        # Run workflow synchronously
        result = workflow_instance.run_workflow(user_request)
        
        if "error" in result:
            return jsonify({
                "status": "failed",
                "error": result["error"],
                "request": user_request
            }), 500
        
        return jsonify({
            "status": "completed",
            "request": user_request,
            "result": result,
            "completed_at": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Error in synchronous analysis: {e}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@app.route("/jobs/<job_id>", methods=["GET"])
def get_job_status(job_id: str):
    """Get job status and results"""
    if job_id not in job_storage:
        return jsonify({"error": "Job not found"}), 404
    
    job_data = job_storage[job_id].copy()
    
    # Add additional metadata if completed
    if job_data["status"] == JobStatus.COMPLETED and job_data.get("result"):
        result = job_data["result"]
        job_data["summary"] = {
            "analysis_mode": result.get("analysis_mode"),
            "region_code": result.get("region_code"),
            "videos_analyzed": len(result.get("video_analysis_results", [])),
            "pdf_available": bool(result.get("pdf_filename")),
            "from_cache": result.get("from_cache", False)
        }
    
    return jsonify(job_data)

@app.route("/jobs/<job_id>/download", methods=["GET"])
def download_pdf(job_id: str):
    """Download PDF report for completed job"""
    if job_id not in job_storage:
        return jsonify({"error": "Job not found"}), 404
    
    job_data = job_storage[job_id]
    
    if job_data["status"] != JobStatus.COMPLETED:
        return jsonify({
            "error": "Job not completed",
            "status": job_data["status"]
        }), 400
    
    if not job_data.get("result") or not job_data["result"].get("pdf_filename"):
        return jsonify({"error": "PDF report not available"}), 404
    
    pdf_filename = job_data["result"]["pdf_filename"]
    
    try:
        if os.path.exists(pdf_filename):
            return send_file(
                pdf_filename,
                as_attachment=True,
                download_name=pdf_filename,
                mimetype='application/pdf'
            )
        else:
            return jsonify({"error": "PDF file not found on disk"}), 404
    
    except Exception as e:
        logger.error(f"❌ Error serving PDF: {e}")
        return jsonify({"error": f"Failed to serve PDF: {str(e)}"}), 500

@app.route("/jobs/<job_id>/results", methods=["GET"])
def get_job_results(job_id: str):
    """Get detailed results for completed job"""
    if job_id not in job_storage:
        return jsonify({"error": "Job not found"}), 404
    
    job_data = job_storage[job_id]
    
    if job_data["status"] != JobStatus.COMPLETED:
        return jsonify({
            "error": "Job not completed",
            "status": job_data["status"]
        }), 400
    
    result = job_data.get("result")
    if not result:
        return jsonify({"error": "No results available"}), 404
    
    # Return formatted results
    formatted_results = {
        "job_id": job_id,
        "request": job_data["request"],
        "analysis_summary": {
            "analysis_mode": result.get("analysis_mode"),
            "region_code": result.get("region_code"),
            "search_term": result.get("search_term"),
            "max_videos": result.get("max_videos"),
            "videos_analyzed": len(result.get("video_analysis_results", [])),
            "from_cache": result.get("from_cache", False),
            "specific_url": result.get("specific_url")
        },
        "video_results": result.get("video_analysis_results", []),
        "final_summary": result.get("final_summary", {}),
        "messages": result.get("messages", []),
        "pdf_filename": result.get("pdf_filename"),
        "completed_at": job_data["completed_at"]
    }
    
    return jsonify(formatted_results)

@app.route("/jobs", methods=["GET"])
def list_jobs():
    """List all jobs with optional filtering"""
    status_filter = request.args.get("status")
    limit = request.args.get("limit", type=int)
    
    jobs_list = []
    for job_id, job_data in job_storage.items():
        if status_filter and job_data["status"] != status_filter:
            continue
        
        # Create summary for list view
        job_summary = {
            "job_id": job_id,
            "request": job_data["request"][:100] + "..." if len(job_data["request"]) > 100 else job_data["request"],
            "status": job_data["status"],
            "created_at": job_data["created_at"],
            "completed_at": job_data.get("completed_at")
        }
        
        if job_data["status"] == JobStatus.COMPLETED and job_data.get("result"):
            job_summary["videos_analyzed"] = len(job_data["result"].get("video_analysis_results", []))
            job_summary["analysis_mode"] = job_data["result"].get("analysis_mode")
        
        jobs_list.append(job_summary)
    
    # Sort by creation time (newest first)
    jobs_list.sort(key=lambda x: x["created_at"], reverse=True)
    
    # Apply limit if specified
    if limit:
        jobs_list = jobs_list[:limit]
    
    return jsonify({
        "jobs": jobs_list,
        "total_count": len(job_storage),
        "filtered_count": len(jobs_list)
    })

@app.route("/jobs/<job_id>", methods=["DELETE"])
def delete_job(job_id: str):
    """Delete job data and associated files"""
    if job_id not in job_storage:
        return jsonify({"error": "Job not found"}), 404
    
    job_data = job_storage[job_id]
    
    # Delete PDF file if it exists
    try:
        if job_data.get("result") and job_data["result"].get("pdf_filename"):
            pdf_filename = job_data["result"]["pdf_filename"]
            if os.path.exists(pdf_filename):
                os.remove(pdf_filename)
                logger.info(f"🗑️ Deleted PDF file: {pdf_filename}")
    except Exception as e:
        logger.warning(f"⚠️ Could not delete PDF file: {e}")
    
    # Remove from storage
    del job_storage[job_id]
    
    logger.info(f"🗑️ Deleted job {job_id}")
    
    return jsonify({
        "message": f"Job {job_id} deleted successfully",
        "deleted_at": datetime.now().isoformat()
    })

@app.route("/jobs/<job_id>/cancel", methods=["POST"])
def cancel_job(job_id: str):
    """Cancel running job (mark as failed)"""
    if job_id not in job_storage:
        return jsonify({"error": "Job not found"}), 404
    
    job_data = job_storage[job_id]
    
    if job_data["status"] in [JobStatus.COMPLETED, JobStatus.FAILED]:
        return jsonify({
            "error": "Cannot cancel job",
            "status": job_data["status"]
        }), 400
    
    # Mark as failed (cancellation)
    job_storage[job_id]["status"] = JobStatus.FAILED
    job_storage[job_id]["error"] = "Job cancelled by user"
    job_storage[job_id]["completed_at"] = datetime.now().isoformat()
    
    return jsonify({
        "message": f"Job {job_id} cancelled",
        "status": JobStatus.FAILED,
        "cancelled_at": job_storage[job_id]["completed_at"]
    })

# Health check endpoint
@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "workflow_initialized": workflow_instance is not None,
        "active_jobs": sum(1 for job in job_storage.values() if job["status"] == JobStatus.RUNNING),
        "total_jobs": len(job_storage),
        "timestamp": datetime.now().isoformat()
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    print("🎬 YouTube Shorts Analysis Flask API")
    print("="*50)
    
    # Initialize workflow
    if initialize_workflow():
        print("✅ Workflow initialized successfully")
    else:
        print("❌ Workflow initialization failed")
        print("⚠️ API will start but analysis endpoints may not work")
    
    print("\n🚀 Starting Flask API...")
    print("📡 Endpoints available:")
    print("   • POST   /analyze           - Start async analysis")
    print("   • POST   /analyze/sync      - Run sync analysis")
    print("   • GET    /jobs/<job_id>     - Get job status")
    print("   • GET    /jobs/<job_id>/download - Download PDF")
    print("   • GET    /jobs              - List all jobs")
    print("   • DELETE /jobs/<job_id>     - Delete job")
    print("   • GET    /health            - Health check")
    print("\n📖 Example requests:")
    print('   curl -X POST http://localhost:5000/analyze -H "Content-Type: application/json" -d \'{"request": "Get 5 trends from US"}\'')
    print("\n🌐 Starting server on http://localhost:5000")
    print("="*50)
    
    # Run Flask app
    app.run(
        host="0.0.0.0",  # Allow external connections
        port=8080,
        debug=True,      # Enable debug mode for development
        threaded=True    # Enable threading for concurrent requests
    )