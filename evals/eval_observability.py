import os
import sys
import json
import time
import requests
from datetime import datetime

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

API_URL = "http://localhost:8000/api/v1/master/query"
LOG_FILE = os.path.join(project_root, "logs", "query_metrics.jsonl")

def get_last_log_line():
    if not os.path.exists(LOG_FILE):
        return None
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if not lines:
            return None
        return json.loads(lines[-1].strip())

def run_eval():
    queries = [
        {"intent": "legal_query", "query": "What is the punishment for murder under BNS?"},
        {"intent": "case_law_search", "query": "Give me supreme court cases on cheque bounce."},
        {"intent": "draft_request", "query": "Draft a legal notice for eviction of a tenant."},
        {"intent": "rights_check", "query": "What are my rights if police arrest me without a warrant?"},
        {"intent": "general", "query": "Hello, who are you?"}
    ]

    results_data = []

    print("Starting Observability Baseline Evaluation...")
    print("-" * 80)

    for item in queries:
        q_text = item["query"]
        expected_intent = item["intent"]
        
        print(f"Testing intent: {expected_intent} | Query: '{q_text}'")
        
        try:
            resp = requests.post(API_URL, json={"query": q_text}, timeout=60)
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            print("Make sure the backend is running at http://localhost:8000")
            sys.exit(1)

        # Wait a small moment to ensure log is flushed
        time.sleep(0.1)

        last_log = get_last_log_line()
        metrics_captured = False
        fields_present = []
        latency = 0
        crag_score = 0
        
        if last_log:
            metrics_captured = True
            expected_fields = ["timestamp", "session_id", "intent", "latency_ms", "citation_status", "scope_status", "crag_score", "chunks_retrieved", "model_used"]
            fields_present = [f for f in expected_fields if f in last_log]
            
            latency = last_log.get("latency_ms", 0)
            crag_score = last_log.get("crag_score", 0)
            
            # Check if all fields are present
            if len(fields_present) == len(expected_fields):
                print(f"  [OK] Metrics captured successfully with all fields. Latency: {latency}ms")
            else:
                missing = set(expected_fields) - set(fields_present)
                print(f"  [WARN] Metrics captured but missing fields: {missing}")
        else:
            print(f"  [FAIL] No metrics log found for query.")

        results_data.append({
            "intent": expected_intent,
            "metrics_captured": metrics_captured,
            "fields_present": fields_present,
            "latency_ms": latency,
            "crag_score": crag_score
        })

    # Summary calculations
    latencies = [r["latency_ms"] for r in results_data if r["latency_ms"] > 0]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    
    expected_field_count = 9 # number of fields in the metric dict
    all_metrics_captured = all(
        r["metrics_captured"] and len(r["fields_present"]) == expected_field_count 
        for r in results_data
    )

    print("-" * 80)
    print(f"Average Latency: {avg_latency:.2f}ms")
    print(f"All Metrics Captured: {all_metrics_captured}")

    # Create results dir if not exists
    results_dir = os.path.join(project_root, "evals", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    output_path = os.path.join(results_dir, "observability_eval.json")
    
    langsmith_active = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true" and bool(os.getenv("LANGCHAIN_API_KEY", "").strip())
    
    final_output = {
        "eval_name": "observability_baseline",
        "run_date": datetime.utcnow().isoformat() + "Z",
        "langsmith_active": langsmith_active,
        "results": results_data,
        "summary": {
            "avg_latency_ms": round(avg_latency, 2),
            "all_metrics_captured": all_metrics_captured,
            "intents_covered": len(queries)
        }
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
        
    print(f"\nResults saved to {output_path}")

if __name__ == '__main__':
    run_eval()
