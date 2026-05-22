import os
import sys
import time
import json
import asyncio
from datetime import datetime

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from agents.case_law_agent import search_cases, summarize_case, search_and_summarize
from rag.llm import llm as groq_client

async def simulate_sequential_search_and_summarize(query: str, groq_client, max_results: int = 3):
    """Simulate the old behaviour by calling summarize sequentially with a sleep."""
    cases = search_cases(query, max_results=max_results)
    enriched = []
    for case in cases:
        summary = await summarize_case(case, groq_client)
        enriched.append({"case": case, "summary": summary})
        time.sleep(0.3)
    return {
        "query": query,
        "results": enriched,
        "total_found": len(enriched),
    }

async def run_eval():
    queries = [
        "Section 302 IPC murder punishment",
        "cheque bounce Section 138 NI Act",
        "Kesavananda Bharati fundamental rights",
        "domestic violence protection order",
        "bail conditions Section 437 CrPC",
    ]

    results_data = []

    print("Starting Case Law Async Performance Evaluation...")
    print(f"Model: {groq_client.model}")
    print("-" * 80)
    print(f"{'Query':<40} | {'Seq (s)':<8} | {'Async (s)':<9} | {'Speedup':<7} | {'Cases':<5}")
    print("-" * 80)

    for query in queries:
        # Run sequential
        start_seq = time.time()
        res_seq = await simulate_sequential_search_and_summarize(query, groq_client)
        time_seq = time.time() - start_seq

        # Run async
        start_async = time.time()
        res_async = await search_and_summarize(query, groq_client)
        time_async = time.time() - start_async

        speedup = time_seq / time_async if time_async > 0 else 0

        cases_returned = res_async["total_found"]
        
        all_cited = True
        all_summarized = True
        
        for item in res_async["results"]:
            if not item["case"].get("citation"):
                all_cited = False
            if not item.get("summary") or len(item["summary"].strip()) < 10:
                all_summarized = False

        print(f"{query[:38]:<40} | {time_seq:<8.2f} | {time_async:<9.2f} | {speedup:<7.2f}x | {cases_returned:<5}")

        results_data.append({
            "query": query,
            "sequential_time_seconds": round(time_seq, 2),
            "async_time_seconds": round(time_async, 2),
            "speedup_factor": round(speedup, 2),
            "cases_returned": cases_returned,
            "all_cited": all_cited,
            "all_summarized": all_summarized,
        })

    avg_seq = sum(r["sequential_time_seconds"] for r in results_data) / len(results_data)
    avg_async = sum(r["async_time_seconds"] for r in results_data) / len(results_data)
    avg_speedup = avg_seq / avg_async if avg_async > 0 else 0

    print("-" * 80)
    print(f"{'AVERAGE':<40} | {avg_seq:<8.2f} | {avg_async:<9.2f} | {avg_speedup:<7.2f}x |")

    # Create results dir if not exists
    results_dir = os.path.join(project_root, "evals", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    output_path = os.path.join(results_dir, "case_law_async_eval.json")
    
    final_output = {
        "eval_name": "case_law_async_performance",
        "run_date": datetime.utcnow().isoformat() + "Z",
        "model_used": groq_client.model,
        "results": results_data,
        "summary": {
            "avg_sequential_time": round(avg_seq, 2),
            "avg_async_time": round(avg_async, 2),
            "avg_speedup_factor": round(avg_speedup, 2),
            "total_queries": len(queries)
        }
    }
    
    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=2)
        
    print(f"\nResults saved to {output_path}")

if __name__ == '__main__':
    asyncio.run(run_eval())
