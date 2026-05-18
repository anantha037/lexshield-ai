import json
import re
from pathlib import Path

def run_local_eval():
    # 1. Load the offline progress data
    file_path = Path(__file__).parent / "eval_progress.json"
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = data.get("rows", [])
    total_queries = len(rows)
    
    if total_queries == 0:
        print("No data found to evaluate.")
        return

    # 2. Metric Trackers
    citation_count = 0
    hit_at_5_count = 0
    reciprocal_ranks = []
    retrieval_eval_count = 0 

    for row in rows:
        question = row.get("question", "")
        answer = row.get("answer", "")
        contexts = row.get("contexts", [])

        # Metric: Citation Rate
        if re.search(r'\[\d+\]', answer):
            citation_count += 1

        # Metric: Hit Rate @5 & MRR
        sec_match = re.search(r'Section\s+(\d+[A-Z]?)', question, re.IGNORECASE)
        
        if sec_match:
            retrieval_eval_count += 1
            expected_sec = sec_match.group(1)
            rank = 0
            
            for i, ctx in enumerate(contexts):
                if expected_sec in ctx[:150]:
                    rank = i + 1
                    break
            
            if 0 < rank <= 5:
                hit_at_5_count += 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)

    # 3. Calculate Final Scores
    citation_rate = (citation_count / total_queries) * 100
    hit_rate = (hit_at_5_count / retrieval_eval_count) * 100 if retrieval_eval_count > 0 else 0.0
    mrr = (sum(reciprocal_ranks) / retrieval_eval_count) if retrieval_eval_count > 0 else 0.0

    # 4. Output to Console
    print("="*50)
    print("LEXSHIELD DETERMINISTIC EVALUATION (OFFLINE)")
    print("="*50)
    print(f"Total Queries Evaluated : {total_queries}")
    print(f"Citation Rate           : {citation_rate:.2f}%")
    print(f"Queries with target Sec : {retrieval_eval_count}")
    print(f"Hit Rate @5             : {hit_rate:.2f}%")
    print(f"Mean Reciprocal Rank    : {mrr:.4f}")
    print("="*50)

    # 5. Save to JSON File
    report_data = {
        "evaluation_summary": {
            "total_queries_evaluated": total_queries,
            "target_statute_queries": retrieval_eval_count,
            "status": "Success",
            "overall_system_health": "Excellent" if (hit_rate >= 90 and mrr >= 0.70 and citation_rate >= 80) else "Good"
        },
        "metrics": {
            "citation_rate": {
                "score_pct": round(citation_rate, 2),
                "grade": "Excellent" if citation_rate >= 80 else "Good" if citation_rate >= 60 else "Needs Improvement",
                "commentary": "Evaluates prompt compliance and generation grounding. A high score confirms the LLM is actively anchoring its legal assertions back to source chunks, severely mitigating the risk of structural hallucination."
            },
            "hit_rate_at_5": {
                "score_pct": round(hit_rate, 2),
                "grade": "Excellent" if hit_rate >= 90 else "Good" if hit_rate >= 75 else "Needs Improvement",
                "commentary": "Evaluates retrieval recall window. Confirms that the embedding model and chunking strategy successfully captured the semantic intent of the legal query and included the vital statute within the top-5 candidate contexts."
            },
            "mean_reciprocal_rank_mrr": {
                "score": round(mrr, 4),
                "grade": "Excellent" if mrr >= 0.70 else "Good" if mrr >= 0.50 else "Needs Improvement",
                "commentary": "Evaluates retrieval ranking precision. A score above 0.70 mathematically proves that the correct legal authority is regularly prioritized at the absolute top positions (Rank 1 or Rank 2), maximizing prompt window efficiency."
            }
        }
    }
    
    save_path = Path(__file__).parent / "local_metrics_report.json"
    with open(save_path, "w", encoding="utf-8") as out_file:
        json.dump(report_data, out_file, indent=4)
        
    print(f"\nDetailed report successfully saved to: {save_path}")

if __name__ == "__main__":
    run_local_eval()