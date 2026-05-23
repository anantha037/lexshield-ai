import json
import time
from datetime import datetime, timezone
import sys
import os
import uuid
import sqlite3

# Ensure LexShield root is in PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.orchestrator import master_orchestrator

TEST_CASES = [
    {
        "name": "full_facts_consumer",
        "query": "Draft a consumer complaint against Reliance Jio for service disruption for 30 days in Kerala. My name is Arjun Menon, account number JIO-2024-KL-8821. I want a refund of 500 rupees.",
        "expected_dynamic": True,
        "expected_questions_asked": 0
    },
    {
        "name": "partial_facts_wage_theft",
        "query": "Draft a wage theft complaint against my employer ABC Corp",
        "expected_dynamic": True,
        "expected_questions_asked": "greater_than_0"
    },
    {
        "name": "vague_query_fallback",
        "query": "Draft a complaint",
        "expected_dynamic": False,
        "expected_questions_asked": "category_menu"
    },
    {
        "name": "full_facts_legal_notice",
        "query": "Draft a legal notice to landlord Mr. Suresh Kumar at 42 MG Road Kochi Kerala for illegal eviction. My name is Priya Nair. I demand reinstatement within 7 days under Transfer of Property Act.",
        "expected_dynamic": True,
        "expected_questions_asked": 0
    },
]

from agents.drafting_agent import _CLARIFYING_QUESTIONS

def run_eval():
    results = []
    print(f"Running Dynamic Fact Extraction Eval on {len(TEST_CASES)} queries...\n")
    
    for tc in TEST_CASES:
        print(f"Testing: {tc['name']}")
        session_id = str(uuid.uuid4())
        
        start_t = time.time()
        resp = master_orchestrator.handle_query(query=tc["query"], session_id=session_id)
        latency = (time.time() - start_t) * 1000
        
        answer = getattr(resp, 'answer', getattr(resp, 'answer_text', str(resp)))
        doc_type = getattr(resp, 'doc_type', '')
        
        # If doc_type wasn't returned, try to deduce it for testing
        if not doc_type:
            if "consumer" in tc["name"]: doc_type = "consumer_complaint"
            elif "wage_theft" in tc["name"]: doc_type = "wage_theft"
            elif "legal_notice" in tc["name"]: doc_type = "illegal_eviction"
            
        total_qs = len(_CLARIFYING_QUESTIONS.get(doc_type, [])) if doc_type else 0
        reached_confirm_stage = "confirm" in answer.lower()
        is_category_menu = "1. **Unpaid Wages / Salary**" in answer
        
        if reached_confirm_stage:
            questions_asked = 0
        else:
            questions_asked = answer.count("?")
            
        use_dynamic_path = False
        questions_saved = 0
        
        if is_category_menu:
            questions_asked_label = "category_menu"
            use_dynamic_path = False
            questions_saved = 0
        else:
            questions_asked_label = questions_asked
            if questions_asked == 0 and reached_confirm_stage:
                use_dynamic_path = True
            elif questions_asked > 0 and total_qs > 0 and questions_asked < total_qs:
                use_dynamic_path = True
            questions_saved = max(0, total_qs - questions_asked)
            
        results.append({
            "name": tc["name"],
            "query": tc["query"],
            "expected_dynamic": tc["expected_dynamic"],
            "use_dynamic_path": use_dynamic_path,
            "questions_asked": questions_asked,
            "questions_asked_label": questions_asked_label,
            "questions_saved": questions_saved,
            "reached_confirm_stage": reached_confirm_stage,
            "latency_ms": latency,
            "answer_preview": answer[:150].replace('\n', ' ') + '...'
        })
        
    # Calculate metrics
    full_facts = [r for r in results if "full_facts" in r["name"]]
    full_facts_skip_rate = sum(1 for r in full_facts if r["reached_confirm_stage"]) / len(full_facts) if full_facts else 0.0
    
    partial_facts = [r for r in results if "partial_facts" in r["name"]]
    partial_facts_reduction_rate = 1.0 if any(r["questions_asked"] > 0 and r["use_dynamic_path"] for r in partial_facts) else 0.0
    
    fallback = [r for r in results if "fallback" in r["name"]]
    fallback_correct = all(not r["use_dynamic_path"] and r["questions_asked_label"] == "category_menu" for r in fallback)
    
    avg_latency = sum(r["latency_ms"] for r in results) / len(results) if results else 0.0
    
    eval_data = {
        "eval_name": "dynamic_fact_extraction_drafting",
        "run_date": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "summary": {
            "full_facts_skip_rate": full_facts_skip_rate,
            "partial_facts_reduction_rate": partial_facts_reduction_rate,
            "fallback_correct": fallback_correct,
            "avg_latency_ms": avg_latency
        }
    }
    
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "dynamic_drafting_eval.json")
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2)
        
    print("\n" + "="*80)
    print(" EVALUATION SUMMARY")
    print("="*80)
    print(f"Full Facts Skip Rate (0 questions):  {full_facts_skip_rate*100:.1f}%")
    print(f"Partial Facts Reduction Rate:        {partial_facts_reduction_rate*100:.1f}%")
    print(f"Fallback Correct (Vague Query):      {'Pass' if fallback_correct else 'Fail'}")
    print(f"Average Latency:                     {avg_latency:.0f} ms")
    print("-"*80)
    print(f"{'Test Case':<30} | {'Dynamic?':<10} | {'Qs Asked':<10} | {'Confirm?':<10}")
    print("-"*80)
    for r in results:
        print(f"{r['name']:<30} | {str(r['use_dynamic_path']):<10} | {str(r['questions_asked_label']):<10} | {str(r['reached_confirm_stage']):<10}")
    print("="*80)
    print(f"Saved detailed results to {out_file}")

if __name__ == "__main__":
    run_eval()
