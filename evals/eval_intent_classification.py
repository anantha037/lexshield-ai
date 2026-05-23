import os
import sys
import json
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.intent_classifier import intent_classifier
from rag.llm import llm as groq_client

TEST_QUERIES = [
    {"query": "What is Section 302 IPC?", "expected_intent": "legal_query", "type": "clear"},
    {"query": "Find landmark cases on cheque bounce Section 138", "expected_intent": "case_law_search", "type": "clear"},
    {"query": "Draft a consumer complaint against Amazon", "expected_intent": "draft_request", "type": "clear"},
    {"query": "What are my rights as a tenant?", "expected_intent": "rights_check", "type": "clear"},
    {"query": "Check the legal risk in this employment contract", "expected_intent": "risk_check", "type": "clear"},
    {"query": "Translate this to Malayalam", "expected_intent": "translation_request", "type": "clear"},
    {"query": "Hello, what can you do?", "expected_intent": "general", "type": "clear"},
    {"query": "I got fired without any notice, what can I do?", "expected_intent": "rights_check", "type": "nuanced"},
    {"query": "My landlord locked me out of my house", "expected_intent": "rights_check", "type": "nuanced"},
    {"query": "Can you help me write a legal notice to my employer?", "expected_intent": "draft_request", "type": "clear"},
    {"query": "Kesavananda Bharati judgment significance", "expected_intent": "case_law_search", "type": "clear"},
    {"query": "Is my non-compete clause enforceable in Kerala?", "expected_intent": "risk_check", "type": "nuanced"}, # The user expects legal_query, let me use the user's expected: legal_query
    {"query": "My employer hasn't paid salary for 3 months", "expected_intent": "rights_check", "type": "nuanced"},
    {"query": "What happened in the Vishaka case?", "expected_intent": "case_law_search", "type": "clear"},
    {"query": "Explain Article 21 of the Constitution", "expected_intent": "legal_query", "type": "clear"},
]

# Correcting the expected_intent for "Is my non-compete clause enforceable in Kerala?" to match user prompt exactly:
TEST_QUERIES[11]["expected_intent"] = "legal_query"

def run_eval():
    print("Running LLM Intent Classification Evaluation...\n")
    print(f"{'Query':<55} | {'Expected':<18} | {'Predicted':<18} | {'Correct':<7} | {'Conf':<5} | {'Path'}")
    print("-" * 125)
    
    results = []
    correct_count = 0
    clear_correct = 0
    clear_total = 0
    nuanced_correct = 0
    nuanced_total = 0
    total_latency = 0
    total_confidence = 0
    llm_path_count = 0
    fallback_path_count = 0

    for item in TEST_QUERIES:
        query = item["query"]
        expected = item["expected_intent"]
        q_type = item["type"]
        
        start_time = time.time()
        result = intent_classifier.classify_with_llm(query, groq_client)
        latency_ms = int((time.time() - start_time) * 1000)
        
        predicted = result.intent
        confidence = getattr(result, "confidence", 0.0)
        is_correct = predicted == expected
        path = "LLM" if type(result).__name__ == "LLMIntentResult" else "Fallback"
        
        reasoning = getattr(result, "reasoning", "")
        detected_sections = list(getattr(result, "detected_sections", []))
        detected_acts = list(getattr(result, "detected_acts", []))
        jurisdiction = getattr(result, "jurisdiction", "")
        query_complexity = getattr(result, "query_complexity", "")
        
        results.append({
            "query": query,
            "expected_intent": expected,
            "predicted_intent": predicted,
            "correct": is_correct,
            "confidence": confidence,
            "reasoning": reasoning,
            "detected_sections": detected_sections,
            "detected_acts": detected_acts,
            "jurisdiction": jurisdiction,
            "query_complexity": query_complexity,
            "latency_ms": latency_ms,
            "path": path,
            "type": q_type
        })
        
        if is_correct:
            correct_count += 1
            if q_type == "clear":
                clear_correct += 1
            else:
                nuanced_correct += 1
                
        if q_type == "clear":
            clear_total += 1
        else:
            nuanced_total += 1
            
        total_latency += latency_ms
        total_confidence += confidence
        
        if path == "LLM":
            llm_path_count += 1
        else:
            fallback_path_count += 1
            
        print(f"{query[:53]:<55} | {expected:<18} | {predicted:<18} | {str(is_correct):<7} | {confidence:.2f} | {path}")

    overall_accuracy = correct_count / len(TEST_QUERIES) if len(TEST_QUERIES) > 0 else 0
    clear_accuracy = clear_correct / clear_total if clear_total > 0 else 0
    nuanced_accuracy = nuanced_correct / nuanced_total if nuanced_total > 0 else 0
    avg_confidence = total_confidence / len(TEST_QUERIES) if len(TEST_QUERIES) > 0 else 0
    avg_latency = total_latency / len(TEST_QUERIES) if len(TEST_QUERIES) > 0 else 0

    print("-" * 125)
    print(f"\nOverall Accuracy: {overall_accuracy*100:.1f}%")
    print(f"Clear Intent Accuracy: {clear_accuracy*100:.1f}%")
    print(f"Nuanced Intent Accuracy: {nuanced_accuracy*100:.1f}%")
    print(f"Average Confidence: {avg_confidence:.2f}")
    print(f"Average Latency: {avg_latency:.1f} ms")
    
    # Save to file
    os.makedirs(os.path.join(os.path.dirname(__file__), 'results'), exist_ok=True)
    out_file = os.path.join(os.path.dirname(__file__), 'results', 'intent_classification_eval.json')
    
    # Need to find actual model name used, fallback to groq_client.model if it has one
    model_name = getattr(groq_client, "model", "llama-3.3-70b-versatile")
    
    eval_data = {
        "eval_name": "llm_intent_classification",
        "run_date": datetime.utcnow().isoformat() + "Z",
        "model_used": model_name,
        "total_queries": len(TEST_QUERIES),
        "results": results,
        "summary": {
            "overall_accuracy": overall_accuracy,
            "clear_intent_accuracy": clear_accuracy,
            "nuanced_intent_accuracy": nuanced_accuracy,
            "avg_confidence": avg_confidence,
            "avg_latency_ms": avg_latency,
            "llm_path_used": llm_path_count,
            "fallback_path_used": fallback_path_count
        }
    }
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2)
        
    print(f"\nResults saved to {out_file}")

if __name__ == "__main__":
    run_eval()
