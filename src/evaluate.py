"""
Stage 8: Evaluation & Monitoring
-----------------------------------
- Latency: runs a set of test queries and reports p50/p95 end-to-end latency
  against the 2-5s target.
- Retrieval quality: if you provide labeled test cases (a question plus the
  expected filename+page that should be retrieved), computes Recall@k and MRR.

Usage:
    python -m src.evaluate
    (edit TEST_QUERIES / LABELED_CASES below, or load from a JSON file)
"""
import json
import statistics
from typing import List, Dict, Optional

from src.rag_chain import RAGChatbot

# Simple smoke-test queries (no ground truth needed) - just measures latency.
TEST_QUERIES = [
    "What is RAG and why does chunking matter?",
    "Summarize chapter 1.",
    "What is mentioned about scanned documents?",
]

# Optional: labeled cases for retrieval quality metrics.
# Fill in with real questions + the filename/page you know contains the answer,
# once you're testing against your real PDF corpus.
LABELED_CASES: List[Dict] = [
    # {"question": "...", "expected_filename": "policy.pdf", "expected_page": 12},
]


def measure_latency(bot: RAGChatbot, queries: List[str]) -> Dict:
    latencies = []
    for q in queries:
        result = bot.ask(q)
        latencies.append(result["latency"]["total_s"])
        print(f"  [{result['latency']['total_s']:.2f}s] {q}")

    latencies_sorted = sorted(latencies)
    p50 = statistics.median(latencies_sorted)
    p95_index = min(int(len(latencies_sorted) * 0.95), len(latencies_sorted) - 1)
    p95 = latencies_sorted[p95_index]

    return {
        "n_queries": len(queries),
        "min_s": round(min(latencies), 2),
        "max_s": round(max(latencies), 2),
        "p50_s": round(p50, 2),
        "p95_s": round(p95, 2),
        "within_2_5s_target": sum(1 for l in latencies if 2 <= l <= 5) / len(latencies),
    }


def recall_at_k_and_mrr(bot: RAGChatbot, cases: List[Dict]) -> Optional[Dict]:
    if not cases:
        return None

    hits = 0
    reciprocal_ranks = []
    for case in cases:
        result = bot.ask(case["question"])
        found_rank = None
        for rank, chunk in enumerate(result["retrieved_chunks"], 1):
            if (chunk["filename"] == case["expected_filename"]
                    and chunk["page"] == case["expected_page"]):
                found_rank = rank
                break
        if found_rank:
            hits += 1
            reciprocal_ranks.append(1 / found_rank)
        else:
            reciprocal_ranks.append(0)

    return {
        "n_cases": len(cases),
        "recall_at_k": round(hits / len(cases), 3),
        "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 3),
    }


def main():
    print("Loading RAG chatbot (embeddings, reranker, vectorstore, LLM)...\n")
    bot = RAGChatbot()

    print("=" * 60)
    print("LATENCY EVALUATION")
    print("=" * 60)
    latency_report = measure_latency(bot, TEST_QUERIES)
    print(f"\n{json.dumps(latency_report, indent=2)}")

    print("\n" + "=" * 60)
    print("RETRIEVAL QUALITY (Recall@k, MRR)")
    print("=" * 60)
    retrieval_report = recall_at_k_and_mrr(bot, LABELED_CASES)
    if retrieval_report:
        print(json.dumps(retrieval_report, indent=2))
    else:
        print("No labeled cases provided - skipping. "
              "Add entries to LABELED_CASES in src/evaluate.py to enable this.")


if __name__ == "__main__":
    main()
