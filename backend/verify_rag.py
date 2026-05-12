from __future__ import annotations

import json
from pathlib import Path

from rag_engine import RAGEngine

engine = RAGEngine()
engine.index_documents()

queries = [
    ("admin", "What is the tuition fee deadline and can I request a payment plan?"),
    ("learning", "I want beginner Python and SQL learning resources for data science."),
    ("peers", "Suggest peers interested in Python and machine learning."),
]

all_results = []
for domain, query in queries:
    results = engine.search(query, domains=[domain])
    entry = {
        "domain": domain,
        "query": query,
        "results": results,
    }
    all_results.append(entry)
    print(f"\n=== {domain.upper()} ===")
    for item in results:
        title = item.get("metadata", {}).get("title", "Unknown")
        score = round(item.get("score", 0.0), 4)
        text_preview = (item.get("text", "")[:140] + "...") if item.get("text") else ""
        print(f"- {title} | score={score} | {text_preview}")

out_path = Path(__file__).resolve().parent / "rag_verification_results.json"
out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nSaved RAG verification results to {out_path}")
