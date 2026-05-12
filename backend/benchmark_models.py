from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from openai import OpenAI
from sentence_transformers import SentenceTransformer, util

from config import settings

MODELS = [
    "Qwen/Qwen3-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "meta-llama/Llama-3.3-70B-Instruct",
]
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_cases(path: str = "test_cases.json") -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_client() -> OpenAI:
    if not settings.hf_token:
        raise SystemExit("HF_TOKEN missing. Put it in backend/.env first.")
    return OpenAI(base_url="https://router.huggingface.co/v1", api_key=settings.hf_token)


def generate_answer(client: OpenAI, model_name: str, prompt: str) -> tuple[str, float, str]:
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful university administrative and recommendation assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=600,
        )
        answer = (response.choices[0].message.content or "").strip()
        return answer, round(time.perf_counter() - start, 3), ""
    except Exception as exc:
        return "", round(time.perf_counter() - start, 3), str(exc)


def keyword_coverage(answer: str, required_keywords: list[str]) -> float:
    if not required_keywords:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in required_keywords if kw.lower() in answer_lower)
    return round(hits / len(required_keywords), 3)


def semantic_similarity(embedder: SentenceTransformer, text1: str, text2: str) -> float:
    if not text1.strip() or not text2.strip():
        return 0.0
    emb1 = embedder.encode(text1, convert_to_tensor=True)
    emb2 = embedder.encode(text2, convert_to_tensor=True)
    return round(float(util.cos_sim(emb1, emb2).item()), 3)


def split_sentences(text: str) -> list[str]:
    pieces = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    return pieces


def rag_faithfulness(embedder: SentenceTransformer, answer: str, context: str, threshold: float = 0.45) -> float:
    answer_sents = split_sentences(answer)
    context_sents = split_sentences(context)
    if not answer_sents or not context_sents:
        return 0.0

    ctx_embeddings = embedder.encode(context_sents, convert_to_tensor=True)
    supported = 0
    for sent in answer_sents:
        sent_emb = embedder.encode(sent, convert_to_tensor=True)
        sims = util.cos_sim(sent_emb, ctx_embeddings)[0]
        if float(sims.max().item()) >= threshold:
            supported += 1
    return round(supported / len(answer_sents), 3)


def format_compliance(answer: str, prompt: str) -> float:
    prompt_lower = prompt.lower()
    if "4 bullet" in prompt_lower or "bullet" in prompt_lower:
        markers = ["- ", "* ", "1.", "2.", "•"]
        return 1.0 if any(m in answer for m in markers) else 0.0
    return 1.0


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["model"], []).append(row)

    summary: dict[str, Any] = {}
    for model, items in grouped.items():
        ok_items = [i for i in items if not i["error"]]
        if not ok_items:
            summary[model] = {
                "avg_latency_seconds": None,
                "avg_keyword_coverage": 0.0,
                "avg_semantic_similarity": 0.0,
                "avg_rag_faithfulness": 0.0,
                "avg_format_compliance": 0.0,
                "failure_rate": 1.0,
                "final_weighted_score": 0.0,
            }
            continue

        avg_latency = sum(i["latency_seconds"] for i in ok_items) / len(ok_items)
        avg_keyword = sum(i["keyword_coverage"] for i in ok_items) / len(ok_items)
        avg_similarity = sum(i["semantic_similarity"] for i in ok_items) / len(ok_items)
        avg_rag = sum(i["rag_faithfulness"] for i in ok_items) / len(ok_items)
        avg_format = sum(i["format_compliance"] for i in ok_items) / len(ok_items)
        failure_rate = 1 - (len(ok_items) / len(items))

        latency_score = min(1.0, 8.0 / max(avg_latency, 0.001))
        final_score = round(
            0.15 * latency_score +
            0.20 * avg_keyword +
            0.25 * avg_similarity +
            0.25 * avg_rag +
            0.10 * avg_format +
            0.05 * (1.0 - failure_rate),
            4,
        )

        summary[model] = {
            "avg_latency_seconds": round(avg_latency, 3),
            "avg_keyword_coverage": round(avg_keyword, 3),
            "avg_semantic_similarity": round(avg_similarity, 3),
            "avg_rag_faithfulness": round(avg_rag, 3),
            "avg_format_compliance": round(avg_format, 3),
            "failure_rate": round(failure_rate, 3),
            "final_weighted_score": final_score,
        }
    return summary


def main() -> None:
    client = make_client()
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    cases = load_cases()

    results: list[dict[str, Any]] = []
    for case in cases:
        for model in MODELS:
            print(f"Evaluating {model} on {case['id']} ...")
            answer, latency, error = generate_answer(client, model, case["prompt"])
            row = {
                "case_id": case["id"],
                "model": model,
                "prompt": case["prompt"],
                "answer": answer,
                "latency_seconds": latency,
                "keyword_coverage": keyword_coverage(answer, case.get("required_keywords", [])),
                "semantic_similarity": semantic_similarity(embedder, answer, case.get("reference_answer", "")),
                "rag_faithfulness": rag_faithfulness(embedder, answer, case.get("retrieved_context", "")),
                "format_compliance": format_compliance(answer, case["prompt"]),
                "error": error,
            }
            results.append(row)
            print(f"  -> latency={latency}s error={'none' if not error else error[:80]}")

    summary = summarize(results)

    out_dir = Path(__file__).resolve().parent / "evaluation_outputs"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "model_benchmark_results.json"
    out_file.write_text(json.dumps({"detailed_results": results, "summary": summary}, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== SUMMARY ===")
    for model, metrics in summary.items():
        print(f"\n{model}")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

    print(f"\nSaved results to: {out_file}")


if __name__ == "__main__":
    main()
