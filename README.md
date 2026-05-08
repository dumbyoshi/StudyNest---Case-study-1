# StudyNest: RAG-Powered Multi-Agent AI Assistant + LLM Benchmark

Multi-agent student-support assistant with RAG retrieval, plus a full benchmark of three open-source LLMs running on the same prompts, same retriever, same scoring rules. Built as a case study for my MSc Applied Data Science (SRH Heidelberg).

The interesting part is not the chatbot. It is the benchmark: the same six metrics applied across all three models so the choice of model is decided on data, not vibes.

## Final results

Three open-source LLMs were tested on the same prompt set with the same RAG context and the same scoring logic.

| Model | Latency (s) | Keyword coverage | Semantic similarity | RAG faithfulness | Failure rate | Weighted score |
|---|---|---|---|---|---|---|
| **Llama-3.3-70B-Instruct** | **1.74** | 0.81 | 0.621 | **0.583** | 0.0 | **0.7631** |
| Qwen3-8B | 4.63 | **0.91** | **0.683** | 0.287 | 0.0 | 0.7244 |
| DeepSeek-R1-Distill-Qwen-32B | 20.83 | 0.49 | 0.476 | 0.421 | 0.0 | 0.5299 |

Plain reading of this:

- Llama wins overall. Fastest of the three (12x faster than DeepSeek, 2.6x faster than Qwen) and best at staying grounded in the retrieved context.
- Qwen is the strongest lightweight backup. Highest keyword coverage and semantic similarity, but slower than Llama and lower RAG faithfulness.
- DeepSeek loses the head-to-head for this use case. Slow and inconsistent, even though the reasoning quality looks fine on individual prompts.

For a live student-support dashboard where latency matters and answers need to stay close to the policy text, Llama is the pick. Qwen is the practical fallback if cost or memory pressure forces a smaller model.

## Why a benchmark, not just "ask the model"

The first iteration of this project (a teammate-recommendation tool from earlier in the semester) just used a single closed model and a hardcoded API key. There was no way to compare models, no way to swap one out, and no record of how good the answers actually were.

This rebuild fixes all three of those:

1. Open-source LLMs only, behind a single Hugging Face router. The model id is one config line, so swapping Qwen for Llama for DeepSeek is one change.
2. A reproducible benchmark script (`benchmark_models.py`) that runs the same prompt set across all candidate models and writes JSON results to disk.
3. RAG verification (`verify_rag.py`) that prints the actual retrieved chunks with similarity scores, so you can see the retrieval is working before blaming the LLM.

## Architecture

```
Frontend (HTML/CSS/JS)
        |
        v
FastAPI backend
   - session memory + login
   - orchestrator routes to admin agent or recommendation agent
   - shared RAG layer (ChromaDB + sentence-transformers)
   - LLM provider wraps Hugging Face router
        |
        v
Open-source LLM (configurable model id)
```

Two agents share one frontend, one session, one RAG layer, one LLM provider. The orchestrator decides which agent handles each message. RAG retrieval is verified separately so retrieval problems don't get hidden inside the LLM's answer.

## Stack

- FastAPI, SQLite, ChromaDB
- sentence-transformers/all-MiniLM-L6-v2 for embeddings
- Hugging Face Router (OpenAI-compatible) for inference
- Plain HTML/CSS/JS frontend

## Project layout

```
backend/
  main.py                    FastAPI app, routes, health
  orchestrator.py            routes admin vs recommendation
  agents/
    admin_agent.py           administrative AI agent
    recommendation_agent.py  recommendation AI agent
  rag_engine.py              shared retrieval (vector + lexical fallback)
  llm_provider.py            HF router wrapper
  benchmark_models.py        run all candidate models on the same prompts
  verify_rag.py              prove retrieval works on real queries
  test_cases.json            shared prompt set
  evaluation_outputs/
    model_benchmark_results.json
  rag_verification_results.json
  data/                      knowledge base
frontend/
  index.html                 single dashboard
```

## Run it locally

Backend:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate         # macOS / Linux
# .venv\Scripts\activate          # Windows
pip install -r requirements.txt
uvicorn main:app --reload
```

Frontend:

```
open frontend/index.html
```

Demo login: `s1001` / `pass1001`

## Enable the LLM

Create `backend/.env`:

```env
HF_TOKEN=your_huggingface_token
HF_MODEL=Qwen/Qwen3-8B:nscale
USE_REMOTE_LLM=true
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RAG_TOP_K=4
```

Without a token the app still runs on a deterministic fallback so you can demo the routing and RAG without hitting any API.

## Reproduce the benchmark

```bash
cd backend
python benchmark_models.py
```

This writes `evaluation_outputs/model_benchmark_results.json` with per-prompt: latency, keyword coverage, semantic similarity, RAG faithfulness, format compliance, and any error message. Aggregated weighted scores match the table above.

To verify RAG retrieval in isolation:

```bash
python verify_rag.py
```

This prints top retrieved chunks per query with similarity scores, and saves `rag_verification_results.json`.

## RAG retrieval check

The retrieval stage was checked separately on three example queries:

| Domain | Query | Top retrieved | Similarity |
|---|---|---|---|
| Admin | tuition fee deadline + payment plan | Tuition Fee and Deadline Policy | 0.675 |
| Learning | beginner Python and SQL for data science | Python for Data Science | 0.591 |
| Peers | peers interested in Python and ML | Ben Kumar profile | 0.551 |

This matters because if retrieval is broken, even the best LLM produces nonsense. Confirming retrieval first lets the benchmark scores actually be about the LLM.

## What I would do next

- Add a small fine-tuned encoder for student-policy text and compare retrieval against the current MiniLM
- Run the same benchmark with two more lightweight models (Phi-3-mini, Mistral-Small) to get a full small-model picture
- Latency under load (the current numbers are single-request, no concurrency)

## Author

Nithin Krishnan, MSc Applied Data Science & Analytics, SRH Heidelberg.

[github.com/dumbyoshi](https://github.com/dumbyoshi) | [linkedin.com/in/nithin-krishnan](https://linkedin.com/in/nithin-krishnan)
