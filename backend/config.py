from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


load_env_file(PROJECT_DIR / '.env')
load_env_file(BASE_DIR / '.env')


@dataclass
class Settings:
    app_name: str = 'StudyNest Unified Open-Source AI Platform'
    hf_token: str | None = os.getenv('HF_TOKEN') or os.getenv('HF_API_TOKEN')
    hf_model: str = os.getenv('HF_MODEL', 'Qwen/Qwen3-8B:nscale')
    use_remote_llm: bool = os.getenv('USE_REMOTE_LLM', 'true').lower() == 'true'
    embedding_model: str = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    rag_collection: str = os.getenv('RAG_COLLECTION', 'studynest_unified_rag')
    top_k: int = int(os.getenv('RAG_TOP_K', '4'))
    llm_timeout_seconds: int = int(os.getenv('LLM_TIMEOUT_SECONDS', '120'))
    default_mode: str = os.getenv('DEFAULT_MODE', 'auto')

    candidate_models: tuple[str, ...] = (
        'Qwen/Qwen3-8B:nscale',
        'Qwen/Qwen3-8B',
        'Qwen/Qwen3.5-9B',
        'meta-llama/Llama-3.3-70B-Instruct',
        'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
        'mistralai/Mistral-Small-3.1-24B-Instruct-2503',
    )


settings = Settings()
