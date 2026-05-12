from __future__ import annotations

from typing import Any

from openai import OpenAI

from config import settings


class OpenSourceLLM:
    def __init__(self) -> None:
        self.model = settings.hf_model
        self.token = settings.hf_token
        self.enabled = bool(self.token) and settings.use_remote_llm
        self.client = None
        if self.enabled:
            self.client = OpenAI(
                base_url='https://router.huggingface.co/v1',
                api_key=self.token,
            )

    def generate_json_or_text(self, system_prompt: str, user_prompt: str, max_tokens: int = 700) -> dict[str, Any]:
        if not self.enabled or self.client is None:
            return {
                'ok': False,
                'provider': 'fallback',
                'model': None,
                'text': '',
                'reason': 'HF_TOKEN or HF_API_TOKEN not configured; using deterministic fallback.',
            }

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': system_prompt.strip()},
                    {'role': 'user', 'content': user_prompt.strip()},
                ],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            text = ''
            if response.choices and response.choices[0].message:
                text = response.choices[0].message.content or ''
            return {
                'ok': bool(text.strip()),
                'provider': 'huggingface_router',
                'model': self.model,
                'text': text.strip(),
                'reason': '' if text.strip() else 'Empty response from model.',
            }
        except Exception as exc:
            return {
                'ok': False,
                'provider': 'huggingface_router',
                'model': self.model,
                'text': '',
                'reason': str(exc),
            }
