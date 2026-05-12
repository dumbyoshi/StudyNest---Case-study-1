from __future__ import annotations

from typing import Any, Dict
import uuid

_SESSION_MEMORY: Dict[str, Dict[str, Any]] = {}


def get_or_create_session(session_id: str | None) -> tuple[str, Dict[str, Any]]:
    if not session_id:
        session_id = str(uuid.uuid4())
    if session_id not in _SESSION_MEMORY:
        _SESSION_MEMORY[session_id] = {
            'student_id': None,
            'preferred_mode': 'auto',
            'last_intent': None,
            'history': [],
            'is_authenticated': False,
        }
    return session_id, _SESSION_MEMORY[session_id]


def set_authenticated(session_id: str, student_id: str) -> None:
    sid, memory = get_or_create_session(session_id)
    memory['student_id'] = student_id
    memory['is_authenticated'] = True


def update_memory(
    session_id: str,
    message: str,
    response: str,
    intent: str,
    student_id: str | None = None,
    preferred_mode: str | None = None,
) -> None:
    _, memory = get_or_create_session(session_id)
    if student_id:
        memory['student_id'] = student_id
    if preferred_mode:
        memory['preferred_mode'] = preferred_mode
    memory['last_intent'] = intent
    memory['history'].append({'message': message, 'response': response, 'intent': intent})
    memory['history'] = memory['history'][-12:]
