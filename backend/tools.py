from __future__ import annotations

import json
import os
from typing import Any

from db import (
    get_student,
    get_enrollment,
    get_missing_documents,
    create_ticket,
    create_appointment,
)

APP_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(APP_DIR, 'data')
CATALOG_PATH = os.path.join(DATA_DIR, 'learning_catalog.json')
PEERS_PATH = os.path.join(DATA_DIR, 'peer_profiles.json')


def _load_json(path: str) -> list[dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def tool_get_student_profile(student_id: str) -> dict | None:
    return get_student(student_id)


def tool_get_enrollment_status(student_id: str) -> dict | None:
    return get_enrollment(student_id, 'WS25')


def tool_get_missing_documents(student_id: str) -> list[dict]:
    return get_missing_documents(student_id)


def tool_create_support_ticket(student_id: str | None, message: str) -> int:
    return create_ticket(student_id, 'Student administrative enquiry', message)


def tool_book_appointment(student_id: str | None, purpose: str) -> dict:
    return create_appointment(student_id, purpose)


def tool_get_learning_catalog() -> list[dict[str, Any]]:
    return _load_json(CATALOG_PATH)


def tool_get_peer_profiles() -> list[dict[str, Any]]:
    return _load_json(PEERS_PATH)
