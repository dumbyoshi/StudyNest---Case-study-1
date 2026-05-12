from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, EmailStr


class LoginIn(BaseModel):
    student_id: str
    password: str


class ChatIn(BaseModel):
    session_id: Optional[str] = None
    student_id: Optional[str] = None
    message: str
    preferred_mode: Optional[str] = None


class ChatOut(BaseModel):
    session_id: str
    agent_mode: str
    intent: str
    answer: str
    tool_trace: List[str] = []
    actions: List[Dict[str, Any]] = []
    data: Optional[Dict[str, Any]] = None


class ContactIn(BaseModel):
    student_id: Optional[str] = None
    name: str
    email: EmailStr
    subject: str
    message: str
