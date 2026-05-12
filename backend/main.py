from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from db import create_enquiry, init_db, list_students, verify_login
from memory import get_or_create_session, set_authenticated, update_memory
from models import ChatIn, ChatOut, ContactIn, LoginIn
from orchestrator import AgentOrchestrator
from rag_engine import RAGEngine

app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

init_db()
orchestrator = AgentOrchestrator()
rag_engine = RAGEngine()
rag_engine.index_documents()


@app.get('/health')
def health() -> dict:
    llm = orchestrator.admin_agent.llm
    return {
        'ok': True,
        'service': settings.app_name,
        'rag_mode': rag_engine.mode,
        'llm_enabled': llm.enabled,
        'llm_model': llm.model if llm.enabled else None,
        'candidate_models': list(settings.candidate_models),
    }


@app.get('/students')
def students() -> list[dict]:
    return list_students()


@app.post('/login')
def login(inp: LoginIn) -> dict:
    user = verify_login(inp.student_id, inp.password)
    if not user:
        raise HTTPException(status_code=401, detail='Invalid student ID or password')
    session_id, _ = get_or_create_session(None)
    set_authenticated(session_id, inp.student_id)
    return {'ok': True, 'session_id': session_id, 'student': user}


@app.post('/chat', response_model=ChatOut)
def chat(inp: ChatIn) -> ChatOut:
    session_id, memory = get_or_create_session(inp.session_id)
    student_id = inp.student_id or memory.get('student_id')
    preferred_mode = inp.preferred_mode or memory.get('preferred_mode') or settings.default_mode
    result = orchestrator.run(inp.message, student_id=student_id, preferred_mode=preferred_mode)
    update_memory(
        session_id,
        inp.message,
        result['answer'],
        result['intent'],
        student_id=student_id,
        preferred_mode=preferred_mode,
    )
    return ChatOut(
        session_id=session_id,
        agent_mode=result['agent_mode'],
        intent=result['intent'],
        answer=result['answer'],
        tool_trace=result.get('tool_trace', []),
        actions=result.get('actions', []),
        data=result.get('data'),
    )


@app.post('/contact')
def contact_org(inp: ContactIn) -> dict:
    eid = create_enquiry(inp.student_id, inp.name, inp.email, inp.subject, inp.message)
    return {'ok': True, 'enquiry_id': eid, 'message': 'Your further enquiry has been recorded for the organisation.'}


@app.post('/rag/reindex')
def reindex_rag() -> dict:
    return rag_engine.index_documents()


@app.get('/rag/debug')
def rag_debug(query: str, domain: str | None = None, top_k: int = 4) -> dict:
    domains = [domain] if domain else None
    results = rag_engine.search(query, top_k=top_k, domains=domains)
    return {
        'ok': True,
        'query': query,
        'domains': domains,
        'count': len(results),
        'results': results,
    }
