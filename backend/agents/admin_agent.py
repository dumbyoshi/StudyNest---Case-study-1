from __future__ import annotations

from tools import (
    tool_book_appointment,
    tool_create_support_ticket,
    tool_get_enrollment_status,
    tool_get_missing_documents,
    tool_get_student_profile,
)
from rag_engine import RAGEngine
from llm_provider import OpenSourceLLM


class AdministrativeAgent:
    name = 'Administrative AI Agent with RAG'

    def __init__(self) -> None:
        self.rag = RAGEngine()
        self.rag.index_documents()
        self.llm = OpenSourceLLM()

    def handle(self, message: str, intent: str, student_id: str | None) -> dict:
        trace: list[str] = [f'intent:{intent}']
        actions: list[dict] = []
        tool_data: dict = {}

        if intent == 'check_status':
            if not student_id:
                return {
                    'intent': intent,
                    'answer': 'Please log in or enter your Student ID so I can check your enrollment status.',
                    'tool_trace': trace,
                    'actions': [{'label': 'Login first', 'type': 'info'}],
                    'data': {'auth_required': True},
                }
            trace.append('tool:get_student_profile')
            profile = tool_get_student_profile(student_id)
            if not profile:
                return {'intent': intent, 'answer': 'I could not find that student ID.', 'tool_trace': trace, 'actions': []}
            trace.append('tool:get_enrollment_status')
            enrollment = tool_get_enrollment_status(student_id)
            trace.append('tool:get_missing_documents')
            docs = tool_get_missing_documents(student_id)
            tool_data = {'student': profile, 'enrollment': enrollment, 'missing_documents': docs}
            actions = [
                {'label': 'View missing documents', 'type': 'suggest', 'payload': 'What documents am I missing?'},
                {'label': 'Book appointment', 'type': 'suggest', 'payload': 'Book an appointment'},
            ]

        elif intent == 'missing_documents':
            if student_id:
                trace.append('tool:get_missing_documents')
                docs = tool_get_missing_documents(student_id)
                trace.append('tool:get_student_profile')
                profile = tool_get_student_profile(student_id)
                tool_data = {'student': profile, 'missing_documents': docs}
            actions = [{'label': 'Book appointment', 'type': 'suggest', 'payload': 'Book an appointment'}]

        elif intent == 'book_appointment':
            trace.append('tool:book_appointment')
            tool_data = {'booking': tool_book_appointment(student_id, 'Administrative advising')}

        elif intent == 'support_escalation':
            trace.append('tool:create_support_ticket')
            tool_data = {'ticket_id': tool_create_support_ticket(student_id, message)}
            actions = [{'label': 'Contact organisation', 'type': 'open_contact'}]

        elif intent == 'contact_org':
            actions = [{'label': 'Open email form', 'type': 'open_contact'}]

        trace.append('rag:retrieve_context')
        contexts = self.rag.search(message, domains=['admin'])
        trace.append('llm:huggingface_open_source' if self.llm.enabled else 'llm:fallback_template')
        answer = self._generate_grounded_answer(message, intent, student_id, tool_data, contexts)
        data = {
            **tool_data,
            'rag_mode': self.rag.mode,
            'rag_sources': [
                {
                    'id': c['id'],
                    'title': c['metadata'].get('title', 'Knowledge Base'),
                    'domain': c['metadata'].get('domain', 'admin'),
                    'score': c.get('score', 0.0),
                    'url': c['metadata'].get('url', ''),
                }
                for c in contexts
            ],
            'llm_enabled': self.llm.enabled,
            'llm_model': self.llm.model if self.llm.enabled else None,
        }
        return {'intent': intent, 'answer': answer, 'tool_trace': trace, 'actions': actions, 'data': data}

    def _generate_grounded_answer(self, message: str, intent: str, student_id: str | None, tool_data: dict, contexts: list[dict]) -> str:
        context_text = '\n\n'.join(
            f"[{idx+1}] {ctx['metadata'].get('title', 'Knowledge Base')}\n{ctx['text']}" for idx, ctx in enumerate(contexts)
        )
        system_prompt = (
            'You are StudyNest, an administrative AI agent for student support. '
            'Answer only with grounded information from the supplied tool results and retrieved knowledge base context. '
            'Be concise, helpful, and practical. Never invent university policies.'
        )
        user_prompt = (
            f'User message: {message}\n'
            f'Intent: {intent}\n'
            f'Student ID available: {bool(student_id)}\n'
            f'Tool results: {tool_data}\n'
            f'Retrieved context:\n{context_text}\n\n'
            'Write a natural response. If helpful, end with a next step.'
        )
        llm_result = self.llm.generate_json_or_text(system_prompt, user_prompt)
        if llm_result.get('ok') and llm_result.get('text'):
            return llm_result['text'].strip()
        return self._fallback_answer(intent, tool_data, contexts)

    def _fallback_answer(self, intent: str, tool_data: dict, contexts: list[dict]) -> str:
        title = contexts[0]['metadata'].get('title', 'Knowledge Base') if contexts else 'Knowledge Base'
        if intent == 'check_status':
            student = tool_data.get('student') or {}
            enrollment = tool_data.get('enrollment') or {}
            docs = tool_data.get('missing_documents') or []
            if not student:
                return 'Please log in or share your Student ID so I can check your enrollment status.'
            status = enrollment.get('status', 'not_found')
            if docs:
                doc_lines = '; '.join(f"{d['doc_name']} (due {d['due_date']})" for d in docs)
                return f"{student.get('name', 'Student')}, your current enrollment status is '{status}'. Missing items on file: {doc_lines}. Next step: upload those documents or book an appointment."
            return f"{student.get('name', 'Student')}, your current enrollment status is '{status}'. No missing documents are currently listed."
        if intent == 'missing_documents':
            docs = tool_data.get('missing_documents') or []
            if docs:
                lines = '\n'.join(f"- {d['doc_name']} (due {d['due_date']})" for d in docs)
                return f"These documents are still missing for your account:\n{lines}\nNext step: upload them or book an appointment for document verification."
            return f"Based on the policy context I retrieved from '{title}', the usual enrollment documents include ID or passport, health insurance certificate, digital photo, and residence permit or visa page for international students."
        if intent == 'fee_policy':
            return f"From '{title}', the tuition fee is €3,000, the deadline is 15 October 2026, and a monthly payment plan can be requested from October to February."
        if intent == 'book_appointment':
            booking = tool_data.get('booking', {})
            return f"Your appointment request has been created for {booking.get('slot')} for '{booking.get('purpose')}'. Reference ID: {booking.get('id')}."
        if intent == 'support_escalation':
            return f"I created a support ticket for your issue. Ticket ID: #{tool_data.get('ticket_id')}. Common holds are usually caused by missing insurance, unpaid fees, or identity verification issues."
        if intent == 'contact_org':
            return 'Use the Email Organisation feature to send a further enquiry. Include your full name, student ID, subject, and a short explanation of the issue.'
        body = contexts[0]['text'] if contexts else 'I could not find a matching policy.'
        return f"Here is the most relevant policy context I found from '{title}':\n{body}"
