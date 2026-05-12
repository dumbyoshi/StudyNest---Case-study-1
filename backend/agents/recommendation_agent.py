from __future__ import annotations

from tools import tool_get_learning_catalog, tool_get_peer_profiles, tool_get_student_profile
from rag_engine import RAGEngine
from llm_provider import OpenSourceLLM


class RecommendationAgent:
    name = 'Recommendation AI Agent with RAG'

    def __init__(self) -> None:
        self.rag = RAGEngine()
        self.rag.index_documents()
        self.llm = OpenSourceLLM()

    def handle(self, message: str, intent: str, student_id: str | None) -> dict:
        trace: list[str] = [f'intent:{intent}']
        tool_data: dict = {}
        actions: list[dict] = []

        if student_id:
            trace.append('tool:get_student_profile')
            tool_data['student'] = tool_get_student_profile(student_id)

        trace.append('tool:get_learning_catalog')
        catalog = tool_get_learning_catalog()
        trace.append('tool:get_peer_profiles')
        peers = tool_get_peer_profiles()
        tool_data['catalog_preview'] = catalog[:5]
        tool_data['peer_preview'] = peers[:4]

        domains = ['learning', 'peers'] if intent == 'peer_recommendation' else ['learning', 'peers']
        trace.append('rag:retrieve_context')
        contexts = self.rag.search(message, domains=domains)
        trace.append('llm:huggingface_open_source' if self.llm.enabled else 'llm:fallback_planner')

        answer, structured = self._generate_recommendation(message, intent, tool_data, contexts)
        actions = [
            {'label': 'Show roadmap', 'type': 'suggest', 'payload': 'Give me a 3 month learning roadmap'},
            {'label': 'Project ideas', 'type': 'suggest', 'payload': 'Suggest portfolio projects for my profile'},
            {'label': 'Peer suggestions', 'type': 'suggest', 'payload': 'Recommend peers or mentors for me'},
        ]
        data = {
            **structured,
            'rag_mode': self.rag.mode,
            'rag_sources': [
                {
                    'id': c['id'],
                    'title': c['metadata'].get('title', 'Learning Base'),
                    'domain': c['metadata'].get('domain', 'learning'),
                    'score': c.get('score', 0.0),
                    'url': c['metadata'].get('url', ''),
                }
                for c in contexts
            ],
            'llm_enabled': self.llm.enabled,
            'llm_model': self.llm.model if self.llm.enabled else None,
        }
        return {'intent': intent, 'answer': answer, 'tool_trace': trace, 'actions': actions, 'data': data}

    def _generate_recommendation(self, message: str, intent: str, tool_data: dict, contexts: list[dict]) -> tuple[str, dict]:
        context_text = '\n\n'.join(
            f"[{idx+1}] {ctx['metadata'].get('title', 'Learning Base')}\n{ctx['text']}" for idx, ctx in enumerate(contexts)
        )
        student = tool_data.get('student') or {}
        system_prompt = (
            'You are StudyNest, a recommendation AI agent for student growth. '
            'Use only the student profile and retrieved learning or peer context. '
            'Return a practical roadmap, relevant resources, project ideas, career roles, and peer suggestions when helpful.'
        )
        user_prompt = (
            f'User message: {message}\nIntent: {intent}\n'
            f'Student profile: {student}\nRetrieved context:\n{context_text}\n\n'
            'Write a concise but useful recommendation response.'
        )
        llm_result = self.llm.generate_json_or_text(system_prompt, user_prompt)
        if llm_result.get('ok') and llm_result.get('text'):
            return llm_result['text'].strip(), {'student_profile': student}
        return self._fallback_recommendation(message, intent, student, contexts)

    def _fallback_recommendation(self, message: str, intent: str, student: dict, contexts: list[dict]) -> tuple[str, dict]:
        learning_docs = [c for c in contexts if c['metadata'].get('domain') == 'learning']
        peer_docs = [c for c in contexts if c['metadata'].get('domain') == 'peers']
        top_learning = learning_docs[:3]
        top_peers = peer_docs[:2]

        roadmap = []
        projects = []
        roles = []
        peer_lines = []

        for doc in top_learning:
            title = doc['metadata'].get('title', 'Learning item')
            url = doc['metadata'].get('url', '')
            roadmap.append(f"- {title}: focus on the concepts and finish the linked resource ({url or 'resource in dashboard data'})")
            text = doc['text']
            if 'Projects:' in text:
                proj_part = text.split('Projects:', 1)[1].split('Resource:', 1)[0].strip()
                projects.extend([p.strip() for p in proj_part.split(',') if p.strip()][:2])
            if title not in roles:
                roles.append(title)

        for doc in top_peers:
            peer_lines.append(f"- {doc['metadata'].get('title', 'Peer')}: {doc['text'].split('Reason:',1)[-1].strip()[:120]}")

        if not roadmap:
            roadmap = ['- Start with Python, SQL, and basic statistics, then move to machine learning and one portfolio project.']
        if not projects:
            projects = ['Student risk prediction dashboard', 'Course recommendation system', 'NLP feedback analyser']
        if not roles:
            roles = ['Data Analyst', 'Junior Data Scientist', 'ML Engineer']

        profile_hint = ''
        if student:
            profile_hint = f"For {student.get('name')}, I used your saved interests ({student.get('interests', 'not provided')}) and target role ({student.get('target_role', 'not set')}).\n\n"

        if intent == 'peer_recommendation':
            answer = profile_hint + 'Here are the best peer or mentor matches I found:\n' + ('\n'.join(peer_lines) if peer_lines else '- A peer with similar focus in Python and data science would be a good fit.')
        elif intent == 'project_recommendation':
            answer = profile_hint + 'These portfolio projects fit your current direction:\n' + '\n'.join(f'- {p}' for p in list(dict.fromkeys(projects))[:4])
        else:
            answer = (
                profile_hint + 'Recommended roadmap:\n' + '\n'.join(roadmap[:4]) +
                '\n\nSuggested projects:\n' + '\n'.join(f'- {p}' for p in list(dict.fromkeys(projects))[:4]) +
                '\n\nRelevant roles:\n' + '\n'.join(f'- {r}' for r in list(dict.fromkeys(roles))[:4]) +
                ('\n\nPeer suggestions:\n' + '\n'.join(peer_lines[:3]) if peer_lines else '')
            )
        return answer, {
            'student_profile': student,
            'recommended_projects': list(dict.fromkeys(projects))[:4],
            'recommended_roles': list(dict.fromkeys(roles))[:4],
        }
