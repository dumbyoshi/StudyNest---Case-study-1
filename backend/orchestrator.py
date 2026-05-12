from __future__ import annotations

from agents.admin_agent import AdministrativeAgent
from agents.recommendation_agent import RecommendationAgent
from agents.router import detect_intent


class AgentOrchestrator:
    def __init__(self) -> None:
        self.admin_agent = AdministrativeAgent()
        self.recommendation_agent = RecommendationAgent()

    def run(self, message: str, student_id: str | None = None, preferred_mode: str | None = None) -> dict:
        route, intent = detect_intent(message, preferred_mode)
        if route == 'recommendation':
            result = self.recommendation_agent.handle(message, intent, student_id)
            result['agent_mode'] = 'recommendation-ai-agent-rag'
        else:
            result = self.admin_agent.handle(message, intent, student_id)
            result['agent_mode'] = 'administrative-ai-agent-rag'
        result.setdefault('intent', intent)
        result.setdefault('route', route)
        return result
