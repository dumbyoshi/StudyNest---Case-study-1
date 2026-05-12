from __future__ import annotations


def detect_intent(message: str, preferred_mode: str | None = None) -> tuple[str, str]:
    mode = preferred_mode or 'auto'
    m = message.lower()

    admin_keywords = ['status', 'enrollment', 'admission', 'document', 'insurance', 'passport', 'photo', 'permit', 'visa', 'fee', 'deadline', 'payment', 'appointment', 'ticket', 'blocked', 'organisation', 'contact']
    rec_keywords = ['roadmap', 'recommend', 'career', 'skill', 'skills', 'project ideas', 'learning path', 'resource', 'peer', 'mentor', 'course', 'ml engineer', 'data analyst', 'nlp', 'python']

    if mode == 'admin':
        route = 'admin'
    elif mode == 'recommendation':
        route = 'recommendation'
    else:
        admin_hits = sum(1 for x in admin_keywords if x in m)
        rec_hits = sum(1 for x in rec_keywords if x in m)
        route = 'recommendation' if rec_hits > admin_hits else 'admin'

    if route == 'admin':
        if any(x in m for x in ['status', 'enrollment status', 'am i enrolled', 'admission status']):
            return route, 'check_status'
        if any(x in m for x in ['document', 'missing', 'upload', 'insurance', 'passport', 'photo', 'permit', 'visa']):
            return route, 'missing_documents'
        if any(x in m for x in ['fee', 'deadline', 'payment', 'tuition', 'payment plan']):
            return route, 'fee_policy'
        if any(x in m for x in ['book', 'appointment', 'meeting', 'advising', 'schedule']):
            return route, 'book_appointment'
        if any(x in m for x in ['help', 'issue', 'problem', 'blocked', 'hold']):
            return route, 'support_escalation'
        if any(x in m for x in ['email', 'contact', 'enquiry', 'organization', 'organisation']):
            return route, 'contact_org'
        return route, 'general_policy'

    if any(x in m for x in ['peer', 'mentor', 'study partner']):
        return route, 'peer_recommendation'
    if any(x in m for x in ['project', 'portfolio']):
        return route, 'project_recommendation'
    if any(x in m for x in ['career', 'role', 'job', 'roadmap', 'recommend', 'learning path', 'resource', 'skills', 'course']):
        return route, 'learning_recommendation'
    return route, 'learning_recommendation'
