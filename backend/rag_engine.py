from __future__ import annotations

import json
import os
from typing import Any

from config import settings

APP_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(APP_DIR, 'data')
RAG_DIR = os.path.join(DATA_DIR, 'rag_store')
SOURCES = {
    'admin': os.path.join(DATA_DIR, 'kb_admin.json'),
    'learning': os.path.join(DATA_DIR, 'learning_catalog.json'),
    'peers': os.path.join(DATA_DIR, 'peer_profiles.json'),
}


class RAGEngine:
    def __init__(self) -> None:
        self.mode = 'lexical'
        self.collection = None
        self.embedder = None
        self._init_backends()

    def _init_backends(self) -> None:
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer

            os.makedirs(RAG_DIR, exist_ok=True)
            client = chromadb.PersistentClient(path=RAG_DIR)
            self.collection = client.get_or_create_collection(name=settings.rag_collection)
            self.embedder = SentenceTransformer(settings.embedding_model)
            self.mode = 'vector'
        except Exception:
            self.collection = None
            self.embedder = None
            self.mode = 'lexical'

    def load_documents(self) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []
        for domain, path in SOURCES.items():
            with open(path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            for item in raw:
                item = dict(item)
                item.setdefault('tags', [])
                item.setdefault('title', item.get('name', item.get('role', 'Knowledge Base')))
                item['domain'] = domain
                item['id'] = item.get('id') or f"{domain}-{len(docs)+1}"
                docs.append(item)
        return docs

    def _doc_to_text(self, doc: dict[str, Any]) -> str:
        if doc['domain'] == 'admin':
            body = doc.get('body', '')
        elif doc['domain'] == 'learning':
            body = (
                f"Level: {doc.get('level')}\nSkills: {', '.join(doc.get('skills', []))}\n"
                f"Summary: {doc.get('summary', '')}\n"
                f"Projects: {', '.join(doc.get('projects', []))}\n"
                f"Resource: {doc.get('resource_url', '')}"
            )
        else:
            body = (
                f"Level: {doc.get('level')}\nFocus: {', '.join(doc.get('focus', []))}\n"
                f"Strengths: {doc.get('strengths', '')}\n"
                f"Reason: {doc.get('reason', '')}"
            )
        return f"Domain: {doc['domain']}\nTitle: {doc['title']}\nTags: {', '.join(doc.get('tags', []))}\nBody: {body}"

    def index_documents(self) -> dict[str, Any]:
        docs = self.load_documents()
        if self.mode != 'vector' or self.collection is None or self.embedder is None:
            return {'ok': True, 'mode': self.mode, 'indexed': len(docs)}

        try:
            existing = self.collection.get(include=[])
            existing_ids = set(existing.get('ids', []))
        except Exception:
            existing_ids = set()

        new_docs = [d for d in docs if d['id'] not in existing_ids]
        if not new_docs:
            return {'ok': True, 'mode': self.mode, 'indexed': 0, 'total': len(docs)}

        texts = [self._doc_to_text(d) for d in new_docs]
        embeddings = self.embedder.encode(texts).tolist()
        metadatas = [
            {
                'title': d['title'],
                'domain': d['domain'],
                'url': d.get('url', d.get('resource_url', '')),
                'tags': ', '.join(d.get('tags', [])),
            }
            for d in new_docs
        ]
        self.collection.add(
            ids=[d['id'] for d in new_docs],
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return {'ok': True, 'mode': self.mode, 'indexed': len(new_docs), 'total': len(docs)}

    def search(self, query: str, top_k: int | None = None, domains: list[str] | None = None) -> list[dict[str, Any]]:
        k = top_k or settings.top_k
        docs = self.load_documents()
        if domains:
            docs = [d for d in docs if d['domain'] in domains]

        if self.mode == 'vector' and self.collection is not None and self.embedder is not None:
            query_embedding = self.embedder.encode([query]).tolist()[0]
            where = {'domain': {'$in': domains}} if domains else None
            results = self.collection.query(query_embeddings=[query_embedding], n_results=k, where=where)
            ids = results.get('ids', [[]])[0]
            distances = results.get('distances', [[]])[0]
            metadatas = results.get('metadatas', [[]])[0]
            documents = results.get('documents', [[]])[0]
            out: list[dict[str, Any]] = []
            for idx, doc_id in enumerate(ids):
                out.append(
                    {
                        'id': doc_id,
                        'text': documents[idx],
                        'score': float(1 / (1 + distances[idx])) if idx < len(distances) else 0.0,
                        'metadata': metadatas[idx] if idx < len(metadatas) else {},
                    }
                )
            return out

        query_tokens = set(query.lower().split())
        scored: list[dict[str, Any]] = []
        for item in docs:
            text = self._doc_to_text(item).lower()
            score = sum(1 for token in query_tokens if token in text)
            if score > 0:
                scored.append(
                    {
                        'id': item['id'],
                        'text': self._doc_to_text(item),
                        'score': float(score),
                        'metadata': {
                            'title': item['title'],
                            'domain': item['domain'],
                            'url': item.get('url', item.get('resource_url', '')),
                            'tags': ', '.join(item.get('tags', [])),
                        },
                    }
                )
        scored.sort(key=lambda x: x['score'], reverse=True)
        return scored[:k]
