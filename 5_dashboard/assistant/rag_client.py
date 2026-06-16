"""
rag_client.py
ChromaDB 1.5.8 HttpClient 기반 벡터 검색
서버: chroma run --path vectordb_v2 --port 8003 (start.bat이 자동 실행)

검색 전략:
1. 약어(대문자 토큰) → glossary 딕셔너리 직접 매칭 (정확도 보장)
2. query_texts로 ChromaDB에 위임 → 임베딩 모델 일치 보장
"""
import re
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

CHROMA_HOST     = "localhost"
CHROMA_PORT     = 8003
COLLECTION_NAME = "hynix_rag"
EMBED_MODEL     = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_client     = None
_collection = None
_ef         = None
_glossary   = {}


def _get_ef():
    global _ef
    if _ef is None:
        _ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return _ef


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        _collection = _client.get_collection(COLLECTION_NAME, embedding_function=_get_ef())
    return _collection


def _load_glossary():
    """glossary 청크를 '괄호 앞 전체 약어(소문자)' → 텍스트 딕셔너리로 로드"""
    global _glossary
    if _glossary:
        return
    try:
        col = _get_collection()
        results = col.get(where={"category": "glossary"}, include=["documents"], limit=10000)
        for doc in results["documents"]:
            m = re.match(r'^([^(]+)\s*\(', doc)
            if m:
                _glossary[m.group(1).strip().lower()] = doc
    except Exception:
        pass


def _glossary_lookup(query: str) -> list[dict]:
    """쿼리에서 영문 토큰을 추출해 glossary 정확 매칭 (긴 N-gram 우선)"""
    _load_glossary()
    hits, seen = [], set()
    tokens = re.findall(r'[A-Za-z][A-Za-z0-9/_\-]*', query)

    candidates = []
    for length in range(min(4, len(tokens)), 0, -1):
        for start in range(len(tokens) - length + 1):
            phrase = " ".join(tokens[start:start + length]).lower()
            candidates.append(phrase)

    for key in candidates:
        if key in _glossary and key not in seen:
            seen.add(key)
            hits.append({
                "text": _glossary[key], "source": "glossary",
                "category": "glossary", "topic": "glossary", "distance": 0.0,
            })
    return hits


def search(query: str, n_results: int = 5, category_filter: str | None = None) -> list[dict]:
    try:
        col = _get_collection()

        # 1. glossary 약어 직접 매칭
        glossary_hits = _glossary_lookup(query) if not category_filter else []

        # 2. ChromaDB 의미 검색 (embedding_function이 자동 임베딩)
        where = {"category": category_filter} if category_filter else None
        fetch = n_results + len(glossary_hits)
        results = col.query(
            query_texts=[query],
            n_results=fetch,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        docs  = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        semantic = [
            {
                "text":     doc,
                "source":   meta.get("title", ""),
                "category": meta.get("category", ""),
                "topic":    meta.get("topic", ""),
                "distance": round(dist, 4),
            }
            for doc, meta, dist in zip(docs, metas, dists)
        ]

        # 3. glossary 우선, 의미 검색으로 채움
        glossary_texts = {h["text"] for h in glossary_hits}
        merged = glossary_hits + [s for s in semantic if s["text"] not in glossary_texts]
        return merged[:n_results]

    except Exception:
        return []


def get_collection_stats() -> dict:
    try:
        col = _get_collection()
        try:
            total = col.count()
        except Exception:
            total = -1
        return {"total_chunks": total, "available": True}
    except Exception:
        return {"total_chunks": 0, "available": False}
