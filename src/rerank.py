from __future__ import annotations
from functools import lru_cache
from typing import Any, Sequence
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny
from sentence_transformers import CrossEncoder

# Suggested by AI for lazy lightweigth loading of crossencoder and caching
@lru_cache(maxsize=1)
def cross_encoder() -> CrossEncoder:
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def expand_children(conn: QdrantClient,nodes: Sequence[Any],collection: str) -> list[Any]:
    ids: set[str] = set()
    for node in nodes:
        payload = node.payload or {}
        children = payload.get("children") or ()
        ids.update(str(child_id) for child_id in children if child_id is not None)
    ret = list(nodes)
    seen = {str(x.id) for x in ret}
    if not ids: return ret

    children, _ = conn.scroll(
        collection_name=collection,
        scroll_filter=Filter(must=[FieldCondition(key="node_id",match=MatchAny(any=sorted(ids)))]),
        limit=len(ids),
        with_payload=True,
        with_vectors=False,
    )

    for c in children:
        id = str(c.id)
        if id not in seen:
            ret.append(c)
            seen.add(id)

    return ret

def rerank(query: str, nodes: Sequence[Any], top_k: int = 10) -> list[Any]:
    if top_k <= 0: raise ValueError("top_k must be positive and non zero")
    if not nodes: return []
    pairs = []
    candidates = []
    for node in nodes:
        payload = node.payload or {}
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip(): continue
        pairs.append((query, text))
        candidates.append(node)
    if not pairs: return []
    reranker = cross_encoder()
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda item: float(item[1]), reverse=True)
    return [node for node, _ in ranked[:top_k]]
