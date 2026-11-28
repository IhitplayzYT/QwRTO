from __future__ import annotations
from typing import Any, Iterable, Sequence
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

def src_label(payload: dict[str, Any]) -> str:
    src = payload.get("source")
    # Check if src in metadata
    if not src:
        metadata = payload.get("metadata") or {}
        src = metadata.get("sources") or "UNKNOWN SOURCE"
    if isinstance(src, (list, tuple)): return ", ".join(str(x) for x in src) or "unknown src"
    return str(src)

def multi_level_retrieve(client: QdrantClient,collection_name: str,query_vector: Sequence[float],levels: Iterable[int],*,per_level: int = 2,rrf_k: int = 60) -> list[Any]:
    if per_level < 1: raise ValueError("per_level retrieval count must be at least 1")
    if rrf_k < 0: raise ValueError("rrf_k must be positive and greater then 0")
    id_map  = {}
    rrf= {}
    for level in sorted(set(levels), reverse=True):
        response = client.query_points(
            collection_name=collection_name,
            query=list(query_vector),
            query_filter=Filter(
                must=[FieldCondition(key="level", match=MatchValue(value=level))]
            ),
            limit=per_level,
            with_payload=True,
            with_vectors=False,
        )

        seen = set()
        rank = 0
        for point in response.points:
            payload = point.payload or {}
            id = str(payload.get("node_id") or point.id)
            if id in seen: continue
            seen.add(id)
            rank += 1
            id_map.setdefault(id, point)
            rrf[id] = rrf.get(id, 0.0) + 1.0 / (rrf_k + rank)

    ranked = sorted(rrf, key=lambda id: rrf[id], reverse=True)
    ret = []
    for id in ranked:
        point = id_map[id]
        score = rrf[id]
        if hasattr(point, "model_copy"): point = point.model_copy(update={"score": score})
        elif hasattr(point, "copy"): point = point.copy(update={"score": score})
        else: point.score = score
        ret.append(point)
    return ret

def build_answer_prompt(query: str, hits: Sequence[Any]) -> str:
    sections = []
    for index, point in enumerate(hits, start=1):
        payload = point.payload or {}
        src = src_label(payload)
        page = payload.get("page")
        loc = f"{src}, page {page}" if page is not None else str(src)
        sections.append(f"[Result {index}; RAPTOR level {payload.get('level', '?')}; {loc}]\n{payload.get('text', '')}")
    context = "\n\n".join(sections)
    return (f"Answer the question using the retrieved context below. Prefer specific details supported by the context. If the context does not contain the answer, say that clearly. Do not invent facts.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:")
