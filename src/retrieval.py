from __future__ import annotations
from typing import Any, Iterable, Sequence
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

def _source_label(payload: dict[str, Any]) -> str:
    src = payload.get("source")
    # Check if src in metadata
    if not src:
        metadata = payload.get("metadata") or {}
        src = metadata.get("sources") or "UNKNOWN SOURCE"
    if isinstance(src, (list, tuple)): return ", ".join(str(x) for x in src) or "unknown src"
    return str(src)

def multi_level_retrieve(client: QdrantClient,collection_name: str,query_vector: Sequence[float],levels: Iterable[int],*,per_level: int = 2) -> list[Any]:
    if per_level < 1: raise ValueError("per_level retrieval count must be atleast 1")
    ret = []
    st= set()
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

        for p in response.points:
            payload = p.payload or {}
            children = tuple(payload.get("children") or ())
            id = f"Level:{payload.get('level')}:{','.join(children)}" if children else str(payload.get("node_id", p.id))
            if id not in st:
                st.add(id)
                ret.append(p)
    return sorted(ret, key=lambda point: point.score, reverse=True)

def build_answer_prompt(query: str, hits: Sequence[Any]) -> str:
    sections = []
    for index, point in enumerate(hits, start=1):
        payload = point.payload or {}
        src = _source_label(payload)
        page = payload.get("page")
        loc = f"{src}, page {page}" if page is not None else str(src)
        sections.append(f"[Result {index}; RAPTOR level {payload.get('level', '?')}; {loc}]\n{payload.get('text', '')}")
    context = "\n\n".join(sections)
    return (f"Answer the question using the retrieved context below. Prefer specific details supported by the context. If the context does not contain the answer, say that clearly. Do not invent facts.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:")
