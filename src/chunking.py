from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import math
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

@dataclass
class RaptorConfig:
    chunk_size: int = 400
    chunk_overlap: int = 60
    max_depth: int = 3
    similarity_thresh: float = 0.45
    max_cluster_size: int = 8

    def __post_init__(self) -> None:
        if self.chunk_size < 1: raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size: raise ValueError("chunk_overlap must be positive and smaller then the chunk_size")
        if self.max_depth < 0: raise ValueError("max_depth must be positive")
        if not -1.0 <= self.similarity_thresh <= 1.0: raise ValueError("similarity_thresh must be in [-1,1]")
        if self.max_cluster_size < 2: raise ValueError("max_cluster_size must be atleast 2")

@dataclass
class RaptorNode:
    id: str
    text: str
    level: int
    source: str | None = None
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    children: tuple[str, ...] = ()

    @property
    def is_leaf(self) -> bool: return not self.children


@dataclass
class RaptorTree:
    nodes: list[RaptorNode]
    roots: list[str]

    @property
    def leaves(self) -> list[RaptorNode]: return [node for node in self.nodes if node.is_leaf]
    def by_id(self) -> dict[str, RaptorNode]: return {node.id: node for node in self.nodes}

# Use overlap based chunking over text  
def split_text(text: str, *, chunk_size: int = 400, chunk_overlap: int = 60) -> list[str]:
    if chunk_size < 1: raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size: raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")
    spans = list(re.finditer(r"\S+", text))
    if not spans: return []
    chunks = []
    for st in range(0, len(spans), chunk_size - chunk_overlap):
        ed = min(st + chunk_size, len(spans))
        if ed == len(spans): break
        x = text[spans[st].start():spans[ed - 1].end()].strip()
        if x: chunks.append(x)
    return chunks


def doc_vals(document: Any) -> tuple[str, str | None, int | None, dict[str, Any]]:
    if isinstance(document, Mapping):
        text = document.get("text", "")
        src = document.get("source")
        page = document.get("page")
        metadata = document.get("metadata", {})
    else:
        text = getattr(document, "text", "")
        src = getattr(document, "source", None)
        page = getattr(document, "page", None)
        metadata = getattr(document, "metadata", {})
    if not isinstance(text, str): raise TypeError("Document text must be a string")
    if src is not None and not isinstance(src, str): src = str(src)
    if page is not None: page = int(page)
    return text, src, page, dict(metadata) if isinstance(metadata, Mapping) else {}


# Use the Documents params to generate randomised ids
def gen_id(level: int, source: str | None, page: int | None, ordinal: int, text: str) -> str: return hashlib.sha256(f"{level}\0{source or ''}\0{page or ''}\0{ordinal}\0{hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]}".encode("utf-8")).hexdigest()[:24]

# Create the lowest level leaf nodes from each doxcuemnt with each node havin gthe resp chunk and doc idx in metadata of node
def chunk_documents(documents: Iterable[Any],*,chunk_size: int = 400,chunk_overlap: int = 60) -> list[RaptorNode]:
    leaves: list[RaptorNode] = []
    for document_index, document in enumerate(documents):
        text,source,page,metadata = document, None, None, {} if isinstance(document,str) else doc_vals(document)
        for chunk_index, chunk in enumerate(split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)):
            metadata = dict(metadata)
            metadata.update({"document_index": document_index, "chunk_index": chunk_index})
            leaves.append(RaptorNode(id=gen_id(0, source, page, len(leaves), chunk),text=chunk,level=0,source=source,page=page,metadata=metadata))
    return leaves

# Embed texts using the Embedding fn
def embed_texts(texts: Sequence[str], embed) -> list[list[float]]:
    if not texts: return []
    x = embed(list(texts))
    vec = [[float(val) for val in vec] for vec in x]
    if len(vec) != len(texts): raise ValueError(f"Embedding fn returned {len(vec)} vecs but there were {len(texts)} texts")
    if not vec or not vec[0]: raise ValueError("Empty embed vector")
    dim = len(vec[0])
    if any(len(vector) != dim for vector in vec): raise ValueError("All embed vectors need to have the samer rdim lklen")
    if any(not math.isfinite(value) for vector in vec for value in vector): raise ValueError("Embed vec vals have to be finite")
    return vec


def cosine_sim(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b): raise ValueError("Vectors must have same dims")
    an = math.sqrt(sum(float(v)**2 for v in a))
    bn = math.sqrt(sum(float(v)**2 for v in b))
    if not an or not bn: return 0.0
    return sum(float(a) * float(b) for a, b in zip(a, b)) / (an * bn)

# KNN style clustering of embeddings
def cluster_embeddings(vectors: Sequence[Sequence[float]],*,similarity_threshold: float = 0.45,max_cluster_size: int = 8) -> list[list[int]]:
    if not -1.0 <= similarity_threshold <= 1.0: raise ValueError("similarity_threshold must be in [-1,1]")
    if max_cluster_size < 2: raise ValueError("max_cluster_size must be at least 2")
    if not vectors: return []
    dimension = len(vectors[0])
    if dimension == 0 or any(len(vector) != dimension for vector in vectors): raise ValueError("All vectors must have same non zero dim")
    clusters: list[list[int]] = []
    centroids: list[list[float]] = []
    for i, vector in enumerate(vectors):
        candidates = [(cosine_sim(vector, centroid), cluster_idx) for cluster_idx, (cluster, centroid) in enumerate(zip(clusters, centroids)) if len(cluster) < max_cluster_size]
        best_sim,best_clust = max(candidates, key=lambda item: (item[0], -item[1])) if candidates else -math.inf, -1
        if best_sim >= similarity_threshold:
            cluster = clusters[best_clust]
            cluster.append(i)
            n = len(cluster) - 1
            centroids[best_clust] = [(centroids[best_clust][x] * n + float(vector[x])) / len(cluster) for x in range(dimension)]
        else:
            clusters.append([i])
            centroids.append([float(value) for value in vector])
    return clusters

def summarize_cluster(texts: Sequence[str], summarize) -> str:
    if not texts: raise ValueError("No summary empty cluster")
    ret = summarize("\n\n".join(text.strip() for text in texts if text.strip()))
    if not isinstance(ret, str) or not ret.strip(): raise ValueError("Emopty summary")
    return ret.strip()


def build_raptor_tree(documents: Iterable[Any],*,embed,summarize,config: RaptorConfig | None = None) -> RaptorTree:
    conf = config or RaptorConfig()
    nodes = chunk_documents(documents,chunk_size=conf.chunk_size,chunk_overlap=conf.chunk_overlap)
    active = list(nodes)
    for level in range(1, conf.max_depth + 1):
        if len(active) < 2: break
        groups = cluster_embeddings(embed_texts([node.text for node in active], embed),similarity_threshold=conf.similarity_threshold,max_cluster_size=conf.max_cluster_size)
        nxt: list[RaptorNode] = []
        merged = False
        for g in groups:
            children = [active[i] for i in g]
            if len(children) < 2:
                nxt.extend(children)
                continue
            merged = True
            summary = summarize_cluster([child.text for child in children], summarize)
            child_ids = tuple(child.id for child in children)
            parent_id = gen_id(level, None, None, len(nodes), summary + "\0".join(child_ids))
            parent = RaptorNode(id=parent_id,text=summary,level=level,metadata={"child_count": len(children), "sources": sorted({child.source for child in children if child.source})},children=child_ids)
            nodes.append(parent)
            nxt.append(parent)
        if not merged: break
        active = nxt 
    return RaptorTree(nodes=nodes, roots=[node.id for node in active])

