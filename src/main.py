import uuid
from qdrant_client import QdrantClient
from qdrant_client.conversions.common_types import PointStruct
from qdrant_client.models import models
from sentence_transformers import SentenceTransformer

from .chunking import RaptorConfig, build_raptor_tree
from .helper import CLI
from .ingestor import load_root
from .prompt_optim import llm
from .retrieval import build_answer_prompt, multi_level_retrieve
from .rerank import rerank

if __name__ == "__main__":
    clargs = CLI().parse()
    DIR = clargs.dir
    DIM = clargs.dim
    COLLECTION = clargs.collection_name
    ENCODER = "all-MiniLM-L6-v2"
    conn = QdrantClient(host="localhost",grpc_port=6334,port=6333,prefer_grpc=True)
    pconf = clargs.prompt_optim_conf
    cconf = clargs.chatbot_conf
    doc_map = load_root(DIR)
    docs = [doc for file_documents in doc_map.values() for doc in file_documents]
    if not any(doc.text.strip() for doc in docs): raise SystemExit(f"No text found in {DIR!r}")
    encoder = SentenceTransformer(ENCODER)
    encoder_dim = encoder.get_embedding_dimension()
    if DIM != encoder_dim: raise SystemExit(f"Configured vector dim: {DIM}, but {ENCODER}: {encoder_dim}")
    if clargs.new_sess: conn.delete_collection(COLLECTION)
    if not conn.collection_exists(COLLECTION): 
        # Create the configuration for the collection
        conn.create_collection(
            collection_name=COLLECTION,
            optimizers_config=models.OptimizersConfigDiff(prevent_unoptimized=True,max_optimization_threads=4,default_segment_number=0,flush_interval_sec=5),
            vectors_config=models.VectorParams(size=DIM,distance=models.Distance.COSINE),
            hnsw_config=models.HnswConfigDiff(m=32,ef_construct=256,full_scan_threshold=10000)
        )
    root = build_raptor_tree(docs,embed=lambda texts: encoder.encode(texts, normalize_embeddings=True),summarize=lambda text: llm(text, pconf, cconf),config=RaptorConfig())
    points = []
    for node in root.nodes:
        vec = encoder.encode(node.text,normalize_embeddings=True).tolist()
        points.append(PointStruct(
            id=str(uuid.uuid7()),
            vector=vec,
            payload={
                "node_id": node.id,
                "text": node.text,
                "level": node.level,
                "children": list(node.children),
                "source": node.source,
                "page": node.page,
                "metadata": node.metadata,
            }
        ))

    if not points: raise SystemExit(f"No chunks produced")
    print(f"No of chunks in {DIR}: {len(points)}")
    conn.upsert(collection_name=COLLECTION,points=points)
    print(f"Total chunks in {COLLECTION} after adding: {conn.count(COLLECTION)}")

    if clargs.query:
        query_vec = encoder.encode(clargs.query, normalize_embeddings=True).tolist()
        retrieved = multi_level_retrieve(conn,COLLECTION,query_vec,(n.level for n in root.nodes))
        retrieved = rerank(clargs.query, retrieved, top_k=10)
        if not retrieved: print("No matching context found for that query.")
        
        else:
            answer = llm(build_answer_prompt(clargs.query, retrieved), pconf, cconf)
            print(f"\nAnswer:\n{answer}")
            print("\nSources:")
            for p in retrieved:
                payload = p.payload or {}
                page = f", page {payload['page']}" if payload.get("page") is not None else ""
                meta = payload.get("metadata") or {}
                src = payload.get("source") or meta.get("sources") or "unknown source"
                if isinstance(src, (list, tuple)): src = ", ".join(str(x) for x in src)
                print(f"- {src}{page} (level {payload.get('level', '?')})")

def encode(s:str) -> list[float]: return encoder.encode(s,normalize_embeddings=True)
