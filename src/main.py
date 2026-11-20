from prompt_optim import llm
from src.chunking import RaptorConfig
from src.chunking import build_raptor_tree
from qdrant_client.conversions.common_types import Points
from sqlalchemy import false
from langchain_openai import ChatOpenAI 
from qdrant_client import QdrantClient 
from qdrant_client.models import models
from sentence_transformers import SentenceTransformer
try:
    from .helper import CLI
    from .ingestor import load_root
    from .prompt_optim import purify
except ImportError:
    from helper import CLI
    from ingestor import load_root
    from prompt_optim import purify

if __name__ == "__main__":
    clargs = CLI().parse()
    QUERY = clargs.query
    DIR = clargs.dir
    DIM = clargs.dim
    COLLECTION = clargs.collection_name
    ENCODER = "all-MiniLM-L6-v2"
    conn = QdrantClient(host="localhost",grpc_port=6334,port=6333,prefer_grpc=True)
    pconf = clargs.prompt_optim_conf
    cconf = clargs.chatbot_conf
    purified = purify(QUERY,pconf,cconf)

    
    if not conn.collection_exists(COLLECTION): 
        # Create the configuration for the collection
        conn.create_collection(
            collection_name=COLLECTION,
            optimizers_config=models.OptimizersConfig(prevent_unoptimized=True,max_optimization_threads=4),
            vectors_config=models.VectorParams(size=DIM,distance=models.Distance.COSINE),
            hnsw_config=models.HnswConfig(m=32,ef_construct=256,hnsw_ef=256)
        )
    encoder = SentenceTransformer(ENCODER)
    doc_map = load_root(DIR)
    root = build_raptor_tree([doc for file in doc_map.values() for doc in file],embed=encoder,summarize=lambda text: llm(text, pconf, cconf),config=RaptorConfig())

def encode(s:str) -> list[float]:
    return encoder.encode(s,normalize_embeddings=True)
