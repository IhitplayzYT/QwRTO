from qdrant_client.conversions.common_types import Points
from sqlalchemy import false
from langchain_openai import ChatOpenAI 
from qdrant_client import QdrantClient 
from qdrant_client.models import models
from helper import CLI

clargs = CLI.parse()
QUERY = clargs.query
DIR = clargs.dir
DIM = clargs.dim
COLLECTION = clargs.collection_name

conn = QdrantClient(host="localhost",grpc_port=6334,port=6333,prefer_grpc=True)
if not conn.collection_exists(COLLECTION): 
    # Create the configuration for the collection
    conn.create_collection(
        collection_name=COLLECTION,
        optimizers_config=models.OptimizersConfig(prevent_unoptimized=True,max_optimization_threads=4),
        vectors_config=models.VectorParams(size=DIM,distance=models.Distance.COSINE),
        hnsw_config=models.HnswConfig(m=32,ef_construct=256,hnsw_ef=256)
    )






