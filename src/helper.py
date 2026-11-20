from argparse import ArgumentParser
from typing import Optional, Sequence
from pydantic.dataclasses import dataclass

@dataclass
class CLI:
    dim: int = 768
    dir: Optional[str] = None
    query: Optional[str] = None
    collection_name: str = "QwRTO"

    def parse(self, argv: Optional[Sequence[str]] = None) -> "CLI":
        parser = ArgumentParser(description="Configure a QwRTO query.")
        parser.add_argument("-d", "--dir", "--directory",dest="dir",metavar="PATH",help="Directory containing documents to index.")
        parser.add_argument("-q", "--query",help="Query to run against the collection.")
        parser.add_argument("--dim",type=int,default=self.dim,metavar="N",help=f"Embedding vector dimension (default: {self.dim}).")
        parser.add_argument("-c", "--collection-name", "--collection",dest="collection_name",default=self.collection_name,metavar="NAME",help=f"Vector collection name (default: {self.collection_name}).")
        args = parser.parse_args(argv)
        if args.dim <= 0: parser.error("--dim must be a positive integer")
        self.dim = args.dim
        self.dir = args.dir
        self.query = args.query
        self.collection_name = args.collection_name
        return self
