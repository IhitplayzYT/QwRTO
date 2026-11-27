from argparse import ArgumentParser
from dataclasses import fields
import json
from pathlib import Path
from typing import Any, Optional, Sequence, TypeVar
from pydantic.dataclasses import dataclass
from .model import ChatbotConf
from .prompt_optim import PromptOptimConf
T = TypeVar("T")


def _load_config(path: str | None, config_type: type[T], name: str) -> T:
    if path is None: return config_type()
    with Path(path).open("r", encoding="utf-8") as config_file: values: Any = json.load(config_file)
    if not isinstance(values, dict): raise ValueError(f"{name} config must contain a JSON object")
    allowed = {field.name for field in fields(config_type)}
    unknown = values.keys() - allowed
    if unknown: raise ValueError(f"Unknown {name} config field(s): {', '.join(sorted(unknown))}")
    if config_type.__name__ == "PromptOptimConf" and "use_multi_query" in values:
        multi_query = values["use_multi_query"]
        if not isinstance(multi_query, (list, tuple)) or len(multi_query) != 3: raise ValueError("prompt optim config use_multi_query must be [enabled, count, limit]")
        values["use_multi_query"] = tuple(multi_query)
    return config_type(**values)

@dataclass
class CLI:
    dim: int = 384
    dir: Optional[str] = None
    query: Optional[str] = None
    collection_name: str = "QwRTO_384"
    chatbot_conf: Any = None
    prompt_optim_conf: Any = None
    new_sess: bool = False

    def parse(self, argv: Optional[Sequence[str]] = None) -> "CLI":
        parser = ArgumentParser(description="Configure a QwRTO query.")
        parser.add_argument("-d", "--dir", "--directory",dest="dir",metavar="PATH",help="Directory containing documents to index.")
        parser.add_argument("-q", "--query",help="Query to run against the collection.")
        parser.add_argument("--dim",type=int,default=self.dim,metavar="N",help=f"Embedding vector dimension (default: {self.dim}).")
        parser.add_argument("-c", "--collection-name", "--collection",dest="collection_name",default=self.collection_name,metavar="NAME",help=f"Vector collection name (default: {self.collection_name}).")
        parser.add_argument("--chatbot-conf", "--chatbot-config", dest="chatbot_conf_path", metavar="JSON", help="Path to a JSON ChatbotConf configuration.")
        parser.add_argument("--prompt-optim-conf", "--prompt-optim-config", dest="prompt_optim_conf_path", metavar="JSON", help="Path to a JSON PromptOptimConf configuration.")
        parser.add_argument("--new","--new-sess","-n",default=False,help="If provided dumps all the points in the collection before upserting",type=bool,dest="new_sess")
        args = parser.parse_args(argv)
        if args.dim <= 0: parser.error("--dim must be a positive integer")

        try:
            self.chatbot_conf = _load_config(args.chatbot_conf_path, ChatbotConf, "chatbot")
            self.prompt_optim_conf = _load_config(args.prompt_optim_conf_path, PromptOptimConf, "prompt optim")
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            parser.error(str(error))
        self.new_sess = args.new_sess
        self.dim = args.dim
        self.dir = args.dir
        self.query = args.query
        self.collection_name = args.collection_name
        return self
