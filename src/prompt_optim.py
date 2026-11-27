from dataclasses import dataclass
from .model import ChatbotConf
from langchain.chat_models import init_chat_model
import json

@dataclass
class PromptOptimConf():
    use_hyde: bool = True
    use_multi_query: tuple[bool,int,int] = (False,4,5)
    use_query2doc: bool = True
    use_stepback: bool = True
    use_decompose: int = 5
    use_qexpand: int = 5

def purify(query:str,pconf:PromptOptimConf,cconf:ChatbotConf) -> dict[str,str]:
    ret:dict[str,str|list[str]] = {"prompt":query}
    if pconf.use_hyde:
        ret["hyde"] = llm(f"Do NOT include any greetings or any other things.\nSimply write the answer to the query.\nWrite a short factual para that could answer the query: \"{query}\"",pconf,cconf)
    if pconf.use_multi_query[0]:
        js = json.loads(llm(f"Do NOT include any greetings or any other things.\nSimply write the answer to the query.\n  Return JSON with key \"queries\" containing{pconf.use_multi_query[0]} semantically diverse search queries for the query:\"{query}\".\nTry to preserve original intent",pconf,cconf))
        ret["multi_query"] = [query] +  [str(x) for x in js] if isinstance(js, list) else [str(x) for x in js["queries"]]
    if pconf.use_query2doc:
        ret["query2doc"] = query + "\n\n"+ llm(f"Do NOT include any greetings or any other things.\nSimply write the answer to the query.\n  Generate a short technical passage relevant to this search query:\"{query}\"\n Include useful terminology and explanations",pconf,cconf)
    if pconf.use_stepback:
        ret["step_back"] = llm(f"Do NOT include any greetings or any other things.\nSimply write the answer to the query.\nRewrite this question into one broader\n  conceptual question that would help answer it.\n  Original: \"{query}\".\n    Return only the broader question",pconf,cconf)
    if pconf.use_decompose:
        js = json.loads(llm(f"Do NOT include any greetings or any other things.\nSimply write the answer to the query.\n Break this complex question into independent\nsearchable subquestions.\nReturn JSON with key \"queries\".\nQuestion: \"{query}\"",pconf,cconf))
        ret["decomposed"] = [str(x) for x in js] if isinstance(js, list) else [str(x) for x in js["queries"]]
    if pconf.use_qexpand:
        js = json.loads(llm(f"Do NOT include any greetings or any other things.\nSimply write the answer to the query.\n     Generate 4 related search formulations using\nsynonyms, abbreviations, and domain terminology.\nPreserve the original meaning.\nReturn JSON with key \"queries\".\nQuery: \"{query}\"",pconf,cconf))
        ret["query_expand"] = [str(x) for x in js] if isinstance(js, list) else [str(x) for x in js["queries"]]
    return ret
    
def llm(prompt:str,pconf:PromptOptimConf,cconf:ChatbotConf) -> str:
    if not cconf.provider.strip() or not cconf.model.strip(): raise ValueError("Chatbot provider and model are required")
    provider_options = {"reasoning": False} if cconf.provider == "ollama" else {}
    return init_chat_model(cconf.model,model_provider=cconf.provider,temperature=cconf.temp,timeout=45,max_retries=3,max_tokens=cconf.max_tokens,**provider_options).invoke(prompt).content




