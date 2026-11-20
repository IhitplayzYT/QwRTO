from dataclasses import dataclass

@dataclass
class ChatbotConf:
    provider: str = ""
    model:str = ""
    temp: float = 0.0
    max_tokens: int = 1000