from dataclasses import dataclass

@dataclass
class Context:
    strategy: str = ""
    event: str = ""
    event_description: str = ""
    time: str = ""