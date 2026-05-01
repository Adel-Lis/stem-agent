from dataclasses import dataclass, field


@dataclass
class Context:
    task_class: str
    task_input: str
    history: list[dict] = field(default_factory=list)

    def add(self, node_id: str, role: str, content: str):
        self.history.append({
            "node_id": node_id,
            "role": role,
            "content": content
        })

    def get_messages(self) -> list[dict]:
        return [{"role": h["role"], "content": h["content"]} for h in self.history]
    
    def reset(self):
        self.history = []