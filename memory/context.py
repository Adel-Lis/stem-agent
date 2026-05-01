from dataclasses import dataclass, field


@dataclass
class Context:
    task_class: str
    task_input: str
    history: list[dict] = field(default_factory=list)

    def add(self, node_id: str, role: str, content: str):
        """Saves a node's output to the history so downstream nodes can use it."""
        self.history.append({
            "node_id": node_id,
            "role": role,
            "content": content
        })

    def get_messages(self) -> list[dict]:
        """Returns the history as a simple list of role/content pairs, without the node ids."""
        return [{"role": h["role"], "content": h["content"]} for h in self.history]

    def reset(self):
        """Clears the history, useful if you want to reuse the context for a fresh run."""
        self.history = []