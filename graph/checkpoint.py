import json
import os
from graph.graph import StemGraph


def save_checkpoint(graph: StemGraph, task_class: str, version: int):
    folder = os.path.join("outputs", task_class.lower().replace(" ", "_"))
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, f"checkpoint_v{version}.json")

    with open(filepath, "w") as f:
        f.write(graph.to_json())

    print(f"Checkpoint [v{version}] saved at : {filepath}")

def load_checkpoint(task_class: str, version: int) -> StemGraph:
    folder = os.path.join("outputs", task_class.lower().replace(" ", "_"))
    filepath = os.path.join(folder, f"checkpoint_v{version}.json")

    with open(filepath, "r") as f:
        data = json.load(f)

    return StemGraph.from_dict(data)

def load_latest_checkpoint(task_class: str) -> tuple[StemGraph, int]:
    folder = os.path.join("outputs", task_class.lower().replace(" ", "_"))

    if not os.path.exists(folder):
        return None, 0
    
    checkpoints = [f for f in os.listdir(folder) if f.startswith("checkpoint_v")]

    if not checkpoints:
        return None, 0
    
    versions = [int(f.replace("checkpoint_v", "").replace(".json", "")) for f in checkpoints]
    latest = max(versions)
    return load_checkpoint(task_class, latest), latest

def rollback(task_class: str, current_version: int) -> StemGraph:
    if current_version <= 1:
        raise ValueError("This is the first version. There are no rollback options")
    
    previous_version = current_version - 1
    print(f"Rolling back to version {previous_version}")
    return load_checkpoint(task_class, previous_version)
