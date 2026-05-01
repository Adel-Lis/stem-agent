import json
import os
from graph.graph import StemGraph
from tools.registry import get_tool_codes, restore_tools_from_codes


def _write_checkpoint(data: dict, filepath: str, label: str):
    """Writes a graph dictionary to a JSON file on disk."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Checkpoint saved: {os.path.basename(filepath)}")

def save_checkpoint(graph: StemGraph, task_class: str, version: int):
    """Saves the current graph as a versioned checkpoint file so progress is not lost between growth cycles."""
    folder = os.path.join("outputs", task_class.lower().replace(" ", "_"))
    os.makedirs(folder, exist_ok=True)

    data = graph.to_inference_dict()
    data["tool_codes"] = get_tool_codes()

    filepath = os.path.join(folder, f"checkpoint_v{version}.json")
    _write_checkpoint(data, filepath, f"v{version}")

def save_final_checkpoint(graph: StemGraph, task_class: str):
    """Saves the fully specialized graph as the final checkpoint, which the traverser will prefer over versioned files when running the agent."""
    folder = os.path.join("outputs", task_class.lower().replace(" ", "_"))
    os.makedirs(folder, exist_ok=True)

    data = graph.to_inference_dict()
    data["tool_codes"] = get_tool_codes()

    filepath = os.path.join(folder, "checkpoint_final.json")
    _write_checkpoint(data, filepath, "final")

def _load_from_file(filepath: str) -> StemGraph:
    """Reads a checkpoint JSON file and rebuilds the graph from it, also restoring any custom tools that were saved."""
    with open(filepath, "r") as f:
        data = json.load(f)
    tool_codes = data.pop("tool_codes", {})
    graph = StemGraph.from_dict(data)
    restore_tools_from_codes(tool_codes)
    return graph

def load_checkpoint(task_class: str, version: int) -> StemGraph:
    """Loads a specific versioned checkpoint for the given domain."""
    folder = os.path.join("outputs", task_class.lower().replace(" ", "_"))
    filepath = os.path.join(folder, f"checkpoint_v{version}.json")
    return _load_from_file(filepath)

def load_latest_checkpoint(task_class: str) -> tuple[StemGraph, int | str]:
    """Loads the best available checkpoint for a domain, it prefers the final one if it exists, otherwise falls back to the highest version number."""
    folder = os.path.join("outputs", task_class.lower().replace(" ", "_"))

    if not os.path.exists(folder):
        return None, 0
    
    # Prefer the fully-specialized checkpoint when it exists
    final_path = os.path.join(folder, "checkpoint_final.json")
    if os.path.exists(final_path):
        print(f"  Loading checkpoint for '{task_class}'")
        return _load_from_file(final_path), "final"

    # Fall back to the highest-versioned intermediate checkpoint
    checkpoints = [f for f in os.listdir(folder) if f.startswith("checkpoint_v")]
    if not checkpoints:
        return None, 0

    versions = [int(f.replace("checkpoint_v", "").replace(".json", "")) for f in checkpoints]
    latest = max(versions)
    return load_checkpoint(task_class, latest), latest

def rollback(task_class: str, current_version: int) -> StemGraph:
    """Loads the checkpoint one version before the current one, used to undo a bad growth step."""
    if current_version <= 1:
        raise ValueError("This is the first version. There are no rollback options")

    previous_version = current_version - 1
    print(f"Rolling back to version {previous_version}")
    return load_checkpoint(task_class, previous_version)
