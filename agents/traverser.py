from openai import OpenAI
from config import MODEL, OPENAI_API_KEY
from graph.graph import StemGraph
from graph.node import NodeType, NodeRole
from graph.edge import EdgeRelation
from graph.checkpoint import load_latest_checkpoint
from memory.context import Context
from tools.registry import get_tool


openai = OpenAI(api_key=OPENAI_API_KEY)


def get_primary_input(graph: StemGraph, node_id: str, context: Context) -> str:
    """Finds what data to pass into a node, it uses the output of its upstream nodes if connected, otherwise falls back to the original user input."""
    feeds_into_sources = [
        e.source_id for e in graph.edges.values()
        if e.target_id == node_id and e.relation == EdgeRelation.feeds_into
    ]
    if feeds_into_sources:
        source_outputs = [
            h["content"] for h in context.history
            if h.get("node_id") in feeds_into_sources
        ]
        if source_outputs:
            combined = "\n\n".join(source_outputs)
            combined += f"\n\n---\nOriginal user input:\n{context.task_input}"
            return combined
    # Fall back to the most recent node output, then the raw task input
    return context.history[-1]["content"] if context.history else context.task_input


def is_contradicted(graph: StemGraph, node_id: str) -> bool:
    """Returns True if another node has marked this node as contradicted, meaning it should be skipped during traversal."""
    return any(
        e.target_id == node_id and e.relation == EdgeRelation.contradicts
        for e in graph.edges.values()
    )


def execute_prompt_node(node_content: str, primary_input: str) -> str:
    """Runs a prompt node by sending its content as the system prompt and the input data as the user message, then returns the LLM response."""
    messages = [
        {"role": "system", "content": node_content},
        {"role": "user", "content": primary_input},
    ]
    response = openai.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content


def execute_tool_node(node_content: str, primary_input: str) -> str:
    """Looks up the tool by name in the registry and calls it with the input data, returning its output as a string."""
    tool_name = node_content.strip()
    try:
        tool = get_tool(tool_name)
        return tool(primary_input)
    except Exception as e:
        return f"Tool execution failed, error: {e}"


def traverse(graph: StemGraph, task_input: str, task_class: str) -> str:
    """Walks through the graph in order, executes each node with the right input, and returns the final node's output as the agent's answer."""
    context = Context(task_class=task_class, task_input=task_input)

    order = graph.get_traversal_order()

    for nID in order:
        node = graph.get_node(nID)

        if node.status == "deprecated":
            continue

        if node.role in (NodeRole.builder, NodeRole.evaluator):
            continue

        if is_contradicted(graph, nID):
            print(f"Skipping {node.role.value} — contradicted")
            continue

        if node.node_type == NodeType.tool:
            tool_label = node.content.strip().split('\n')[0].replace('def ', '').split('(')[0]
            print(f"[tool] {tool_label}")
        elif node.role == NodeRole.domain_kn:
            print(f"[domain_knowledge] {node.content[:80].rstrip()}...")
        else:
            print(f"[strategy] {node.content[:100].rstrip()}...")

        primary_input = get_primary_input(graph, nID, context)

        if node.node_type == NodeType.prompt:
            output = execute_prompt_node(node.content, primary_input)
        elif node.node_type == NodeType.tool:
            output = execute_tool_node(node.content, primary_input)
        else:
            continue

        context.add(node_id=node.id, role="assistant", content=output)

    final = context.history[-1]["content"] if context.history else "No output made"
    return final


def run_traverser(task_class: str, task_input: str) -> str:
    """Loads the saved checkpoint for a domain and runs the specialized agent on the given input."""
    graph, _ = load_latest_checkpoint(task_class)

    if graph is None:
        raise ValueError(f"No checkpoint found for task class '{task_class}'. Run the builder first.")

    return traverse(graph, task_input, task_class)
