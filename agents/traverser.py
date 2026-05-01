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
    return any(
        e.target_id == node_id and e.relation == EdgeRelation.contradicts
        for e in graph.edges.values()
    )


def execute_prompt_node(node_content: str, primary_input: str) -> str:
    messages = [
        {"role": "system", "content": node_content},
        {"role": "user", "content": primary_input},
    ]
    response = openai.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content


def execute_tool_node(node_content: str, primary_input: str) -> str:
    tool_name = node_content.strip()
    try:
        tool = get_tool(tool_name)
        return tool(primary_input)
    except Exception as e:
        return f"Tool execution failed, error: {e}"


def traverse(graph: StemGraph, task_input: str, task_class: str) -> str:
    context = Context(task_class=task_class, task_input=task_input)

    order = graph.get_traversal_order()

    for nID in order:
        node = graph.get_node(nID)

        if node.status == "deprecated":
            continue

        if node.role in (NodeRole.builder, NodeRole.evaluator):
            continue

        if is_contradicted(graph, nID):
            print(f"[TRAVERSER] Skipping node {node.role.value} — incoming contradicts edge signals unresolved conflict")
            continue

        print(f"[TRAVERSER] Executing node: {node.role} ({node.node_type})")

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
    graph, version = load_latest_checkpoint(task_class)

    if graph is None:
        raise ValueError(f"No checkpoint found for task class '{task_class}'. Run the builder first.")

    print(f"[TRAVERSER] Loaded checkpoint version {version} for '{task_class}'")
    print(f"[TRAVERSER] Start traversing")
    return traverse(graph, task_input, task_class)
