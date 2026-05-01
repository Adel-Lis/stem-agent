from openai import OpenAI
from config import MODEL, OPENAI_API_KEY
from graph.graph import StemGraph
from graph.node import NodeType, NodeRole
from graph.checkpoint import load_latest_checkpoint
from memory.context import Context
from tools.registry import get_tool


openai = OpenAI(api_key=OPENAI_API_KEY)


def execute_prompt_node(node_content: str, context: Context) -> str:
    messages = [{"role": "system", "content": node_content}] + context.get_messages()

    response = openai.chat.completions.create(
        model=MODEL,
        messages=messages
    )

    return response.choices[0].message.content

def execute_tool_node(node_content: str, context: Context) -> str:
    tool_name = node_content.strip()
    last_output = context.history[-1]["content"] if context.history else context.task_input

    try:
        tool = get_tool(tool_name)
        return tool(last_output)
    except Exception as e:
        return f"Tool execution failed, error : {e}"
    
def traverse(graph: StemGraph, task_input: str, task_class: str) -> str:
    context = Context(task_class=task_class, task_input=task_input)
    context.add(node_id="input", role="user", content=task_input)

    order = graph.get_traversal_order()

    for nID in order:
        node = graph.get_node(nID)

        if node.status == "deprecated":
            continue

        if node.role in (NodeRole.builder, NodeRole.evaluator):
            continue

        print(f"[TRAVERSER] Executing node: {node.role} ({node.node_type})")

        if node.node_type == NodeType.prompt:
            output = execute_prompt_node(node.content, context)
        elif node.node_type == NodeType.tool:
            output = execute_tool_node(node.content, context)
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
