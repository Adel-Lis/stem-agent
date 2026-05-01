import json
from openai import OpenAI
from config import MODEL_EVAL, OPENAI_API_KEY
from graph.graph import StemGraph
from graph.node import Node

openai = OpenAI(api_key=OPENAI_API_KEY)
EVAL_RULE: dict = {}


def define_evaluation(task_class: str) -> dict:
    global EVAL_RULE

    response = openai.chat.completions.create(
        model=MODEL_EVAL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an evaluator agent. Given a task domain, define a rubric "
                    "to assess whether an AI agent's knowledge graph is fully specialized for that domain. "
                    "The graph nodes can ONLY have these roles: builder, evaluator, domain_knowledge, strategy, tool. "
                    "The graph nodes can ONLY have these types: prompt, tool. "
                    "You MUST include these four criteria exactly as written: "
                    "'The graph contains at least one tool type node', "
                    "'All strategy node content is an actionable LLM instruction (begins with Given/Analyze/Extract/You have received/etc.) and not a general description', "
                    "'Every tool node has at least one outgoing feeds_into edge to a strategy node (tool nodes are never the last node in the pipeline)', "
                    "'The terminal strategy node (the strategy node with no outgoing feeds_into edges) instructs natural language output and does not say Output as JSON or Return a JSON object'. "
                    "Do NOT add any further criteria beyond these four. "
                    "Do NOT invent new node types or roles. "
                    "Respond ONLY with a valid JSON object with one field: "
                    "criteria (a list of strings). No explanation or markdown."
                )
            },
            {
                "role": "user",
                "content": f"Task domain: {task_class}"
            }
        ]
    )
    raw = response.choices[0].message.content.strip()

    try:
        EVAL_RULE = json.loads(raw)
        print(f"[EVALUATOR] Rubric is devined: {EVAL_RULE}")
    except json.JSONDecodeError:
        print("Failed to parse rubric, using default")
        EVAL_RULE = {"criteria": [
            "The graph contains at least one strategy node",
            "The graph contains at least two domain_knowledge nodes",
            "All tool type nodes are edge nodes",
            "The graph is fully connected with no isolated nodes"
        ]}

    return EVAL_RULE

def evaluate(graph: StemGraph, task_class: str, cycle: int, new_node: Node) -> tuple[bool, str]:
    global EVAL_RULE

    if not EVAL_RULE:
        define_evaluation(task_class)

    response = openai.chat.completions.create(
        model=MODEL_EVAL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an evaluator agent. Your only job is to assess whether "
                    "the node provided below is a valid and useful addition to the graph. "
                    "Do NOT check whether the full graph is complete or satisfies all criteria. "
                    "Check ALL of the following and fail if any one does not hold: "
                    "1. The node has a valid type (prompt or tool) and a valid role (domain_knowledge, strategy, or tool). "
                    "2. The node is connected to at least one other node in the graph. "
                    "3. If the node is type 'prompt': its content must be an actionable LLM instruction — "
                    "it must tell the model what to do with an input (e.g. begins with Given/Analyze/Extract/You have received/etc.). "
                    "REJECT if the content reads like a general description, methodology overview, or advice. "
                    "4. If the node is type 'tool': its content must be either an existing tool name or valid Python code. "
                    "5. If the node is a strategy node and has no outgoing feeds_into edges in the full graph (it is the terminal node the user will read): "
                    "its content must NOT instruct JSON output. REJECT if the content contains phrases like 'Output as JSON', 'Return a JSON object', 'Output a JSON array', or similar. "
                    "Respond ONLY with a valid JSON object with two fields: "
                    "passed (boolean) and reason (string). No explanation or markdown."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Task domain: {task_class}\n"
                    f"Cycle: {cycle}\n"
                    f"Node to evaluate:\n{json.dumps(new_node.model_dump(), indent=2)}\n"
                    f"Full graph context:\n{graph.to_json()}"
                )
            }
        ]
    )
    raw = response.choices[0].message.content.strip()

    try:
        result = json.loads(raw)
        passed = result.get("passed", False)
        reason = result.get("reason", "No reason provided")
        return passed, reason
    except json.JSONDecodeError:
        return False, "Evaluator produced invalid response"

def get_eval_rule() -> dict:
    return EVAL_RULE

