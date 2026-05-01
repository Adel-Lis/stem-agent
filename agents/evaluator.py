import json
from openai import OpenAI
from config import MODEL, OPENAI_API_KEY
from graph.graph import StemGraph

openai = OpenAI(api_key=OPENAI_API_KEY)
EVAL_RULE: dict = {}


def define_evaluation(task_class: str) -> dict:
    global EVAL_RULE

    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an evaluator agent. Given a task domain, define a rubric "
                    "to assess whether an AI agent's knowledge graph is correctly specializing for that domain. "
                    "The graph nodes can ONLY have these roles: builder, evaluator, domain_knowledge, strategy, tool. "
                    "The graph nodes can ONLY have these types: prompt, tool. "
                    "Your criteria must ONLY reference these exact roles and types, nothing else. "
                    "Define a maximum of 4 simple criteria. Examples of good criteria: "
                    "'The graph contains at least one strategy node', "
                    "'The graph contains at least two domain_knowledge nodes', "
                    "'All tool type nodes are edge nodes', "
                    "'The graph is fully connected with no isolated nodes'. "
                    "Do NOT invent new node types or roles. Do NOT check for content quality. "
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
            "Graph contains at least one domain knowledge node",
            "Graph contains at least one strategy node",
            "Graph is connected with no isolated nodes",
            "All tool nodes are edge nodes"
        ]}

    return EVAL_RULE

def evaluate(graph: StemGraph, task_class: str, cycle: int) -> tuple[bool, str]:
    global EVAL_RULE

    if not EVAL_RULE:
        define_evaluation(task_class)

    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an evaluator agent. Given a knowledge graph and a rubric, "
                    "assess whether the graph passes all criteria. "
                    "Respond ONLY with a valid JSON object with two fields: "
                    "passed (boolean) and reason (string explaining why it passed or failed). "
                    "Do not include any explanation or markdown."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Task domain: {task_class}\n"
                    f"Cycle: {cycle}\n"
                    f"Evaluation Rules Rubric: {json.dumps(EVAL_RULE)}\n"
                    f"Current graph:\n{graph.to_json()}"
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

