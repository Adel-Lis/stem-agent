
import json
from openai import OpenAI
from config import MODEL, MAX_SEARCH_RESULTS, MAX_GROWTH_CYCLES, MIN_GROWTH_CYCLES, OPENAI_API_KEY
from graph.graph import StemGraph
from graph.node import Node, NodeType, NodeRole
from graph.edge import Edge, EdgeRelation
from graph.checkpoint import save_checkpoint
from tools.registry import web_search_raw, format_search_results, register_tool_from_code, list_tools
from agents.evaluator import evaluate, define_evaluation, get_eval_rule


openai = OpenAI(api_key=OPENAI_API_KEY)


def birth_phase(graph: StemGraph, task_class: str) -> StemGraph:
    print(f"Stem cell starts learning about '{task_class}'")

    raw_results = web_search_raw(f"how to build an AI agent for {task_class}", max_results=MAX_SEARCH_RESULTS)
    search_summary = format_search_results(raw_results)

    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a domain analyst. Given some search results about a given task domain, extract the key concepts, typical approaches, tools commonly used, and what good output looks like. Be concise."
            },
            {
                "role": "user",
                "content": f"Task domain: {task_class}\n\nSearch Results:\n{search_summary}"
            }
        ]
    )
    domain_summary = response.choices[0].message.content

    domain_node = Node(
        node_type=NodeType.prompt,
        role=NodeRole.domain_kn,
        content=domain_summary,
        version=0
    )
    graph.add_node(domain_node)

    print(f"Nodes in graph at birth phase: {[(id, n.role) for id, n in graph.nodes.items()]}")
    builder_node_id = next(idx for idx, n in graph.nodes.items() if n.role == NodeRole.builder)
    graph.add_edge(Edge(
        source_id=builder_node_id,
        target_id=domain_node.id,
        relation=EdgeRelation.feeds_into
    ))

    print(f"Domain knowledge node created: [{domain_node.id}] | Stem cell expanded")
    return graph


def propose_node(graph: StemGraph, task_class: str, cycle: int, evaluator_feedback: str = None) -> tuple | None:
    graph_summary = graph.to_json()
    tools_available = list_tools()

    feedback_section = ''
    if evaluator_feedback:
        existing_nodes = [(nid, n.role.value) for nid, n in graph.nodes.items()]
        feedback_section = (
            f"\nThe evaluator rejected the last proposal. Reason: {evaluator_feedback}"
            f"\nExisting nodes you can connect to: {existing_nodes}"
            f"\nTake this into account and propose something that addresses the rejection reason directly."
        )

    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the builder of an AI agent that is growing into a specialist for a given task domain. "
                    "Each cycle you propose exactly one new node to add to the knowledge graph. "

                    "NODE TYPES AND ROLES: "
                    "A node can be type 'prompt' (role: domain_knowledge or strategy) or type 'tool' (role: tool). "
                    "Tool nodes must always be edge nodes — they can only be targets, never sources. "

                    "NODE CONTENT RULES: "
                    "For prompt nodes: content must be an actionable instruction telling an LLM exactly what to do "
                    "with the input it receives at inference time. "
                    "Example of GOOD content: 'Given the research question provided, identify the key sub-topics "
                    "and output a numbered list of specific search queries to investigate each one.' "
                    "Example of BAD content: 'Deep research involves gathering information from multiple sources.' "

                    "For tool nodes there are two cases: "
                    "1. Referencing an existing tool from the registry — set content to ONLY the exact function name "
                    "(e.g. 'web_search') and set tool_name to the same name. "
                    "2. Creating a new tool NOT in the registry — set content to the complete Python function "
                    "definition and set tool_name to the function name. "
                    "The function must accept exactly one string argument and return a string. "
                    "Available imports inside the function body: json, re, ddgs "
                    "(usage: list(DDGS().text(query, max_results=5)) — import inside function body). "

                    "TOOLS ALREADY IN REGISTRY (reference these by name only, do not recreate them): "
                    f"{tools_available} "

                    "CONNECTIVITY: "
                    f"Existing nodes you can connect to: {[(nid, n.role.value) for nid, n in graph.nodes.items()]} "
                    "Always connect your new node to an existing node. Never leave it isolated. "

                    "RESPONSE FORMAT: "
                    "Respond ONLY with a valid JSON object with these fields: "
                    "type (prompt or tool), "
                    "role (domain_knowledge, strategy, or tool), "
                    "content (actionable instruction for prompt nodes; function name or full code for tool nodes), "
                    "connect_to (the full node id to connect this node to), "
                    "relation (feeds_into, requires, validates, or contradicts), "
                    "tool_name (only include this field if type is tool, must match the function name). "
                    "No explanation, no markdown, no extra fields."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Task domain: {task_class}\n"
                    f"Current cycle: {cycle}\n"
                    f"Available tools: {tools_available}\n"
                    f"Current graph:\n{graph_summary}"
                    f"{feedback_section}"
                )
            }
        ]
    )
    raw = response.choices[0].message.content.strip()
    print(f"[BUILDER RAW RESPONSE] {raw}")

    try:
        proposal = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"Builder produced invalid JSON, cycle is skipped. Error: {e}")
        return None

    print(f"[BUILDER PROPOSAL] {proposal}")

    if proposal.get("type") == "tool":
        tool_name = proposal.get("tool_name") or proposal.get("content", "").strip()

        if tool_name in list_tools():
            # Existing tool — just ensure content is the clean name
            proposal["content"] = tool_name
        else:
            # New tool — content must be Python function code
            success = register_tool_from_code(tool_name, proposal["content"])
            if not success:
                print(f"Tool registration failed for '{tool_name}', cycle is skipped.")
                return None

    node = Node(
        node_type=NodeType(proposal["type"]),
        role=NodeRole(proposal["role"]),
        content=proposal["content"],
        version=cycle
    )

    print(f"[BUILDER] Node created: {node.role} ({node.node_type})")
    return node, proposal.get("connect_to"), proposal.get("relation")


def growth_loop(graph: StemGraph, task_class: str) -> StemGraph:
    print(f"\nGrowth Loop Started for '{task_class}' - stem cell is learning")
    cycle = 1
    feedback = None

    while cycle <= MAX_GROWTH_CYCLES:
        print(f"\n-- Cycle {cycle} --")

        isolated = graph.get_isolated_nodes()
        if isolated:
            connectivity_feedback = f"Warning: these nodes are isolated with no edges: {isolated}. Connect your new node to an existing node"
        else:
            connectivity_feedback = "Graph is fully connected"

        complete_feedback = f"{feedback}\n{connectivity_feedback}" if feedback else connectivity_feedback
        result = propose_node(graph, task_class, cycle, complete_feedback)

        if result is None:
            print("No valid proposal, continuing to next cycle")
            cycle += 1
            continue

        node, connect_to_id, relation = result

        graph.add_node(node)

        if connect_to_id and relation:
            try:
                edge = Edge(
                    source_id=connect_to_id,
                    target_id=node.id,
                    relation=EdgeRelation(relation)
                )
                graph.add_edge(edge)
            except Exception as e:
                print(f"Edge creation failed: {e}")
                del graph.nodes[node.id]
                graph.graph.remove_node(node.id)
                cycle += 1
                continue

        print(f"[GRAPH STATE] Nodes: {[(nid[:8], n.role.value) for nid, n in graph.nodes.items()]}")
        print(f"[GRAPH STATE] Edges: {[(e.source_id[:8], e.relation.value, e.target_id[:8]) for e in graph.edges.values()]}")

        passed, reason = evaluate(graph, task_class, cycle)

        if passed:
            print(f"Cycle {cycle} approved")
            feedback = None
            if cycle >= MIN_GROWTH_CYCLES:
                save_checkpoint(graph, task_class, cycle)
                if is_fully_specialized(graph, task_class):
                    print(f"Agent fully specialized after {cycle} cycles")
                    return graph
        else:
            print(f"Cycle {cycle} rejected: {reason}")
            feedback = reason
            graph.graph.remove_node(node.id)
            del graph.nodes[node.id]
            graph.edges = {
                eid: e for eid, e in graph.edges.items()
                if e.source_id != node.id and e.target_id != node.id
            }

        cycle += 1

    print("Max growth cycles reached")
    return graph


def is_fully_specialized(graph: StemGraph, task_class: str) -> bool:
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are assessing whether an AI agent's knowledge graph is fully specialized "
                    "for its target domain. Check ALL criteria in the rubric simultaneously. "
                    "Respond ONLY with a valid JSON object with two fields: "
                    "done (boolean) and reason (string)."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Task domain: {task_class}\n"
                    f"Rubric criteria to satisfy: {json.dumps(get_eval_rule())}\n"
                    f"Current graph:\n{graph.to_json()}"
                )
            }
        ]
    )
    raw = response.choices[0].message.content.strip()

    try:
        result = json.loads(raw)

        if result.get("done"):
            print(f"[EVALUATOR] Graph fully specialized: {result.get('reason')}")

        return result.get("done", False)
    except json.JSONDecodeError:
        return False


def run_builder(task_class: str) -> StemGraph:
    graph = StemGraph()

    builder_node = Node(
        node_type=NodeType.prompt,
        role=NodeRole.builder,
        content=f"You are a builder agent growing into a specialist for the domain: {task_class}.",
        version=0
    )

    evaluator_node = Node(
        node_type=NodeType.prompt,
        role=NodeRole.evaluator,
        content=f"You are an evaluator agent. Your job is to assess whether the graph is correctly specializing for: {task_class}.",
        version=0
    )

    graph.add_node(builder_node)
    graph.add_node(evaluator_node)

    graph.add_edge(Edge(
        source_id=builder_node.id,
        target_id=evaluator_node.id,
        relation=EdgeRelation.validates
    ))

    # Define the rubric before growth starts so the evaluator has criteria from cycle 1
    define_evaluation(task_class)

    graph = birth_phase(graph, task_class)
    graph = growth_loop(graph, task_class)

    return graph
