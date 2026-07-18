# Stem Agent

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white) ![OpenAI](https://img.shields.io/badge/OpenAI-GPT-412991?logo=openai)

> A framework for growing self-specializing LLM agents: instead of hand-writing an agent for every domain, Stem Agent starts from a generic "stem" and grows it into a domain specialist. For the full technical report, see `Adel_Lis_Stem_Agent_Report.pdf` in this repository.

---

## What is Stem Agent?

The architecture to better represent a Stem Agent is a directed knowledge graph. Each node is either a `prompt` (a specific instruction for the LLM to follow) or a `tool` (a Python function the agent can call). Edges define how data flows between nodes. When you grow an agent for a domain like "deep research" or "python analyzer", the system searches the web to learn how that domain is typically approached, then enters a growth loop where a `builder` LLM proposes new nodes one at a time. An `evaluator` LLM checks each proposal against a quality rubric and either approves it or rejects it with a reason. Rejected nodes are removed and the builder tries again with the feedback.

Growth stops when the evaluator confirms the graph meets all specialization criteria, things like having at least one tool node, having actionable strategy nodes, and ending with a human-readable output. The result is saved as a _checkpoint_. When you run the agent on a real input, a `traverser` walks the graph in topological order, pipes each node's output into the next, and returns the final answer. For a different domain, you start a new stem agent from scratch.

---

## Requirements

- [uv](https://docs.astral.sh/uv/) — used to manage the Python environment and dependencies
- Python 3.11+
- An OpenAI API key

---

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/Adel-Lis/stem-agent
cd stem_agent
```

**2. Create the environment and install dependencies**

```bash
uv sync
```

**3. Create a `.env` file in the project root**

```
OPENAI_API_KEY=your_key_here
```

---

## Usage

### Grow a new agent

This command builds a specialized agent for a given domain. It runs the birth phase (web search + domain knowledge) and the growth loop, and saves the result as a checkpoint under `outputs/`.

```bash
uv run main.py grow "<domain>"
```

Examples:

```bash
uv run main.py grow "deep research"
uv run main.py grow "python analyzer"
uv run main.py grow "log analyst"
```

---

### Run a specialized agent

Once an agent has been grown, use this command to run it on a specific input. The agent must have been grown first.

```bash
uv run main.py run "<domain>" --input "<your question or task>"
```

Examples:

```bash
uv run main.py run "deep research" --input "What are the latest advances in quantum computing?"
uv run main.py run "python analyzer" --input "Is this function correct? def area(r): return 3.14 * r * r * r"
uv run main.py run "log analyst" --input "2026-05-01T10:22:11Z ERROR DB connection timeout after 30s"
```

---

## Project Structure

```
stem_agent/
├── agents/
│   ├── builder.py       # Birth phase, growth loop, node proposals
│   ├── evaluator.py     # Node validation and specialization rubric
│   └── traverser.py     # Runs the specialized agent at inference time
├── graph/
│   ├── graph.py         # StemGraph — the knowledge graph data structure
│   ├── node.py          # Node model and roles
│   ├── edge.py          # Edge model and relation types
│   └── checkpoint.py    # Save and load graph checkpoints
├── tools/
│   └── registry.py      # Tool registry with sandboxed dynamic tool creation
├── memory/
│   └── context.py       # Execution context used during traversal
├── outputs/             # Saved agent checkpoints (one folder per domain)
├── config.py            # Model names and global settings
├── main.py              # CLI entry point
└── .env                 # Your OpenAI API key (not committed)
```

---

© 2026 Adel Lis. All rights reserved.
