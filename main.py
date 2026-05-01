
import argparse
from agents.builder import run_builder
from agents.traverser import run_traverser



def grow(task_class: str):
    """Builds and grows a new stem agent for the given domain, saving the result as a checkpoint when done."""
    print(f"Starting stem agent for domain: '{task_class}'")
    graph = run_builder(task_class)
    print(f"\nGrowth complete. Agent specialized for '{task_class}'")
    return graph

def run(task_class: str, task_input: str):
    """Loads the specialized agent for the given domain and runs it on the provided input, printing the final answer."""
    print(f"\nRunning specialized agent for domain: '{task_class}'")
    output = run_traverser(task_class, task_input)
    print(f"\nAgent output:\n{output}")
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stem Agent")

    parser.add_argument("mode", choices=["grow", "run"], help="grow: grow the agent for a domain | run: run the specialized agent on a task")
    parser.add_argument("task_class", type=str, help="The domain to specialize for e.g. 'deep_research'")
    parser.add_argument("--input", type=str, help="Task input for run mode", default=None)

    args = parser.parse_args()

    if args.mode == "grow":
        grow(args.task_class)
    elif args.mode == "run":
        if not args.input:
            print("Error: --input is required when agent is run in 'run' mode")
        else:
            run(args.task_class, args.input)