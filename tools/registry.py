import builtins as _builtins
from ddgs import DDGS


# Restricted execution environment for dynamically registered tools.
# Prevents access to os, subprocess, open, eval, etc. while keeping
# the primitives and the three explicitly allowed third-party imports.
_SAFE_BUILTINS = {
    name: getattr(_builtins, name)
    for name in [
        'abs', 'bool', 'chr', 'dict', 'enumerate', 'filter', 'float',
        'frozenset', 'int', 'isinstance', 'issubclass', 'iter', 'len',
        'list', 'map', 'max', 'min', 'next', 'ord', 'print', 'range',
        'reversed', 'round', 'set', 'slice', 'sorted', 'str', 'sum',
        'tuple', 'type', 'zip', 'True', 'False', 'None',
        'Exception', 'ValueError', 'TypeError', 'KeyError', 'IndexError',
    ]
}
_ALLOWED_IMPORTS = {"json", "re", "ddgs"}

def _safe_import(name, *args, **kwargs):
    if name not in _ALLOWED_IMPORTS:
        raise ImportError(f"Import of '{name}' is not permitted in tool code")
    return __import__(name, *args, **kwargs)

_SAFE_BUILTINS["__import__"] = _safe_import


# Internal low-level functions that is used in the builder birth phase
def web_search_raw(query: str, max_results: int = 5) -> list[dict]:
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    return results

def format_search_results(results: list[dict]) -> str:
    return "\n\n".join(
        f"Title: {r.get('title', '')}\nSnippet: {r.get('body', '')}"
        for r in results
    )

# Tool-compatible version registered for traversal: accepts str, returns str
def web_search(query: str) -> str:
    results = web_search_raw(query)
    return format_search_results(results)


TOOL_REGISTRY: dict[str, callable] = {
    "web_search": web_search,
}

# Stores source code for dynamically registered tools
TOOL_CODE_REGISTRY: dict[str, str] = {}


def get_tool(name: str) -> callable:
    """Returns the callable tool function by name, raising an error if it hasn't been registered."""
    if name not in TOOL_REGISTRY:
        raise ValueError(f"Tool '{name}' is not in the Tool Registry")
    return TOOL_REGISTRY[name]

def register_tool_from_code(name: str, code: str) -> bool:
    """Takes a Python function as a string, runs it in a sandbox to make sure it works safely, and adds it to the registry if it passes."""
    try:
        namespace = {"__builtins__": _SAFE_BUILTINS}
        exec(code, namespace)
        func = namespace.get(name)
        if func is None or not callable(func):
            print(f"Tool '{name}' not found or not callable in generated code")
            return False
        test_result = func("test")
        if not isinstance(test_result, str):
            print(f"Tool '{name}' must return a string")
            return False
        TOOL_REGISTRY[name] = func
        TOOL_CODE_REGISTRY[name] = code
        print(f"  Tool registered: {name}")
        return True
    except Exception as e:
        print(f"  Failed to register tool '{name}': {e}")
        return False

def restore_tools_from_codes(tool_codes: dict[str, str]):
    """Re-registers all custom tools from their saved source code, called when loading a checkpoint so the tools are available again."""
    for name, code in tool_codes.items():
        if name not in TOOL_REGISTRY:
            register_tool_from_code(name, code)

def get_tool_codes() -> dict[str, str]:
    """Returns the source code of all dynamically created tools so they can be saved in a checkpoint."""
    return dict(TOOL_CODE_REGISTRY)

def register_tool(name: str, func: callable):
    """Directly registers a callable function as a tool by name, for tools defined in code rather than generated at runtime."""
    TOOL_REGISTRY[name] = func
    print(f"  Tool registered: {name}")

def list_tools() -> list[str]:
    """Returns the names of all tools currently in the registry, so the builder knows what's already available."""
    return list(TOOL_REGISTRY.keys())
