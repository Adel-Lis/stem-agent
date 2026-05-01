from ddgs import DDGS


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
    if name not in TOOL_REGISTRY:
        raise ValueError(f"Tool '{name}' is not in the Tool Registry")
    return TOOL_REGISTRY[name]

def register_tool_from_code(name: str, code: str) -> bool:
    try:
        namespace = {}
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
        print(f"Tool registered: {name}")
        return True
    except Exception as e:
        print(f"Failed to register tool '{name}': {e}")
        return False

def restore_tools_from_codes(tool_codes: dict[str, str]):
    for name, code in tool_codes.items():
        if name not in TOOL_REGISTRY:
            register_tool_from_code(name, code)

def get_tool_codes() -> dict[str, str]:
    return dict(TOOL_CODE_REGISTRY)

def register_tool(name: str, func: callable):
    TOOL_REGISTRY[name] = func
    print(f"Tool {name} registered!")

def list_tools() -> list[str]:
    return list(TOOL_REGISTRY.keys())
