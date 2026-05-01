from ddgs import DDGS


def web_search(query: str, max_results: int = 5) -> list[dict]:
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    
    # print(f"[TOOL DDGS] Outuput: {results}")
    return results

def summarize_results(results: list[dict]) -> str:
    return "\n\n".join(
        f"Title: {r.get('title', '')}\nSnippet: {r.get('body', '')}"
        for r in results
    )

TOOL_REGISTRY: dict[str, callable] = {
    "web_search": web_search,
    "summarize_results": summarize_results
}

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
        print(f"Tool registered: {name}")
        return True
    except Exception as e:
        print(f"Failed to register tool '{name}': {e}")
        return False

def register_tool(name: str, func: callable):
    TOOL_REGISTRY[name] = func
    print(f"Tool {name} registered !")

def list_tools() -> list[str]:
    return list(TOOL_REGISTRY.keys())