"""Sample tools for LangGraph sub-agents.

These are **prototype stubs** — they return hardcoded or minimal data.
Replace each with a real API integration for production:

- ``search_tool``     -> Bing Search API, Azure AI Search, etc.
- ``calculator_tool`` -> already functional (eval with allowlist).
- ``weather_tool``    -> OpenWeatherMap, Azure Maps Weather, etc.

Each tool is decorated with ``@tool`` so LangChain can auto-generate
the JSON schema that the LLM uses for tool calling.
"""
from langchain_core.tools import tool


@tool
def search_tool(query: str) -> str:
    """Search for information on a topic. Returns relevant information."""
    # Stub — replace with actual search API in production
    responses = {
        "azure": "Azure is Microsoft's cloud platform with 200+ services.",
        "langgraph": "LangGraph is a framework for building stateful multi-agent apps.",
    }
    for key, val in responses.items():
        if key in query.lower():
            return val
    return f"Search results for: {query} — No specific results found. This is a prototype stub."


@tool
def calculator_tool(expression: str) -> str:
    """Evaluate a mathematical expression. Input should be a valid Python math expression."""
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: Only numeric expressions with +, -, *, /, (, ) are allowed."
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"


@tool
def weather_tool(location: str) -> str:
    """Get current weather for a location."""
    # Stub — replace with actual weather API in production
    return f"Weather in {location}: 72°F, partly cloudy, humidity 45%. (Prototype stub data)"
