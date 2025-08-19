from langchain_core.tools import tool

@tool
def get_weather() -> str:
    """Returns current weather (mock)."""
    return "It’s sunny and 72°F."
