from langchain_core.tools import tool

@tool
def multiply(x: int, y: int) -> int:
    """Multiplies two numbers."""
    return x * y
