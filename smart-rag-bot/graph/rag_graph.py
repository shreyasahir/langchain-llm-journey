# graph/rag_graph.py
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
import re

from tools.math_tool import multiply
from tools.weather_tool import get_weather


def build_rag_graph(
    retrieve_documents,
    generate_response,
    prompt_user
):
    class ChatState(TypedDict):
        messages: List
        query: str
        context: str
        continue_chat: bool
        intent: str

    def classify_node(state: ChatState) -> ChatState:
        query = state["messages"][-1].content.lower()
        state["query"] = query

        if any(word in query for word in ["weather", "temperature", "sunny", "forecast"]):
            state["intent"] = "weather"
        elif any(word in query for word in ["multiply", "product", "times", "x", "*"]) and any(c.isdigit() for c in query):
            state["intent"] = "math"
        else:
            state["intent"] = "rag"
        return state

    def retrieve_node(state: ChatState) -> ChatState:
        context = retrieve_documents(state["query"])
        state["context"] = context
        return state

    def chat_node(state: ChatState) -> ChatState:
        question = state["query"]
        context = state["context"]
        answer = generate_response(question, context)
        state["messages"].append(AIMessage(content=answer))
        print("\n🤖 Bot:", answer)
        return state

    def math_node(state: ChatState) -> ChatState:
        query = state["query"]
        nums = re.findall(r'\d+', query)
        if len(nums) >= 2:
            x, y = int(nums[0]), int(nums[1])
            result = multiply.invoke({"x": x, "y": y})
            answer = f"The result of {x} × {y} is {result}"
        else:
            answer = "Please provide two numbers to multiply."
        state["messages"].append(AIMessage(content=answer))
        print("\n🧮 Bot:", answer)
        return state

    def weather_node(state: ChatState) -> ChatState:
        weather = get_weather.invoke({})
        state["messages"].append(AIMessage(content=weather))
        print("\n🌦️ Bot:", weather)
        return state

    def router_node(state: ChatState) -> ChatState:
        user_input = prompt_user()
        if user_input.strip().lower() in ["exit", "quit"]:
            state["continue_chat"] = False
        else:
            state["continue_chat"] = True
            state["messages"].append(HumanMessage(content=user_input))
        return state

    def intent_router(state: ChatState) -> str:
        return state.get("intent", "rag")

    def loop_router(state: ChatState) -> str:
        return "classify" if state.get("continue_chat", True) else "end"

    # Build the graph
    builder = StateGraph(ChatState)

    builder.add_node("classify", classify_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("chat", chat_node)
    builder.add_node("math", math_node)
    builder.add_node("weather", weather_node)
    builder.add_node("router", router_node)

    builder.set_entry_point("classify")

    # Routing based on intent
    builder.add_conditional_edges("classify", intent_router, {
        "rag": "retrieve",
        "math": "math",
        "weather": "weather",
    })

    # RAG flow
    builder.add_edge("retrieve", "chat")
    builder.add_edge("chat", "router")

    # Tool flows
    builder.add_edge("math", "router")
    builder.add_edge("weather", "router")

    # Loop controller
    builder.add_conditional_edges("router", loop_router, {
        "classify": "classify",
        "end": END,
    })

    return builder.compile()
