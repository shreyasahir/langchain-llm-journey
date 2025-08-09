
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from typing import TypedDict, List
from langchain_core.messages import HumanMessage, AIMessage


#1 Define state structure

class ChatState(TypedDict):
    messages: List

# 2 Create LLM

llm = ChatOllama(model="llama3")

# 3 Define node , function that modifies state

def chat_node(state: ChatState) -> ChatState:
    response = llm.invoke(state["messages"])
    state["messages"].append(response)
    return state

# 4 Build langGraph with one node

builder = StateGraph(ChatState)
builder.add_node("chat", chat_node)
builder.set_entry_point("chat")
builder.set_finish_point("chat")

graph = builder.compile()

# 5 Run it

if __name__ == "__main__":
    user_input = input("You: ")
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
    }
    final_state = graph.invoke(initial_state)
    print("Bot:", final_state["messages"][-1].content)
