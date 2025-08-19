from typing import TypedDict, List
import re

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END


class ChatState(TypedDict):
    messages: List
    query: str
    context: str
    continue_chat: bool
    intent: str  # Added to track user intent


# Document processing setup
loader = TextLoader("data/data.txt")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.from_documents(docs, embeddings)

# Create LLM
llm = ChatOllama(model="llama3")


# Define tools
@tool
def multiply(x: int, y: int) -> int:
    """Multiplies two numbers."""
    return x * y


@tool
def get_weather() -> str:
    """Get the current weather information"""
    return "Weather is sunny and 72°F"


def classify_intent(state: ChatState) -> ChatState:
    """Classify the user's intent to determine which node to route to."""
    query = state["messages"][-1].content.lower()
    print(f"CLASSIFY: Processing query: '{query}'")

    # Simple intent classification
    weather_keywords = ["weather", "temperature", "sunny", "rain", "climate", "forecast", "hot", "cold", "warm"]
    math_keywords = ["multiply", "times", "product", "calculate", "*", "x"]

    if any(word in query for word in weather_keywords):
        state["intent"] = "weather"
        print(f"CLASSIFY: Set intent to weather")
    elif any(word in query for word in math_keywords) and any(char.isdigit() for char in query):
        state["intent"] = "math"
        print(f"CLASSIFY: Set intent to math")
    else:
        state["intent"] = "document_qa"
        print(f"CLASSIFY: Set intent to document_qa")

    return state


def retrieve_node(state: ChatState) -> ChatState:
    """Retrieve relevant documents based on the user's query."""
    query = state["messages"][-1].content
    state["query"] = query
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    state["context"] = context
    return state


def chat_node(state: ChatState) -> ChatState:
    """Generate response using retrieved context."""
    context = state["context"]
    question = state["query"]

    prompt = f"""You are a helpful assistant answering questions based on documentation.

Context:
{context}

Question:
{question}

Answer:"""

    response = llm.invoke([HumanMessage(content=prompt)])
    state["messages"].append(response)
    print("Bot:", response.content)
    return state


def weather_node(state: ChatState) -> ChatState:
    """Handle weather-related queries."""
    try:
        # Invoke the tool properly with empty input (since get_weather takes no parameters)
        weather_info = get_weather.invoke({})
        response = AIMessage(content=f"Here's the current weather: {weather_info}")
    except Exception as e:
        response = AIMessage(content=f"Sorry, I couldn't get the weather information. Error: {e}")

    state["messages"].append(response)
    print("Bot:", response.content)
    return state


def math_node(state: ChatState) -> ChatState:
    """Handle mathematical operations."""
    query = state["messages"][-1].content

    # Extract numbers from the query
    numbers = re.findall(r'\d+', query)

    if len(numbers) >= 2:
        try:
            x, y = int(numbers[0]), int(numbers[1])
            # Invoke the tool properly with the required parameters
            result = multiply.invoke({"x": x, "y": y})
            response = AIMessage(content=f"The result of {x} × {y} = {result}")
        except ValueError:
            response = AIMessage(content="I couldn't parse the numbers. Please provide two clear numbers to multiply.")
        except Exception as e:
            response = AIMessage(content=f"Sorry, I couldn't perform the calculation. Error: {e}")
    else:
        response = AIMessage(content="I need two numbers to multiply. For example: 'multiply 5 and 3'")

    state["messages"].append(response)
    print("Bot:", response.content)
    return state


def router_node(state: ChatState) -> ChatState:
    """Handle user input for continuing the chat."""
    user_input = input("\nAsk another question (or type 'exit' to quit): ")

    if user_input.strip().lower() in ['exit', 'quit']:
        state["continue_chat"] = False
        print("Goodbye!")
    else:
        state["continue_chat"] = True
        # Add the new user message to continue the conversation
        state["messages"].append(HumanMessage(content=user_input))
        # Reset intent for new classification
        state["intent"] = ""

    return state


def intent_condition(state: ChatState) -> str:
    """Route based on classified intent."""
    intent = state.get("intent", "document_qa")
    print(f"INTENT_CONDITION: Routing to {intent}")

    if intent == "weather":
        return "weather"
    elif intent == "math":
        return "math"
    else:
        return "retrieve"


def router_condition(state: ChatState) -> str:
    """Determine the next node based on continue_chat flag."""
    if state["continue_chat"]:
        return "classify"  # Go to intent classification
    else:
        return "end"  # End the conversation


# Build the graph (ONLY ONCE!)
builder = StateGraph(ChatState)

# Add nodes
builder.add_node("classify", classify_intent)
builder.add_node("retrieve", retrieve_node)
builder.add_node("chat", chat_node)
builder.add_node("weather", weather_node)
builder.add_node("math", math_node)
builder.add_node("router", router_node)

# Set entry point
builder.set_entry_point("classify")

# Add conditional edges from classify to different handlers
builder.add_conditional_edges(
    "classify",
    intent_condition,
    {
        "weather": "weather",
        "math": "math",
        "retrieve": "retrieve"
    }
)

# All paths lead to router
builder.add_edge("retrieve", "chat")
builder.add_edge("chat", "router")
builder.add_edge("weather", "router")
builder.add_edge("math", "router")

# Router decides whether to continue or end
builder.add_conditional_edges(
    "router",
    router_condition,
    {
        "classify": "classify",  # Loop back for intent classification
        "end": END  # End the conversation
    }
)

# Compile the graph
graph = builder.compile()

# Run the chatbot
if __name__ == "__main__":
    print("Multi-Function Chatbot")
    print("I can help with:")
    print("- Document-based Q&A")
    print("- Weather information")
    print("- Math calculations (multiplication)")
    print()

    user_input = input("You: ")

    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "query": "",
        "context": "",
        "continue_chat": True,
        "intent": ""
    }

    try:
        final_state = graph.invoke(initial_state)
        print("\nChat ended. Goodbye!")
    except Exception as e:
        print(f"An error occurred: {e}")