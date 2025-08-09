from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from typing import TypedDict, List
from langchain_core.messages import HumanMessage, AIMessage

loader = TextLoader("langgraph_doc.txt")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.from_documents(docs, embeddings)


# 1 Define state structure
class ChatState(TypedDict):
    messages: List
    query: str
    context: str
    continue_chat: bool  # Add flag to control flow


# 2 Create LLM
llm = ChatOllama(model="llama3")


# 3 Define nodes - functions that modify state

def retrieve_node(state: ChatState) -> ChatState:
    query = state["messages"][-1].content
    state["query"] = query
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    state["context"] = context
    return state


def chat_node(state: ChatState) -> ChatState:
    context = state["context"]
    question = state["query"]

    prompt = f"""You are a helpful assistant answering questions based on LangGraph documentation.

Context:
{context}

Question:
{question}

Answer:"""

    response = llm.invoke([HumanMessage(content=prompt)])
    state["messages"].append(response)

    # Print the response immediately
    print("Bot:", response.content)

    return state


def router_node(state: ChatState) -> ChatState:
    user_input = input("\n👤 Ask another question (or type 'exit' to quit): ")
    if user_input.strip().lower() in ["exit", "quit"]:
        state["continue_chat"] = False
    else:
        state["messages"].append(HumanMessage(content=user_input))
        state["continue_chat"] = True
    return state


# Router function that determines next step
def route_condition(state: ChatState) -> str:
    if state.get("continue_chat", True):
        return "retrieve"
    else:
        return "end"


# 4 Build LangGraph
builder = StateGraph(ChatState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("chat", chat_node)
builder.add_node("router", router_node)

builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "chat")
builder.add_edge("chat", "router")

# Use the correct router function
builder.add_conditional_edges(
    "router",
    route_condition,  # This is the routing function
    {
        "retrieve": "retrieve",
        "end": END,
    }
)

graph = builder.compile()

# 5 Run it
if __name__ == "__main__":
    user_input = input("You: ")
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "query": "",
        "context": "",
        "continue_chat": True,
    }

    try:
        final_state = graph.invoke(initial_state)
        print("\nChat ended. Goodbye!")
    except Exception as e:
        print(f"An error occurred: {e}")