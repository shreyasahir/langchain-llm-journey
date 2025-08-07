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


#1 Define state structure

class ChatState(TypedDict):
    messages: List
    query: str
    context: str

# 2 Create LLM

llm = ChatOllama(model="llama3")

# 3 Define node , function that modifies state

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
    return state

def retrieve_node(state: ChatState) -> ChatState:
    query = state["messages"][-1].content
    state["query"] = query
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    state["context"] = context
    return state
# 4 Build langGraph with one node

builder = StateGraph(ChatState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("chat", chat_node)
builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "chat")
builder.set_finish_point("chat")

graph = builder.compile()

# 5 Run it

if __name__ == "__main__":
    user_input = input("You: ")
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "query": "",
        "context": "",
    }
    final_state = graph.invoke(initial_state)
    print("Bot:", final_state["messages"][-1].content)
