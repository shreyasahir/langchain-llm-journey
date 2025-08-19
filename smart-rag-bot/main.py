from langchain_core.messages import HumanMessage
from graph.rag_graph import build_rag_graph
from embeddings.retriever import DocumentRetriever
from agents.chat import generate_response, prompt_user
from langchain_core.tracers.context import tracing_v2_enabled

from contextlib import nullcontext
import os

if __name__ == "__main__":
    print("💬 Welcome to the Smart RAG Bot!")
    user_input = input("Ask a question: ")
    retriever = DocumentRetriever(filepath="data/langgraph_doc.txt")

    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "query": "",
        "context": "",
        "continue_chat": True,
        "intent": ""
    }
    tracing_context = tracing_v2_enabled(project_name=os.getenv("LANGCHAIN_PROJECT")) \
        if os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true" \
        else nullcontext()
    try:
        graph = build_rag_graph(
            retrieve_documents=retriever.search,
            generate_response=generate_response,
            prompt_user=prompt_user
        )
        with tracing_context:
            graph.invoke(initial_state)
        print("\n✅ Chat ended.")
    except Exception as e:
        print(f"❌ Error: {e}")
