from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3")

def generate_response(query: str, context: str) -> str:
    prompt = f"""You are a helpful assistant answering questions based on context.

Context:
{context}

Question:
{query}

Answer:"""
    response = llm.invoke(prompt)
    return response.content.strip()

def prompt_user() -> str:
    return input("\n👤 Ask another question (or type 'exit' to quit): ")
