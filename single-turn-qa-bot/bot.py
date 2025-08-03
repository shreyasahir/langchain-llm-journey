from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser



loader = TextLoader("data/data.txt")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.from_documents(docs, embeddings)

prompt = PromptTemplate.from_template("Ask something on langchain?")
chat = ChatOllama(model="llama3")

query = input("🔍 Ask a question about LangChain: ")



relevant_docs = vectorstore.similarity_search(query, k =3)
context = "\n\n".join([doc.page_content for doc in relevant_docs])
# Prompt template that includes the context and user question
template = """
You are a helpful AI assistant. Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:"""

prompt = PromptTemplate.from_template(template)
final_prompt = prompt.format(context=context, question=query)

response = chat.invoke(final_prompt)
print("🤖", response.content)