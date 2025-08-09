from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_tool_calling_agent

@tool
def multiply(x: int, y: int) -> int:
    """Multiplies two numbers."""
    return x * y

llm = ChatOllama(model ="openchat")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use tools if needed."),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])
tools = [multiply]

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": "What is 13 times 7?"})
