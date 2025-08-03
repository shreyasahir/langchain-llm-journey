# 🧠 Week 1: Single-Turn Q&A Bot using LangChain + Ollama

This is a one-turn question-answering app that loads a `.txt` file as context and answers questions using local LLMs and embeddings.

---

## 🔍 Features

- Local-only: no OpenAI or API keys needed
- Text-based knowledge base
- Powered by:
  - `nomic-embed-text` for embeddings
  - `llama3` for generation
  - FAISS vector search

---

## 🚀 Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt

#Pull required Ollama models

ollama pull llama3
ollama pull nomic-embed-text

#Run the bot

python bot.py

#Example

🔍 Ask a question about LangChain: How does LangChain help with prompt creation?
🤖 LangChain helps with prompt creation through its PromptTemplate component...
