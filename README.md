# ChatAI

**ChatAI** is a powerful, interactive web interface built with **Streamlit** that integrates advanced LLM capabilities, web search, document-based RAG (Retrieval-Augmented Generation), and deep multi-agent research workflows. It supports both light and agentic interactions, powered by cutting-edge libraries like **LangGraph**, **Chroma DB**, **LangChain**, and **Tavily**.

---

## 🚀 Features

- **🧠 Simple Chat:** Interact with a local or cloud-based LLM using tunable generation parameters.
- **🌐 Web Search (ReAct Agent):** Ask real-time questions and receive LLM-augmented responses from the internet.
- **📄 Upload PDF for Contextual RAG:** Perform context-aware question answering from user-uploaded PDFs using Chroma DB + LangChain.
- **🔍 Deep Web Research (Multi-Agent):** Trigger an agentic deep-research pipeline with LangGraph for complex, multi-hop questions.

---

## 🛠️ Tech Stack

| Library     | Purpose                                     |
|-------------|---------------------------------------------|
| **Streamlit** | Web Interface                             |
| **LangChain** | LLM Orchestration                         |
| **Chroma DB** | Vector Store for RAG                      |
| **LangGraph** | Multi-Agent Workflow & Control            |
| **Tavily**    | Web Search Integration                    |

---

## ⚙️ Controls

All tools include interactive controls to customize the behavior of the LLM:

- `Temperature`: Controls randomness (0.0 = deterministic, 2.0 = creative)
- `Top-k`: Limits sampling to top-K tokens
- `Top-p`: Nucleus sampling for diversity

---

## 📁 Modes

| Mode             | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| **Simple Chat**   | Basic conversation with an LLM                                              |
| **Web Search**    | Real-time question answering using Tavily and ReAct-based agents           |
| **Upload PDF**    | Ask questions based on content inside uploaded PDFs using RAG              |
| **Deep Web Search** | LangGraph-based multi-agent research with long-form answers               |


## How Multi Agent Deep Search Works

<img src="./Multi Agent Deep Search (1).png" style="height:320; width:320">

---

## 🔧 Setup

### 1. Clone the Repository
```bash
git clone <REPOSITORY>
cd REPOSITORY
```

#### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run app.py
```


## 🧠 Credits
Built with ❤️ using Streamlit, LangChain, LangGraph, Tavily, and Chroma DB