# Agentic AI – Practical Implementations

This repository contains **practical implementations of Agentic AI concepts**, demonstrated through simple and modular examples.
It was created as a companion to a **national-level workshop on *Agentic AI and its Applications***.

The goal of this repo is to help learners understand **how agents are built in practice**, beyond theory and hype.

---

## 📌 What This Repository Covers

* Basics of **LLM-driven agents**
* **Simple agent loops**
* **ReAct-style agents** (Reason + Act)
* Function calling–based agent behavior
* Agent control flow and reasoning graphs
* Minimal examples to understand agent architectures

---

## 🧠 Agent Implementations

### 1. Simple LLM Agent

* File: `simpleLLM.py`
* Demonstrates a basic prompt → response interaction using an LLM.

### 2. Agent with Tool Usage

* File: `AIAgent.py`
* Shows how an agent can reason and take actions using tools/functions.

### 3. ReAct Agent

* File: `ReAct.py`
* Implements the **ReAct pattern**, where the agent alternates between:

  * Reasoning (thoughts)
  * Actions (tool calls)
  * Observations

---

## 📊 Agent Flow Visualizations

The `graphs/` folder contains visual representations of agent workflows:

* `graph_simple.png` – Simple agent loop
* `graph_react.png` – ReAct-style agent flow
* `graph_agent.png` – General agent architecture

These diagrams help in understanding **decision loops and control flow** in agentic systems.

---

## 🧪 Notebooks

* `openai_gpt.ipynb`
A notebook used to experiment with early language models (e.g., GPT-1) to illustrate how modern LLMs have evolved over time.

---

## ⚙️ Setup Instructions

1. Clone the repository:

   ```bash
   git clone https://github.com/Vishnu-3101/Agentic-AI.git
   cd Agentic-AI
   ```

2. Create and activate a virtual environment (optional but recommended)

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Add your API keys in a `.env` file. I used Groq API since it free to use.

   ```env
   GROQ_API_KEY=your_api_key_here
   ```

---

## 🎯 Who Is This For?

* Students exploring **AI agents**
* Researchers prototyping agent architectures
* Engineers experimenting with **LLM-based autonomous systems**
* Anyone curious about **Agentic AI in practice**

---

## 🚧 Disclaimer

These implementations are **educational and experimental**.
They are intentionally kept simple to focus on **core agent concepts**, not production readiness.

---

## 📬 Feedback & Contributions

Feel free to open issues, suggest improvements, or extend the examples.
Happy to discuss and collaborate on agentic system design.
