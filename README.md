# Retail Analytics Copilot

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![DSPy](https://img.shields.io/badge/DSPy-2.4+-green.svg)](https://github.com/dspy-ai/dspy)

A local AI-powered Retail Analytics Copilot that answers business questions using:

* **Document Retrieval (RAG)** from policy/marketing/KPI docs
* **SQL queries** over a local SQLite Northwind-like database
* **Hybrid reasoning**, combining both RAG and SQL results

Built with **Python**, **DSPy**, **Ollama LLM**, and **Rich** for interactive CLI outputs.

---

## Features

* Classifies questions into **RAG**, **SQL**, or **Hybrid** automatically
* Generates **valid SQL** with schema constraints and date/category filters
* Retrieves **relevant documents** for policy and KPI questions
* Synthesizes **final answers** in the requested format (`int`, `float`, `dict`, `list`)
* Provides **confidence scores** and **citations** for each answer
* Fully **batchable** via JSONL input files

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ahmed-el-shenawy/retail-analytics-copilot.git
cd retail-analytics-copilot
```

### 2. Create a Python environment

```bash
conda create -n app_task python=3.12
conda activate app_task
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Dependencies include:**
> `dspy`, `click`, `rich`, and other core packages.

### 4. Setup Ollama LLM

1. Ensure Ollama is installed: [https://ollama.com](https://ollama.com)
2. Run Ollama server:

```bash
ollama serve
```

3. Pull the required model:

```bash
ollama pull qwen3:4b-instruct
```

---

## Usage

### Run in batch mode

```bash
python run_agent_hybrid.py \
    --batch sample_questions_hybrid_eval.jsonl \
    --out outputs_hybrid.jsonl
```

* `--batch`: JSONL file containing questions
* `--out`: JSONL file to save results

### Input JSONL format

Each line is a JSON object:

```json
{
  "id": "rag_policy_beverages_return_days",
  "question": "According to the product policy, what is the return window (days) for unopened Beverages?",
  "format_hint": "int"
}
```

### Output JSONL format

Each line is a JSON object:

```json
{
  "id": "rag_policy_beverages_return_days",
  "final_answer": 14,
  "sql": "",
  "confidence": 0.82,
  "explanation": "Retrieved from policy docs",
  "citations": ["product_policy.md::chunk0"]
}
```
 