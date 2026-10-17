# Project 2 Part 1 — Intelligent AI Coding Assistant (Week 4, Cellula)

A Streamlit chatbot that answers programming questions. Every request first
goes through an LLM intent classifier, then an intelligent router picks one
of two pipelines:

- **Code Explanation** (Route 1): the LLM analyzes the pasted code directly.
  No RAG, no embeddings, no vector database search on this path.
- **Code Generation** (Route 2): retrieves programming knowledge from a
  Chroma vector database, a second LLM acting as a relevance evaluator
  checks whether the retrieved context can answer the request. If relevant,
  the code is generated from the context. If not relevant, the assistant
  does not hallucinate: it asks the user for the correct solution, embeds
  it and inserts it into Chroma (human feedback learning), so the knowledge
  base keeps expanding.

The assistant also keeps long term conversation memory (previous questions,
generated code, language and framework preferences) and can execute the code
it generated through a separate code runner tool that captures stdout and
stderr.

## Structure

| File | Role |
|---|---|
| `app.py` | Streamlit UI: chat, history, streaming, Run-code buttons, memory panel |
| `assistant.py` | Router: wires classifier, RAG path and explain path together |
| `intent_classifier.py` | LLM classifies each query as EXPLAIN or GENERATE |
| `knowledge.py` | Chroma retrieval, LLM relevance check, feedback insertion |
| `build_kb.py` | Offline RAG phase: chunk data docs, embed, store in Chroma |
| `memory.py` | Conversation memory: turns, code snippets, preferences |
| `code_runner.py` | Executes generated code in a subprocess with timeout |
| `prompts.py` | All prompt templates in one place |
| `data/` | Knowledge base source documents (markdown) |
| `chroma_db/` | Persistent Chroma database (created by build_kb.py) |

The knowledge base is built entirely from the openai/openai_humaneval
dataset on Hugging Face: fetch_humaneval.py downloads all 164 problems and
stores each one chunked and embedded directly in the Chroma database
(chroma_db/chroma.sqlite3), with metadata holding the problem id and
function name. There is no intermediate file, the dataset lives in sqlite.

## Setup and run

```bash
pip3 install -r requirements.txt
cp app.env.example app.env   # then put your OpenRouter key inside
python3 fetch_humaneval.py   # downloads HumanEval into chroma_db, only once
streamlit run app.py
```

The LLM calls (classifier, relevance evaluator, generation) use the free
OpenRouter model openrouter/free through the same api/v1 base url as last
week's project.

## Notes

- Embeddings: sentence-transformers all-MiniLM-L6-v2 (384 dims), cosine
  similarity with HNSW index inside Chroma.
- Chunking: recursive splitter with separators tuned for markdown/code so
  chunks do not cut functions in half, 1000 chars with 150 overlap.
- The relevance evaluator follows the CRAG idea of a retrieval evaluator
  that gates generation instead of trusting the retrieval blindly.
