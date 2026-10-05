# Week 5 Exercise: Agentic RAG

This exercise extends Week 5's `pro_implementation` into a tool-using RAG
agent for the Insurellm knowledge base.

## What changed

The pro implementation follows a fixed pipeline:

1. rewrite the query;
2. retrieve the original and rewritten queries;
3. rerank the chunks;
4. answer.

This version gives the model retrieval tools and lets it choose the next step:

1. decide which aspect of the question to investigate;
2. run one or more focused semantic searches;
3. refine weak or ambiguous searches;
4. optionally read a complete source document;
5. answer from evidence and cite source paths.

The implementation runs locally with **Llama 3.2 through Ollama** and uses
the local Hugging Face `all-MiniLM-L6-v2` model for embeddings. No cloud API
keys are needed.

Other improvements include deterministic chunk IDs, batched embeddings,
paragraph-aware overlapping chunks, source/type metadata filters, cosine
distance, bounded agent steps, retries, path-safe document reading, and a
Gradio UI that shows the agent steps and retrieved sources.

## Project structure

```text
week_5_exercise_agentic_rag/
├── .env.example       # required environment variables
├── config.py          # paths and model settings
├── ingest.py          # builds the local Chroma database
├── agentic_rag.py     # tools and agent loop
├── app.py             # Gradio chat application
└── README.md
```

The source documents remain in `week5/knowledge-base`. The generated Chroma
database is stored locally in `week_5_exercise_agentic_rag/agentic_db`.

## Requirements

- Python 3.11 or newer
- The course environment installed with `uv sync`
- [Ollama](https://ollama.com/download) installed and running
- `llama3.2` downloaded locally

The root `pyproject.toml` already includes Ollama, LiteLLM, Chroma,
Sentence Transformers, Gradio, Tenacity, python-dotenv, and tqdm.

## Run it

Run all commands from the `llm_engineering` directory.

### 1. Install the course dependencies

```powershell
uv sync
```

### 2. Download the local models

Make sure the Ollama application is running, then execute:

```powershell
ollama pull llama3.2
```

Confirm both models are available:

```powershell
ollama list
```

No API keys are required. You may optionally create `llm_engineering/.env`
to change the Ollama server or local models:

```env
OLLAMA_BASE_URL=http://localhost:11434
AGENTIC_RAG_MODEL=ollama/llama3.2
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

`AGENTIC_RAG_MODEL` can be another Ollama model that supports tool calling.
Larger Llama models generally plan and cite more reliably when your computer
has enough RAM. The embedding model downloads automatically on first use and
is then cached locally.

### 3. Build the vector database

```powershell
uv run python week5/week_5_exercise_agentic_rag/ingest.py
```

This reads the Week 5 Markdown knowledge base, creates overlapping chunks,
embeds them, and writes a Chroma collection under `agentic_db`.

Run ingestion again whenever documents or chunk settings change. It replaces
the old collection to avoid duplicate records.

### 4. Start the application

```powershell
uv run python week5/week_5_exercise_agentic_rag/app.py
```

Gradio opens in the browser. If it does not open automatically, use the local
URL printed in the terminal, normally `http://127.0.0.1:7860`.

### 5. Try questions that require agent behavior

- `What products does Insurellm offer?`
- `Who is Avery Lancaster and what is their role?`
- `Compare the contracts for Carllm and identify their termination terms.`
- `Which employee works on Healthllm, and what does that product do?`

The last two should cause multiple searches and may cause the agent to read
full source documents.

## How the agent loop works

`AgenticRAG.answer()` sends the question and two tool definitions to the
model. When the model requests a tool:

- `search_knowledge_base` embeds a focused query and retrieves up to ten
  Chroma chunks, optionally filtered by document type.
- `read_source` safely loads one complete Markdown file returned by search.

Tool outputs are returned to the model. It can call more tools or produce the
final cited answer. The loop is capped at six steps to control cost and avoid
unbounded behavior.

## Troubleshooting

- **Vector database not found:** run the ingestion command before the app.
- **Connection refused:** start the Ollama application and verify
  `http://localhost:11434` is available.
- **Model not found:** run `ollama pull llama3.2`.
- **Slow responses:** use a smaller Llama model or reduce
  `DEFAULT_SEARCH_RESULTS` and `MAX_AGENT_STEPS` in `config.py`.
- **Port already in use:** stop the other Gradio process or change
  `demo.launch()` to include another `server_port`.
- **Answers lack evidence:** ask a more specific question, inspect the listed
  retrieved sources, or increase `DEFAULT_SEARCH_RESULTS` in `config.py`.
