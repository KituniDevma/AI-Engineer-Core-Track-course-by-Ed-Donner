"""Shared configuration for the Week 5 agentic RAG exercise."""

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(override=True)

HERE = Path(__file__).resolve().parent
WEEK5_DIR = HERE.parent

KNOWLEDGE_BASE_PATH = WEEK5_DIR / "knowledge-base"
DB_PATH = HERE / "agentic_db"
COLLECTION_NAME = "insurellm_docs"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
AGENT_MODEL = os.getenv("AGENTIC_RAG_MODEL", "ollama/llama3.2")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

CHUNK_SIZE = 1_800
CHUNK_OVERLAP = 300
EMBEDDING_BATCH_SIZE = 100
DEFAULT_SEARCH_RESULTS = 6
MAX_AGENT_STEPS = 6
