"""Tool-using RAG agent for the Insurellm knowledge base."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chromadb import PersistentClient
from litellm import completion
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential

from config import (
    AGENT_MODEL,
    COLLECTION_NAME,
    DB_PATH,
    DEFAULT_SEARCH_RESULTS,
    EMBEDDING_MODEL,
    KNOWLEDGE_BASE_PATH,
    MAX_AGENT_STEPS,
    OLLAMA_BASE_URL,
)


SYSTEM_PROMPT = """
You are an expert knowledge worker for Insurellm.

Answer questions only from evidence in the company knowledge base. You have tools to:
1. search the knowledge base semantically;
2. read a complete source document when a search extract is insufficient.

Work agentically:
- Always search before answering a knowledge-base question.
- Break multi-part questions into multiple focused searches.
- If results are weak or ambiguous, refine the query and search again.
- Read a full source when exact wording, contract terms, or surrounding context matters.
- Never invent facts. Say what could not be found when evidence is missing.
- Cite factual claims inline using the exact source path in square brackets, for example
  [products/Carllm.md].
- Keep the final answer concise but complete.
""".strip()


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Semantic search over Insurellm products, employees, contracts, and company "
                "documents. Call more than once with different queries for multi-part questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A focused, context-rich semantic search query.",
                    },
                    "document_type": {
                        "type": ["string", "null"],
                        "enum": ["products", "employees", "contracts", "company", None],
                        "description": "Optional knowledge-base folder filter.",
                    },
                    "n_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "default": DEFAULT_SEARCH_RESULTS,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_source",
            "description": (
                "Read a complete knowledge-base document. Use an exact source path returned "
                "by search_knowledge_base."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Exact relative source path, such as products/Carllm.md.",
                    }
                },
                "required": ["source"],
                "additionalProperties": False,
            },
        },
    },
]


@dataclass
class AgentResult:
    answer: str
    sources: list[str] = field(default_factory=list)
    steps: int = 0


class AgenticRAG:
    """An LLM agent that chooses and iterates over retrieval tools."""

    def __init__(self) -> None:
        if not DB_PATH.exists():
            raise RuntimeError(
                f"Vector database not found at {DB_PATH}. Run `python ingest.py` first."
            )
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        chroma = PersistentClient(path=str(DB_PATH))
        try:
            self.collection = chroma.get_collection(COLLECTION_NAME)
        except Exception as exc:
            raise RuntimeError(
                f"Collection {COLLECTION_NAME!r} is missing. Run `python ingest.py` first."
            ) from exc

    def search_knowledge_base(
        self,
        query: str,
        document_type: str | None = None,
        n_results: int = DEFAULT_SEARCH_RESULTS,
    ) -> dict[str, Any]:
        embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True,
        ).tolist()

        query_args: dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results": max(1, min(n_results, 10)),
            "include": ["documents", "metadatas", "distances"],
        }
        if document_type:
            query_args["where"] = {"type": document_type}

        results = self.collection.query(**query_args)
        matches = []
        for document, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            matches.append(
                {
                    "source": metadata["source"],
                    "chunk": metadata["chunk"],
                    "relevance": round(1 - distance, 4),
                    "content": document,
                }
            )
        return {"query": query, "matches": matches}

    def read_source(self, source: str) -> dict[str, str]:
        requested = (KNOWLEDGE_BASE_PATH / source).resolve()
        root = KNOWLEDGE_BASE_PATH.resolve()
        if root not in requested.parents or requested.suffix.lower() != ".md":
            return {"error": "The source must be a Markdown file inside the knowledge base."}
        if not requested.is_file():
            return {"error": f"Source not found: {source}"}
        return {"source": source, "content": requested.read_text(encoding="utf-8")}

    def execute_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "search_knowledge_base":
            return self.search_knowledge_base(**arguments)
        if name == "read_source":
            return self.read_source(**arguments)
        return {"error": f"Unknown tool: {name}"}

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _complete(self, messages: list[dict], tools: list[dict] | None = None):
        kwargs: dict[str, Any] = {
            "model": AGENT_MODEL,
            "messages": messages,
            "api_base": OLLAMA_BASE_URL,
        }
        if tools:
            kwargs.update({"tools": tools, "tool_choice": "auto"})
        return completion(**kwargs)

    def answer(self, question: str, history: list[dict] | None = None) -> AgentResult:
        if not question.strip():
            return AgentResult(answer="Please ask a question.")

        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history[-8:])
        messages.append({"role": "user", "content": question})

        sources: set[str] = set()
        for step in range(1, MAX_AGENT_STEPS + 1):
            response = self._complete(messages, tools=TOOLS)
            message = response.choices[0].message
            tool_calls = message.tool_calls or []

            if not tool_calls:
                return AgentResult(
                    answer=message.content or "I could not produce an answer.",
                    sources=sorted(sources),
                    steps=step,
                )

            messages.append(message.model_dump(exclude_none=True))
            for tool_call in tool_calls:
                try:
                    arguments = json.loads(tool_call.function.arguments)
                    result = self.execute_tool(tool_call.function.name, arguments)
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    result = {"error": f"Invalid tool arguments: {exc}"}

                if tool_call.function.name == "search_knowledge_base":
                    sources.update(
                        match["source"] for match in result.get("matches", [])
                    )
                elif tool_call.function.name == "read_source" and "source" in result:
                    sources.add(result["source"])

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

        messages.append(
            {
                "role": "user",
                "content": (
                    "You have reached the tool-call limit. Give the best grounded answer now, "
                    "cite the sources you used, and state any uncertainty."
                ),
            }
        )
        final = self._complete(messages)
        return AgentResult(
            answer=final.choices[0].message.content or "I could not produce an answer.",
            sources=sorted(sources),
            steps=MAX_AGENT_STEPS,
        )
