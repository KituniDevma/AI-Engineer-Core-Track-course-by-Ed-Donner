"""Gradio chat interface for the Week 5 agentic RAG exercise."""

import gradio as gr

from agentic_rag import AgenticRAG
from config import AGENT_MODEL


agent: AgenticRAG | None = None


def get_agent() -> AgenticRAG:
    global agent
    if agent is None:
        agent = AgenticRAG()
    return agent


def chat(message: str, history: list[dict]) -> str:
    """Answer a chat message and expose retrieval diagnostics."""
    try:
        result = get_agent().answer(message, history)
    except Exception as exc:
        return (
            f"Unable to answer: {exc}\n\n"
            "Make sure Ollama is running, the local models are installed, and run `uv run python "
            "week5/week_5_exercise_agentic_rag/ingest.py` first."
        )

    source_list = "\n".join(f"- `{source}`" for source in result.sources)
    diagnostics = f"\n\n---\n**Agent steps:** {result.steps}"
    if source_list:
        diagnostics += f"\n\n**Retrieved sources:**\n{source_list}"
    return result.answer + diagnostics


demo = gr.ChatInterface(
    fn=chat,
    type="messages",
    title="Insurellm Agentic RAG",
    description=(
        "A Week 5 knowledge worker that decides how to search, refines retrieval, "
        "reads source documents, and answers with citations."
    ),
    examples=[
        "What products does Insurellm offer?",
        "Who is Avery Lancaster and what is their role?",
        "Compare the contracts for Carllm and identify their termination terms.",
    ],
    additional_inputs=[],
)


if __name__ == "__main__":
    print(f"Starting agent with model: {AGENT_MODEL}")
    demo.launch(inbrowser=True)
