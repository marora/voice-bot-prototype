"""LangGraph agent public interface.

This package contains the multi-agent reasoning pipeline.  The
Orchestrator calls ``stream_agent(text)`` to process a user utterance
and receive the response token-by-token.

Architecture:
    A **supervisor** node inspects the user message and routes it to
    one of four specialist sub-agents:

    - **search** — factual lookups (stub in this prototype).
    - **calculator** — arithmetic evaluation.
    - **weather** — weather queries (stub).
    - **general** — catch-all conversational responses.

    Each sub-agent is a ``create_react_agent`` with its own tools.
    The graph is defined in ``graph.py``; tools live in ``tools.py``.
"""
import logging
from typing import AsyncIterator

from .graph import graph

logger = logging.getLogger("voicebot.langgraph")

_GRAPH_CONFIG = {"configurable": {"thread_id": "voice-session"}}


async def invoke_agent(user_text: str) -> str:
    """Invoke the LangGraph agent and return the full response text."""
    logger.info(f"[LANGGRAPH] invoke_agent() called — routing user text to supervisor graph")
    logger.info(f"[LANGGRAPH] Input: {user_text!r}")
    result = await graph.ainvoke({"messages": [("user", user_text)]}, _GRAPH_CONFIG)
    response = result["messages"][-1].content
    logger.info(f"[LANGGRAPH] invoke_agent() response ({len(response)} chars): {response!r}")
    return response


async def stream_agent(user_text: str) -> AsyncIterator[str]:
    """Stream LangGraph agent response token-by-token.

    Uses ``astream_events`` to capture LLM streaming tokens across the
    *entire* execution tree — including nested sub-agents created with
    ``create_react_agent``.  The previous ``astream(stream_mode="messages")``
    approach only captured direct LLM calls within the outer graph nodes,
    missing all responses from sub-agent graphs.
    """
    logger.info(f"[LANGGRAPH] stream_agent() called — starting streaming from supervisor graph")
    logger.info(f"[LANGGRAPH] Input: {user_text!r}")
    full_response = ""
    async for event in graph.astream_events(
        {"messages": [("user", user_text)]},
        _GRAPH_CONFIG,
        version="v2",
    ):
        if event["event"] != "on_chat_model_stream":
            continue
        # Skip the supervisor's routing LLM call — it only emits a
        # one-word routing label (e.g. "search") that must not reach TTS.
        node = event.get("metadata", {}).get("langgraph_node", "")
        if node == "supervisor":
            continue
        chunk = event["data"]["chunk"]
        if hasattr(chunk, "content") and chunk.content and isinstance(chunk.content, str):
            full_response += chunk.content
            yield chunk.content
    logger.info(f"[LANGGRAPH] stream_agent() complete ({len(full_response)} chars): {full_response!r}")
