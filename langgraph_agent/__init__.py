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

from langchain_core.messages import AIMessageChunk

from .graph import graph

logger = logging.getLogger("voicebot.langgraph")


async def invoke_agent(user_text: str) -> str:
    """Invoke the LangGraph agent and return the full response text."""
    logger.info(f"[LANGGRAPH] invoke_agent() called — routing user text to supervisor graph")
    logger.info(f"[LANGGRAPH] Input: {user_text!r}")
    result = await graph.ainvoke({"messages": [("user", user_text)]})
    response = result["messages"][-1].content
    logger.info(f"[LANGGRAPH] invoke_agent() response ({len(response)} chars): {response!r}")
    return response


async def stream_agent(user_text: str) -> AsyncIterator[str]:
    """Stream LangGraph agent response token-by-token."""
    logger.info(f"[LANGGRAPH] stream_agent() called — starting streaming from supervisor graph")
    logger.info(f"[LANGGRAPH] Input: {user_text!r}")
    full_response = ""
    async for chunk in graph.astream(
        {"messages": [("user", user_text)]},
        stream_mode="messages",
    ):
        message, metadata = chunk
        if isinstance(message, AIMessageChunk) and message.content:
            full_response += message.content
            yield message.content
    logger.info(f"[LANGGRAPH] stream_agent() complete ({len(full_response)} chars): {full_response!r}")
