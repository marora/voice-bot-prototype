"""LangGraph supervisor agent with sub-agent routing.

Graph topology::

    START -> supervisor -> conditional_edges -> {search, calculator, weather, general} -> END

The supervisor is a lightweight LLM call that classifies the user
message into one of the ``MEMBERS`` categories.  Each sub-agent is a
``create_react_agent`` with its own system prompt and tool set.

To add a new specialist:
    1. Create its tool(s) in ``tools.py``.
    2. Build a ``create_react_agent`` with those tools.
    3. Add a node + edge in the graph below.
    4. Update ``MEMBERS`` and the supervisor prompt.
"""
import logging
from typing import Literal

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent

from config import config
from .tools import search_tool, calculator_tool, weather_tool

logger = logging.getLogger("voicebot.langgraph")


def _create_llm() -> AzureChatOpenAI:
    """Create an AzureChatOpenAI instance from app config."""
    return AzureChatOpenAI(
        azure_endpoint=config.OPENAI_ENDPOINT,
        api_key=config.OPENAI_API_KEY,
        azure_deployment=config.OPENAI_DEPLOYMENT,
        api_version=config.OPENAI_API_VERSION,
    )


# ── Sub-agents ──────────────────────────────────────────────────
# Each sub-agent is a self-contained ReAct agent with its own tools
# and system prompt.  They receive the user message from the supervisor
# and return a single response message.

_search_agent = create_react_agent(
    _create_llm(),
    tools=[search_tool],
    prompt="You are a search specialist. Use the search tool to find information. Be concise.",
)

_calc_agent = create_react_agent(
    _create_llm(),
    tools=[calculator_tool],
    prompt="You are a calculator specialist. Use the calculator tool for math. Return just the answer.",
)

_weather_agent = create_react_agent(
    _create_llm(),
    tools=[weather_tool],
    prompt="You are a weather specialist. Provide weather info concisely.",
)

# ── Routing ─────────────────────────────────────────────────────
# The supervisor is a zero-tool LLM that classifies the user message
# into one of the MEMBERS categories.  It replies with a single word.

MEMBERS = ["search", "calculator", "weather", "general"]

_supervisor_llm = _create_llm().bind_tools([], tool_choice="none")

_SUPERVISOR_PROMPT = f"""You are a routing supervisor. Given the user's message, decide which specialist should handle it.
Reply with ONLY one word — the specialist name.

Specialists: {', '.join(MEMBERS)}
- search: factual questions, lookups, information retrieval
- calculator: math, arithmetic, calculations
- weather: weather inquiries
- general: everything else (greetings, conversation, opinions)
"""


async def _supervisor(state: MessagesState) -> dict:
    """Route to the appropriate sub-agent."""
    messages = [SystemMessage(content=_SUPERVISOR_PROMPT)] + state["messages"]
    response = await _supervisor_llm.ainvoke(messages)
    route = response.content.strip().lower()
    logger.info(f"Supervisor routed to: {route}")
    return {"messages": state["messages"], "_route": route}


def _route_from_supervisor(state: dict) -> str:
    route = state.get("_route", "general")
    if route in MEMBERS:
        return route
    return "general"


async def _search_node(state: MessagesState) -> dict:
    result = await _search_agent.ainvoke({"messages": state["messages"]})
    return {"messages": result["messages"][-1:]}


async def _calc_node(state: MessagesState) -> dict:
    result = await _calc_agent.ainvoke({"messages": state["messages"]})
    return {"messages": result["messages"][-1:]}


async def _weather_node(state: MessagesState) -> dict:
    result = await _weather_agent.ainvoke({"messages": state["messages"]})
    return {"messages": result["messages"][-1:]}


async def _general_node(state: MessagesState) -> dict:
    llm = _create_llm()
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}


# ── Build the StateGraph ────────────────────────────────────────
# Nodes: supervisor -> conditional routing -> specialist -> END
workflow = StateGraph(MessagesState)
workflow.add_node("supervisor", _supervisor)
workflow.add_node("search", _search_node)
workflow.add_node("calculator", _calc_node)
workflow.add_node("weather", _weather_node)
workflow.add_node("general", _general_node)

workflow.add_edge(START, "supervisor")
workflow.add_conditional_edges("supervisor", _route_from_supervisor, MEMBERS)
for member in MEMBERS:
    workflow.add_edge(member, END)

memory = InMemorySaver()
graph = workflow.compile(checkpointer=memory)
