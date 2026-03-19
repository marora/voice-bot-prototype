"""Structured logging utilities for the voice bot pipeline.

Provides ``start_timer`` / ``log_stage`` helpers for measuring latency
across pipeline stages (STT, LangGraph, TTS).  Timers are stored in a
module-level dict and consumed on the next ``log_stage`` call for the
same stage name.

Usage::

    from logger import start_timer, log_stage

    start_timer("langgraph")
    # ... do work ...
    log_stage("langgraph", "Agent finished")
    # logs: [LANGGRAPH] [142ms] Agent finished
"""

import logging
import time
from typing import Optional

_stage_timers: dict[str, float] = {}


def start_timer(stage: str) -> None:
    """Start a latency timer for a pipeline stage."""
    _stage_timers[stage] = time.perf_counter()


def log_stage(
    stage: str,
    message: str,
    logger_name: str = "voicebot.pipeline",
    level: int = logging.INFO,
    extra: Optional[dict] = None,
) -> None:
    """Log a pipeline stage with optional latency from timer start."""
    log = logging.getLogger(logger_name)
    elapsed = ""
    if stage in _stage_timers:
        dt = time.perf_counter() - _stage_timers.pop(stage)
        elapsed = f" [{dt*1000:.0f}ms]"
    parts = [f"[{stage.upper()}]{elapsed} {message}"]
    if extra:
        parts.append(" ".join(f"{k}={v}" for k, v in extra.items()))
    log.log(level, " | ".join(parts))
