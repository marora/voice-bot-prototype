"""Voice Live session management — STT + VAD only (no auto-LLM).

This module owns the WebSocket lifecycle to Azure Voice Live.
The session is configured with ``create_response=False`` so that
Voice Live performs only:

- **Semantic VAD** — detects when the user finishes speaking.
- **STT** — transcribes the utterance and emits a transcript event.

It does *not* auto-invoke the realtime LLM for a response.  Instead,
the Orchestrator receives the transcript, runs LangGraph, and
explicitly calls ``trigger_tts()`` to synthesise audio from the
LangGraph response text.

Key functions:
    create_session  — async context manager that connects and configures
                      the Voice Live WebSocket.
    trigger_tts     — sends text to Voice Live's ``response.create()``
                      for audio synthesis.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from azure.core.credentials import AzureKeyCredential
from azure.ai.voicelive.aio import connect, VoiceLiveConnection
from azure.ai.voicelive.models import (
    RequestSession,
    Modality,
    InputAudioFormat,
    OutputAudioFormat,
    AzureSemanticVad,
    AzureStandardVoice,
    AudioInputTranscriptionOptions,
    AudioEchoCancellation,
    AudioNoiseReduction,
    ResponseCreateParams,
)

from config import config

logger = logging.getLogger("voicebot.session")


def _build_session_config() -> RequestSession:
    """Build STT-only session config with semantic VAD.

    Important settings:
        - ``create_response=False`` — disables the automatic LLM
          response that Voice Live would normally generate after
          end-of-turn.  This gives the Orchestrator full control.
        - ``threshold`` — VAD sensitivity (0.0–1.0).  Higher values
          require more confidence before triggering end-of-turn.
        - ``silence_duration_ms`` — how long the user must pause
          before VAD considers the turn complete.
    """
    return RequestSession(
        modalities=[Modality.TEXT, Modality.AUDIO],
        instructions=(
            "You are a text-to-speech engine. When given text in the instructions, "
            "repeat it exactly word for word. Do not add, remove, or change any words."
        ),
        voice=AzureStandardVoice(name=config.VOICE_NAME),
        input_audio_format=InputAudioFormat.PCM16,
        output_audio_format=OutputAudioFormat.PCM16,
        turn_detection=AzureSemanticVad(
            create_response=False,  # CRITICAL: disable auto-LLM
            threshold=0.5,
            prefix_padding_ms=300,
            silence_duration_ms=500,
            remove_filler_words=True,
        ),
        input_audio_transcription=AudioInputTranscriptionOptions(
            model=config.STT_MODEL,
            language=config.STT_LANGUAGE,
        ),
        input_audio_echo_cancellation=AudioEchoCancellation(),
        input_audio_noise_reduction=AudioNoiseReduction(
            type="azure_deep_noise_suppression",
        ),
        tools=[],  # No function tools
    )


@asynccontextmanager
async def create_session() -> AsyncIterator[VoiceLiveConnection]:
    """Create and configure a Voice Live session."""
    logger.info(f"Connecting to Voice Live: {config.VOICELIVE_ENDPOINT}")
    async with connect(
        endpoint=config.VOICELIVE_ENDPOINT,
        credential=AzureKeyCredential(config.VOICELIVE_API_KEY),
        model=config.VOICELIVE_MODEL,
    ) as conn:
        session_config = _build_session_config()
        await conn.session.update(session=session_config)
        logger.info("Voice Live session configured (STT-only, create_response=False)")
        yield conn


async def trigger_tts(conn: VoiceLiveConnection, text: str) -> None:
    """Send text to Voice Live for audio synthesis (TTS).

    Uses ``response.create()`` with instructions that ask the model
    to repeat the text verbatim.  The model may occasionally
    paraphrase slightly (~95 %% fidelity) — acceptable for POCs but must validate this assumption.

    The synthesised audio arrives as ``RESPONSE_AUDIO_DELTA`` events
    handled by the EventDispatcher.
    """
    logger.info(f"TTS trigger: {text[:80]}...")
    await conn.response.create(
        response=ResponseCreateParams(
            modalities=[Modality.AUDIO, Modality.TEXT],
            instructions=(
                "Repeat the following text exactly, word for word, "
                f"with no additions or changes:\n\n{text}"
            ),
        ),
        additional_instructions="Do not paraphrase, summarize, or add commentary. Speak the exact text given.",
    )
