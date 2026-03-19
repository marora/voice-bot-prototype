"""Voice Live event dispatcher — maps server events to callbacks.

The EventDispatcher is the single consumer of the Voice Live WebSocket
event stream.  It translates low-level server events into higher-level
callbacks that the Orchestrator registers:

    on_transcript     — user's speech has been fully transcribed (STT).
    on_audio_delta    — a chunk of synthesised audio is ready (TTS).
    on_speech_started — VAD detected the user started speaking
                        (used for barge-in).
    on_response_done  — the server finished generating a response.

Why ``asyncio.create_task`` for transcripts?
    Transcript handling involves a blocking LangGraph call.  Wrapping
    it in a task prevents it from stalling the event loop and blocking
    subsequent events (e.g. barge-in during processing).
"""

import asyncio
import logging
from typing import Awaitable, Callable, Optional

from azure.ai.voicelive.aio import VoiceLiveConnection
from azure.ai.voicelive.models import ServerEventType

logger = logging.getLogger("voicebot.events")


class EventDispatcher:
    """Dispatches Voice Live server events to registered handlers."""

    def __init__(self):
        self._on_transcript: Optional[Callable[[str], Awaitable[None]]] = None
        self._on_audio_delta: Optional[Callable[[bytes], None]] = None
        self._on_speech_started: Optional[Callable[[], Awaitable[None]]] = None
        self._on_response_done: Optional[Callable[[], Awaitable[None]]] = None
        self._transcript_task: Optional[asyncio.Task] = None

    def on_transcript(self, handler: Callable[[str], Awaitable[None]]) -> None:
        """Register handler for completed STT transcripts."""
        self._on_transcript = handler

    def on_audio_delta(self, handler: Callable[[bytes], None]) -> None:
        """Register handler for TTS audio chunks (raw PCM16 bytes)."""
        self._on_audio_delta = handler

    def on_speech_started(self, handler: Callable[[], Awaitable[None]]) -> None:
        """Register handler for VAD speech-start (barge-in trigger)."""
        self._on_speech_started = handler

    def on_response_done(self, handler: Callable[[], Awaitable[None]]) -> None:
        """Register handler for server response completion."""
        self._on_response_done = handler

    def cancel_transcript_task(self) -> None:
        """Cancel the currently running transcript handler task, if any."""
        if self._transcript_task and not self._transcript_task.done():
            self._transcript_task.cancel()
            self._transcript_task = None

    async def run(self, conn: VoiceLiveConnection) -> None:
        """Main event loop — receive and dispatch Azure Voice Live server events.

        Iterates over the server-sent event stream from the Voice Live WebSocket
        connection. Events are pushed by the service in response to:
        - Session lifecycle (created/updated)
        - Voice Activity Detection (speech start/stop)
        - Speech-to-text transcription (partial deltas and completed)
        - LLM response generation
        - Text-to-speech audio synthesis (deltas and completed)

        This is the single consumer of the connection's event stream. All
        application logic is triggered from handlers registered on this dispatcher.

        See: https://learn.microsoft.com/azure/ai-services/openai/realtime-audio-reference

        Args:
            conn: An active VoiceLiveConnection whose session has already been
                configured via session.update().
        """        
        logger.info("Event dispatcher started")
        async for event in conn:
            # --- Session lifecycle ---
            if event.type == ServerEventType.SESSION_CREATED:
                logger.info("Session created")

            elif event.type == ServerEventType.SESSION_UPDATED:
                logger.info("Session updated")

            # --- VAD (Voice Activity Detection) — server-side ---
            elif event.type == ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STARTED:
                logger.info("Speech started (VAD detected)")
                if self._on_speech_started:
                    await self._on_speech_started()

            elif event.type == ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STOPPED:
                logger.info("Speech stopped (VAD end-of-turn)")

            # --- STT (Speech-to-Text) — server-side transcription ---
            # Fired when the service finishes transcribing a complete utterance.
            # The transcription model and language are set via
            # AudioInputTranscriptionOptions in the session config.
            elif event.type == ServerEventType.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED:
                transcript = event.transcript
                logger.info(f"STT transcript: {transcript!r}")
                if self._on_transcript:
                    # Cancel previous transcript task to prevent stale handlers
                    # from racing with the new one after a barge-in.
                    if self._transcript_task and not self._transcript_task.done():
                        self._transcript_task.cancel()
                    self._transcript_task = asyncio.create_task(self._on_transcript(transcript))

            elif event.type == ServerEventType.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_DELTA:
                logger.debug(f"STT partial: {event.delta!r}")

            # --- TTS (Text-to-Speech) — server-side audio synthesis ---
            elif event.type == ServerEventType.RESPONSE_AUDIO_DELTA:
                if self._on_audio_delta:
                    self._on_audio_delta(event.delta)

            elif event.type == ServerEventType.RESPONSE_AUDIO_DONE:
                logger.info("TTS audio stream complete")

            elif event.type == ServerEventType.RESPONSE_AUDIO_TRANSCRIPT_DELTA:
                logger.debug(f"TTS transcript delta: {event.delta!r}")

            elif event.type == ServerEventType.RESPONSE_AUDIO_TRANSCRIPT_DONE:
                logger.info(f"TTS spoke: {event.transcript!r}")

            # --- Response lifecycle ---
            elif event.type == ServerEventType.RESPONSE_DONE:
                logger.info("Response complete")
                if self._on_response_done:
                    await self._on_response_done()

            elif event.type == ServerEventType.ERROR:
                logger.error(f"Voice Live error: {event.error}")

            else:
                logger.debug(f"Unhandled event: {event.type}")
