"""Orchestrator — connects STT -> LangGraph -> TTS pipeline.

This is the central coordination module.  It wires together:

- **EventDispatcher** (voice_live/events.py) — receives server events
  from the Voice Live WebSocket (transcripts, audio deltas, etc.).
- **AudioManager** (voice_live/audio.py) — captures mic input and
  plays synthesised audio through the speaker.
- **LangGraph agent** (langgraph_agent/) — processes user utterances
  and returns text responses.
- **trigger_tts** (voice_live/session.py) — sends response text to
  Voice Live for audio synthesis.

Concurrency model:
    Two long-running async tasks execute in parallel:
    1. ``AudioManager.capture_mic_loop`` — reads mic frames and pushes
       them to Voice Live.
    2. ``EventDispatcher.run`` — iterates over server-sent events and
       dispatches to registered callbacks.

    Audio *playback* runs on a dedicated background thread so that
    writing PCM chunks to the speaker never blocks the async event loop
    (critical for responsive barge-in handling).
"""
import asyncio
import logging
import time
from typing import Optional

from azure.ai.voicelive.aio import VoiceLiveConnection

from voice_live.session import trigger_tts
from voice_live.audio import AudioManager
from voice_live.events import EventDispatcher
from langgraph_agent import stream_agent

logger = logging.getLogger("voicebot.orchestrator")


class Orchestrator:
    """Coordinates the voice bot pipeline.

    Lifecycle of a single conversational turn:
        1. Voice Live fires ``on_transcript`` with the user's utterance.
        2. ``_handle_transcript`` streams the text through LangGraph.
        3. The full LangGraph response is sent to Voice Live via TTS.
        4. Audio deltas arrive and are enqueued for playback.
        5. ``on_response_done`` marks the turn complete.

    Barge-in:
        If the user speaks while the bot is still responding,
        ``on_speech_started`` fires.  The handler immediately
        interrupts local playback, cancels the server-side response,
        and resets state so the new utterance can be processed.
    """

    GREETING = "Hello! I'm your voice assistant. How can I help you today?"

    def __init__(self, conn: VoiceLiveConnection, audio: AudioManager):
        self._conn = conn
        self._audio = audio
        self._dispatcher = EventDispatcher()
        self._processing = False  # True while handling a turn
        self._current_task: Optional[asyncio.Task] = None
        self._barge_in_event = asyncio.Event()

        # Wire up event handlers
        self._dispatcher.on_transcript(self._handle_transcript)
        self._dispatcher.on_audio_delta(self._audio.play_audio)
        self._dispatcher.on_speech_started(self._handle_barge_in)
        self._dispatcher.on_response_done(self._handle_response_done)

    async def run(self) -> None:
        """Start mic capture and event processing concurrently."""
        mic_task = asyncio.create_task(self._audio.capture_mic_loop(self._conn))
        event_task = asyncio.create_task(self._dispatcher.run(self._conn))

        # Proactive greeting on launch
        self._processing = True
        await trigger_tts(self._conn, self.GREETING)
        logger.info("[PIPELINE] Greeting TTS triggered")

        logger.info("Orchestrator running — speak into your microphone")
        try:
            await asyncio.gather(mic_task, event_task)
        except asyncio.CancelledError:
            logger.info("Orchestrator shutting down")

    async def _handle_transcript(self, transcript: str) -> None:
        """Handle completed STT transcript — route to LangGraph."""
        if not transcript.strip():
            logger.debug("Empty transcript, skipping")
            return

        self._processing = True
        self._barge_in_event.clear()
        self._audio.resume()  # ensure playback is allowed after any prior interrupt
        t_start = time.perf_counter()
        logger.info(f"[STT] Transcript received: {transcript!r}")
        logger.info(f"[PIPELINE] Routing to LangGraph agent")

        try:
            full_response = ""
            async for token in stream_agent(transcript):
                if self._barge_in_event.is_set():
                    logger.info("[PIPELINE] Barge-in detected during LangGraph streaming, aborting")
                    return
                full_response += token

            t_langgraph = time.perf_counter()
            logger.info(
                f"[LANGGRAPH] Response complete ({t_langgraph - t_start:.2f}s, "
                f"{len(full_response)} chars): {full_response[:100]}..."
            )

            # Check barge-in one more time before triggering TTS
            if self._barge_in_event.is_set():
                logger.info("[PIPELINE] Barge-in detected before TTS trigger, aborting")
                return

            await trigger_tts(self._conn, full_response)
            logger.info(f"[TTS] Triggered ({time.perf_counter() - t_start:.2f}s total pipeline)")

        except asyncio.CancelledError:
            logger.info("[PIPELINE] Processing cancelled by barge-in")
        except Exception as e:
            logger.error(f"[PIPELINE] Error: {e}", exc_info=True)
        finally:
            if not self._barge_in_event.is_set():
                pass  # _processing cleared by _handle_response_done
            else:
                self._processing = False

    async def _handle_barge_in(self) -> None:
        """Handle user interruption — cancel server response and local playback."""
        if not self._processing:
            return
        logger.info("[BARGE-IN] User interrupted — cancelling response and clearing buffers")
        # Signal processing to abort
        self._barge_in_event.set()
        # Stop local audio immediately
        self._audio.interrupt()
        # Cancel server-side response (may be a no-op if LangGraph still running)
        try:
            await self._conn.response.cancel()
        except Exception as e:
            logger.debug(f"[BARGE-IN] response.cancel() error (expected if no active response): {e}")
        try:
            await self._conn.output_audio_buffer.clear()
        except Exception as e:
            logger.debug(f"[BARGE-IN] output_audio_buffer.clear() error: {e}")
        self._processing = False
        logger.info("[BARGE-IN] Interruption handled — ready for new input")

    async def _handle_response_done(self) -> None:
        """Handle response completion."""
        self._processing = False
        self._audio.resume()
        logger.info("[PIPELINE] Turn complete — ready for next input")
