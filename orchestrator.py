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
import re
import time
from azure.ai.voicelive.aio import VoiceLiveConnection

from voice_live.session import trigger_tts
from voice_live.audio import AudioManager
from voice_live.events import EventDispatcher
from langgraph_agent import stream_agent

logger = logging.getLogger("voicebot.orchestrator")

# Sentence-boundary chunking constants for progressive TTS
_SENTENCE_END = re.compile(r'[.!?;]\s|[.!?;]$|\n')
_MAX_CHUNK_CHARS = 200
_MIN_CHUNK_CHARS = 10


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
        self._turn_id = 0  # monotonically increasing turn counter
        self._tts_turn_id = 0  # turn_id of the most recently triggered TTS
        self._pending_tts_count = 0  # in-flight TTS chunks awaiting RESPONSE_DONE
        self._barge_in_event = asyncio.Event()
        self._tts_slot = asyncio.Semaphore(1)  # guards sequential trigger_tts() calls

        # Wire up event handlers
        self._dispatcher.on_transcript(self._handle_transcript)
        self._first_audio_logged = False

        def _audio_with_log(data: bytes):
            if not self._first_audio_logged:
                self._first_audio_logged = True
                logger.info("[PLAYBACK] First audio delta received — playback starting")
            self._audio.play_audio(data)

        self._dispatcher.on_audio_delta(_audio_with_log)
        self._dispatcher.on_speech_started(self._handle_barge_in)
        self._dispatcher.on_response_done(self._handle_response_done)

    async def run(self) -> None:
        """Start mic capture and event processing concurrently."""
        mic_task = asyncio.create_task(self._audio.capture_mic_loop(self._conn))
        event_task = asyncio.create_task(self._dispatcher.run(self._conn))

        # Proactive greeting on launch
        self._processing = True
        self._tts_turn_id = self._turn_id
        self._pending_tts_count = 1
        await self._tts_slot.acquire()
        await trigger_tts(self._conn, self.GREETING)
        logger.info("[PIPELINE] Greeting TTS triggered")

        logger.info("Orchestrator running — speak into your microphone")
        try:
            await asyncio.gather(mic_task, event_task)
        except asyncio.CancelledError:
            logger.info("Orchestrator shutting down")

    async def _handle_transcript(self, transcript: str) -> None:
        """Handle completed STT transcript — route to LangGraph with progressive TTS."""
        if not transcript.strip():
            logger.debug("Empty transcript, skipping")
            return

        self._turn_id += 1
        my_turn = self._turn_id

        self._processing = True
        self._barge_in_event.clear()
        self._audio.resume()
        t_start = time.perf_counter()
        logger.info(f"[STT] Transcript received (turn {my_turn}): {transcript!r}")

        try:
            buffer = ""
            full_response = ""
            chunk_index = 0
            self._pending_tts_count = 0
            self._first_audio_logged = False

            logger.info("[PIPELINE] Starting progressive TTS — streaming from LangGraph")

            async for token in stream_agent(transcript):
                if self._barge_in_event.is_set() or self._turn_id != my_turn:
                    logger.info("[PIPELINE] Barge-in detected during LangGraph streaming, aborting")
                    return
                buffer += token
                full_response += token

                # Check for sentence boundary
                match = _SENTENCE_END.search(buffer)
                if match and len(buffer) >= _MIN_CHUNK_CHARS:
                    end_pos = match.end()
                    chunk_text = buffer[:end_pos].strip()
                    buffer = buffer[end_pos:]
                    if chunk_text:
                        chunk_index += 1
                        sent = await self._flush_chunk(chunk_text, chunk_index, my_turn, t_start)
                        if not sent:
                            return
                elif len(buffer) >= _MAX_CHUNK_CHARS:
                    chunk_text = buffer.strip()
                    buffer = ""
                    if chunk_text:
                        chunk_index += 1
                        sent = await self._flush_chunk(chunk_text, chunk_index, my_turn, t_start)
                        if not sent:
                            return

            # Final flush
            if buffer.strip() and not self._barge_in_event.is_set() and self._turn_id == my_turn:
                chunk_index += 1
                await self._flush_chunk(buffer.strip(), chunk_index, my_turn, t_start)

            logger.info(
                f"[PIPELINE] Progressive TTS complete: {chunk_index} chunks, "
                f"{len(full_response)} total chars, {time.perf_counter() - t_start:.2f}s"
            )

            # Guard: if LangGraph produced no output, reset pipeline immediately
            if chunk_index == 0 and not self._barge_in_event.is_set() and self._turn_id == my_turn:
                logger.warning(
                    "[PIPELINE] Empty agent response — no TTS chunks to send, resetting pipeline"
                )
                self._processing = False
                self._audio.resume()
                return

        except asyncio.CancelledError:
            logger.info("[PIPELINE] Processing cancelled by barge-in")
        except Exception as e:
            logger.error(f"[PIPELINE] Error: {e}", exc_info=True)
        finally:
            if self._turn_id == my_turn and not self._barge_in_event.is_set():
                if self._pending_tts_count == 0:
                    # Safety net: no pending TTS means no RESPONSE_DONE will come
                    self._processing = False
                    self._audio.resume()
                # else: _processing cleared by _handle_response_done
            else:
                self._processing = False

    async def _flush_chunk(
        self, text: str, index: int, my_turn: int, t_start: float
    ) -> bool:
        """Acquire the TTS slot, send a chunk, and return True on success.

        Returns False (and does not send) if a barge-in occurred or
        the turn has moved on while waiting for the slot.
        """
        self._pending_tts_count += 1
        self._tts_turn_id = my_turn
        await self._tts_slot.acquire()
        if self._barge_in_event.is_set() or self._turn_id != my_turn:
            self._tts_slot.release()
            return False
        logger.info(
            f"[CHUNK {index}] TTS triggered at +{time.perf_counter() - t_start:.2f}s "
            f"({len(text)} chars): {text[:80]}..."
        )
        await trigger_tts(self._conn, text)
        return True

    async def _handle_barge_in(self) -> None:
        """Handle user interruption — cancel server response and local playback."""
        if not self._processing and not self._audio.is_playing():
            return
        logger.info("[BARGE-IN] User interrupted — cancelling response and clearing buffers")
        # Signal processing to abort and invalidate the current turn
        self._barge_in_event.set()
        self._turn_id += 1
        self._pending_tts_count = 0
        # Release the TTS slot so nothing blocks forever
        try:
            self._tts_slot.release()
        except ValueError:
            pass  # already released
        # Stop local audio immediately
        self._audio.interrupt()
        # Cancel the in-flight transcript handler task
        self._dispatcher.cancel_transcript_task()
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
        """Handle response completion.

        Each TTS chunk fires its own RESPONSE_DONE.  We decrement the
        pending counter and only mark the turn complete when all chunks
        have finished.
        """
        if self._turn_id != self._tts_turn_id:
            logger.debug("[PIPELINE] Stale response_done (turn moved on), ignoring")
            return
        self._pending_tts_count = max(0, self._pending_tts_count - 1)
        # Release the TTS slot so the next chunk can fire
        try:
            self._tts_slot.release()
        except ValueError:
            pass  # already released (e.g. greeting or barge-in reset)
        if self._pending_tts_count == 0:
            self._processing = False
            self._audio.resume()
            logger.info("[PIPELINE] All TTS chunks complete — ready for next input")
        else:
            logger.debug(f"[PIPELINE] TTS chunk complete, {self._pending_tts_count} remaining")
