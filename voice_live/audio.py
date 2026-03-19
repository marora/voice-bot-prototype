"""Audio I/O — PyAudio mic capture and speaker playback.

Design notes
------------
- **Mic capture** runs in the async event loop, offloading blocking
  ``stream.read()`` calls to a thread-pool executor.
- **Speaker playback** runs on a dedicated daemon thread that pulls
  PCM chunks from a ``queue.Queue``.  This decouples playback from the
  event loop so that barge-in events can be processed without waiting
  for the current audio chunk to finish writing.
- **Interrupt / resume** — ``interrupt()`` sets a flag and drains the
  queue so playback stops within one chunk (~50 ms).  ``resume()``
  clears the flag before the next response starts.
- ALSA/JACK stderr noise on headless Linux is suppressed by installing
  a no-op error handler on ``libasound.so.2`` before PyAudio init.
"""

import asyncio
import base64
import ctypes
import logging
import queue
import threading
from typing import Optional

import pyaudio

from config import config

logger = logging.getLogger("voicebot.audio")

# Audio format — must match Voice Live's PCM16 mono 24 kHz format.
FORMAT = pyaudio.paInt16
CHANNELS = config.CHANNELS
RATE = config.SAMPLE_RATE
CHUNK_SIZE = config.chunk_size  # samples per frame (e.g. 1200 for 50 ms)


def _suppress_alsa_errors():
    """Redirect ALSA/JACK stderr noise during PortAudio init."""
    try:
        asound = ctypes.cdll.LoadLibrary("libasound.so.2")
        c_error_handler = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int,
                                           ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
        asound.snd_lib_error_set_handler(c_error_handler(lambda *_: None))
    except OSError:
        pass  # libasound not available — nothing to suppress


class AudioManager:
    """Manages PyAudio input/output streams with interrupt-safe playback."""

    def __init__(self):
        self._pa: Optional[pyaudio.PyAudio] = None
        self._input_stream: Optional[pyaudio.Stream] = None
        self._output_stream: Optional[pyaudio.Stream] = None
        self._running = False
        # Playback queue + thread for non-blocking, interruptible output
        self._playback_queue: queue.Queue[Optional[bytes]] = queue.Queue()
        self._playback_thread: Optional[threading.Thread] = None
        self._interrupted = threading.Event()

    def start(self) -> None:
        """Open mic + speaker streams and launch the playback thread."""
        _suppress_alsa_errors()
        self._pa = pyaudio.PyAudio()
        self._input_stream = self._pa.open(
            format=FORMAT, channels=CHANNELS, rate=RATE,
            input=True, frames_per_buffer=CHUNK_SIZE,
        )
        self._output_stream = self._pa.open(
            format=FORMAT, channels=CHANNELS, rate=RATE,
            output=True, frames_per_buffer=CHUNK_SIZE,
        )
        self._running = True
        self._playback_thread = threading.Thread(
            target=self._playback_worker, daemon=True, name="audio-playback",
        )
        self._playback_thread.start()
        logger.info(f"Audio started: {RATE}Hz, {CHANNELS}ch, chunk={CHUNK_SIZE}")

    def stop(self) -> None:
        """Shut down audio: signal playback thread, close streams."""
        self._running = False
        # Signal playback thread to exit
        self._interrupted.set()
        self._playback_queue.put(None)  # sentinel
        if self._playback_thread:
            self._playback_thread.join(timeout=2)
        if self._input_stream:
            self._input_stream.stop_stream()
            self._input_stream.close()
        if self._output_stream:
            self._output_stream.stop_stream()
            self._output_stream.close()
        if self._pa:
            self._pa.terminate()
        logger.info("Audio stopped")

    def interrupt(self) -> None:
        """Immediately stop playback and discard queued audio."""
        self._interrupted.set()
        # Drain the queue
        while not self._playback_queue.empty():
            try:
                self._playback_queue.get_nowait()
            except queue.Empty:
                break
        logger.info("Audio playback interrupted and queue cleared")

    def resume(self) -> None:
        """Allow playback again after an interruption."""
        self._interrupted.clear()

    def _playback_worker(self) -> None:
        """Background thread that plays audio chunks from the queue."""
        while self._running:
            try:
                chunk = self._playback_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if chunk is None:
                break  # sentinel — shutdown
            if self._interrupted.is_set():
                continue  # discard chunk
            try:
                self._output_stream.write(chunk)
            except Exception as e:
                if self._running:
                    logger.error(f"Playback error: {e}")

    async def capture_mic_loop(self, conn) -> None:
        """Continuously read mic audio and send to Voice Live."""
        loop = asyncio.get_event_loop()
        logger.info("Mic capture started")
        while self._running:
            try:
                raw = await loop.run_in_executor(
                    None, self._input_stream.read, CHUNK_SIZE, False
                )
                b64 = base64.b64encode(raw).decode("utf-8")
                await conn.input_audio_buffer.append(audio=b64)
            except Exception as e:
                if self._running:
                    logger.error(f"Mic capture error: {e}")
                break

    def play_audio(self, audio_bytes: bytes) -> None:
        """Enqueue audio bytes for playback (non-blocking)."""
        if self._running and not self._interrupted.is_set():
            self._playback_queue.put(audio_bytes)
