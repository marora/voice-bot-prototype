"""Voice Bot Prototype — Azure Voice Live + LangGraph.

Entry point for the voice bot.  Boots the audio subsystem, opens a
Voice Live WebSocket session, and hands off to the Orchestrator which
coordinates the STT -> LangGraph -> TTS pipeline.

Lifecycle:
    1. Configure Python logging (timestamps, level, module name).
    2. Instantiate AudioManager (opens mic + speaker via PyAudio).
    3. Open a Voice Live session (WebSocket to Azure).
    4. Create the Orchestrator and run until Ctrl-C / SIGTERM.
    5. Tear down audio and exit.

Run::

    python main.py
"""
import asyncio
import logging
import signal

from voice_live.session import create_session
from voice_live.audio import AudioManager
from orchestrator import Orchestrator

_LOG_FMT = "%(asctime)s [%(name)-25s] %(levelname)-7s %(filename)s:%(lineno)d — %(message)s"
_LOG_DATEFMT = "%H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=_LOG_FMT,
    datefmt=_LOG_DATEFMT,
)

# Also log to a rotating file so sessions can be reviewed after the fact.
_file_handler = logging.FileHandler("voicebot.log", mode="a", encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter(_LOG_FMT, datefmt=_LOG_DATEFMT))
logging.getLogger().addHandler(_file_handler)

logger = logging.getLogger("voicebot.main")

# Reduce noise from dependencies
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("aiohttp").setLevel(logging.WARNING)


async def main():
    """Boot audio, connect to Voice Live, and run the orchestrator."""
    audio = AudioManager()
    try:
        audio.start()
        async with create_session() as conn:
            orchestrator = Orchestrator(conn, audio)

            loop = asyncio.get_event_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, lambda: asyncio.create_task(_shutdown(audio)))

            logger.info("=" * 60)
            logger.info("Voice Bot ready — speak into your microphone")
            logger.info("Press Ctrl+C to stop")
            logger.info("=" * 60)

            await orchestrator.run()
    except KeyboardInterrupt:
        pass
    finally:
        audio.stop()
        logger.info("Voice Bot stopped")


async def _shutdown(audio: AudioManager):
    """Graceful shutdown — stop audio then cancel all async tasks."""
    audio.stop()
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for t in tasks:
        t.cancel()


if __name__ == "__main__":
    asyncio.run(main())
