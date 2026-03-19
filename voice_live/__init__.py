"""Voice Live integration — session, audio I/O, and event dispatch.

This package wraps the Azure Voice Live SDK into three focused modules:

- **session** — WebSocket connection lifecycle and TTS triggering.
- **audio** — PyAudio mic capture / speaker playback with barge-in.
- **events** — Server event dispatcher with callback registration.
"""

from .session import create_session, trigger_tts
from .audio import AudioManager
from .events import EventDispatcher

__all__ = [
    "create_session",
    "trigger_tts",
    "AudioManager",
    "EventDispatcher",
]
