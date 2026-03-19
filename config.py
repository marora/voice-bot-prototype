"""Centralised configuration loaded from environment variables.

All settings are read once at import time via the singleton ``config``
object.  Required variables (no default) raise ``KeyError`` on startup
if missing — this is intentional so misconfigurations fail fast.

See .env.example for a full template with descriptions.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration backed by environment variables.

    Two Azure resources are required:

    1. **Voice Live** — provides the real-time WebSocket for VAD, STT,
       and TTS.  Set via ``AZURE_VOICELIVE_*`` env vars.
    2. **Azure OpenAI** — powers the LangGraph reasoning agents.
       Set via ``AZURE_OPENAI_*`` env vars.
    """

    # ── Voice Live (required) ───────────────────────────────────
    VOICELIVE_ENDPOINT: str = os.environ["AZURE_VOICELIVE_ENDPOINT"]
    VOICELIVE_API_KEY: str = os.environ["AZURE_VOICELIVE_API_KEY"]
    VOICELIVE_MODEL: str = os.getenv("AZURE_VOICELIVE_MODEL", "gpt-4o")

    # ── Azure OpenAI for LangGraph (required) ───────────────────
    OPENAI_ENDPOINT: str = os.environ["AZURE_OPENAI_ENDPOINT"]
    OPENAI_API_KEY: str = os.environ["AZURE_OPENAI_API_KEY"]
    OPENAI_DEPLOYMENT: str = os.getenv(
        "AZURE_OPENAI_DEPLOYMENT", "gpt-4o",
    )
    OPENAI_API_VERSION: str = os.getenv(
        "AZURE_OPENAI_API_VERSION", "2024-12-01-preview",
    )

    # ── Voice / STT settings ────────────────────────────────────
    VOICE_NAME: str = os.getenv(
        "VOICE_NAME", "en-US-AvaMultilingualNeural",
    )
    STT_LANGUAGE: str = os.getenv("STT_LANGUAGE", "en-US")
    STT_MODEL: str = os.getenv("STT_MODEL", "azure-speech")

    # ── Audio hardware settings ─────────────────────────────────
    # Must match Voice Live's expected format: PCM16, mono, 24 kHz.
    SAMPLE_RATE: int = int(os.getenv("AUDIO_SAMPLE_RATE", "24000"))
    CHANNELS: int = int(os.getenv("AUDIO_CHANNELS", "1"))
    CHUNK_MS: int = int(os.getenv("AUDIO_CHUNK_MS", "50"))

    @property
    def chunk_size(self) -> int:
        """Number of samples per audio frame.

        Calculated from sample rate and chunk duration.
        Example: 24000 Hz * 50 ms / 1000 = 1200 samples.
        """
        return int(self.SAMPLE_RATE * self.CHUNK_MS / 1000)


# Singleton — import this from other modules.
config = Config()
