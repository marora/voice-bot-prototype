---
title: "Voice Bot Prototype: Azure Voice Live + LangGraph"
description: "Real-time voice bot using Azure Voice Live API for STT/VAD with create_response=False, enabling external LangGraph orchestration for reasoning. Demonstrates a pattern with no public Azure sample equivalent."
author: Mani Arora
ms.date: 2026-03-20
ms.topic: concept
keywords:
  - azure-voice-live
  - create_response=false
  - langgraph
  - external-orchestration
  - semantic-vad
  - realtime-api
---

Real-time voice bot using Azure Voice Live API for STT/VAD with external LangGraph orchestration for reasoning.

## Why This Exists

No public Azure sample demonstrates `create_response=False` with semantic VAD and an external reasoning pipeline. Existing samples either use default auto-response behavior or push-to-talk without VAD. This prototype fills that gap by combining Voice Live's STT/VAD with a LangGraph supervisor agent for reasoning and tool use.

## Architecture

```text
            [ Mic ]
               |
               |  PCM16 audio
               v
+-----------------------------+
|   Azure Voice Live (WSS)    |
|-----------------------------|
|  Semantic VAD               |
|  STT (create_response=False)|
+--------------+--------------+
               |
               |  transcript
               v
    +---------------------+
    | Python Orchestrator  |
    |---------------------|
    | Sentence chunking    |
    | Barge-in control     |
    +----------+----------+
               |
               |  query
               v
    +---------------------+
    | LangGraph Supervisor |
    |---------------------|
    | Search  | Calculator |
    | Weather | General    |
    +----------+----------+
               |
               |  streamed tokens
               |  (chunked by Orchestrator)
               v
+-----------------------------+
|  Voice Live TTS             |
|-----------------------------|
|  response.create(chunk)     |
|  (same WSS connection)      |
+--------------+--------------+
               |
               |  audio deltas
               v
          [ Speaker ]
```

```mermaid
flowchart TD
    Mic(["🎤 Mic"]) -->|"PCM16 audio"| VL
    VL["Azure Voice Live · WSS<br/>Semantic VAD · STT<br/>create_response=False"] -->|"transcript"| Orch
    Orch["Python Orchestrator<br/>Sentence chunking · Barge-in"] -->|"query"| LG
    LG -->|"streamed tokens"| Orch
    Orch -->|"response.create(chunk)"| VL
    VL -->|"audio deltas"| Spk(["🔊 Speaker"])

    subgraph LG["LangGraph Supervisor"]
        direction LR
        S["Search"] ~~~ C["Calculator"] ~~~ W["Weather"] ~~~ G["General"]
    end
```

### Data Flow

1. **Mic** captures audio and streams PCM16 to Voice Live via WebSocket.
2. **Voice Live** runs semantic VAD to detect end-of-turn, then transcribes speech (`create_response=False` prevents auto-LLM).
3. The **orchestrator** routes the transcript through the LangGraph supervisor, which delegates to the appropriate sub-agent (search, calculator, weather, or general).
4. The **orchestrator** streams the LangGraph response token-by-token, chunking at sentence boundaries, and sends each chunk to Voice Live via `response.create(instructions=chunk)` for progressive audio synthesis. Audio deltas stream to the speaker as each chunk is synthesized.
5. If the user speaks during playback (barge-in), the response is cancelled and a new turn begins.

## Prerequisites

- **Python 3.9+**
- **PortAudio** system library (for PyAudio):
  ```bash
  # Ubuntu/Debian
  sudo apt-get install portaudio19-dev libasound2-dev
  # macOS
  brew install portaudio
  ```
- **Azure OpenAI resource** with:
  - Voice Live model (e.g., `gpt-4o`), fully managed, no deployment needed
  - `gpt-4o` deployment (for LangGraph agents via Azure OpenAI)
- **Microphone and speaker** (or headphones to avoid echo)

## Setup

```bash
# 1. Create and activate virtual environment
cd voice-bot-prototype
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your Azure credentials (see "Getting API Keys" below)
```

### Getting API Keys

```bash
# 1. Authenticate
az login

# 2. Retrieve key
az cognitiveservices account keys list \
  --name <RESOURCE_NAME> \
  --resource-group <RESOURCE_GROUP> \
  --query key1 -o tsv
```

<details>
<summary>Troubleshooting: disableLocalAuth is enabled</summary>

Azure AI Services resources provisioned through Microsoft Foundry may have local authentication disabled by default. Enable it with:

```bash
az rest --method patch \
  --url "https://management.azure.com/subscriptions/<SUB_ID>/resourceGroups/<RG>/providers/Microsoft.CognitiveServices/accounts/<NAME>?api-version=2024-10-01" \
  --body '{ "properties": { "disableLocalAuth": false } }'
```

Then retry the key retrieval command. The toggle may revert if an Azure Policy enforces it. For production, use Entra ID authentication.

</details>

### Required Environment Variables

| Variable | Description |
|---|---|
| `AZURE_VOICELIVE_ENDPOINT` | WSS endpoint for Voice Live (e.g., `wss://your-resource.openai.azure.com`) |
| `AZURE_VOICELIVE_API_KEY` | API key for the Voice Live resource |
| `AZURE_OPENAI_ENDPOINT` | HTTPS endpoint for Azure OpenAI (e.g., `https://your-resource.openai.azure.com/`) |
| `AZURE_OPENAI_API_KEY` | API key for Azure OpenAI |

Optional variables with defaults are documented in `.env.example`.

## Usage

```bash
source venv/bin/activate
python main.py
```

Speak into your microphone. The bot will:
- Greet you proactively on launch
- Detect when you stop speaking (semantic VAD)
- Transcribe your speech
- Route to the appropriate LangGraph agent
- Speak the response back through your speaker

All sessions are logged to `voicebot.log` for post-session review.

Press **Ctrl+C** to stop.

## Project Structure

```text
voice-bot-prototype/
├── main.py               # Entry point: logging, lifecycle, signal handling
├── orchestrator.py        # Core pipeline: STT → LangGraph → TTS
├── config.py              # Environment variable loading
├── logger.py              # Structured stage-level logging with latency tracking
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variable template
├── voice_live/
│   ├── __init__.py        # Package exports (create_session, trigger_tts)
│   ├── session.py         # Voice Live session setup (create_response=False)
│   ├── audio.py           # PyAudio mic capture + speaker playback
│   └── events.py          # Server event dispatcher with callbacks
└── langgraph_agent/
    ├── __init__.py         # Public API (stream_agent, invoke_agent)
    ├── graph.py            # Supervisor StateGraph + sub-agent routing
    └── tools.py            # Sample tools (search, calculator, weather stubs)
```

## Architecture Decision: Path A vs Path B

This prototype uses **Path A**, a single Voice Live WebSocket for both STT and TTS:

| Aspect | Path A (Prototype) | Path B (Production) |
|---|---|---|
| **TTS mechanism** | Voice Live `response.create(instructions=text)` | Azure Speech SDK text streaming |
| **Verbatim fidelity** | Model may paraphrase (~95% accurate) | 100% verbatim guaranteed |
| **Connections** | Single WebSocket | Two WebSockets (Voice Live + Speech SDK) |
| **Barge-in** | Native support | Manual implementation needed |
| **Cost** | Higher (Voice Live model for TTS) | Lower (Azure Neural TTS pricing) |
| **Latency** | Model inference per chunk (progressive chunking reduces time-to-first-audio) | ~200-500ms first byte with text streaming |
| **Complexity** | Low | Medium |

### Path B Production Upgrade

For production verbatim TTS, replace `response.create` with Azure Speech SDK text streaming (`SpeechSynthesisRequest(input_type=TextStream)` feeding LangGraph tokens progressively). This gives 100% fidelity, lower cost, and better latency.

## Key Learnings and Gotchas

### SDK discriminator fields: always set `type` explicitly

Azure Voice Live model classes use a `type` discriminator for polymorphic serialization, but not all classes auto-populate it (SDK v1.1.0). Notably, `AudioNoiseReduction()` without `type` serializes to `{}`, causing the service to reject the config and close the WebSocket after 5 retries. Always pass `type` explicitly (valid values: `"azure_deep_noise_suppression"`, `"near_field"`, `"far_field"`) and verify with `obj.as_dict()`.

### Barge-in requires careful state management

When the user speaks during bot playback:

1. Voice Live fires `input_audio_buffer.speech_started`.
2. The orchestrator must cancel the in-flight `response`, clear the local audio playback queue, and reset turn state.
3. Expected errors like `response_cancel_not_active` and `already_has_active_response` appear during barge-in and can be safely ignored.
4. A debounce window (1.5 s in this prototype) prevents rapid re-triggers.

### Progressive TTS improves perceived latency

Rather than accumulating the full LangGraph response before triggering TTS, this prototype chunks the response at sentence boundaries and sends multiple `response.create` calls. Each chunk starts synthesizing immediately, reducing time-to-first-audio. The tradeoff is managing multiple in-flight responses and handling barge-in across chunk boundaries.

### `astream_events` v2 for nested sub-agents

LangGraph's `astream(stream_mode="messages")` only captures LLM tokens from direct graph nodes. For supervisor patterns with `create_react_agent` sub-agents, use `graph.astream_events(version="v2")` and filter for `on_chat_model_stream` events. Skip the supervisor node's routing tokens (they emit one-word labels like "search") by checking `event.metadata.langgraph_node`.

## Known Limitations

- Search, calculator, and weather tools are stubs returning fake data. Replace with real APIs for production.
- Audio uses PyAudio for local mic/speaker. For browser clients, consider WebRTC or a server-side WebSocket relay.
- STT is configured for `en-US` only. Multi-language support requires dynamic `AzureSemanticVad.languages` configuration.
- LangGraph uses `InMemorySaver` for checkpoints with no persistence across restarts.

## References

| Resource | What it covers |
|---|---|
| [Azure OpenAI Realtime Audio Reference](https://learn.microsoft.com/azure/ai-services/openai/realtime-audio-reference) | Full event catalog, `RealtimeTurnDetection` schema including `create_response` boolean, `response.create` event structure |
| [How to use the Realtime API](https://learn.microsoft.com/azure/ai-services/openai/how-to/realtime-audio) | "VAD without automatic response generation" section with JSON example of `create_response: false` |
| [Azure Voice Live API Overview](https://learn.microsoft.com/azure/ai-services/openai/concepts/voice-live-api) | Managed speech-to-speech layer, supported models (gpt-realtime, gpt-4o, gpt-4.1, gpt-5, phi4), noise suppression, echo cancellation, advanced end-of-turn detection |
| [AzureSemanticVad Python SDK Reference](https://learn.microsoft.com/python/api/azure-ai-voicelive/azure.ai.voicelive.models.azuresemanticvad) | SDK class docs: `create_response`, `interrupt_response`, `threshold`, `silence_duration_ms`, `prefix_padding_ms`, `eagerness`, `languages`, `remove_filler_words` |
| [OpenAI Realtime API Guide](https://platform.openai.com/docs/guides/realtime) | Upstream OpenAI docs for the Realtime API protocol that Azure Voice Live is compatible with |

### Related Samples

| Sample | Pattern | How it differs from this prototype |
|---|---|---|
| [VoiceRAG (aisearch-openai-rag-audio)](https://github.com/Azure-Samples/aisearch-openai-rag-audio) | Middle-tier WebSocket proxy that intercepts tool calls server-side and injects RAG results | Uses `create_response=True` (default). The realtime model still does all reasoning; the middle tier just handles tool execution. |
| [aoai-realtime-audio-sdk](https://github.com/Azure-Samples/aoai-realtime-audio-sdk) | Official Python/JS SDK samples for the Realtime Audio API | Demonstrates `NoTurnDetection()` for push-to-talk and `ServerVAD` for auto-detection, but no sample sets `create_response=False` with VAD. |
| [azure-ai-voice-live-samples](https://github.com/Azure-Samples/azure-ai-voice-live-samples) | Voice Live SDK quickstarts and feature demos | Showcases Voice Live features (noise suppression, echo cancellation, semantic VAD) with default auto-response behavior. |

This prototype fills a gap by combining VAD-based turn detection with full external control over the reasoning and response pipeline.
