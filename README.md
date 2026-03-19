# Voice Bot Prototype — Azure Voice Live + LangGraph

Real-time voice bot using Azure Voice Live API for STT/VAD with external LangGraph orchestration for reasoning.

## Architecture

```text
┌──────────┐     ┌─────────────────────────────────────────────────┐     ┌──────────┐
│          │     │              Python Orchestrator                │     │          │
│   Mic    │────▶│                                                 │     │ Speaker  │
│          │     │  ┌──────────────────────────────────────────┐  │     │          │
└──────────┘     │  │  Azure Voice Live (WSS)                  │  │     └──────────┘
                 │  │  • Semantic VAD (end-of-turn detection)  │  │          ▲
                 │  │  • STT (create_response=False)           │  │          │
                 │  │  • TTS via response.create(instructions) │──┼──────────┘
                 │  └──────────────┬───────────────────────────┘  │   audio deltas
                 │                 │ transcript                    │
                 │                 ▼                               │
                 │  ┌──────────────────────────────────────────┐  │
                 │  │  LangGraph Supervisor Agent               │  │
                 │  │  ┌──────────┐ ┌────────┐ ┌───────────┐  │  │
                 │  │  │ Search   │ │ Calc   │ │ Weather   │  │  │
                 │  │  │ Agent    │ │ Agent  │ │ Agent     │  │  │
                 │  │  └──────────┘ └────────┘ └───────────┘  │  │
                 │  └──────────────────────────────────────────┘  │
                 └─────────────────────────────────────────────────┘
```

### Data Flow

1. **Mic** captures audio → streams to Voice Live via WebSocket
2. **Voice Live** runs semantic VAD to detect end-of-turn, then transcribes speech (STT)
3. **Orchestrator** intercepts the transcript (`create_response=False` prevents auto-LLM)
4. **LangGraph Supervisor** routes to the appropriate sub-agent (search, calculator, weather, or general)
5. **Sub-agent** processes the query and returns a text response
6. **Orchestrator** sends the response text to Voice Live via `response.create(instructions=text)`
7. **Voice Live** synthesizes audio (TTS via gpt-4o-realtime) and streams audio deltas
8. **Speaker** plays the audio deltas in real time
9. **Barge-in**: if the user speaks during playback, the response is cancelled

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
  - Voice Live model (e.g., `gpt-4o`) — fully managed, no deployment needed
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

Azure AI Services resources created through Microsoft Foundry may have local authentication disabled by default (`disableLocalAuth=true`) due to tenant governance policies. Follow these steps to retrieve your API key:

```bash
# 1. Authenticate via Azure CLI
az login

# 2. Attempt to list keys
az cognitiveservices account keys list \
  --name mani-voice-live-bot-resource \
  --resource-group rg-mani-voice-live \
  --query key1 -o tsv
```

If you get `(BadRequest) Failed to list key. disableLocalAuth is set to be true`, enable local auth first:

```bash
# 3. Enable local authentication on the resource
az rest \
  --method patch \
  --url "https://management.azure.com/subscriptions/<SUBSCRIPTION_ID>/resourceGroups/<RESOURCE_GROUP>/providers/Microsoft.CognitiveServices/accounts/<RESOURCE_NAME>?api-version=2024-10-01" \
  --body '{ "properties": { "disableLocalAuth": false } }'

# 4. Verify the setting was updated
az rest \
  --method get \
  --url "https://management.azure.com/subscriptions/<SUBSCRIPTION_ID>/resourceGroups/<RESOURCE_GROUP>/providers/Microsoft.CognitiveServices/accounts/<RESOURCE_NAME>?api-version=2024-10-01" \
  --query "properties.disableLocalAuth"
# Expected output: false

# 5. Now retrieve the key
az cognitiveservices account keys list \
  --name mani-voice-live-bot-resource \
  --resource-group rg-mani-voice-live \
  --query key1 -o tsv
```

> **Note**: The `disableLocalAuth` toggle may revert if an Azure Policy enforces it at the subscription or management group level. For production, consider using Entra ID (managed identity) authentication instead of API keys.

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
- Detect when you stop speaking (semantic VAD)
- Transcribe your speech
- Route to the appropriate LangGraph agent
- Speak the response back through your speaker

Press **Ctrl+C** to stop.

## Project Structure

```text
voice-bot-prototype/
├── main.py               # Entry point — logging, lifecycle, signal handling
├── orchestrator.py        # Core pipeline: STT → LangGraph → TTS
├── config.py              # Environment variable loading
├── logger.py              # Structured stage-level logging with latency tracking
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variable template
├── voice_live/
│   ├── session.py         # Voice Live session setup (create_response=False)
│   ├── audio.py           # PyAudio mic capture + speaker playback
│   └── events.py          # Server event dispatcher with callbacks
└── langgraph_agent/
    ├── graph.py            # Supervisor StateGraph + sub-agent routing
    └── tools.py            # Sample tools (search, calculator, weather stubs)
```

## Architecture Decision: Path A vs Path B

This prototype uses **Path A** — a single Voice Live WebSocket for both STT and TTS:

| Aspect | Path A (Prototype) | Path B (Production) |
|---|---|---|
| **TTS mechanism** | Voice Live `response.create(instructions=text)` | Azure Speech SDK text streaming |
| **Verbatim fidelity** | Model may paraphrase (~95% accurate) | 100% verbatim guaranteed |
| **Connections** | Single WebSocket | Two WebSockets (Voice Live + Speech SDK) |
| **Barge-in** | Native support | Manual implementation needed |
| **Cost** | Higher (Voice Live model for TTS) | Lower (Azure Neural TTS pricing) |
| **Latency** | Model inference per TTS call | ~200-500ms first byte with text streaming |
| **Complexity** | Low | Medium |

### Path B Production Upgrade

For production, replace the TTS mechanism with Azure Speech SDK text streaming:

1. Install `azure-cognitiveservices-speech`
2. Configure Speech SDK with `wss://{region}.tts.speech.microsoft.com/cognitiveservices/websocket/v2`
3. Use `SpeechSynthesisRequest(input_type=TextStream)` to feed LangGraph tokens progressively
4. Set `Raw24Khz16BitMonoPcm` output format (matches Voice Live's PCM16)
5. Connect `Synthesizing` event callback to stream audio chunks to the client

Benefits: 100% verbatim output, lower cost, better latency with progressive text feeding.

## Known Limitations

- **TTS paraphrasing**: The gpt-4o-realtime model may slightly rephrase the LangGraph response despite verbatim prompting. Acceptable for demos.
- **Double LLM cost**: Both LangGraph agents and the realtime model consume tokens because Voice Live uses the model for audio synthesis.
- **No progressive TTS**: The full LangGraph response is accumulated before TTS starts. Sentence-level chunking is a planned optimization.
- **Stub tools**: Search, calculator, and weather tools return fake data. Replace with real APIs for production.
- **Local audio only**: Uses PyAudio for mic/speaker. For browser clients, consider WebRTC.

## Quick Reference for Developers

### How a Single Turn Works

1. `main.py` boots audio + Voice Live session, hands off to `Orchestrator`
2. `AudioManager.capture_mic_loop()` streams PCM16 mic audio to Voice Live
3. Voice Live detects end-of-turn via semantic VAD and emits a transcript event
4. `EventDispatcher` fires `on_transcript` → `Orchestrator._handle_transcript()`
5. Orchestrator streams the transcript through `langgraph_agent.stream_agent()`
6. The supervisor graph routes to the right sub-agent (search/calc/weather/general)
7. Full response text is sent to Voice Live via `trigger_tts()` (`response.create`)
8. Voice Live synthesizes audio and pushes `RESPONSE_AUDIO_DELTA` events
9. `EventDispatcher` fires `on_audio_delta` → `AudioManager.play_audio()` (enqueued)
10. Playback thread writes PCM chunks to speaker; `RESPONSE_DONE` marks turn end

### Key Design Decisions

- **`create_response=False`** in VAD config prevents Voice Live from auto-generating an LLM response, giving us full control over the reasoning pipeline.
- **Playback runs on a separate thread** so the async event loop stays responsive for barge-in detection.
- **ALSA errors are suppressed** at the C library level since they are harmless on headless Linux servers.
