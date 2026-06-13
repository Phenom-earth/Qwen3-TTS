# Qwen3-TTS Voice Server: Setup Guide (Phenom)

This guide is for **Logan**. It sets up the Qwen3-TTS server that gives PAI (and
every developer's Code Server) a single shared voice endpoint.

## Architecture

```
   Mac Studio (bare metal, Apple Silicon)
   ┌──────────────────────────────────────────┐
   │  Qwen3-TTS server  (this repo)            │
   │  uvicorn :8880   bind 0.0.0.0             │
   │  POST /v1/audio/speech  (OpenAI-style)    │
   └───────────────▲──────────────────────────┘
                   │  host.docker.internal:8880
   ┌───────────────┼──────────────────────────┐
   │  OrbStack     │                           │
   │  ┌──────────┐ │  ┌──────────┐  ┌────────┐ │
   │  │ dev-alice│─┘  │ dev-bob  │  │ dev-…  │ │   each Code Server container
   │  │ (PAI)    │    │ (PAI)    │  │        │ │   reaches the host server
   │  └──────────┘    └──────────┘  └────────┘ │
   └──────────────────────────────────────────┘
```

The model needs native Metal acceleration, so the server runs on the **host
(bare metal)**, not in a container. Each Code Server container reaches it across
the container-to-host boundary at `http://host.docker.internal:8880`.

## Prerequisites

- macOS on Apple Silicon (Mac Studio).
- `python3` 3.10+ (Xcode Command Line Tools or `brew install python@3.11`).
- Optional but recommended: [`uv`](https://docs.astral.sh/uv/) (`brew install uv`).
- ~10 GB free disk for the 1.7B model (use the 0.6B model for less).
- A Hugging Face account is not required for public Qwen models.

## 1. Set up the server (one command)

```bash
git clone https://github.com/Phenom-earth/Qwen3-TTS.git
cd Qwen3-TTS
./scripts/setup-macstudio.sh
```

This creates a virtualenv, installs `qwen_tts` plus the server requirements,
downloads the model + tokenizer into `./models`, and writes a `.env`. To use the
smaller model:

```bash
QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice ./scripts/setup-macstudio.sh
```

## 2. Run the server

```bash
./scripts/run-server.sh                 # foreground
# or, auto-start at login and keep alive:
./scripts/run-server.sh --install-launchd
```

## 3. Verify (on the Mac Studio)

```bash
curl -s http://127.0.0.1:8880/readyz                       # {"status":"ready",...}
curl -s http://127.0.0.1:8880/v1/voices                    # supported speakers + languages
curl -s http://127.0.0.1:8880/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Phenom voice online.","voice":"narrator","response_format":"wav"}' \
  --output /tmp/test.wav && afplay /tmp/test.wav
```

## 4. Wire the Code Server containers

Each container reaches the host server at `host.docker.internal`. The Phenom
dev-environment image is already wired for this: it sets `QWEN_TTS_URL` and adds
the `host.docker.internal` host mapping in `docker-compose.yml`. Nothing to do
per developer beyond having the server running on the Mac Studio.

From inside any container you can confirm reachability:

```bash
curl -s http://host.docker.internal:8880/readyz
```

If you run a container by hand (not via the compose file), add the host mapping
and point PAI at the endpoint:

```bash
docker run \
  --add-host host.docker.internal:host-gateway \
  -e QWEN_TTS_URL=http://host.docker.internal:8880/v1 \
  ghcr.io/phenom-earth/dev-environment:latest
```

## 5. Point PAI's voice at the server

PAI's voice layer speaks the OpenAI "audio/speech" dialect, so it just needs the
base URL. Set it from `QWEN_TTS_URL` (already present in the container):

- Endpoint: `POST $QWEN_TTS_URL/audio/speech`
- The PAI voice name (e.g. `narrator`) is sent as the `voice` field.

Map PAI's `narrator` to a real Qwen speaker on the **server** via `.env`:

```bash
# .env on the Mac Studio; see the model's own speakers with /v1/voices
QWEN_TTS_VOICE_ALIASES={"narrator":"Ethan"}
```

Unknown voices fall back to the default speaker, so generic OpenAI voice names
still produce audio.

## Security

When the server binds `0.0.0.0` it is reachable by anything on the local
network. Set a shared bearer token in `.env` and the server will require it:

```bash
QWEN_TTS_API_KEY=some-long-random-token
```

Clients (PAI / containers) then send `Authorization: Bearer some-long-random-token`.
The server logs a warning at startup if it binds a non-loopback host without a token.

## API reference

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/v1/audio/speech` | OpenAI-compatible TTS. Body: `input`, `voice`, `response_format` (wav/pcm/flac/ogg/opus), `instructions` (optional), `language` (optional). Returns audio bytes. |
| GET | `/v1/voices` | The model's supported speakers + languages + configured aliases. |
| GET | `/v1/models` | Minimal OpenAI-style model list. |
| GET | `/healthz` | Liveness. |
| GET | `/readyz` | Readiness (model loaded). |

## Configuration (`.env`)

| Variable | Default | Notes |
|----------|---------|-------|
| `QWEN_TTS_MODEL` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | HF id or local dir |
| `QWEN_TTS_HOST` | `0.0.0.0` | must be non-loopback for containers to reach it |
| `QWEN_TTS_PORT` | `8880` | |
| `QWEN_TTS_DEVICE` | `mps` | `mps` on Apple Silicon, `cpu` to force CPU |
| `QWEN_TTS_DTYPE` | `bfloat16` | |
| `QWEN_TTS_DEFAULT_SPEAKER` | (first supported) | |
| `QWEN_TTS_DEFAULT_LANGUAGE` | `Auto` | |
| `QWEN_TTS_VOICE_ALIASES` | `{}` | JSON map of voice name to Qwen speaker |
| `QWEN_TTS_API_KEY` | (none) | bearer token; set when binding `0.0.0.0` |

## Troubleshooting

- **Container cannot reach the server**: confirm the server is bound `0.0.0.0`
  (not `127.0.0.1`) on the Mac Studio, and that the container has the
  `host.docker.internal` mapping (`curl http://host.docker.internal:8880/readyz`).
- **`mps` out of memory**: switch to the 0.6B model, or set `QWEN_TTS_DEVICE=cpu`.
- **First request is slow**: the model loads at startup; `/readyz` returns
  `loading` until it is ready, then `ready`.
- **401 from the server**: a `QWEN_TTS_API_KEY` is set; send the matching
  `Authorization: Bearer` header.
