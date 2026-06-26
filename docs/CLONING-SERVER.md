# File-ref voice-cloning server (Phenom dev-environment)

This fork carries **two** OpenAI-compatible servers on top of the Qwen3-TTS models:

| Server | Files | Cloning model | What it serves |
| :-- | :-- | :-- | :-- |
| **CustomVoice** | `server/qwen_tts_server.py` | `1.7B-CustomVoice` | The model's 9 built-in premium timbres via `/v1/audio/speech`. No per-user file cloning. |
| **File-ref cloning** | `tts_server.py` + `backends/` + `engine.py` | `*-Base` (mlx-audio) | Per-developer **cloned** voices from `<id>_ref.wav` + `<id>_ref.txt`. |

The **Phenom dev-environment access layer** (`Phenom-earth/sablier-weblogon`) drives the
**file-ref cloning server** — it is the one that implements the contract the enrollment
service and the persona voice-gate depend on:

- `GET  /v1/voices` — lists the **file-cloned** voices (`{voices:[{id}, ...]}`).
- `POST /v1/voices/reload` — rescans the voices directory.
- `POST /v1/voices/{id}/reclone` — rebuilds a voice's embedding after a re-enrollment.

The server code is the **proven SanMarcSoft reference deploy**, brought in unchanged
(verified end-to-end: Cloudflare/Keycloak SSO → provision → enroll → clone → editor).

## Voices directory

`engine.py` scans **its own directory** (the repo root) for `<id>_ref.wav` + `<id>_ref.txt`.
On the Mac Studio the `sablier-weblogon` voice-enrollment container bind-mounts that
directory as `/tts-voices`, writes the reference pair there, then calls `/v1/voices/reload`
+ `/v1/voices/{id}/reclone`.

## Model — quality vs footprint

`TTS_QWEN_MODEL` selects the mlx-audio **Base** model used for cloning:

| Host | Default | Why |
| :-- | :-- | :-- |
| **Mac Studio** (Phenom) | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` | Higher-quality clone; the Mac Studio has the headroom. |
| Constrained box | `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit` | Lighter; what the RAM-tight reference box ran. |

## Bring-up (Mac Studio)

```bash
./scripts/setup-cloning-macstudio.sh                 # venv + deps + 1.7B bf16 model
./scripts/run-cloning-server.sh --install-launchd    # KeepAlive LaunchAgent: com.phenom.qwen3-tts

# Verify the cloning contract:
curl -s http://127.0.0.1:8880/v1/voices
curl -s -X POST http://127.0.0.1:8880/v1/voices/reload
```

The `sablier-weblogon` installer (`bin/install.sh`) clones this fork and runs the two
scripts above as its Qwen3-TTS provisioning step.
