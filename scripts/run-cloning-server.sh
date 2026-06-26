#!/usr/bin/env bash
# Run the FILE-REF voice-cloning server (tts_server.py, mlx-audio backend) on :8880.
#
#   ./scripts/run-cloning-server.sh                 # run in the foreground
#   ./scripts/run-cloning-server.sh --install-launchd    # KeepAlive LaunchAgent (auto-start)
#   ./scripts/run-cloning-server.sh --uninstall-launchd  # remove the LaunchAgent
#
# Runs from the repo root so engine.py scans THIS dir for <id>_ref.wav/.txt — the same
# directory the sablier-weblogon voice-enrollment container bind-mounts as /tts-voices.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT"
VENV_DIR="${QWEN_TTS_CLONING_VENV:-$REPO_ROOT/.venv-cloning}"
LABEL="com.phenom.qwen3-tts"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY="$(command -v python3)"
[ -f "$REPO_ROOT/.env-cloning" ] && { set -a; . "$REPO_ROOT/.env-cloning"; set +a; }
: "${TTS_QWEN_MODEL:=mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16}"

case "${1:-}" in
  --install-launchd)
    cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array>
    <string>$PY</string><string>$REPO_ROOT/tts_server.py</string>
  </array>
  <key>WorkingDirectory</key><string>$REPO_ROOT</string>
  <key>EnvironmentVariables</key><dict>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    <key>TTS_QWEN_MODEL</key><string>$TTS_QWEN_MODEL</string>
    <key>TTS_DEFAULT_BACKEND</key><string>qwen</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$REPO_ROOT/qwen3-tts.stdout.log</string>
  <key>StandardErrorPath</key><string>$REPO_ROOT/qwen3-tts.stderr.log</string>
</dict></plist>
PL
    launchctl unload "$PLIST" 2>/dev/null || true
    launchctl load "$PLIST"
    echo "installed + loaded $LABEL (model=$TTS_QWEN_MODEL)"; exit 0 ;;
  --uninstall-launchd)
    launchctl unload "$PLIST" 2>/dev/null || true; rm -f "$PLIST"; echo "removed $LABEL"; exit 0 ;;
esac

echo "Starting Qwen3-TTS cloning server :8880 (model=$TTS_QWEN_MODEL, voices dir=$REPO_ROOT)"
exec "$PY" "$REPO_ROOT/tts_server.py"
