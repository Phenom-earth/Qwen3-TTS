#!/usr/bin/env bash
# =============================================================================
# Qwen3-TTS: run the OpenAI-compatible voice server (Mac Studio, bare metal)
# =============================================================================
# Loads .env (written by setup-macstudio.sh) and launches the server.
#
#   ./scripts/run-server.sh                 # run in the foreground
#   ./scripts/run-server.sh --install-launchd   # install + load a LaunchAgent (auto-start at login)
#   ./scripts/run-server.sh --uninstall-launchd # remove the LaunchAgent
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR="${QWEN_TTS_VENV:-$REPO_ROOT/.venv}"
PY="$VENV_DIR/bin/python"
LABEL="com.phenom.qwen3tts"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

[ -x "$PY" ] || { echo "venv missing; run ./scripts/setup-macstudio.sh first" >&2; exit 1; }
# Load .env if present (export every assignment).
if [ -f "$REPO_ROOT/.env" ]; then set -a; . "$REPO_ROOT/.env"; set +a; fi

case "${1:-}" in
  --install-launchd)
    mkdir -p "$HOME/Library/LaunchAgents"
    cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$REPO_ROOT/scripts/run-server.sh</string>
  </array>
  <key>WorkingDirectory</key><string>$REPO_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$REPO_ROOT/qwen3tts.out.log</string>
  <key>StandardErrorPath</key><string>$REPO_ROOT/qwen3tts.err.log</string>
</dict>
</plist>
EOF
    launchctl unload "$PLIST" 2>/dev/null || true
    launchctl load "$PLIST"
    echo "LaunchAgent installed and loaded: $PLIST"
    echo "Logs: $REPO_ROOT/qwen3tts.{out,err}.log"
    exit 0
    ;;
  --uninstall-launchd)
    launchctl unload "$PLIST" 2>/dev/null || true
    rm -f "$PLIST"
    echo "LaunchAgent removed."
    exit 0
    ;;
esac

echo "Starting Qwen3-TTS server on ${QWEN_TTS_HOST:-0.0.0.0}:${QWEN_TTS_PORT:-8880} (model=${QWEN_TTS_MODEL:-default})"
exec "$PY" server/qwen_tts_server.py
