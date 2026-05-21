#!/usr/bin/env bash
# Download embedding weights for eval (if not already cached).
# Emotion/sentiment: set EMOTION_MODEL_PATH / SENTIMENT_MODEL_PATH in .env to local folders.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HF="${ROOT}/.venv/bin/hf"

if [[ ! -x "$HF" ]]; then
  echo "Missing hf CLI. Run: source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

# shellcheck disable=SC1091
source "${ROOT}/.env" 2>/dev/null || true
EMBED="${EMBED_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"

echo "Embedding model (shared with inference): $EMBED"
echo "Cache: ${HF_HOME:-$HOME/.cache/huggingface}"
echo ""

if [[ -n "${EMOTION_MODEL_PATH:-}" && -d "$EMOTION_MODEL_PATH" ]]; then
  echo "Skipping emotion download (EMOTION_MODEL_PATH=$EMOTION_MODEL_PATH)"
else
  echo "Tip: set EMOTION_MODEL_PATH in .env to a local model folder to skip hub download."
fi

if [[ -n "${SENTIMENT_MODEL_PATH:-}" && -d "$SENTIMENT_MODEL_PATH" ]]; then
  echo "Skipping sentiment download (SENTIMENT_MODEL_PATH=$SENTIMENT_MODEL_PATH)"
else
  echo "Tip: set SENTIMENT_MODEL_PATH in .env to a local model folder to skip hub download."
fi

echo ""
echo "==> $EMBED"
"$HF" download "$EMBED" --exclude "*.h5" --exclude "tf_*"
echo ""
echo "Done."
