#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

CMD=(python3 main.py chat \
  --system S8_FULL \
  --models miniLM,e5,bge_m3,vietnamese_bi \
  --vector-store qdrant \
  --qdrant-mode local \
  --qdrant-path ./qdrant_storage \
  --device auto \
  --history-k 3 \
  --history-file results/chat_history.json)

if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  CMD+=(--openai)
fi

CMD+=("$@")

exec "${CMD[@]}"
