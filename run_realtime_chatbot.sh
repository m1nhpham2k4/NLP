#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

SYSTEM="S8_FULL"

CMD=(python3 main.py serve-chat \
  --host 127.0.0.1 \
  --port 7860 \
  --system "$SYSTEM" \
  --models miniLM,e5,bge_m3,vietnamese_bi \
  --vector-store qdrant \
  --qdrant-mode local \
  --qdrant-path ./qdrant_storage \
  --device auto \
  --history-k 3 \
  --history-file results/web_chat_history.json)

if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  CMD+=(--openai)
fi

CMD+=("$@")

echo "Đang chạy chế độ: $SYSTEM"
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  echo "OpenAI: bật cho S8_FULL"
else
  echo "OpenAI: tắt tự động; có thể thêm --openai nếu muốn bật thủ công"
fi

exec "${CMD[@]}"
