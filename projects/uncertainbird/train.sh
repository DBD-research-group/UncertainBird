#!/usr/bin/env bash
set -euo pipefail

# Use your usual interpreter (override with: PYTHON_BIN="poetry run python" ./train.sh ...)
PYTHON_BIN="${PYTHON_BIN:-python}"

# Toggle debugging by exporting DEBUGPY=1
DEBUG_ARGS=()
if [[ "${DEBUGPY:-0}" == "1" ]]; then
  DBG_HOST="${DEBUGPY_HOST:-localhost}"   # use 0.0.0.0 if attaching from outside a container/SSH without VS Code Remote
  DBG_PORT="${DEBUGPY_PORT:-5678}"
  if [[ "${DEBUGPY_WAIT:-1}" == "1" ]]; then WAIT_FLAG=(--wait-for-client); else WAIT_FLAG=(); fi

  # If you ever run multi-process (DDP), only make rank 0 wait:
  if [[ "${LOCAL_RANK:-0}" == "0" ]]; then
    DEBUG_ARGS=(-m debugpy --listen "${DBG_HOST}:${DBG_PORT}" "${WAIT_FLAG[@]}")
  fi
fi

exec ${PYTHON_BIN} "${DEBUG_ARGS[@]}" \
  birdset/train.py \
  --config-path '../projects/uncertainbird/configs' \
  --config-dir 'configs' \
  logger=wandb_UncertainBird \
  "$@"
