#!/bin/bash
# Serve a registered TPU preset under the chip-0 queue.
#
# Set RESIN_TPU_QUEUE to tpu-queue.sh. This wrapper does not take the lock
# itself. The queue sources tpu-env.sh, logs the command, and holds chip 0
# until the server exits.
#
#   RESIN_TPU_QUEUE=/path/to/tpu-queue.sh scripts/tpu_serve.sh \
#       --model qwen3-32b --checkpoint PATH --workload 1k/8k

if [ "${BASH_SOURCE[0]}" != "$0" ]; then
  echo "tpu_serve.sh: run it; do not source it" >&2
  return 1
fi

set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
queue=${RESIN_TPU_QUEUE:-}
if [ -z "$queue" ] || [ ! -x "$queue" ]; then
  echo "tpu_serve.sh: set RESIN_TPU_QUEUE to the executable tpu-queue.sh" >&2
  exit 1
fi

exec "$queue" env RESIN_TPU_QUEUED=1 /opt/pearl/tpu/bin/python "$here/tpu_serve.py" "$@"
