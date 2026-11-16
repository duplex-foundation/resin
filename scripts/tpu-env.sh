# Source before any TPU job: `source tpu-env.sh`.
# scripts/tpu-queue.sh sources this file while it holds the chip lock.
# You can also source it yourself and run the job under
# `flock /tmp/resin-tpu-chip0.lock`.
# Pins the process to our claimed slot, 1x (A) on dev-tpu-tpu7x8b-0 chip 0,
# per the Notion TPU allocation table. Never edit to use other chips without
# Daniel's explicit permission.

if [ "$(hostname -s)" != "dev-tpu-tpu7x8b-0" ]; then
  echo "tpu-env.sh: our slot is on dev-tpu-tpu7x8b-0, not $(hostname -s); refusing." >&2
  return 1 2>/dev/null || exit 1
fi

export TPU_VISIBLE_CHIPS=0
export TPU_CHIPS_PER_HOST_BOUNDS=1,1,1
export TPU_RUNTIME_METRICS_PORTS=8431

# The host is part of a 2-host slice; without these JAX tries to join host -1.
export TPU_HOST_BOUNDS=1,1,1
export TPU_WORKER_ID=0
unset TPU_PROCESS_ADDRESSES TPU_WORKER_HOSTNAMES JAX_COORDINATOR_ADDRESS
