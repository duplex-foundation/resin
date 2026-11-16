#!/bin/bash
# Queue one chip-0 job.
#
# Takes the same exclusive flock other agents take on
# /tmp/resin-tpu-chip0.lock, sources tpu-env.sh, appends a JSONL record
# of the image, commit, and command, then runs the command.
#
#   scripts/tpu-queue.sh /opt/pearl/tpu/bin/python -c '...'
#   RESIN_IMAGE=vllm/vllm-tpu:nightly scripts/tpu-queue.sh docker run ...
#
# Run this script. Do not source it, and do not start another flock on
# this file from inside the job. That second flock waits on the one this
# script already holds.
#
# TPU_ENV_SH      env file (default: tpu-env.sh beside this script)
# RESIN_TPU_LOCK  lock file (default: /tmp/resin-tpu-chip0.lock)
# RESIN_TPU_LOG   JSONL path (default: /tmp/resin-tpu-chip0.jsonl)
# RESIN_IMAGE     image name when the command does not name one

if [ "${BASH_SOURCE[0]}" != "$0" ]; then
  echo "tpu-queue.sh: run it; do not source it" >&2
  return 1
fi

set -euo pipefail

lock=${RESIN_TPU_LOCK:-/tmp/resin-tpu-chip0.lock}
log=${RESIN_TPU_LOG:-/tmp/resin-tpu-chip0.jsonl}
here=$(cd "$(dirname "$0")" && pwd)
env_sh=${TPU_ENV_SH:-$here/tpu-env.sh}

if [ "${RESIN_TPU_QUEUE_HELD:-}" != 1 ]; then
  if [ $# -eq 0 ]; then
    echo "usage: tpu-queue.sh command [args...]" >&2
    exit 2
  fi
  if [ ! -f "$env_sh" ]; then
    echo "tpu-queue.sh: no env file at $env_sh (set TPU_ENV_SH)" >&2
    exit 1
  fi
  # The re-exec must see this. Without it, the script flocks again and
  # waits on the lock it already holds.
  export RESIN_TPU_QUEUE_HELD=1
  export TPU_ENV_SH=$env_sh
  export RESIN_TPU_LOCK=$lock
  export RESIN_TPU_LOG=$log
  # Same `flock <file> <command>` other agents run. Do not pass --close:
  # that drops the lock before the command starts.
  exec flock "$lock" "$0" "$@"
fi

# shellcheck disable=SC1090
source "$env_sh"

cmd=("$@")
host_short=$(hostname -s)

record() {
  EVENT=$1 STATUS=${2-} QUEUE_PID=$$ HOST_SHORT=$host_short START_EPOCH=${start_epoch-} \
    python3 - "$log" "${cmd[@]}" <<'PY'
import json, os, subprocess, sys, time

log_path = sys.argv[1]
argv = sys.argv[2:]
event = os.environ["EVENT"]

def git(*args):
    try:
        done = subprocess.run(
            ["git", "--no-optional-locks", *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if done.returncode != 0:
        return None
    return done.stdout

def image_from(argv):
    # ponytail: common `docker run` value-flags only. Set RESIN_IMAGE
    # when a flag this list misses would be read as the image name.
    named = os.environ.get("RESIN_IMAGE")
    if named:
        return named
    if not argv or os.path.basename(argv[0]) not in {"docker", "podman"}:
        return None
    try:
        index = argv.index("run", 1)
    except ValueError:
        return None
    index += 1
    takes_value = {
        "-e", "--env", "-v", "--volume", "-p", "--publish", "-w", "--workdir",
        "--name", "--entrypoint", "-u", "--user", "--network", "--net",
        "--mount", "-l", "--label", "--hostname", "--runtime", "--pull",
        "--platform", "-m", "--memory", "--cpus", "--device", "--ipc",
        "--pid", "--ulimit", "--add-host", "--env-file", "--group-add",
        "--gpus", "--restart", "--shm-size", "--security-opt", "--tmpfs",
        "--sysctl", "--cap-add", "--cap-drop", "--log-driver", "--log-opt",
    }
    while index < len(argv):
        arg = argv[index]
        if arg == "--":
            return None
        if arg in takes_value:
            index += 2
            continue
        if arg.startswith("-"):
            index += 1
            continue
        return arg
    return None

record = {
    "event": event,
    "pid": int(os.environ["QUEUE_PID"]),
    "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "argv": argv,
}
if event == "start":
    head = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain")
    record.update(
        host=os.environ["HOST_SHORT"],
        cwd=os.getcwd(),
        commit=head.strip() if head else None,
        dirty=dirty.splitlines() if dirty else [],
        image=image_from(argv),
        tpu_visible_chips=os.environ.get("TPU_VISIBLE_CHIPS"),
        tpu_chips_per_host_bounds=os.environ.get("TPU_CHIPS_PER_HOST_BOUNDS"),
        tpu_host_bounds=os.environ.get("TPU_HOST_BOUNDS"),
        tpu_worker_id=os.environ.get("TPU_WORKER_ID"),
    )
else:
    record["status"] = int(os.environ["STATUS"])
    record["elapsed_s"] = round(time.time() - float(os.environ["START_EPOCH"]), 3)

with open(log_path, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(record) + "\n")
PY
}

record start
start_epoch=$(date +%s.%N)
# A child that calls this script again must take the lock itself.
# Inheriting the flag would let that call skip the queue.
unset RESIN_TPU_QUEUE_HELD
set +e
"${cmd[@]}"
status=$?
set -e
record end "$status" || true
exit "$status"
