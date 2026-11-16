"""Chip-0 queue: one flock, then a record of image, commit, and flags.

Run with system Python. The script does not import Resin.

    python3 tests/test_tpu_queue.py
"""

from __future__ import annotations

import json
import os
import signal
import stat
import subprocess
import tempfile
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "scripts" / "tpu-queue.sh"
LOCK = "/tmp/resin-tpu-chip0.lock"
HOST = subprocess.check_output(["hostname", "-s"], text=True).strip()


def run(cmd, *, env, timeout=10):
    return subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def base_env(tmp, env_sh, *, lock=None, log=None, image=None, path=None):
    env = os.environ.copy()
    env["TPU_ENV_SH"] = str(env_sh)
    env["RESIN_TPU_LOCK"] = str(lock or (tmp / "chip.lock"))
    env["RESIN_TPU_LOG"] = str(log or (tmp / "jobs.jsonl"))
    env.pop("RESIN_TPU_QUEUE_HELD", None)
    env.pop("RESIN_IMAGE", None)
    if image is not None:
        env["RESIN_IMAGE"] = image
    if path is not None:
        env["PATH"] = path
    return env


def write_env(tmp, body="export TPU_VISIBLE_CHIPS=0\nexport TPU_QUEUE_MARKER=chip0\n"):
    path = tmp / "tpu-env.sh"
    path.write_text(body)
    return path


def records(log):
    text = Path(log).read_text() if Path(log).exists() else ""
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_default_lock_is_the_shared_file():
    """The default lock path is the file other agents pass to plain flock."""
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        env_sh = write_env(tmp)
        seen = tmp / "flock-args"
        fake_bin = tmp / "bin"
        fake_bin.mkdir()
        fake = fake_bin / "flock"
        fake.write_text(
            textwrap.dedent(
                f"""\
                #!/bin/bash
                printf '%s\\n' "$@" > {seen}
                exit 0
                """
            )
        )
        fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
        env = base_env(tmp, env_sh, path=f"{fake_bin}{os.pathsep}{os.environ['PATH']}")
        del env["RESIN_TPU_LOCK"]
        result = run([str(QUEUE), "true"], env=env)
        assert result.returncode == 0, result.stderr
        args = seen.read_text().splitlines()
        # A fake flock on PATH exits before locking, so the real lock file is untouched.
        assert args[0] == LOCK
        assert "true" in args


def test_plain_flock_and_the_queue_take_turns():
    """A plain `flock <lock> sleep` blocks the queue, and the queue blocks `flock -n`."""
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        env_sh = write_env(tmp)
        lock = tmp / "chip.lock"
        order = tmp / "order"
        env = base_env(tmp, env_sh, lock=lock)
        holder = subprocess.Popen(
            [
                "/usr/bin/flock",
                str(lock),
                "bash",
                "-c",
                f"sleep 0.4; echo plain >> {order}",
            ]
        )
        time.sleep(0.1)
        blocked = subprocess.run(
            ["/usr/bin/flock", "-n", str(lock), "true"],
            check=False,
        )
        assert blocked.returncode != 0
        queued = run(
            [str(QUEUE), "bash", "-c", f"echo queue >> {order}"],
            env=env,
            timeout=5,
        )
        assert queued.returncode == 0, queued.stderr
        holder.wait(timeout=5)
        assert order.read_text().splitlines() == ["plain", "queue"]


def test_job_sees_sourced_env_and_log_has_commit_image_and_flags():
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        env_sh = write_env(tmp)
        log = tmp / "jobs.jsonl"
        marker = tmp / "marker"
        env = base_env(tmp, env_sh, log=log, image="vllm/vllm-tpu:nightly")
        result = run(
            [
                str(QUEUE),
                "bash",
                "-c",
                f'printf %s "$TPU_QUEUE_MARKER" > {marker}',
                "--tensor-parallel-size=2",
            ],
            env=env,
        )
        assert result.returncode == 0, result.stderr
        assert marker.read_text() == "chip0"
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        rows = records(log)
        assert [row["event"] for row in rows] == ["start", "end"]
        start = rows[0]
        assert start["commit"] == head
        assert isinstance(start["dirty"], list)
        assert start["image"] == "vllm/vllm-tpu:nightly"
        assert start["argv"] == [
            "bash",
            "-c",
            f'printf %s "$TPU_QUEUE_MARKER" > {marker}',
            "--tensor-parallel-size=2",
        ]
        assert start["host"] == HOST
        assert start["tpu_visible_chips"] == "0"
        assert rows[1]["status"] == 0
        assert rows[1]["pid"] == start["pid"]
        assert rows[1]["elapsed_s"] >= 0


def test_docker_run_image_is_recorded_without_running_docker():
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        env_sh = write_env(tmp)
        log = tmp / "jobs.jsonl"
        fake_bin = tmp / "bin"
        fake_bin.mkdir()
        docker = fake_bin / "docker"
        docker.write_text("#!/bin/bash\nexit 3\n")
        docker.chmod(docker.stat().st_mode | stat.S_IEXEC)
        env = base_env(tmp, env_sh, log=log, path=f"{fake_bin}{os.pathsep}{os.environ['PATH']}")
        result = run(
            [
                str(QUEUE),
                "docker",
                "run",
                "--rm",
                "--gpus",
                "all",
                "-e",
                "FOO=1",
                "-v",
                "/tmp:/tmp",
                "vllm/vllm-tpu:nightly-20260914",
                "echo",
                "hi",
            ],
            env=env,
        )
        assert result.returncode == 3, result.stderr
        start = records(log)[0]
        assert start["image"] == "vllm/vllm-tpu:nightly-20260914"
        assert records(log)[1]["status"] == 3


def test_failed_env_does_not_run_the_command():
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        env_sh = write_env(tmp, "echo refused >&2\nreturn 1\n")
        lock = tmp / "chip.lock"
        marker = tmp / "ran"
        # Hold nothing. If the queue locks before sourcing, a later flock -n fails.
        env = base_env(tmp, env_sh, lock=lock)
        result = run([str(QUEUE), "bash", "-c", f"echo ran > {marker}"], env=env)
        assert result.returncode != 0
        assert not marker.exists()
        # Usage errors and env failures must not leave the lock held.
        probe = subprocess.run(["/usr/bin/flock", "-n", str(lock), "true"], check=False)
        assert probe.returncode == 0


def test_usage_exits_before_locking():
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        env_sh = write_env(tmp)
        lock = tmp / "chip.lock"
        holder = subprocess.Popen(
            ["/usr/bin/flock", str(lock), "sleep", "2"],
            start_new_session=True,
        )
        time.sleep(0.1)
        started = time.monotonic()
        env = base_env(tmp, env_sh, lock=lock)
        result = run([str(QUEUE)], env=env, timeout=2)
        elapsed = time.monotonic() - started
        os.killpg(holder.pid, signal.SIGTERM)
        holder.wait(timeout=2)
        assert result.returncode == 2
        assert elapsed < 1
        assert "usage:" in result.stderr


def test_nested_queue_waits_on_the_same_lock():
    """A second queue inside the job blocks, the same way a second plain flock does."""
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        env_sh = write_env(tmp)
        env = base_env(tmp, env_sh)
        proc = subprocess.Popen(
            [str(QUEUE), str(QUEUE), "true"],
            cwd=ROOT,
            env=env,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1)
        still_waiting = proc.poll() is None
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=5)
        assert still_waiting


def test_real_env_pins_chip_zero_on_this_host():
    if HOST != "dev-tpu-tpu7x8b-0":
        return
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        log = tmp / "jobs.jsonl"
        env = base_env(tmp, ROOT / "scripts" / "tpu-env.sh", log=log)
        result = run([str(QUEUE), "true"], env=env)
        assert result.returncode == 0, result.stderr
        start = records(log)[0]
        assert start["tpu_visible_chips"] == "0"
        assert start["tpu_chips_per_host_bounds"] == "1,1,1"
        assert start["tpu_host_bounds"] == "1,1,1"
        assert start["tpu_worker_id"] == "0"
        assert start["commit"]


def main():
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
        print("ok", test.__name__)


if __name__ == "__main__":
    main()
