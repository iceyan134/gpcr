"""Run monitor: heartbeat + stall detection (industrialization P1-5).

Codifies the metal-hang lesson (notes/gcgr-vanlscreen-results.md §1):
"CPU 100% + no progress != computing" — a background thread samples
self-utime (/proc/self/stat) and output-file mtimes; when utime stops
advancing AND no output file changes for `stall_min` minutes, a stall
warning is appended to heartbeat.jsonl. Tolerant by design: every probe
degrades to null outside Linux/without GPU tooling.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path


def read_utime(pid: int | None = None) -> float | None:
    try:
        pid = pid or os.getpid()
        with open(f"/proc/{pid}/stat", "rb") as f:
            parts = f.read().split()
        return (int(parts[13]) + int(parts[14])) / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        return None


def read_gpu_mem() -> float | None:
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return float(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return None


def detect_stall(prev_utime, cur_utime, last_output_mtime, now,
                 stall_min: float, utime_eps: float = 0.5) -> bool:
    """Pure logic (unit-testable): stalled iff CPU-time flat AND no new output."""
    if prev_utime is None or cur_utime is None:
        return False
    cpu_flat = abs(cur_utime - prev_utime) < utime_eps
    io_flat = (now - last_output_mtime) > stall_min * 60 if last_output_mtime else False
    return cpu_flat and io_flat


class RunMonitor:
    """Background heartbeat writer bound to a run directory."""

    def __init__(self, run_dir: str | Path, interval_s: float = 30.0,
                 stall_min: float = 10.0):
        self.run_dir = Path(run_dir)
        self.interval_s = interval_s
        self.stall_min = stall_min
        self.stage = "init"
        self.stage_t0 = time.time()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._prev_utime = read_utime()
        self._last_output_mtime = time.time()

    def start(self, stage: str | None = None, **stage_info):
        if stage:
            self.stage = stage
            self.stage_t0 = time.time()
            self._prev_utime = read_utime()
            self._last_output_mtime = time.time()
        self._thread = self._thread or threading.Thread(target=self._loop, daemon=True)
        if not self._thread.is_alive():
            self._thread.start()

    def log_event(self, event: str, **data):
        self._write({"event": event, **data})

    def _write(self, record: dict):
        self._touch()
        with open(self.run_dir / "heartbeat.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def _touch(self):
        self._last_output_mtime = time.time()

    def _latest_output_mtime(self) -> float:
        mt = self._last_output_mtime
        for p in self.run_dir.rglob("*"):
            try:
                mt = max(mt, p.stat().st_mtime)
            except OSError:
                continue
        return mt

    def _loop(self):
        while not self._stop.wait(self.interval_s):
            cur = read_utime()
            rec = {"event": "heartbeat", "stage": self.stage,
                   "stage_elapsed_s": round(time.time() - self.stage_t0, 1),
                   "utime_s": cur, "gpu_mem_mb": read_gpu_mem(),
                   "ts": time.time()}
            if detect_stall(self._prev_utime, cur,
                            self._latest_output_mtime(), time.time(),
                            self.stall_min):
                rec["stall_suspected"] = True
                rec["hint"] = ("utime flat AND no output change > "
                               f"{self.stall_min}min — suspect UFFTYPER-style "
                               "hang; check subprocess isolation")
            self._write(rec)
            self._prev_utime = cur

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval_s + 5)
        self._write({"event": "monitor_stopped", "stage": self.stage})
