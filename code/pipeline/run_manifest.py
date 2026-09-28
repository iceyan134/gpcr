"""Run manifest: make every run auditable and replayable in principle.

Audit lesson (third occurrence 2026-08-31): GCGR L1 scores lost, GPR146 444
Nesso scores not persisted, no run-level record of config/code/model state —
three same-class incidents means a systemic gap, not carelessness. A manifest
records everything needed to answer "what exactly produced this result?".

Failure-tolerant by design: every probe (git, nvidia-smi, docker env) degrades
to a recorded null instead of raising, so manifest writing can never break a
screening run.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Files whose hashes fingerprint the pipeline code (repo is not git-managed).
CODE_FILES = [
    "app/cascade.py", "app/cascade_config.py", "app/confidence_gate.py",
    "app/consensus.py", "app/boltz_scorer.py", "app/local_inference.py",
    "app/l1_prescreen.py", "app/topk_selector.py", "app/hitlist.py",
    "app/drugclip_scorer.py", "app/models.py",
]


def _sha256(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except OSError:
        return None


def _git_hint() -> dict:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=5, cwd=Path(__file__).parent)
        if sha.returncode == 0:
            return {"git_commit": sha.stdout.strip()}
    except (OSError, subprocess.TimeoutExpired):
        pass
    return {"git_commit": None}


def _gpu_hint() -> dict:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version",
                              "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=5)
        if out.returncode == 0:
            return {"gpu": out.stdout.strip()}
    except (OSError, subprocess.TimeoutExpired):
        pass
    return {"gpu": None}


def _lib_versions() -> dict:
    vers = {"python": sys.version.split()[0]}
    for mod in ("numpy", "rdkit"):
        try:
            m = __import__(mod)
            vers[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            vers[mod] = None
    return vers


def build_manifest(
    run_id: str,
    config: dict | None = None,
    inputs: dict[str, str] | None = None,
    extra: dict | None = None,
    root: Path | None = None,
) -> dict:
    """Assemble the manifest dict. inputs: {label: path} to hash."""
    root = root or Path(__file__).parent.parent
    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "docker_image": os.environ.get("ESMFOLD_IMAGE_TAG")
            or os.environ.get("HOSTNAME", None),
        },
        "code": {
            **_git_hint(),
            "file_hashes": {f: _sha256(root / f) for f in CODE_FILES
                            if (root / f).exists()},
        },
        "env": {
            **_lib_versions(),
            **_gpu_hint(),
            "key_env": {k: os.environ.get(k) for k in
                        ("HF_HUB_OFFLINE", "NESSO_CACHE", "ESMCFOLD_CCD_PATH")},
        },
        "config": config or {},
        "inputs": {label: {"path": p, "sha256_16": _sha256(Path(p))}
                   for label, p in (inputs or {}).items()},
        "extra": extra or {},
    }
    return manifest


def write_manifest(out_path: Path, **kwargs) -> dict:
    manifest = build_manifest(**kwargs)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    tmp.replace(out_path)  # atomic-ish write
    return manifest


def load_manifest(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
