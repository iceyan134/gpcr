"""
Gnina CNN --score_only rescoring of the ESMFold2 co-fold pose.

The pose is preserved exactly (no re-docking) — this resolves the
"co-fold then Vina re-dock" methodological contradiction.

Reference: docs/screening_cascade.py:403-439
"""
from __future__ import annotations

import logging
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("cascade.gnina")


@dataclass
class GninaResult:
    cnn_affinity: float     # pK, higher = stronger binding
    cnn_score: float        # pose quality 0-1
    vina_affinity: float    # kcal/mol reference
    raw: str = ""


def gnina_rescore(
    receptor_pdb: Path,
    ligand_sdf: Path,
    gnina_bin: str = "gnina",
    cnn_model: str = "crossdock_default2018",
    cnn_scoring: str = "rescore",
    use_gpu: bool = True,
    local_minimize: bool = False,
    seed: int = 42,
    workdir: Path | None = None,
    timeout_s: int = 3600,
) -> GninaResult:
    """Score a protein-ligand complex pose with Gnina CNN.

    The key flag is --score_only: Gnina evaluates the given pose without
    moving the ligand. This preserves the ESMFold2 co-fold geometry.

    Parameters
    ----------
    local_minimize : bool
        If True, run a gentle local minimization before scoring (relieves
        small clashes). Default False to honour the exact ESMFold2 pose.
    """
    if workdir is None:
        workdir = Path.cwd()
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    _require_gnina(gnina_bin)

    out_sdf = workdir / "gnina_scored.sdf"
    cmd = [
        gnina_bin,
        "-r", str(receptor_pdb),
        "-l", str(ligand_sdf),
        "-o", str(out_sdf),
        "--cnn", cnn_model,
        "--cnn_scoring", cnn_scoring,
        "--seed", str(seed),
    ]

    if local_minimize:
        cmd.append("--minimize")
    else:
        cmd.append("--score_only")

    if not use_gpu:
        cmd.append("--no_gpu")

    logger.debug("RUN: %s", " ".join(cmd))
    proc = subprocess.run(
        cmd, cwd=str(workdir),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=timeout_s,
    )
    stdout = proc.stdout
    if proc.returncode != 0:
        raise RuntimeError(f"gnina failed ({proc.returncode}): {' '.join(cmd)}\n{stdout[-2000:]}")

    cnn_aff = cnn_score = vina_aff = math.nan
    for line in stdout.splitlines():
        s = line.strip()
        if s.startswith("CNNaffinity:"):
            cnn_aff = float(s.split()[-1])
        elif s.startswith("CNNscore:"):
            cnn_score = float(s.split()[-1])
        elif s.startswith("Affinity:"):
            vina_aff = float(s.split()[1])

    if math.isnan(cnn_aff):
        raise RuntimeError(f"Could not parse CNNaffinity from gnina output:\n{stdout[-2000:]}")

    return GninaResult(
        cnn_affinity=cnn_aff,
        cnn_score=cnn_score,
        vina_affinity=vina_aff,
        raw=stdout,
    )


def _require_gnina(gnina_bin: str):
    if shutil.which(gnina_bin) is None:
        raise FileNotFoundError(
            f"{gnina_bin} not found on PATH. "
            "Install gnina: https://github.com/gnina/gnina/releases"
        )
