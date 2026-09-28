"""
Orthogonal rescue (Leg 3) — non-ESMFold2 pose source for compounds where
Leg 1 (pocket-off) and Leg 2 (pocket-on) both fail or are weak.

Preferred: Gnina pocket-directed docking (open-source, stays in existing stack).
Alternative: Boltz-2 / Boltz-1x as pure pose generator (poses only, NOT affinity).

Reference: docs/REBUILD_PLAN_addendum_A_*.md §1, Leg 3
"""
from __future__ import annotations

import logging
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("cascade.rescue")


@dataclass
class RescueResult:
    """Result from one orthogonal rescue attempt."""
    engine: str                    # "gnina" | "boltz"
    site_label: str = ""           # pocket site identifier
    status: str = "failed"         # "ok" | "failed"
    cnn_affinity: float = math.nan
    cnn_score: float = math.nan
    vina_affinity: float = math.nan
    poses: list[dict] = field(default_factory=list)
    best_pose_sdf: Path | None = None
    error: str | None = None
    raw_output: str = ""


# ── Gnina docking rescue ──────────────────────────────────────────────────

def gnina_rescue_dock(
    receptor_pdb: Path,
    smiles: str,
    pocket_centers: list[tuple[float, float, float]],
    workdir: Path,
    gnina_bin: str = "gnina",
    cnn_model: str = "crossdock_default2018",
    box_size: tuple = (25, 25, 25),
    exhaustiveness: int = 16,
    seed: int = 42,
    use_gpu: bool = True,
    timeout_s: int = 3600,
) -> RescueResult:
    """Run Gnina docking restricted to candidate pocket sites.

    This is a RESCUE path, not the primary pipeline. Compounds reaching here
    have already failed Leg 1 (pocket-off ESMFold2 co-fold) and Leg 2
    (PocketConditioning co-fold). We use Gnina docking as an orthogonal
    (non-ESMFold2) method to generate poses.

    Parameters
    ----------
    pocket_centers : list of (x, y, z) tuples
        Docking box centers for each candidate pocket site.
    """
    import shutil
    import subprocess

    if shutil.which(gnina_bin) is None:
        return RescueResult(
            engine="gnina", status="failed",
            error=f"{gnina_bin} not found on PATH",
        )

    workdir.mkdir(parents=True, exist_ok=True)

    # Prepare ligand from SMILES
    lig_sdf = _smiles_to_sdf(smiles, workdir)

    best_affinity = math.nan
    best_score = math.nan
    best_vina = math.nan
    best_pose_sdf = None
    all_results = []
    best_overall = -float("inf")

    for i, (cx, cy, cz) in enumerate(pocket_centers):
        site_label = f"site_{i}"
        out_sdf = workdir / f"gnina_rescue_{site_label}.sdf"

        cmd = [
            gnina_bin,
            "-r", str(receptor_pdb),
            "-l", str(lig_sdf),
            "-o", str(out_sdf),
            "--cnn", cnn_model,
            "--cnn_scoring", "rescore",
            "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
            "--size_x", str(box_size[0]), "--size_y", str(box_size[1]), "--size_z", str(box_size[2]),
            "--exhaustiveness", str(exhaustiveness),
            "--num_modes", "9",
            "--seed", str(seed),
        ]
        if not use_gpu:
            cmd.append("--no_gpu")

        try:
            proc = subprocess.run(
                cmd, cwd=str(workdir),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=timeout_s,
            )
            stdout = proc.stdout

            cnn_aff = cnn_sc = vina_aff = math.nan
            for line in stdout.splitlines():
                s = line.strip()
                if s.startswith("CNNaffinity:"):
                    cnn_aff = float(s.split()[-1])
                elif s.startswith("CNNscore:"):
                    cnn_sc = float(s.split()[-1])
                elif s.startswith("Affinity:"):
                    vina_aff = float(s.split()[1])

            if not math.isnan(cnn_aff):
                all_results.append({
                    "site": site_label, "center": (cx, cy, cz),
                    "cnn_affinity": cnn_aff, "cnn_score": cnn_sc,
                    "vina_affinity": vina_aff,
                })
                if cnn_aff > best_overall:
                    best_overall = cnn_aff
                    best_affinity = cnn_aff
                    best_score = cnn_sc
                    best_vina = vina_aff
                    best_pose_sdf = out_sdf

        except Exception as e:
            logger.warning("Gnina rescue docking at %s failed: %s", site_label, e)
            all_results.append({"site": site_label, "error": str(e)})

    if best_pose_sdf is None:
        return RescueResult(
            engine="gnina", status="failed",
            error=f"Gnina docking failed at all {len(pocket_centers)} sites",
        )

    return RescueResult(
        engine="gnina", status="ok",
        cnn_affinity=best_affinity, cnn_score=best_score,
        vina_affinity=best_vina,
        poses=all_results, best_pose_sdf=best_pose_sdf,
    )


# ── Boltz rescue stub ─────────────────────────────────────────────────────

def boltz_rescue_score(
    protein_sequence: str,
    smiles: str,
    workdir: Path,
    timeout_s: int = 600,
) -> RescueResult:
    """Score a rescue compound with Boltz-2 affinity module.

    Generates a Boltz-2 YAML from protein sequence + SMILES, runs boltz predict
    with minimal steps, and returns the affinity score.

    This is the scoring-only rescue (no docking pose needed).
    """
    import subprocess
    import json
    import os
    from pathlib import Path

    workdir.mkdir(parents=True, exist_ok=True)

    yaml_text = f"""version: 1
sequences:
  - protein:
      id: A
      sequence: {protein_sequence}
      msa: empty
  - ligand:
      id: L
      smiles: '{smiles}'
properties:
  - affinity:
      binder: L
"""
    yaml_path = workdir / "rescue_boltz.yaml"
    yaml_path.write_text(yaml_text)

    out_dir = workdir / "boltz_out"
    env = {**os.environ, "BOLTZ_CACHE": os.environ.get("BOLTZ_CACHE", "/cache/boltz")}

    try:
        r = subprocess.run(
            ["/opt/boltz-venv/bin/python", "-m", "boltz.main", "predict",
             str(yaml_path), "--out_dir", str(out_dir),
             "--recycling_steps", "1", "--diffusion_samples", "1",
             "--sampling_steps", "200", "--devices", "1",
             "--accelerator", "gpu", "--no_kernels"],
            check=True, capture_output=True, text=True, timeout=timeout_s, env=env,
        )
    except subprocess.TimeoutExpired:
        return RescueResult(engine="boltz", status="failed", error="timeout")
    except subprocess.CalledProcessError as e:
        return RescueResult(engine="boltz", status="failed", error=e.stderr[-500:])

    # Parse affinity JSON
    pred_dir = out_dir / "boltz_results_rescue_boltz" / "predictions" / "rescue_boltz"
    if not pred_dir.exists():
        return RescueResult(engine="boltz", status="failed", error="no_pred_dir")

    affinity_files = list(pred_dir.glob("affinity_*.json"))
    if not affinity_files:
        return RescueResult(engine="boltz", status="failed", error="no_affinity_json")

    with open(affinity_files[0]) as f:
        data = json.load(f)

    return RescueResult(
        engine="boltz", status="ok",
        cnn_affinity=float(data.get("affinity_probability_binary", 0.0)),
        cnn_score=float(data.get("affinity_pred_value", 0.0)),
    )


# ── Helpers ───────────────────────────────────────────────────────────────

def _smiles_to_sdf(smiles: str, workdir: Path) -> Path:
    """Convert SMILES to 3D SDF via RDKit."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)

    out = workdir / "rescue_lig.sdf"
    Chem.MolToMolFile(mol, str(out))
    return out


def compute_pocket_center(
    protein_pdb: Path,
    pocket_residues: list[int],
) -> tuple[float, float, float] | None:
    """Compute CA centroid of pocket residues for docking box center."""
    import numpy as np

    coords = []
    with open(protein_pdb) as f:
        for line in f:
            if not line.startswith("ATOM") and not line.startswith("HETATM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            resnum = int(line[22:26].strip())
            if resnum in pocket_residues:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append((x, y, z))

    if not coords:
        return None
    arr = np.asarray(coords)
    return tuple(float(v) for v in arr.mean(axis=0))
