"""
GenPack-Lite: pocket refinement via iterative centroid convergence.
Achieves better pocket localization without a generative model.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser

logger = logging.getLogger("cascade.genpack")

FPOCKET_BIN = "fpocket"


def _iterative_centroid(spheres, min_spheres=5, max_iter=5, tail_pct=10):
    coords = spheres.copy()
    for _ in range(max_iter):
        center = coords.mean(axis=0)
        dists = np.linalg.norm(coords - center, axis=1)
        cutoff = np.percentile(dists, 100 - tail_pct)
        keep = dists <= cutoff
        if keep.sum() < min_spheres or keep.all():
            break
        coords = coords[keep]
    return coords


def _extract_residues_near_spheres(pdb_path, spheres, radius=6.0):
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("p", str(pdb_path))
    chain = struct[0]["A"]
    residues = set()
    for res in chain:
        for atom in res.get_atoms():
            if atom.element == "H":
                continue
            if np.linalg.norm(spheres - atom.get_coord(), axis=1).min() <= radius:
                residues.add(res.id[1])
                break
    return sorted(residues)


def run_fpocket_with_genpack(
    pdb_path, workdir, top_n=3, min_quality=0.20,
    pocket_threshold=6.0, refinement_tail_pct=10,
):
    """Run fpocket on receptor PDB, refine top pockets via GenPack-Lite.

    Returns list[list[int]] compatible with cascade's pocket_sites format.
    """
    from app.drugclip_pocket import pqr_parser, get_binding_pockets, compute_quality

    pdb_path = Path(pdb_path)
    workdir = Path(workdir)
    fp_workdir = workdir / "fpocket"
    fp_workdir.mkdir(parents=True, exist_ok=True)

    input_copy = fp_workdir / pdb_path.name
    shutil.copy(str(pdb_path), str(input_copy))

    result = subprocess.run(
        [FPOCKET_BIN, "-f", str(input_copy.name)],
        cwd=str(fp_workdir), capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        logger.warning("fpocket failed: %s", result.stderr[-500:])
        return []

    outdir = fp_workdir / f"{input_copy.stem}_out"
    pockets_dir = outdir / "pockets"
    if not pockets_dir.exists():
        return []

    pqr_files = sorted(pockets_dir.glob("*.pqr"))
    if not pqr_files:
        return []

    parser = PDBParser(QUIET=True)
    pdb_struct = parser.get_structure("protein", str(pdb_path))

    all_pockets = []
    for pqr_file in pqr_files:
        try:
            spheres, score, volume = pqr_parser(str(pqr_file), return_score=True)
            if spheres.shape[0] < 5:
                continue
            _, residue_set, atom_count = get_binding_pockets(
                deepcopy(pdb_struct), spheres, pocket_threshold,
            )
            p = {
                "residue_ids": sorted(residue_set),
                "score": round(score, 4),
                "volume": round(volume, 1),
                "alpha_spheres": spheres,
                "pqr_file": str(pqr_file),
            }
            p["quality_score"] = compute_quality(p)
            all_pockets.append(p)
        except Exception as e:
            logger.debug("PQR parse failed %s: %s", pqr_file, e)

    all_pockets.sort(key=lambda p: -(p.get("quality_score", 0)))
    filtered = [p for p in all_pockets if p.get("quality_score", 0) >= min_quality]
    top = filtered[:top_n]

    refined_sites = []
    for pocket in top:
        try:
            spheres = pocket.get("alpha_spheres")
            if spheres is None or spheres.shape[0] < 5:
                refined_sites.append(pocket["residue_ids"])
                continue

            retained = _iterative_centroid(
                spheres, max_iter=5, tail_pct=refinement_tail_pct,
            )
            residues = _extract_residues_near_spheres(
                pdb_path, retained, radius=6.0,
            )
            logger.info(
                "GenPack: Q=%.3f s=%.1f v=%.0f %dres (%d spheres)",
                pocket["quality_score"], pocket["score"], pocket["volume"],
                len(residues), len(retained),
            )
            refined_sites.append(residues)
        except Exception as e:
            logger.warning("GenPack refine failed: %s", e)
            refined_sites.append(pocket["residue_ids"])

    return refined_sites
