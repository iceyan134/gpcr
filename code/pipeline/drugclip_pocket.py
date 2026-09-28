"""
Pocket extraction utilities for DrugCLIP.
Uses fpocket PQR vertices + 6A residue extraction (same as dtwg_pocket).
Then applies quality filtering and returns clean pocket PDBs.

This is equivalent to the "Exp. Pocket" level in the DrugCLIP paper,
as long as the input structure is accurate enough.
"""
from __future__ import annotations
import logging, subprocess, tempfile, os
import numpy as np
from pathlib import Path
from copy import deepcopy
from Bio.PDB import PDBParser, PDBIO, Chain, Model, Structure

logger = logging.getLogger("cascade.drugclip.pocket")

FPOCKET_BIN = os.environ.get("FPOCKET_BIN", "fpocket")


def pqr_parser(filename: str, return_score: bool = False):
    """Parse fpocket PQR output to get alpha sphere coordinates and scores."""
    with open(filename, 'r') as f:
        data = f.readlines()
    coord = []
    score = 0.0
    volume = 0.0
    for line in data:
        if "Pocket Score" in line:
            score = float(line.split()[-1])
        elif "Real volume" in line:
            volume = float(line.split()[-1])
        elif line[:4] == 'ATOM':
            coord.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    coord = np.array(coord) if coord else np.empty((0, 3))
    if return_score:
        return coord, score, volume
    return coord


def get_binding_pockets(pdb_struct, lig_coord, threshold: float = 6.0):
    """Extract protein residues within threshold A of ligand/PQR coordinates.
    Returns (pocket_structure, residue_id_set, atom_count)."""
    chain = pdb_struct[0]['A']
    tmp_chain = Chain.Chain('A')
    resid = set()
    for res in chain:
        res_coord = np.array([atom.get_coord() for atom in res.get_atoms()])
        dist = np.linalg.norm(res_coord[:, None, :] - lig_coord[None, :, :], axis=-1).min()
        if dist <= threshold:
            tmp_chain.add(res.copy())
            resid.add(res.id[1])
    tmp_structure = Structure.Structure(pdb_struct.id)
    tmp_model = Model.Model(0)
    tmp_structure.add(tmp_model)
    tmp_model.add(tmp_chain)
    atom_num = sum(1 for a in tmp_chain.get_atoms() if a.element != 'H')
    return tmp_structure, resid, atom_num


def _volume_score(volume: float) -> float:
    if volume <= 0: return 0.0
    if volume < 200: return volume / 200.0 * 0.3
    if volume <= 400: return 0.3 + (volume - 200) / 200.0 * 0.5
    if volume <= 1200: return 0.8 + (volume - 400) / 800.0 * 0.2
    if volume <= 2500: return 1.0 - (volume - 1200) / 1300.0 * 0.8
    return max(0.1, 0.2 - (volume - 2500) / 2500.0 * 0.1)


def _sphere_count_score(n: int) -> float:
    if n < 15: return 0.1
    if n <= 40: return 0.1 + (n - 15) / 25.0 * 0.6
    if n <= 100: return 0.7 + (n - 40) / 60.0 * 0.25
    return min(1.0, 0.95 + (n - 100) / 200.0 * 0.05)


def _residue_count_score(n: int) -> float:
    if n < 5: return 0.1
    if n <= 10: return 0.1 + (n - 5) / 5.0 * 0.5
    if n <= 25: return 0.6 + (n - 10) / 15.0 * 0.35
    return min(1.0, 0.95 + (n - 25) / 50.0 * 0.05)


def compute_quality(pocket: dict) -> float:
    raw_score = pocket.get("score", 0)
    fp_score = 1.0 / (1.0 + max(0, 15.0 - raw_score) / 10.0) if raw_score > 0 else 0.0
    v_score = _volume_score(pocket.get("volume", 0))
    s_score = _sphere_count_score(pocket.get("alpha_sphere_count", 0))
    r_score = _residue_count_score(len(pocket.get("residue_ids", [])))
    return round(0.50 * fp_score + 0.20 * v_score + 0.15 * s_score + 0.15 * r_score, 4)


def extract_pockets_from_pdb(
    receptor_pdb: str | Path, workdir: str | Path | None = None,
    sphere_radius: float = 10.0, top_n: int = 5,
    min_quality: float = 0.20, min_volume: float = 100.0, min_residues: int = 5,
) -> list[Path]:
    """Run fpocket, extract quality-filtered pockets, return list of pocket PDB paths.

    Uses PQR vertex analysis (same as DTWG) to extract pocket residues within 6A
    of alpha spheres, then filters by structural quality.

    Returns list of Path objects to pocket PDB files, sorted by quality descending.
    """
    import shutil
    receptor_pdb = Path(receptor_pdb)
    workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="fp_"))
    workdir.mkdir(parents=True, exist_ok=True)
    input_copy = workdir / receptor_pdb.name
    shutil.copy(str(receptor_pdb), str(input_copy))

    # Run fpocket
    result = subprocess.run(
        [FPOCKET_BIN, "-f", str(input_copy.name)],
        cwd=str(workdir), capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        logger.warning("fpocket failed: %s", result.stderr[-500:])
        return []

    fp_dir = workdir / f"{input_copy.stem}_out"
    pockets_dir = fp_dir / "pockets" if (fp_dir / "pockets").exists() else fp_dir
    if not pockets_dir.exists():
        return []

    pqr_files = sorted(pockets_dir.glob("*.pqr"))
    if not pqr_files:
        # fallback: use atm.pdb files
        atm_files = sorted(pockets_dir.glob("pocket*_atm.pdb"),
            key=lambda p: int(p.name.replace("pocket","").replace("_atm.pdb","")))
        return atm_files[:top_n]

    # Parse PQR files and extract pocket residues
    parser = PDBParser(QUIET=True)
    pdb_struct = parser.get_structure("prot", str(input_copy))
    all_pockets = []

    for pqr_file in pqr_files:
        try:
            coords, score, volume = pqr_parser(str(pqr_file), return_score=True)
            if len(coords) == 0:
                continue
            pdb_copy = deepcopy(pdb_struct)
            pocket_struct, resid_set, atom_count = get_binding_pockets(pdb_copy, coords, 6.0)

            info = {
                "score": round(score, 4), "volume": round(volume, 1),
                "residue_ids": sorted(resid_set), "atom_count": atom_count,
                "alpha_sphere_count": len(coords), "structure": pocket_struct,
            }
            info["quality"] = compute_quality(info)

            # Apply quality filters
            if volume < min_volume: continue
            if len(resid_set) < min_residues: continue
            if info["quality"] < min_quality: continue

            # Save pocket PDB
            pocket_pdb = workdir / f"pocket_{len(all_pockets)}.pdb"
            io = PDBIO()
            io.set_structure(pocket_struct)
            io.save(str(pocket_pdb))
            info["pocket_pdb"] = pocket_pdb
            all_pockets.append(info)
        except Exception as e:
            logger.debug("Failed to parse %s: %s", pqr_file, e)
            continue

    # Sort by quality descending
    all_pockets.sort(key=lambda p: -(p.get("quality", 0)))
    logger.info("fpocket: %d/%d pockets passed quality filter", len(all_pockets), len(pqr_files))

    return [p["pocket_pdb"] for p in all_pockets[:top_n]]


def _get_pocket_residues_from_pqr(
    receptor_pdb: str | Path, workdir: str | Path | None = None,
    top_n: int = 3, min_quality: float = 0.20,
) -> list[list[int]]:
    """Run fpocket on receptor, return residue lists for PocketConditioning.
    
    Returns list of [resid1, resid2, ...] for each quality pocket.
    """
    import shutil, tempfile
    receptor_pdb = Path(receptor_pdb)
    workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="fp_"))
    workdir.mkdir(parents=True, exist_ok=True)
    input_copy = workdir / receptor_pdb.name
    shutil.copy(str(receptor_pdb), str(input_copy))

    result = subprocess.run(
        [FPOCKET_BIN, "-f", str(input_copy.name)],
        cwd=str(workdir), capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        return []

    fp_dir = workdir / f"{input_copy.stem}_out"
    pockets_dir = fp_dir / "pockets" if (fp_dir / "pockets").exists() else fp_dir
    if not pockets_dir.exists():
        return []

    pqr_files = sorted(pockets_dir.glob("*.pqr"))
    if not pqr_files:
        return []

    parser = PDBParser(QUIET=True)
    pdb_struct = parser.get_structure("prot", str(input_copy))
    all_sites = []

    for pqr_file in pqr_files:
        try:
            coords, score, volume = pqr_parser(str(pqr_file), return_score=True)
            if len(coords) == 0: continue
            pdb_copy = deepcopy(pdb_struct)
            _, resid_set, atom_count = get_binding_pockets(pdb_copy, coords, 6.0)
            info = {"score": round(score, 4), "volume": round(volume, 1),
                    "residue_ids": sorted(resid_set), "atom_count": atom_count,
                    "alpha_sphere_count": len(coords)}
            info["quality"] = compute_quality(info)
            if volume < 100.0: continue
            if len(resid_set) < 5: continue
            if info["quality"] < min_quality: continue
            all_sites.append(sorted(resid_set))
        except Exception:
            continue

    all_sites.sort(key=lambda s: -len(s))
    return all_sites[:top_n]