"""
DTWG-integrated pocket detection.
Uses fpocket PQR vertices + TM-align template matching for
high-precision binding pocket identification.
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
from Bio.PDB import PDBParser, Chain, Model, Structure, PDBIO
from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.Residue import DisorderedResidue, Residue
from Bio.PDB.Atom import DisorderedAtom

logger = logging.getLogger(__name__)

FPOCKET_BIN = os.environ.get("FPOCKET_BIN", "fpocket")
TMALIGN_BIN = os.environ.get("TMALIGN_BIN", str(Path(__file__).parent.parent / "Pocket-Detection-of-DTWG-main" / "TMalign"))


# ══════════════════════════════════════════════════════════════════════
# PQR parser (from DTWG utils.py)
# ══════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════
# Binding pocket extraction (from DTWG utils.py)
# ══════════════════════════════════════════════════════════════════════

def get_binding_pockets(pdb_struct, lig_coord, threshold: float = 6.0):
    """
    Extract protein residues within threshold Å of ligand/PQR coordinates.
    Returns (pocket_structure, residue_id_set, atom_count).
    """
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


# ══════════════════════════════════════════════════════════════════════
# Enhanced fpocket with PQR vertex analysis (DTWG getfpockets)
# ══════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════
# Structural quality scoring for pocket filtering
# ══════════════════════════════════════════════════════════════════════

def _volume_score(volume: float) -> float:
    """Score pocket volume on a bell curve: optimal ~400-1200 Å³."""
    if volume <= 0:
        return 0.0
    if volume < 200:
        return volume / 200.0 * 0.3
    if volume <= 400:
        return 0.3 + (volume - 200) / 200.0 * 0.5
    if volume <= 1200:
        return 0.8 + (volume - 400) / 800.0 * 0.2
    if volume <= 2500:
        return 1.0 - (volume - 1200) / 1300.0 * 0.8
    return max(0.1, 0.2 - (volume - 2500) / 2500.0 * 0.1)


def _sphere_count_score(n_spheres: int) -> float:
    """Score alpha sphere count: more spheres = better defined pocket."""
    if n_spheres < 15:
        return 0.1
    if n_spheres <= 40:
        return 0.1 + (n_spheres - 15) / 25.0 * 0.6
    if n_spheres <= 100:
        return 0.7 + (n_spheres - 40) / 60.0 * 0.25
    return min(1.0, 0.95 + (n_spheres - 100) / 200.0 * 0.05)


def _residue_count_score(n_residues: int) -> float:
    """Score residue count: enough residues for meaningful binding site."""
    if n_residues < 5:
        return 0.1
    if n_residues <= 10:
        return 0.1 + (n_residues - 5) / 5.0 * 0.5
    if n_residues <= 25:
        return 0.6 + (n_residues - 10) / 15.0 * 0.35
    return min(1.0, 0.95 + (n_residues - 25) / 50.0 * 0.05)


def compute_structural_quality(pocket: dict) -> float:
    """
    Compute composite structural quality score for a pocket.

    Combines fpocket score (alpha sphere geometry), volume, alpha sphere count,
    and residue count into a single 0-1 quality metric.

    fpocket scores typically range 1-30 (good pockets 10+, excellent 20+).
    Normalized with a sigmoid-like curve centered at 10.

    Weights: fpocket_score=0.50, volume=0.20, sphere_count=0.15, residue_count=0.15
    """
    raw_score = pocket.get("score", 0)
    # Sigmoid-like: score 5 → 0.27, 10 → 0.62, 15 → 0.82, 20 → 0.92
    fp_score = 1.0 / (1.0 + max(0, 15.0 - raw_score) / 10.0) if raw_score > 0 else 0.0

    v_score = _volume_score(pocket.get("volume", 0))
    s_score = _sphere_count_score(pocket.get("alpha_sphere_count", 0))
    r_score = _residue_count_score(len(pocket.get("residue_ids", [])))

    quality = (
        0.50 * fp_score
        + 0.20 * v_score
        + 0.15 * s_score
        + 0.15 * r_score
    )
    return round(quality, 4)


_DEFAULT_QUALITY_MIN = 0.20
_DEFAULT_MIN_SCORE = -999.0
_DEFAULT_MIN_VOLUME = 100.0
_DEFAULT_MIN_RESIDUES = 5


def filter_quality_pockets(
    pockets: list[dict],
    min_quality: float = _DEFAULT_QUALITY_MIN,
    min_volume: float = _DEFAULT_MIN_VOLUME,
    min_residues: int = _DEFAULT_MIN_RESIDUES,
) -> list[dict]:
    """
    Filter pockets by structural quality thresholds.

    Pockets must pass ALL criteria:
      - composite quality_score >= min_quality
      - volume >= min_volume (Å³)
      - residue count >= min_residues

    The composite quality_score already weights fpocket score at 50%, so
    a separate score floor is not needed. Negative fpocket scores (fpocket 4.0+)
    will naturally reduce the composite quality.

    Returns filtered list sorted by quality_score descending.
    """
    filtered = []
    for p in pockets:
        score = p.get("score", 0)
        volume = p.get("volume", 0)
        n_residues = len(p.get("residue_ids", []))
        n_spheres = p.get("alpha_sphere_count", 0)
        quality = compute_structural_quality(p)

        logger.info(
            f"Pocket raw: score={score:.1f}, volume={volume:.0f}, "
            f"spheres={n_spheres}, residues={n_residues}, quality={quality:.4f}"
        )

        p["quality_score"] = quality
        if volume < min_volume:
            logger.info(f"  Rejected: volume {volume:.0f} < {min_volume}")
            continue
        if n_residues < min_residues:
            logger.info(f"  Rejected: residues {n_residues} < {min_residues}")
            continue
        if quality < min_quality:
            logger.info(f"  Rejected: quality {quality:.4f} < {min_quality}")
            continue
        filtered.append(p)

    filtered.sort(key=lambda p: -(p.get("quality_score", 0)))
    for i, p in enumerate(filtered):
        p["rank"] = i + 1

    logger.info(
        f"Pocket quality filter: {len(pockets)} → {len(filtered)} "
        f"(min_quality={min_quality}, "
        f"min_volume={min_volume}, min_residues={min_residues})"
    )
    for p in filtered:
        logger.info(
            f"  Pocket {p['rank']}: quality={p['quality_score']:.4f}, "
            f"fpocket_score={p['score']:.1f}, volume={p['volume']:.0f}Å³, "
            f"spheres={p.get('alpha_sphere_count', 0)}, residues={len(p.get('residue_ids', []))}"
        )
    return filtered


def detect_pockets_fpocket_pqr(
    pdb_text: str,
    pocket_threshold: float = 6.0,
    top_n: int = 3,
) -> list[dict]:
    """
    Run fpocket, parse PQR vertices, and extract pocket residues.

    Uses PQR vertex files which contain alpha sphere coordinates and structural
    pocket scores (based on alpha sphere geometry, hydrophobic density, etc.).

    Returns list of dicts with:
        - rank, score, volume, quality_score
        - residues: [(chain, res_id), ...]
        - residue_ids: [res_id, ...]
        - alpha_sphere_count, atom_count
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        infile = tmpdir / "input.pdb"
        infile.write_text(pdb_text)

        try:
            subprocess.run(
                [FPOCKET_BIN, "-f", str(infile)],
                cwd=str(tmpdir), capture_output=True, timeout=120, check=True,
            )
        except Exception as e:
            logger.error(f"fpocket failed: {e}")
            return []

        outdir = tmpdir / "input_out"
        pockets_dir = outdir / "pockets"
        if not pockets_dir.exists():
            return []

        parser = PDBParser()
        pdb_struct = parser.get_structure("protein", str(infile))

        all_pockets = []
        for pqr_file in sorted(pockets_dir.glob("*.pqr")):
            try:
                alpha_sphere_coords, score, volume = pqr_parser(str(pqr_file), return_score=True)
                if len(alpha_sphere_coords) == 0:
                    continue

                pdb_struct_copy = deepcopy(pdb_struct)
                _, residue_set, atom_count = get_binding_pockets(
                    pdb_struct_copy, alpha_sphere_coords, pocket_threshold
                )

                all_pockets.append({
                    "residue_ids": sorted(residue_set),
                    "residues": [("A", r) for r in sorted(residue_set)],
                    "score": round(score, 4),
                    "volume": round(volume, 1),
                    "atom_count": atom_count,
                    "alpha_sphere_count": len(alpha_sphere_coords),
                })
            except Exception as e:
                logger.info(f"Failed to parse {pqr_file}: {e}")
                continue

        # Compute quality score for each pocket, sort by quality
        for p in all_pockets:
            p["quality_score"] = compute_structural_quality(p)

        all_pockets.sort(key=lambda p: -(p.get("quality_score", 0)))
        for i, p in enumerate(all_pockets):
            p["rank"] = i + 1

        return all_pockets[:top_n]


# ══════════════════════════════════════════════════════════════════════
# TM-align template matching (from DTWG TMalign.py)
# ══════════════════════════════════════════════════════════════════════

class TMaligner:
    """Align two protein structures using TM-align."""

    def align(self, protein_file: str, ref_protein_file: str) -> Optional[dict]:
        try:
            rotation_file = f"/tmp/tmalign_{os.getpid()}.txt"
            out_bytes = subprocess.check_output(
                [TMALIGN_BIN, protein_file, ref_protein_file, "-m", rotation_file],
                timeout=120,
            )
            out_text = out_bytes.decode('utf-8').strip().split("\n")

            return {
                "TMscore1": float(out_text[12].split(" ")[1]),
                "TMscore2": float(out_text[13].split(" ")[1]),
                "seq_protein": out_text[17],
                "seq_ref_protein": out_text[19],
                "rotation_matrix": self._get_rotate_matrix(rotation_file),
            }
        except Exception as e:
            logger.warning(f"TM-align failed: {e}")
            return None

    def _get_rotate_matrix(self, rotation_file: str) -> tuple[np.ndarray, np.ndarray]:
        with open(rotation_file, "r") as f:
            data = f.readlines()
        u, t = [], []
        for i in range(2, 5):
            parts = [float(x) for x in data[i].split(" ") if x != ""]
            t.append(parts[1])
            u.append(parts[2:])
        try:
            os.remove(rotation_file)
        except Exception:
            pass
        return np.array(u), np.array(t)


def detect_pocket_by_template(
    query_pdb_text: str,
    template_pdb_text: str,
    template_ligand_sdf: Optional[str] = None,
    threshold: float = 6.0,
    min_tm_score: float = 0.5,
) -> Optional[dict]:
    """
    Detect binding pocket by aligning a known co-crystal structure.

    Args:
        query_pdb_text: PDB text of the folded query protein
        template_pdb_text: PDB text of the template (known co-crystal)
        template_ligand_sdf: Optional SDF of template ligand (superimposed)
        threshold: distance threshold for pocket extraction (Å)
        min_tm_score: minimum TM-score for accepting alignment

    Returns dict with pocket residues and alignment info, or None.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        query_file = tmpdir / "query.pdb"
        template_file = tmpdir / "template.pdb"
        query_file.write_text(query_pdb_text)
        template_file.write_text(template_pdb_text)

        aligner = TMaligner()
        result = aligner.align(str(template_file), str(query_file))

        if result is None:
            return None

        if result["TMscore2"] < min_tm_score:
            logger.info(f"TM-score {result['TMscore2']:.3f} below threshold {min_tm_score}")
            return None

        # Parse query protein
        parser = PDBParser()
        query_struct = parser.get_structure("query", str(query_file))

        if template_ligand_sdf:
            from rdkit import Chem
            mol = Chem.MolFromMolFile(template_ligand_sdf)
            if mol is None:
                logger.warning("Failed to read template ligand SDF")
                return None
            ligand_coords = mol.GetConformer().GetPositions()

            # Apply rotation to ligand coordinates
            u, t = result["rotation_matrix"]
            rotated_coords = np.dot(ligand_coords, u.T) + t

            # Extract pocket residues around rotated ligand
            pocket_struct, residue_set, atom_count = get_binding_pockets(
                query_struct, rotated_coords, threshold
            )

            return {
                "TMscore": round(result["TMscore2"], 4),
                "residue_ids": sorted(residue_set),
                "residues": [("A", r) for r in sorted(residue_set)],
                "atom_count": atom_count,
                "method": "template_matching",
            }

        return {
            "TMscore": round(result["TMscore2"], 4),
            "method": "template_alignment_only",
        }


# ══════════════════════════════════════════════════════════════════════
# Unified pocket detection (use all available methods)
# ══════════════════════════════════════════════════════════════════════

def detect_pocket_residues(
    mmcif_or_pdb_text: str,
    chain_id: str = "A",
    top_n: int = 3,
    use_template: bool = False,
    template_pdb: Optional[str] = None,
    template_ligand_sdf: Optional[str] = None,
) -> list[dict]:
    """
    Unified pocket detection using fpocket PQR analysis + optional template matching.

    Returns list of pocket dicts ready for ESMFold2 pocket conditioning.
    """
    pockets = detect_pockets_fpocket_pqr(mmcif_or_pdb_text, top_n=top_n)

    if pockets:
        logger.info(f"fpocket found {len(pockets)} pockets, top score={pockets[0]['score']:.4f}, "
                     f"volume={pockets[0]['volume']:.1f}Å³")
        for p in pockets:
            logger.info(f"  Pocket {p['rank']}: {len(p['residue_ids'])} residues, "
                         f"score={p['score']:.4f}, volume={p['volume']:.1f}Å³")

    # If template is provided, also do template matching
    if use_template and template_pdb:
        tm_result = detect_pocket_by_template(mmcif_or_pdb_text, template_pdb, template_ligand_sdf)
        if tm_result:
            logger.info(f"Template match: TM-score={tm_result['TMscore']}, "
                         f"{len(tm_result.get('residue_ids', []))} residues")
            # Insert template result at top if it has residues
            if tm_result.get("residue_ids"):
                tm_result["rank"] = 0
                tm_result["score"] = tm_result["TMscore"]
                tm_result["volume"] = 0
                pockets.insert(0, tm_result)

    return pockets
