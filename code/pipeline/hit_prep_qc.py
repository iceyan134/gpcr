"""Step 1.5 PoseBusters + Step 1.6 Pocket QC — independent functions.

Design principles (audit 2026-09-02):
- Zero modification to existing hit_prep.py functions
- Each function returns dict or None (never raises)
- Results are annotations, not elimination criteria
- PoseBusters: 5 physically meaningful checks only (skip N/A + generative
  model artifacts)
- Pocket QC: convex hull volume ratio (ligand/pocket), pure numpy+scipy
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("cascade.hit_prep")

# ── Step 1.5: PoseBusters (filtered) ────────────────────────────────────

# Only these 5 checks are physically meaningful for ESMFold2 co-folded
# structures. The other ~25 checks are either:
# - N/A (no waters/cofactors/crystal reference in our structures)
# - Systematic generative model artifacts (aromatic ring planarity)
# - Too approximate to discriminate (bond lengths from diffusion model)
POSEBUSTERS_RELEVANT = [
    "volume_overlap_with_protein",       # any steric overlap with protein
    "most_extreme_clash_protein",        # severe steric clash
    "passes_valence_checks",             # chemical valence legality
    "sanitization",                      # RDKit can process the ligand
    "protein-ligand_maximum_distance",   # ligand not unreasonably far
]


def run_posebusters_filtered(lig_sdf: Path, rec_pdb: Path) -> dict | None:
    """Run PoseBusters with only the 5 physically meaningful checks.

    Returns {"n_pass": int, "n_total": int, "failed": [str], "checks": {name: bool}}
    or None on any failure (never raises).
    """
    try:
        from posebusters import PoseBusters
        pb = PoseBusters(config="dock")
        df = pb.bust([str(lig_sdf)], None, str(rec_pdb), full_report=True)
        # extract only boolean columns that match our relevant list
        checks = {}
        for col in POSEBUSTERS_RELEVANT:
            if col in df.columns and str(df[col].dtype) == "bool":
                checks[col] = bool(df[col].iloc[0])
            else:
                logger.debug("PoseBusters check '%s' not in output, skipping", col)
        if not checks:
            return {"n_pass": 0, "n_total": 0, "failed": ["no_matching_checks"],
                    "checks": {}}
        n_pass = sum(checks.values())
        failed = [k for k, v in checks.items() if not v]
        return {"n_pass": n_pass, "n_total": len(checks),
                "failed": failed, "checks": checks}
    except ImportError:
        logger.warning("PoseBusters not installed")
        return None
    except Exception as e:
        logger.warning("PoseBusters failed: %s", str(e)[:100])
        return None


# ── Step 1.6: Pocket volume ratio ───────────────────────────────────────

# approximate van der Waals volume per heavy atom (Å³)
# C~16, N~14, O~12, S~24, halogen~20 → weighted mean ~15
_VDW_VOL_PER_ATOM = 15.0


def _estimate_ligand_volume(lig_sdf: Path) -> float | None:
    """Estimate ligand volume from heavy atom count × approximate vdW volume."""
    try:
        from rdkit import Chem
        m = Chem.SDMolSupplier(str(lig_sdf), removeHs=True)[0]
        if m is None:
            return None
        n_heavy = m.GetNumHeavyAtoms()
        return n_heavy * _VDW_VOL_PER_ATOM
    except Exception:
        return None


def _estimate_pocket_volume(rec_pdb: Path, lig_sdf: Path,
                            contact_cutoff: float = 6.0) -> float | None:
    """Estimate pocket volume as convex hull of protein atoms near ligand.

    Uses scipy.spatial.ConvexHull on all protein atoms within
    contact_cutoff Å of any ligand heavy atom.
    """
    try:
        from rdkit import Chem
        from scipy.spatial import ConvexHull

        # ligand coordinates
        m = Chem.SDMolSupplier(str(lig_sdf), removeHs=True)[0]
        if m is None or m.GetNumConformers() == 0:
            return None
        conf = m.GetConformer()
        lig = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y,
                         conf.GetAtomPosition(i).z]
                        for i in range(m.GetNumAtoms())])

        # protein atoms within cutoff of ligand
        pocket_atoms = []
        for line in Path(rec_pdb).read_text(errors="ignore").splitlines():
            if not line.startswith("ATOM"):
                continue
            try:
                xyz = np.array([float(line[30:38]), float(line[38:46]),
                                float(line[46:54])])
            except (ValueError, IndexError):
                continue
            d = np.linalg.norm(lig - xyz, axis=1).min()
            if d <= contact_cutoff:
                pocket_atoms.append(xyz)

        if len(pocket_atoms) < 4:  # ConvexHull needs ≥4 points in 3D
            return None

        hull = ConvexHull(np.array(pocket_atoms))
        return float(hull.volume)
    except Exception:
        return None


def pocket_volume_ratio(lig_sdf: Path, rec_pdb: Path) -> dict | None:
    """Compute ligand volume / pocket volume ratio.

    Returns {"ligand_vol": float, "pocket_vol": float, "ratio": float,
             "verdict": str} or None.
    Verdict:
      "too_small"   ratio < 0.05  (ligand occupies <5% of pocket)
      "loose"       0.05 ≤ ratio < 0.15  (weak complementarity)
      "good"        0.15 ≤ ratio ≤ 0.60  (reasonable complementarity)
      "tight"       ratio > 0.60  (potential steric strain)
    """
    lig_vol = _estimate_ligand_volume(lig_sdf)
    if lig_vol is None or lig_vol <= 0:
        return None
    pocket_vol = _estimate_pocket_volume(rec_pdb, lig_sdf)
    if pocket_vol is None or pocket_vol <= 0:
        return None
    ratio = lig_vol / pocket_vol
    if ratio < 0.05:
        verdict = "too_small"
    elif ratio < 0.15:
        verdict = "loose"
    elif ratio <= 0.60:
        verdict = "good"
    else:
        verdict = "tight"
    return {"ligand_vol": round(lig_vol, 1), "pocket_vol": round(pocket_vol, 1),
            "ratio": round(ratio, 3), "verdict": verdict}
