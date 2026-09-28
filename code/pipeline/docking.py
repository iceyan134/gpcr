"""
ESMFold2 + AutoDock Vina hybrid pipeline.

Workflow:
  1. ESMFold2 folds protein (apo)
  2. fpocket detects binding pockets → structural quality scoring → filter
  3. ESMFold2 co-folds protein + ligand (constrained to top-quality pocket)
  4. Vina scores the co-folded complex → ΔG (kcal/mol)
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Protein fold cache (sequence+seed → pdb_text + pockets) ──────────
_fold_cache: dict[str, dict] = {}


def _fold_cache_key(sequence: str, num_loops: int, num_steps: int, seed: int) -> str:
    return f"{hash(sequence)}_{num_loops}_{num_steps}_{seed}"


# ── Ligand property computation ───────────────────────────────────────

def _compute_ligand_properties(smiles: str) -> dict:
    """Compute molecular descriptors for ligand efficiency normalization.

    Also counts charged/polar functional groups for electrostatic correction,
    since Vina underestimates electrostatic contributions from phosphates,
    carboxylates, and other charged moieties.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"n_heavy_atoms": 1, "mw": 1.0, "logp": None, "tpsa": None, "n_rotatable": 0}

        n_heavy = mol.GetNumHeavyAtoms()
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        n_rotatable = Descriptors.NumRotatableBonds(mol)

        # Count charged groups for electrostatic correction.
        # Vina underestimates electrostatic contributions from charged groups.
        # Phosphate groups: count P atoms (each phosphate has one P)
        n_phosphate = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'P')
        # Carboxylate/carboxylic acid: C(=O)[O,OH]
        carboxylate_pattern = Chem.MolFromSmarts("[C](=O)[O-,OH]")
        n_carboxylate = len(mol.GetSubstructMatches(carboxylate_pattern, uniquify=True)) if carboxylate_pattern else 0
        # Sulfonate/SO2 (e.g. in Lapatinib): S(=O)(=O)
        sulfone_pattern = Chem.MolFromSmarts("[SX4](=O)(=O)")
        n_sulfone = len(mol.GetSubstructMatches(sulfone_pattern, uniquify=True)) if sulfone_pattern else 0
        n_charged_groups = n_phosphate + n_carboxylate + n_sulfone

        # Electrostatic correction per group type
        electro_correction = n_phosphate * 1.2 + n_carboxylate * 0.6 + n_sulfone * 0.5

        return {
            "n_heavy_atoms": n_heavy,
            "mw": round(mw, 1),
            "logp": round(logp, 2),
            "tpsa": round(tpsa, 1),
            "n_rotatable": n_rotatable,
            "n_phosphate": n_phosphate,
            "n_carboxylate": n_carboxylate,
            "n_sulfone": n_sulfone,
            "n_charged_groups": n_charged_groups,
            "electro_correction": round(electro_correction, 2),
        }
    except Exception:
        return {"n_heavy_atoms": 1, "mw": 1.0, "logp": None, "tpsa": None, "n_rotatable": 0,
                "n_phosphate": 0, "n_carboxylate": 0, "n_sulfone": 0,
                "n_charged_groups": 0, "electro_correction": 0.0}


# ── Composite binding score (multi-term, physiologically weighted) ─────

def compute_binding_score(
    vina_affinity: float,
    n_heavy: int,
    mw: float,
    logp: float,
    tpsa: float,
    n_phosphate: int,
    n_carboxylate: int,
    n_sulfone: int,
    n_rotatable: int,
) -> dict:
    """
    Multi-term binding score that corrects Vina's known biases.

    Terms:
      1. Vina raw (35%): empirical docking score
      2. Ligand Efficiency (25%): size-normalized, penalizes brute-force hydrophobics
      3. Electrostatic matching (25%): charged-group contributions Vina underestimates
      4. Specificity penalty (15%): penalize non-specific hydrophobic binders
         (high logP + low TPSA = likely promiscuous)

    Returns dict with all sub-scores and the composite.
    """
    if vina_affinity is None:
        return {"composite_score": None}

    # 1. Vina term: normalize to ~0-1 scale (typical range 0 to -8 kcal/mol)
    vina_term = max(0.0, min(1.0, abs(vina_affinity) / 6.0))

    # 2. Ligand Efficiency term
    le = abs(vina_affinity) / max(n_heavy, 1)
    le_term = max(0.0, min(1.0, le / 0.25))  # LE=0.25 → 1.0 (excellent)

    # 3. Electrostatic matching term
    #    Each charged group that CAN form salt bridges in a pocket contributes favorably.
    #    Phosphates in kinase pockets are coordinated by Lys/Mg²⁺ → net favorable.
    #    Carboxylates can form salt bridges with Arg/Lys.
    electro_contrib = (n_phosphate * 3.0 + n_carboxylate * 1.5 + n_sulfone * 1.0)
    electro_term = max(0.0, min(1.0, electro_contrib / 10.0))

    # 4. Specificity penalty: high logP + low TPSA = promiscuous hydrophobic binding
    #    Penalty scales with TPSA deficit and logP excess.
    #    TPSA > 50: no penalty (sufficient polar character for specific binding)
    #    TPSA 0-50: penalty proportional to (1 - TPSA/50) × logP_excess
    tpsa_safe = tpsa if tpsa is not None else 0
    logp_safe = logp if logp is not None else 0
    if logp_safe > 2.0 and tpsa_safe < 50:
        logp_excess = logp_safe - 2.0
        tpsa_factor = max(0.0, 1.0 - tpsa_safe / 50.0)  # 1.0 at TPSA=0, 0.0 at TPSA=50
        specificity_penalty = min(0.8, logp_excess * 0.15 * tpsa_factor + tpsa_factor * 0.20)
    else:
        specificity_penalty = 0.0
    specificity_term = max(0.0, 1.0 - specificity_penalty)

    # 5. Entropic cost: rotatable bonds beyond 5 incur entropic penalty (~0.6 kcal/mol each)
    n_rot_safe = n_rotatable if n_rotatable is not None else 0
    entropy_penalty = max(0.0, min(0.4, (n_rot_safe - 5) * 0.05))
    entropy_term = 1.0 - entropy_penalty

    # ── Composite (0-1 scale, higher = better binder) ──
    # Weights: Vina 25% + LE 20% + Electro 25% + Specificity 20% + Entropy 10%
    # Higher specificity weight penalizes promiscuous hydrophobic binders
    composite = (
        0.25 * vina_term
        + 0.20 * le_term
        + 0.25 * electro_term
        + 0.20 * specificity_term
        + 0.10 * entropy_term
    )

    return {
        "composite_score": round(composite, 4),
        "vina_term": round(vina_term, 4),
        "le_term": round(le_term, 4),
        "electro_term": round(electro_term, 4),
        "specificity_term": round(specificity_term, 4),
        "entropy_term": round(entropy_term, 4),
        "specificity_penalty": round(specificity_penalty, 4),
    }


class VinaDockingPipeline:
    def __init__(self, output_dir: str = "/output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def dock(
        self,
        protein_sequence: str,
        smiles: str,
        num_loops: int = 3,
        num_sampling_steps: int = 32,
        exhaustiveness: int = 8,
        box_size: tuple = (25, 25, 25),
        seed: int = 42,
        min_quality: float = 0.20,
        min_volume: float = 100.0,
        min_residues: int = 5,
        top_n_pockets: int = 5,
    ) -> dict:
        """
        [DEPRECATED] SMILES → fpocket (quality-filtered) → ESMFold2 co-fold → Vina scoring.

        Prefer ScreeningCascade (app/cascade.py) or POST /cascade for the new
        pose-confidence-gated Gnina + MM-GBSA pipeline.

        ALL quality pockets are scored. The best pocket (by composite binding score)
        is selected as the primary result. Per-pocket results are included for comparison.

        Args:
            min_quality: Minimum composite structural quality score (0-1)
            min_volume: Minimum pocket volume (Å³)
            min_residues: Minimum pocket-lining residues
            top_n_pockets: Max pockets to co-fold (all must pass quality filter)

        Returns:
            best_affinity, composite_score, pocket_results[], etc.
        """
        import warnings
        warnings.warn(
            "VinaDockingPipeline.dock() is deprecated. Use ScreeningCascade.run_single() "
            "or the /cascade endpoint for the new Gnina + MM-GBSA pipeline.",
            DeprecationWarning, stacklevel=2,
        )

        t0 = time.time()
        job_id = uuid.uuid4().hex[:12]

        # Step 1: ESMFold2 fold protein → detect + filter pockets by quality
        logger.info("Step 1: Folding protein + detecting pockets (quality-filtered)...")
        fold_result = self._fold_and_detect_quality(
            protein_sequence, num_loops, num_sampling_steps, seed,
            min_quality=min_quality,
            min_volume=min_volume, min_residues=min_residues,
        )
        if fold_result is None:
            return {"error": "Protein folding failed", "job_id": job_id}

        pdb_text = fold_result["pdb_text"]
        quality_pockets = fold_result["pockets"]

        if not quality_pockets:
            logger.warning("No quality pockets found, falling back to blind docking")
            return self._dock_blind(
                job_id, protein_sequence, smiles, pdb_text,
                num_loops, num_sampling_steps, exhaustiveness, box_size, seed, t0,
            )

        # Limit to top-N
        quality_pockets = quality_pockets[:top_n_pockets]

        logger.info(
            f"Scoring {len(quality_pockets)} quality pockets for docking. "
            f"Top: quality={quality_pockets[0].get('quality_score', 'N/A'):.4f}, "
            f"fpocket_score={quality_pockets[0]['score']:.1f}"
        )

        # Step 2: Co-fold + Vina + composite score for ALL quality pockets
        return self._dock_all_pockets(
            job_id, protein_sequence, smiles, pdb_text, quality_pockets,
            num_loops, num_sampling_steps, exhaustiveness, box_size, seed, t0,
        )

    # ══════════════════════════════════════════════════════════════════
    # Step 1: Fold + detect pockets with quality filtering
    # ══════════════════════════════════════════════════════════════════

    def _fold_and_detect_quality(
        self, sequence: str, num_loops: int, num_steps: int, seed: int,
        min_quality: float = 0.25, min_volume: float = 150.0, min_residues: int = 5,
    ) -> Optional[dict]:
        """Fold protein, detect pockets with fpocket, filter by structural quality.

        Uses a global cache so all compounds against the same protein get the SAME fold
        and pocket detection — eliminating non-determinism across compounds.
        If the first fold produces no quality pockets, retries with different seeds.
        """
        cache_key = _fold_cache_key(sequence, num_loops, num_steps, seed)
        cached = _fold_cache.get(cache_key)
        if cached is not None:
            if cached is False:  # sentinel: all retries failed previously
                return {"pdb_text": "", "pockets": []}
            logger.info("Using cached protein fold + pocket detection")
            return cached

        from app.models import ChainInput, FoldingConfig, MoleculeType, PredictionRequest
        from app.local_inference import get_engine
        from app.pocket_detector import _mmcif_to_pdb_fragment, detect_pocket_residues

        for attempt in range(3):
            trial_seed = seed + attempt * 7
            logger.info(f"Fold+Pocket attempt {attempt+1}/3 (seed={trial_seed})")

            chains = [ChainInput(id="A", sequence=sequence, type=MoleculeType.protein)]
            config = FoldingConfig(num_loops=num_loops, num_sampling_steps=num_steps, seed=trial_seed)
            request = PredictionRequest(name="apo", chains=chains, config=config)
            result = get_engine().predict(request)

            pdb_text = _mmcif_to_pdb_fragment(result.mmcif)

            pockets = detect_pocket_residues(
                result.mmcif,
                top_n=5,
                filter_by_quality=True,
                min_quality=min_quality,
                min_volume=min_volume,
                min_residues=min_residues,
            )

            if pockets:
                logger.info(f"Fold+Pocket attempt {attempt+1}: {len(pockets)} quality pockets found")
                cached = {"pdb_text": pdb_text, "pockets": pockets}
                _fold_cache[cache_key] = cached
                return cached
            logger.info(f"Fold+Pocket attempt {attempt+1}: no quality pockets, retrying...")

        # All attempts failed — store False sentinel to prevent infinite retries
        logger.warning("All 3 fold attempts failed to find quality pockets")
        _fold_cache[cache_key] = False
        return {"pdb_text": "", "pockets": []}

    # ══════════════════════════════════════════════════════════════════
    # Step 2: Dock ALL quality pockets → rank by composite score
    # ══════════════════════════════════════════════════════════════════

    def _dock_all_pockets(
        self, job_id: str, sequence: str, smiles: str, pdb_text: str,
        pockets: list[dict], num_loops: int, num_steps: int,
        exhaustiveness: int, box_size: tuple, seed: int, t0: float,
    ) -> dict:
        """Co-fold + Vina for ALL quality pockets. Select best by composite score."""
        lig_props = _compute_ligand_properties(smiles) if smiles else {}
        all_pocket_results = []

        for i, pocket in enumerate(pockets):
            logger.info(
                f"Pocket {i+1}/{len(pockets)}: rank={pocket.get('rank')}, "
                f"quality={pocket.get('quality_score', 0):.4f}, "
                f"volume={pocket.get('volume', 0):.0f}Å³"
            )
            pocket_residues = pocket.get("residue_ids", [])[:25]

            # Co-fold protein + ligand with pocket constraint
            cofold_pdb, mmcif_text = self._cofold(
                sequence, smiles, pocket_residues, num_loops, num_steps, seed
            )

            if cofold_pdb is None:
                all_pocket_results.append({
                    "pocket_rank": pocket.get("rank"),
                    "pocket_quality": pocket.get("quality_score"),
                    "pocket_volume": pocket.get("volume"),
                    "pocket_residues": pocket_residues[:10],
                    "vina_affinity": None,
                    "composite_score": None,
                    "error": "Co-folding failed",
                })
                continue

            # Vina scoring
            pocket_center = self._get_pocket_center_from_residues(pdb_text, pocket_residues)
            dock_result = self._vina_dock(
                cofold_pdb, smiles, pocket_center, box_size, exhaustiveness, seed
            )

            vina_aff = dock_result.get("best_affinity")

            # Compute composite binding score for this pocket
            binding_score = compute_binding_score(
                vina_affinity=vina_aff,
                n_heavy=lig_props.get("n_heavy_atoms", 1),
                mw=lig_props.get("mw", 1.0),
                logp=lig_props.get("logp") or 0,
                tpsa=lig_props.get("tpsa") or 0,
                n_phosphate=lig_props.get("n_phosphate", 0),
                n_carboxylate=lig_props.get("n_carboxylate", 0),
                n_sulfone=lig_props.get("n_sulfone", 0),
                n_rotatable=lig_props.get("n_rotatable", 0),
            )

            all_pocket_results.append({
                "pocket_rank": pocket.get("rank"),
                "pocket_quality": pocket.get("quality_score"),
                "pocket_volume": pocket.get("volume"),
                "pocket_residues": pocket_residues[:10],
                "vina_affinity": vina_aff,
                "vina_poses": dock_result.get("all_affinities", []),
                "num_poses": dock_result.get("num_poses", 0),
                "composite_score": binding_score.get("composite_score"),
                "score_breakdown": binding_score,
            })

        # Select best pocket by composite score (higher = better)
        valid_results = [r for r in all_pocket_results if r.get("composite_score") is not None]
        if valid_results:
            valid_results.sort(key=lambda r: -(r["composite_score"] or 0))
            best = valid_results[0]
        else:
            best = None

        elapsed = time.time() - t0

        if best is None:
            return {
                "job_id": job_id, "smiles": smiles,
                "protein_length": len(sequence),
                "error": "All pocket docking attempts failed",
                "pocket_results": all_pocket_results,
                "wall_time_s": round(elapsed, 1),
            }

        result = {
            "job_id": job_id,
            "smiles": smiles,
            "protein_length": len(sequence),
            # Best pocket summary
            "best_affinity": best["vina_affinity"],
            "all_affinities": best.get("vina_poses", []),
            "num_poses": best.get("num_poses", 0),
            "composite_score": best["composite_score"],
            "score_breakdown": best.get("score_breakdown", {}),
            # Ligand properties
            "ligand_n_heavy": lig_props.get("n_heavy_atoms"),
            "ligand_mw": lig_props.get("mw"),
            "ligand_logp": lig_props.get("logp"),
            "ligand_tpsa": lig_props.get("tpsa"),
            "ligand_n_phosphate": lig_props.get("n_phosphate", 0),
            "ligand_n_carboxylate": lig_props.get("n_carboxylate", 0),
            "ligand_n_sulfone": lig_props.get("n_sulfone", 0),
            "ligand_n_charged": lig_props.get("n_charged_groups", 0),
            "electro_correction": lig_props.get("electro_correction", 0),
            "ligand_efficiency": round(best["vina_affinity"] / max(lig_props.get("n_heavy_atoms", 1), 1), 4) if best["vina_affinity"] else None,
            "corrected_affinity": round(best["vina_affinity"] - lig_props.get("electro_correction", 0), 2) if best["vina_affinity"] else None,
            "corrected_le": round((best["vina_affinity"] - lig_props.get("electro_correction", 0)) / max(lig_props.get("n_heavy_atoms", 1), 1), 4) if best["vina_affinity"] else None,
            # Best pocket info
            "best_pocket_rank": best["pocket_rank"],
            "pocket_quality": best["pocket_quality"],
            "pocket_volume": best["pocket_volume"],
            "pocket_residues": best.get("pocket_residues", []),
            # All pockets summary
            "num_pockets_scored": len(all_pocket_results),
            "pocket_results": all_pocket_results,
            "docking_mode": "pocket_guided",
            "wall_time_s": round(elapsed, 1),
        }

        out_path = self.output_dir / f"dock_{job_id}.json"
        out_path.write_text(json.dumps(result, indent=2, default=str))

        logger.info(
            f"Dock done: {smiles[:30]} → Vina={best['vina_affinity']} kcal/mol, "
            f"Composite={best['composite_score']:.3f} "
            f"(best pocket rank={best['pocket_rank']}, {elapsed:.1f}s)"
        )
        return result

    # ══════════════════════════════════════════════════════════════════
    # Blind docking fallback (no quality pockets found)
    # ══════════════════════════════════════════════════════════════════

    def _dock_blind(
        self, job_id: str, sequence: str, smiles: str, pdb_text: str,
        num_loops: int, num_steps: int, exhaustiveness: int,
        box_size: tuple, seed: int, t0: float,
    ) -> dict:
        """Fallback: co-fold without pocket constraint, dock at protein center."""
        logger.info("Blind docking: co-folding without pocket constraint...")
        cofold_pdb, mmcif_text = self._cofold(
            sequence, smiles, None, num_loops, num_steps, seed
        )

        if cofold_pdb is None:
            elapsed = time.time() - t0
            return {"job_id": job_id, "smiles": smiles,
                    "protein_length": len(sequence),
                    "error": "Co-folding failed", "wall_time_s": round(elapsed, 1)}

        # Use geometric center of protein as docking center
        center = self._get_protein_center(pdb_text)
        dock_result = self._vina_dock(
            cofold_pdb, smiles, center, box_size, exhaustiveness, seed
        )

        elapsed = time.time() - t0
        lig_props = _compute_ligand_properties(smiles) if smiles else {}
        n_heavy = lig_props.get("n_heavy_atoms", 1)
        mw = lig_props.get("mw", 1.0)
        raw_aff = dock_result.get("best_affinity")

        binding_score = compute_binding_score(
            vina_affinity=raw_aff,
            n_heavy=n_heavy, mw=mw,
            logp=lig_props.get("logp") or 0,
            tpsa=lig_props.get("tpsa") or 0,
            n_phosphate=lig_props.get("n_phosphate", 0),
            n_carboxylate=lig_props.get("n_carboxylate", 0),
            n_sulfone=lig_props.get("n_sulfone", 0),
            n_rotatable=lig_props.get("n_rotatable", 0),
        )

        return {
            "job_id": job_id, "smiles": smiles,
            "protein_length": len(sequence),
            "best_affinity": raw_aff,
            "all_affinities": dock_result.get("all_affinities", []),
            "num_poses": dock_result.get("num_poses", 0),
            "composite_score": binding_score.get("composite_score"),
            "score_breakdown": binding_score,
            "ligand_efficiency": round(raw_aff / n_heavy, 4) if raw_aff is not None and n_heavy else None,
            "corrected_affinity": round(raw_aff - lig_props.get("electro_correction", 0), 2) if raw_aff else None,
            "corrected_le": round((raw_aff - lig_props.get("electro_correction", 0)) / max(n_heavy, 1), 4) if raw_aff else None,
            "ligand_n_heavy": n_heavy,
            "ligand_mw": round(mw, 1),
            "ligand_logp": lig_props.get("logp"),
            "ligand_tpsa": lig_props.get("tpsa"),
            "ligand_n_phosphate": lig_props.get("n_phosphate", 0),
            "ligand_n_carboxylate": lig_props.get("n_carboxylate", 0),
            "ligand_n_sulfone": lig_props.get("n_sulfone", 0),
            "ligand_n_charged": lig_props.get("n_charged_groups", 0),
            "electro_correction": lig_props.get("electro_correction", 0),
            "pocket_quality": None,
            "pocket_volume": None,
            "pocket_residues": [],
            "pocket_results": [],
            "num_pockets_scored": 0,
            "docking_mode": "blind",
            "wall_time_s": round(elapsed, 1),
        }

    # ══════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════

    def _get_pocket_center_from_residues(self, pdb_text: str, residue_ids: list) -> Optional[list]:
        """Compute pocket center from CA atoms of pocket-lining residues in apo PDB."""
        coords = []
        for line in pdb_text.split("\n"):
            if not line.startswith("ATOM") or "CA" not in line:
                continue
            try:
                res_seq = int(line[22:26].strip())
                if res_seq in residue_ids:
                    coords.append([
                        float(line[30:38]), float(line[38:46]), float(line[46:54])
                    ])
            except (ValueError, IndexError):
                continue
        if coords:
            return np.mean(coords, axis=0).tolist()
        return None

    def _get_protein_center(self, pdb_text: str) -> list:
        """Get geometric center of all CA atoms."""
        coords = []
        for line in pdb_text.split("\n"):
            if line.startswith("ATOM") and "CA" in line:
                try:
                    coords.append([
                        float(line[30:38]), float(line[38:46]), float(line[46:54])
                    ])
                except (ValueError, IndexError):
                    continue
        if coords:
            return np.mean(coords, axis=0).tolist()
        return [0.0, 0.0, 0.0]

    # ══════════════════════════════════════════════════════════════════
    # Co-fold protein + ligand with pocket constraint
    # ══════════════════════════════════════════════════════════════════

    def _cofold(
        self, sequence: str, smiles: str, pocket_residues: list,
        num_loops: int, num_steps: int, seed: int,
    ) -> tuple[Optional[str], Optional[str]]:
        """Co-fold protein + ligand constrained to detected pocket."""
        from app.models import ChainInput, FoldingConfig, MoleculeType, PredictionRequest
        from app.local_inference import get_engine
        from esm.utils.structure.input_builder import PocketConditioning
        from app.pocket_detector import _mmcif_to_pdb_fragment

        chains = [
            ChainInput(id="A", sequence=sequence, type=MoleculeType.protein),
            ChainInput(id="L", sequence="", type=MoleculeType.ligand, smiles=smiles),
        ]
        config = FoldingConfig(num_loops=num_loops, num_sampling_steps=num_steps, seed=seed)
        request = PredictionRequest(name="cofold", chains=chains, config=config)

        pocket = None
        if pocket_residues:
            pocket = PocketConditioning(
                binder_chain_id="L",
                contacts=[("A", r) for r in pocket_residues],
            )

        from app.local_inference import get_engine as _get_engine
        result = _get_engine().predict(request, pocket=pocket)

        pdb_text = _mmcif_to_pdb_fragment(result.mmcif)
        return pdb_text, result.mmcif

    # ══════════════════════════════════════════════════════════════════
    # Vina scoring on co-folded complex
    # ══════════════════════════════════════════════════════════════════

    def _vina_dock(
        self, pdb_text: str, smiles: str, center: list,
        box_size: tuple, exhaustiveness: int, seed: int,
    ) -> dict:
        """Score the co-folded complex with Vina."""
        try:
            from vina import Vina
        except ImportError:
            return {"best_affinity": None, "all_affinities": [], "num_poses": 0}

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)

            receptor_pdb = td / "receptor.pdb"
            receptor_pdb.write_text(self._extract_chain_a(pdb_text))

            pdbqt_path = td / "receptor.pdbqt"
            try:
                subprocess.run(
                    ["obabel", str(receptor_pdb), "-O", str(pdbqt_path),
                     "-xr", "-xp", "7.4", "--partialcharge", "gasteiger"],
                    capture_output=True, text=True, timeout=60, check=True,
                )
            except Exception as e:
                logger.error(f"Receptor PDBQT failed: {e}")
                return {"best_affinity": None, "all_affinities": [], "num_poses": 0}

            lig_pdbqt = td / "ligand.pdbqt"
            try:
                self._smiles_to_pdbqt(smiles, td, lig_pdbqt)
            except Exception:
                return {"best_affinity": None, "all_affinities": [], "num_poses": 0}

            if not lig_pdbqt.exists():
                return {"best_affinity": None, "all_affinities": [], "num_poses": 0}

            if center is None:
                logger.warning("Vina docking center is None, skipping")
                return {"best_affinity": None, "all_affinities": [], "num_poses": 0}

            v = Vina(sf_name="vina", seed=seed, verbosity=0)
            try:
                v.set_receptor(str(pdbqt_path))
                v.set_ligand_from_file(str(lig_pdbqt))
                v.compute_vina_maps(center=center, box_size=list(box_size))
                v.dock(exhaustiveness=exhaustiveness, n_poses=9)
                raw = v.energies(n_poses=9)
                if raw is None:
                    return {"best_affinity": None, "all_affinities": [], "num_poses": 0}
                flat = raw.tolist() if hasattr(raw, 'tolist') else raw
                energies = [float(e[0] if isinstance(e, (list, tuple)) else e) for e in flat]
                return {
                    "best_affinity": round(energies[0], 2),
                    "all_affinities": [round(e, 2) for e in energies],
                    "num_poses": len(energies),
                }
            except Exception as e:
                logger.error(f"Vina failed: {e}")
                return {"best_affinity": None, "all_affinities": [], "num_poses": 0}

    def _smiles_to_pdbqt(self, smiles: str, td: Path, out_path: Path):
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        sdf = td / "lig.sdf"
        Chem.SDWriter(str(sdf)).write(mol)
        subprocess.run(
            ["obabel", str(sdf), "-O", str(out_path),
             "--gen3d", "--partialcharge", "gasteiger"],
            capture_output=True, text=True, timeout=120, check=True,
        )

    def _extract_chain_a(self, pdb_text: str) -> str:
        """Keep only chain A atoms from co-folded PDB."""
        lines = []
        last_res = ""
        for line in pdb_text.split("\n"):
            if line.startswith("ATOM"):
                chain = line[21:22] if len(line) > 21 else " "
                if chain.strip() in ("", "A"):
                    res = line[22:27]
                    if last_res and res != last_res:
                        try:
                            if int(line[22:26]) < int(last_res[0:4]):
                                lines.append("TER")
                        except ValueError:
                            pass
                    last_res = res
                    lines.append(line)
        lines.append("TER\nEND")
        return "\n".join(lines)


_pipeline: Optional[VinaDockingPipeline] = None


def get_docking() -> VinaDockingPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = VinaDockingPipeline()
    return _pipeline
