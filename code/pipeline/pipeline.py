"""
SMILES → Protein-Ligand Binding Prediction Pipeline.
ESMFold2 natively supports SMILES for ligands — no CCD lookup required.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional

from app.models import ChainInput, FoldingConfig, MoleculeType, PredictionRequest, PredictionResult

logger = logging.getLogger(__name__)


class CompoundBindingPipeline:
    def __init__(self, output_dir: str = "/output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def predict_binding(
        self,
        protein_sequence: str,
        smiles: Optional[str] = None,
        ccd: Optional[list[str]] = None,
        protein_id: str = "A",
        ligand_id: str = "L",
        num_loops: int = 3,
        num_sampling_steps: int = 32,
        seed: int = 42,
        pocket_residues: Optional[list[tuple[str, int]]] = None,
        covalent_bonds: Optional[list[dict]] = None,
    ) -> dict:
        """
        Predict protein-ligand complex structure.

        Args:
            protein_sequence: Amino acid sequence
            smiles: SMILES string of the ligand (primary method, no CCD needed)
            ccd: CCD codes for known PDB ligands (fallback)
            pocket_residues: Optional list of (chain_id, residue_idx) for pocket targeting
            covalent_bonds: Optional list of {chain1, res1, atom1, chain2, res2, atom2}

        Returns dict with job_id, mmcif_path, plddt_mean, ptm, iptm, etc.
        """
        t0 = time.time()
        job_id = uuid.uuid4().hex[:12]

        # Validate input
        if not smiles and not ccd:
            raise ValueError("Either smiles or ccd must be provided for the ligand")

        has_smiles = bool(smiles)

        # Build chains
        chains = [
            ChainInput(id=protein_id, sequence=protein_sequence, type=MoleculeType.protein),
        ]

        if smiles:
            chains.append(ChainInput(
                id=ligand_id, sequence="", type=MoleculeType.ligand,
                smiles=smiles,
            ))
        elif ccd:
            chains.append(ChainInput(
                id=ligand_id, sequence="", type=MoleculeType.ligand,
                ccd=ccd,
            ))

        config = FoldingConfig(
            num_loops=num_loops,
            num_sampling_steps=num_sampling_steps,
            seed=seed,
        )
        request = PredictionRequest(name=f"bind_{job_id}", chains=chains, config=config)

        # Build pocket conditioning
        pocket = None
        if pocket_residues:
            logger.info(f"Applying pocket constraint: {len(pocket_residues)} residues")
            from esm.utils.structure.input_builder import PocketConditioning
            pocket = PocketConditioning(
                binder_chain_id=ligand_id,
                contacts=pocket_residues,
            )

        # Build covalent bonds
        cov_bonds = None
        if covalent_bonds:
            from esm.models.esmfold2 import CovalentBond
            cov_bonds = [
                CovalentBond(
                    chain_id1=b["chain1"], res_idx1=b["res1"], atom_idx1=b["atom1"],
                    chain_id2=b["chain2"], res_idx2=b["res2"], atom_idx2=b["atom2"],
                )
                for b in covalent_bonds
            ]

        # Run prediction
        mode = os.environ.get("ESMFOLD2_MODE", "local")
        fold_start = time.time()

        if mode == "local":
            from app.local_inference import get_engine
            pred_result = get_engine().predict(request, pocket=pocket, covalent_bonds=cov_bonds)
        else:
            from app.api_client import get_api_client
            pred_result = get_api_client().predict(request)

        fold_time = time.time() - fold_start

        return self._save_result(pred_result, job_id, smiles, ccd, protein_sequence, num_loops, num_sampling_steps, seed, fold_time, t0)

    def predict_binding_auto(
        self,
        protein_sequence: str,
        smiles: Optional[str] = None,
        ccd: Optional[list[str]] = None,
        protein_id: str = "A",
        ligand_id: str = "L",
        num_loops: int = 3,
        num_sampling_steps: int = 32,
        seed: int = 42,
        min_quality: float = 0.20,
        min_volume: float = 100.0,
        min_residues: int = 5,
    ) -> dict:
        """
        [DEPRECATED] Fold protein → fpocket pockets → quality filter → co-fold.

        Prefer ScreeningCascade (app/cascade.py) or POST /cascade for the new
        Gnina + MM-GBSA pipeline.
        """
        from app.pocket_detector import detect_pocket_residues

        # Step 1: Fold protein alone to get structure
        chains = [ChainInput(id=protein_id, sequence=protein_sequence, type=MoleculeType.protein)]
        config = FoldingConfig(num_loops=num_loops, num_sampling_steps=num_sampling_steps, seed=seed)
        request = PredictionRequest(name=f"apo_{protein_id}", chains=chains, config=config)

        logger.info("Step 1: Folding protein to detect pockets...")
        mode = os.environ.get("ESMFOLD2_MODE", "local")
        if mode == "local":
            from app.local_inference import get_engine
            apo_result = get_engine().predict(request)
        else:
            from app.api_client import get_api_client
            apo_result = get_api_client().predict(request)

        # Step 2: Detect pockets with structural quality filtering
        logger.info("Step 2: Detecting binding pockets (quality-filtered)...")
        pockets = detect_pocket_residues(
            apo_result.mmcif, chain_id=protein_id, top_n=3,
            filter_by_quality=True,
            min_quality=min_quality,
            min_volume=min_volume, min_residues=min_residues,
        )

        if pockets:
            best_pocket = pockets[0]
            pocket_residues = [(protein_id, r[1]) for r in best_pocket.get("residues", [])[:20]]
            logger.info(f"Found {len(pockets)} quality pockets. "
                         f"Top: quality={best_pocket.get('quality_score', 'N/A'):.4f}, "
                         f"fpocket_score={best_pocket['score']:.1f}, "
                         f"volume={best_pocket['volume']:.0f}Å³, "
                         f"residues={pocket_residues[:5]}...")
        else:
            pocket_residues = None
            logger.info("No pockets detected, running unconstrained co-folding")

        # Step 3: Co-fold with pocket constraint
        logger.info("Step 3: Co-folding protein + ligand...")
        return self.predict_binding(
            protein_sequence=protein_sequence,
            smiles=smiles,
            ccd=ccd,
            protein_id=protein_id,
            ligand_id=ligand_id,
            num_loops=num_loops,
            num_sampling_steps=num_sampling_steps,
            seed=seed,
            pocket_residues=pocket_residues,
        )

    def _save_result(self, pred_result, job_id, smiles, ccd, protein_sequence, num_loops, num_sampling_steps, seed, fold_time, t0):
        has_smiles = bool(smiles)
        out_path = self.output_dir / f"{pred_result.name}_{pred_result.job_id}"
        mmcif_path = out_path.with_suffix(".cif")
        mmcif_path.write_text(pred_result.mmcif)

        summary = {
            "job_id": job_id,
            "input_type": "smiles" if has_smiles else "ccd",
            "smiles": smiles,
            "ccd": ccd,
            "protein_sequence": protein_sequence,
            "plddt_mean": pred_result.plddt_mean,
            "ptm": pred_result.ptm,
            "iptm": pred_result.iptm,
            "num_residues": pred_result.num_residues,
            "wall_time_s": round(time.time() - t0, 1),
            "fold_time_s": round(fold_time, 1),
            "config": {"num_loops": num_loops, "num_sampling_steps": num_sampling_steps, "seed": seed},
        }
        json_path = out_path.with_suffix(".json")
        json_path.write_text(json.dumps(summary, indent=2))

        return {
            "job_id": job_id,
            "input_type": summary["input_type"],
            "smiles": smiles,
            "ccd": ccd,
            "plddt_mean": pred_result.plddt_mean,
            "ptm": pred_result.ptm,
            "iptm": pred_result.iptm,
            "num_residues": pred_result.num_residues,
            "wall_time_s": round(time.time() - t0, 1),
            "fold_time_s": round(fold_time, 1),
            "mmcif_path": str(mmcif_path),
            "summary_path": str(json_path),
        }


    def cascade(
        self,
        protein_sequence: str,
        smiles: str,
        seed: int = 42,
        num_loops: int = 8,
        num_sampling_steps: int = 32,
    ):
        """Convenience: run a single compound through the cascade pipeline."""
        from app.cascade import get_cascade
        return get_cascade().run_single(
            protein_sequence, smiles, seed=seed,
            num_loops=num_loops, num_sampling_steps=num_sampling_steps,
        )


_pipeline: Optional[CompoundBindingPipeline] = None


def get_pipeline() -> CompoundBindingPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = CompoundBindingPipeline()
    return _pipeline
