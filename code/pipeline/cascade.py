"""
ScreeningCascade — recall-first, three-leg core engine.

Leg 1 (pocket_off):   Unconstrained ESMFold2 co-fold → honest PAE signal
Leg 2 (pocket_on):    PocketConditioning co-fold per candidate site
Leg 3 (orthogonal):   Gnina docking rescue (non-ESMFold2 pose source)

Merge = union of all legs. Gate = triage (route, never kill).
MM-GBSA on ALL gate survivors. Parked bucket for truly hopeless compounds.

References:
- docs/REBUILD_PLAN_v2.md
- docs/REBUILD_PLAN_addendum_A_*.md
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.cascade_config import CascadeConfig
from app.models import (
    BoltzScoreDetail,
    CascadeResponse,
    CompoundMergeResult,
    GateSignalDetail,
    GninaScoreDetail,
    LegResult,
    MMGBSADetail,
)

logger = logging.getLogger("cascade")


# ── ESMFold2 CoFolder adapter ─────────────────────────────────────────────


class ESMFold2CoFolder:
    """Adapts LocalInferenceEngine to the cascade's co-fold protocol."""

    def __init__(self, engine=None):
        if engine is None:
            from app.local_inference import get_engine

            engine = get_engine()
        self._engine = engine

    def __call__(
        self,
        protein_sequence: str,
        smiles: str,
        *,
        seed: int = 42,
        num_loops: int,
        num_sampling_steps: int = 32,
        n_consistency_samples: int,
        pocket_residues: list[int] | None = None,
    ):
        """Co-fold. If pocket_residues given, use PocketConditioning (Leg 2)."""
        from app.models import (
            ChainInput,
            FoldingConfig,
            MoleculeType,
            PredictionRequest,
        )

        chains = [
            ChainInput(id="A", sequence=protein_sequence, type=MoleculeType.protein),
            ChainInput(id="L", sequence="", type=MoleculeType.ligand, smiles=smiles),
        ]
        config = FoldingConfig(
            num_loops=num_loops,
            num_sampling_steps=num_sampling_steps,
            num_diffusion_samples=1,
            seed=seed,
        )
        request = PredictionRequest(name="cascade_cofold", chains=chains, config=config)

        return self._engine.predict_with_details(
            request,
            n_samples=n_consistency_samples,
            smiles=smiles,
            pocket_residues=pocket_residues,
            binder_chain_id="L",
        )

    def fold_apo_protein(
        self,
        protein_sequence: str,
        seed: int = 42,
        num_loops: int = 3,
        num_sampling_steps: int = 32,
    ) -> str:
        """Fold apo protein (no ligand) for Leg 3 receptor."""
        from app.models import (
            ChainInput,
            FoldingConfig,
            MoleculeType,
            PredictionRequest,
        )

        chains = [
            ChainInput(id="A", sequence=protein_sequence, type=MoleculeType.protein)
        ]
        config = FoldingConfig(
            num_loops=num_loops,
            num_sampling_steps=num_sampling_steps,
            num_diffusion_samples=1,
            seed=seed,
        )
        request = PredictionRequest(name="apo_receptor", chains=chains, config=config)
        result = self._engine.predict(request)
        return result.mmcif


class ScreeningCascade:
    """Recall-first three-leg cascade."""

    def __init__(self, engine=None, config: CascadeConfig | None = None):
        self.cofolder = ESMFold2CoFolder(engine)
        self.cfg = config or CascadeConfig()

    # ═══════════════════════════════════════════════════════════════════════
    # Leg 1 — Unconstrained co-fold (pocket-off)
    # ═══════════════════════════════════════════════════════════════════════

    def run_leg1(
        self,
        protein_sequence: str,
        smiles: str,
        num_loops: int,
        seed: int = 42,
        num_sampling_steps: int = 32,
    ) -> LegResult:
        """Leg 1: unconstrained co-fold — unbiased site discovery + honest PAE."""
        cfg = self.cfg
        cid = _short_id(smiles)
        t0 = time.time()
        leg = LegResult(leg="pocket_off", status="running")
        routing_log: list[str] = []

        try:
            logger.info("[%s] Leg 1: unconstrained co-fold", cid)
            fold_result = self.cofolder(
                protein_sequence,
                smiles,
                seed=seed,
                num_loops=num_loops,
                num_sampling_steps=num_sampling_steps,
                n_consistency_samples=cfg.consistency_samples,
            )

            # Gate signal — Leg 1 is the ONLY honest source
            from app.confidence_gate import triage_gate

            triage = triage_gate(fold_result, cfg, source_leg="pocket_off")

            leg.gate = GateSignalDetail(
                source_leg="pocket_off",
                passed=triage.gate.passed,
                pocket_residues=triage.gate.pocket_residues,
                pocket_plddt=triage.gate.pocket_plddt,
                ligand_plddt=triage.gate.ligand_plddt,
                interface_pae=triage.gate.interface_pae,
                ligand_iptm=triage.gate.ligand_iptm,
                reasons=triage.gate.reasons,
                posebusters_passed=triage.gate.posebusters_passed,
                posebusters_failed=triage.gate.posebusters_failed,
                pose_rmsd_max=triage.gate.pose_rmsd_max,
                pose_rmsd_samples=triage.gate.pose_rmsd_samples,
            )
            leg.status = triage.status
            routing_log = triage.routing_actions.copy()

            # If ok or weak, run scoring
            if triage.status in ("ok", "weak") and fold_result.complex_path:
                wd = self._workdir_for(cid, "leg1")
                if cfg.use_boltz:
                    rec_raw, lig_sdf = self._complex_prep(fold_result, wd)
                    leg._rec_raw = str(rec_raw)
                    leg._lig_sdf = str(lig_sdf)
                    leg._fold_complex_path = (
                        str(fold_result.complex_path)
                        if fold_result.complex_path
                        else None
                    )
                    leg._pocket_residues = triage.gate.pocket_residues
                    leg = self._boltz_score(
                        leg, fold_result, protein_sequence, smiles, wd
                    )
                else:
                    leg = self._gnina_score(leg, fold_result, wd)
                    leg._fold_complex_path = (
                        str(fold_result.complex_path)
                        if fold_result.complex_path
                        else None
                    )
                    leg._pocket_residues = triage.gate.pocket_residues

            leg.routing_log = routing_log

        except Exception as e:
            logger.exception("[%s] Leg 1 failed: %s", cid, e)
            leg.status = "failed"
            leg.error = f"{type(e).__name__}: {e}"
            leg.routing_log = ["try_orthogonal_rescue"]

        leg.wall_time_s = round(time.time() - t0, 1)
        return leg

    # ═══════════════════════════════════════════════════════════════════════
    # Leg 2 — PocketConditioning co-fold (pocket-on)
    # ═══════════════════════════════════════════════════════════════════════

    def run_leg2(
        self,
        protein_sequence: str,
        smiles: str,
        candidate_sites: list[list[int]],  # list of pocket residue lists
        num_loops: int,
        seed: int = 42,
        num_sampling_steps: int = 32,
    ) -> list[LegResult]:
        """Leg 2: PocketConditioning co-fold for each candidate site.

        Returns one LegResult per candidate site.
        """
        cfg = self.cfg
        cid = _short_id(smiles)
        results: list[LegResult] = []

        for site_idx, pocket_residues in enumerate(candidate_sites):
            site_label = f"site_{site_idx}"
            t0 = time.time()
            leg = LegResult(leg="pocket_on", site_label=site_label, status="running")
            routing_log: list[str] = []

            try:
                logger.info(
                    "[%s] Leg 2 / %s: PocketConditioning with %d residues",
                    cid,
                    site_label,
                    len(pocket_residues),
                )

                fold_result = self.cofolder(
                    protein_sequence,
                    smiles,
                    seed=seed,
                    num_loops=num_loops,
                    num_sampling_steps=num_sampling_steps,
                    n_consistency_samples=max(1, cfg.consistency_samples // 2),
                    pocket_residues=pocket_residues,
                )

                # Gate — note: Leg 2 PAE is contaminated, downgraded
                from app.confidence_gate import triage_gate

                triage = triage_gate(
                    fold_result,
                    cfg,
                    source_leg="pocket_on",
                    pocket_residues=pocket_residues,
                )

                leg.gate = GateSignalDetail(
                    source_leg="pocket_on",
                    passed=triage.gate.passed,
                    pocket_residues=pocket_residues,
                    pocket_plddt=triage.gate.pocket_plddt,
                    ligand_plddt=triage.gate.ligand_plddt,
                    interface_pae=triage.gate.interface_pae,
                    ligand_iptm=triage.gate.ligand_iptm,
                    reasons=triage.gate.reasons,
                    posebusters_passed=triage.gate.posebusters_passed,
                    posebusters_failed=triage.gate.posebusters_failed,
                    pose_rmsd_max=triage.gate.pose_rmsd_max,
                    pose_rmsd_samples=triage.gate.pose_rmsd_samples,
                )
                leg.status = triage.status
                routing_log = triage.routing_actions.copy()

                if triage.status in ("ok", "weak") and fold_result.complex_path:
                    wd = self._workdir_for(cid, f"leg2_{site_label}")
                    if cfg.use_boltz:
                        rec_raw, lig_sdf = self._complex_prep(fold_result, wd)
                        leg._rec_raw = str(rec_raw)
                        leg._lig_sdf = str(lig_sdf)
                        leg._fold_complex_path = str(fold_result.complex_path)
                        leg._pocket_residues = pocket_residues
                        leg = self._boltz_score(
                            leg, fold_result, protein_sequence, smiles, wd
                        )
                    else:
                        leg = self._gnina_score(leg, fold_result, wd)
                        leg._fold_complex_path = str(fold_result.complex_path)
                        leg._pocket_residues = pocket_residues

                leg.routing_log = routing_log

            except Exception as e:
                logger.exception("[%s] Leg 2 / %s failed: %s", cid, site_label, e)
                leg.status = "failed"
                leg.error = f"{type(e).__name__}: {e}"
                routing_log.append("try_orthogonal_rescue")
                leg.routing_log = routing_log

            leg.wall_time_s = round(time.time() - t0, 1)
            results.append(leg)

        return results

    # ═══════════════════════════════════════════════════════════════════════
    # Leg 3 — Orthogonal rescue (non-ESMFold2)
    # ═══════════════════════════════════════════════════════════════════════

    def run_leg3(
        self,
        protein_sequence: str,
        smiles: str,
        pocket_sites: list[list[int]],
        num_loops: int,
        seed: int = 42,
        num_sampling_steps: int = 32,
    ) -> LegResult:
        """Leg 3: orthogonal rescue via Boltz-2 affinity (independent of ESMFold2)."""
        cfg = self.cfg
        cid = _short_id(smiles)
        t0 = time.time()
        leg = LegResult(leg="orthogonal", status="running")

        try:
            logger.info("[%s] Leg 3: orthogonal rescue (Boltz-2 affinity)", cid)

            wd = self._workdir_for(cid, "leg3")

            if cfg.use_boltz:
                from app.ortho_rescue import boltz_rescue_score

                rescue = boltz_rescue_score(
                    protein_sequence,
                    smiles,
                    wd,
                )
            else:
                # Fallback to Gnina rescue
                mmcif = self.cofolder.fold_apo_protein(
                    protein_sequence,
                    seed=seed + 1000,
                    num_loops=num_loops,
                    num_sampling_steps=num_sampling_steps,
                )
                from app.pocket_detector import _mmcif_to_pdb_fragment

                pdb_text = _mmcif_to_pdb_fragment(mmcif)
                rec_pdb = wd / "receptor.pdb"
                rec_pdb.write_text(pdb_text)

                from app.ortho_rescue import compute_pocket_center, gnina_rescue_dock

                centers = []
                for site_res in pocket_sites:
                    c = compute_pocket_center(rec_pdb, site_res)
                    if c is not None:
                        centers.append(c)
                if not centers:
                    from app.docking import _get_protein_center

                    centers = [tuple(_get_protein_center(pdb_text))]

                rescue = gnina_rescue_dock(
                    rec_pdb,
                    smiles,
                    centers,
                    wd,
                    gnina_bin=cfg.gnina_bin,
                    cnn_model=cfg.gnina_cnn_model,
                    box_size=cfg.rescue_box_size,
                    exhaustiveness=cfg.rescue_exhaustiveness,
                    seed=seed,
                    use_gpu=cfg.gnina_use_gpu,
                )

            if rescue.status == "ok":
                leg.status = "ok"
                if cfg.use_boltz:
                    leg.boltz = BoltzScoreDetail(
                        affinity_binary=rescue.cnn_affinity,
                        affinity_value=rescue.cnn_score,
                    )
                else:
                    leg.gnina = GninaScoreDetail(
                        cnn_affinity=rescue.cnn_affinity,
                        cnn_score=rescue.cnn_score,
                        vina_affinity=rescue.vina_affinity,
                        raw_output=rescue.raw_output,
                    )
            else:
                leg.status = "failed"
                leg.error = rescue.error

            leg.routing_log = [f"orthogonal_{rescue.engine}_rescue"]

        except Exception as e:
            logger.exception("[%s] Leg 3 failed: %s", cid, e)
            leg.status = "failed"
            leg.error = f"{type(e).__name__}: {e}"
            leg.routing_log = ["orthogonal_failed"]

        leg.wall_time_s = round(time.time() - t0, 1)
        return leg

    # ═══════════════════════════════════════════════════════════════════════
    # Merge — union of all legs, triage-based confidence tier
    # ═══════════════════════════════════════════════════════════════════════

    def merge(
        self,
        smiles: str,
        leg1: LegResult,
        leg2_results: list[LegResult] = None,
        leg3: LegResult | None = None,
    ) -> CompoundMergeResult:
        """Merge results from all legs into a single compound result.

        Union: any leg/site with a usable result is kept.
        Triage: confidence_tier = convergent | divergent | rescue | parked.
        """
        cid = _short_id(smiles)
        leg2_results = leg2_results or []

        all_legs: dict[str, list[LegResult]] = {
            "pocket_off": [leg1],
            "pocket_on": leg2_results,
        }
        if leg3 is not None:
            all_legs["orthogonal"] = [leg3]

        routing_log: list[str] = []
        for leg_result in [leg1] + leg2_results + ([leg3] if leg3 else []):
            routing_log.extend(leg_result.routing_log)

        # Collect all successful Gnina / Boltz scores
        gnina_scores = []
        boltz_scores = []
        for leg_list in all_legs.values():
            for lr in leg_list:
                if lr.gnina is not None and not _is_nan(lr.gnina.cnn_affinity):
                    gnina_scores.append(lr.gnina)
                if lr.boltz is not None and lr.boltz.affinity_binary > 0:
                    boltz_scores.append(lr.boltz)

        best_gnina = gnina_scores[0] if gnina_scores else None
        if gnina_scores:
            gnina_scores.sort(key=lambda g: g.cnn_affinity, reverse=True)
            best_gnina = gnina_scores[0]

        best_boltz = boltz_scores[0] if boltz_scores else None
        if boltz_scores:
            boltz_scores.sort(key=lambda b: b.affinity_binary, reverse=True)
            best_boltz = boltz_scores[0]

        # ── Determine status and confidence tier ────────────────────────
        leg1_ok = leg1.status == "ok"
        leg2_ok = any(lr.status == "ok" for lr in leg2_results)
        leg3_ok = leg3 is not None and leg3.status == "ok"
        leg1_weak = leg1.status == "weak"
        leg2_weak = any(lr.status == "weak" for lr in leg2_results)

        any_ok = leg1_ok or leg2_ok or leg3_ok
        any_usable = any_ok or leg1_weak or leg2_weak

        if leg1_ok and leg2_ok:
            confidence_tier = "convergent"
            agreement = "convergent"
        elif leg1_ok and leg2_ok is False and leg2_weak is False:
            confidence_tier = "rescue" if leg3_ok else "divergent"
            agreement = "divergent"
        elif any_ok:
            confidence_tier = (
                "rescue" if (leg3_ok and not leg1_ok and not leg2_ok) else "divergent"
            )
            agreement = "partial"
        elif any_usable:
            confidence_tier = "divergent"
            agreement = "divergent"
        elif leg3_ok:
            confidence_tier = "rescue"
            agreement = "single_source"
        else:
            confidence_tier = "parked"
            agreement = "none"

        status = "parked" if confidence_tier == "parked" else "kept"

        # Calibrated consensus score (additive; see app/consensus.py and
        # notes/consensus-calibration-results.md). Uses leg1 gate signals
        # per gate_signal_from="pocket_off" HARD rule.
        consensus_val = None
        try:
            from app.consensus import consensus_score, inputs_from_merge_result
            _tmp = CompoundMergeResult(compound_id=cid, smiles=smiles,
                                       legs=all_legs, best_boltz=best_boltz)
            _inp = inputs_from_merge_result(_tmp)
            if _inp is not None:
                consensus_val = consensus_score(
                    _inp.affinity, _inp.interface_pae,
                    _inp.ligand_plddt, _inp.ligand_iptm)
        except Exception:
            consensus_val = None

        # Park reasons
        park_reasons = []
        if status == "parked":
            park_reasons = [
                f"leg1: {leg1.status}",
                f"leg2: {[lr.status for lr in leg2_results]}",
                f"leg3: {leg3.status if leg3 else 'not_run'}",
            ]

        # Build gate summary
        gate_summary = {
            "leg1_pae": leg1.gate.interface_pae if leg1.gate else None,
            "leg1_iptm": leg1.gate.ligand_iptm if leg1.gate else None,
            "leg1_status": leg1.status,
            "leg2_sites_tried": len(leg2_results),
            "leg2_ok": sum(1 for lr in leg2_results if lr.status == "ok"),
            "leg3_triggered": leg3 is not None,
            "leg3_status": leg3.status if leg3 else "not_triggered",
        }

        return CompoundMergeResult(
            compound_id=cid,
            smiles=smiles,
            status=status,
            confidence_tier=confidence_tier,
            legs=all_legs,
            best_gnina=best_gnina,
            best_boltz=best_boltz,
            best_dg_gb=None,  # filled after MM-GBSA
            gate_summary=gate_summary,
            routing_log=routing_log,
            agreement=agreement,
            park_reasons=park_reasons,
            consensus_score=consensus_val,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Top-level run — all legs → merge → MM-GBSA → ranking
    # ═══════════════════════════════════════════════════════════════════════

    def run(
        self,
        protein_sequence: str,
        smiles_list: list[str],
        seed: int = 42,
        num_loops: int = 8,
        num_sampling_steps: int = 32,
        candidate_sites: list[list[int]] | None = None,
    ) -> CascadeResponse:
        """Run the full three-leg recall-first cascade on a compound library.

        Parameters
        ----------
        candidate_sites : list of pocket residue lists for Leg 2.
            If None, auto-detect via fpocket on first fold.
        """
        cfg = self.cfg
        job_id = uuid.uuid4().hex[:12]
        t0 = time.time()

        # Auto-detect candidate sites if needed (deferred if DrugCLIP L0 will do it)
        if candidate_sites is None and "pocket_on" in cfg.run_modes:
            if cfg.run_l1 and cfg.l1_method == "drugclip":
                # DrugCLIP L0 will detect pockets; skip auto-detect for now
                pass
            else:
                candidate_sites = self._auto_detect_sites(
                    protein_sequence, seed, num_loops, num_sampling_steps
                )

        # ── L1 pre-screen: reduce library before expensive L2 co-folding ──
        l1_survivors = smiles_list
        l1_result = None
        if cfg.run_l1 and len(smiles_list) > 3:
            from app.l1_prescreen import L1DockingPreScreen

            l1 = L1DockingPreScreen(
                gnina_bin=cfg.gnina_bin,
                cnn_model=cfg.l1_cnn_model,
                exhaustiveness=cfg.l1_exhaustiveness,
                use_gpu=cfg.gnina_use_gpu,
                seed=cfg.gnina_seed,
            )
            l1_result = l1.screen_library(
                protein_sequence,
                smiles_list,
                workdir=cfg.workdir / "l1",
                method=cfg.l1_method,
                top_fraction=cfg.l1_top_fraction,
                reference_ligand_path=cfg.l1_reference_ligand,
                rf_model_path=cfg.l1_rf_model_path,
                # gnina-specific (only used when method="gnina")
                n_conformers=cfg.l1_n_conformers,
                num_loops=num_loops,
                num_sampling_steps=num_sampling_steps,
                # drugclip-specific (only used when method="drugclip")
                drugclip_ensemble=cfg.l1_drugclip_ensemble,
                receptor_pdb=cfg.l1_drugclip_receptor_pdb,
                pocket_pdb=cfg.l1_drugclip_pocket_pdb,
                top_n_pockets=cfg.l1_drugclip_top_n_pockets,
                min_quality=cfg.l1_drugclip_min_quality,
            )
            l1_survivors = l1_result.survivors
            logger.info(
                "L1 (%s): %d -> %d compounds (%.1f%%)",
                cfg.l1_method,
                l1_result.n_input,
                l1_result.n_survivors,
                100 * l1_result.n_survivors / max(1, l1_result.n_input),
            )

            # ── DrugCLIP/GenPack pocket → ESMFold2 PocketConditioning ──
            if cfg.l1_method == "drugclip" and l1_result.pocket_sites:
                candidate_sites = l1_result.pocket_sites
                n_res = sum(len(s) for s in candidate_sites)
                logger.info(
                    "DrugCLIP pocket: %d site(s), %d total residues → L2 PocketConditioning",
                    len(candidate_sites),
                    n_res,
                )

            # ── Release DrugCLIP GPU memory before ESMFold2 loads ──
            if cfg.l1_method == "drugclip":
                logger.info("Releasing DrugCLIP GPU memory...")
                import torch
                import gc

                # Clear DrugCLIP singleton
                from app.drugclip_scorer import DrugCLIPScorer

                DrugCLIPScorer._instance = None
                gc.collect()
                torch.cuda.empty_cache()

        # Auto-detect candidate sites if DrugCLIP didn't find any
        if candidate_sites is None and "pocket_on" in cfg.run_modes:
            candidate_sites = self._auto_detect_sites(
                protein_sequence, seed, num_loops, num_sampling_steps
            )

        merged_results: list[CompoundMergeResult] = []
        parked: list[CompoundMergeResult] = []

        # ── Process each compound ───────────────────────────────────────
        for i, smi in enumerate(l1_survivors):
            cid = _short_id(smi)
            logger.info("══ [%d/%d] %s ══", i + 1, len(smiles_list), cid)

            # Leg 1 — always run (gate signal source)
            leg1 = self.run_leg1(
                protein_sequence,
                smi,
                seed=seed + i,
                num_loops=num_loops,
                num_sampling_steps=num_sampling_steps,
            )

            # Leg 2 — pocket-on if configured and sites available
            leg2_results: list[LegResult] = []
            if "pocket_on" in cfg.run_modes and candidate_sites:
                leg2_results = self.run_leg2(
                    protein_sequence,
                    smi,
                    candidate_sites,
                    seed=seed + i,
                    num_loops=num_loops,
                    num_sampling_steps=num_sampling_steps,
                )

            # Leg 3 — orthogonal rescue only when needed
            leg3 = None
            need_rescue = (
                cfg.orthogonal_rescue
                and leg1.status in ("failed", "invalid_pose")
                and not any(lr.status == "ok" for lr in leg2_results)
            )
            if need_rescue:
                rescue_sites = self._collect_sites_from_legs(leg1, leg2_results)
                leg3 = self.run_leg3(
                    protein_sequence,
                    smi,
                    rescue_sites,
                    seed=seed + i,
                    num_loops=num_loops,
                    num_sampling_steps=num_sampling_steps,
                )

            # Merge immediately (scoring already done in each leg)
            merged = self.merge(smi, leg1, leg2_results, leg3)
            if merged.status == "parked":
                parked.append(merged)
            else:
                merged_results.append(merged)

            # ── Periodic GPU memory cleanup ──
            if (i + 1) % 5 == 0:
                import gc, torch

                gc.collect()
                torch.cuda.empty_cache()

        # ── Release GPU: unload ESMFold2 so OpenMM can use CUDA ────────
        from app.local_inference import get_engine

        eng = get_engine()
        if hasattr(eng, "_model") and eng._model is not None:
            eng._model = None
            eng._builder = None
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # ── MM-GBSA: Two-phase batch (Phase1 CPU-parallel, Phase2 GPU-serial) ─
        if cfg.mmgbsa_selection == "all_gate_survivors" and merged_results:
            _mmgbsa_batch(merged_results, cfg)

        # ── Sort by consensus score (higher = better) ───────────────────
        from app.cascade_config import compute_consensus_score

        for mr in merged_results:
            pae = plddt = iptm = cnn = dg = None
            leg1_list = mr.legs.get("pocket_off", [])
            if leg1_list and leg1_list[0].gate:
                g = leg1_list[0].gate
                pae, plddt, iptm = g.interface_pae, g.ligand_plddt, g.ligand_iptm
            if mr.best_gnina is not None:
                cnn = mr.best_gnina.cnn_affinity
            elif mr.best_boltz is not None:
                cnn = mr.best_boltz.affinity_binary
            if mr.best_dg_gb is not None:
                dg = mr.best_dg_gb
            mr._consensus = compute_consensus_score(
                pae,
                plddt,
                iptm,
                cnn,
                dg,
                gate_penalty=cfg.gate_penalty_enabled,
                equal_weights=cfg.consensus_equal_weights,
            )
        merged_results.sort(key=lambda mr: -getattr(mr, "_consensus", 0))

        elapsed = time.time() - t0

        n_leg1_ok = sum(
            1
            for mr in merged_results
            if mr.legs.get("pocket_off", [None])[0]
            and mr.legs["pocket_off"][0].status == "ok"
        )
        n_leg2_ok = sum(
            1
            for mr in merged_results
            if any(lr.status == "ok" for lr in mr.legs.get("pocket_on", []))
        )
        n_leg3 = sum(1 for mr in merged_results if "orthogonal" in mr.legs)
        n_rescue = sum(1 for mr in merged_results if mr.confidence_tier == "rescue")

        return CascadeResponse(
            job_id=job_id,
            status="completed",
            results=merged_results,
            parked=parked,
            funnel_summary={
                "n_input": len(smiles_list),
                "n_l1_survivors": len(l1_survivors),
                "n_kept": len(merged_results),
                "n_parked": len(parked),
                "n_leg1_ok": n_leg1_ok,
                "n_leg2_ok": n_leg2_ok,
                "n_leg3_rescued": n_leg3,
                "n_rescue_tier": n_rescue,
                "n_convergent": sum(
                    1 for mr in merged_results if mr.confidence_tier == "convergent"
                ),
                "n_divergent": sum(
                    1 for mr in merged_results if mr.confidence_tier == "divergent"
                ),
                "n_other": sum(
                    1
                    for mr in merged_results
                    if mr.confidence_tier not in ("convergent", "divergent")
                ),
            },
            wall_time_s=round(elapsed, 1),
        )

    def run_single(
        self,
        protein_sequence: str,
        smiles: str,
        seed: int = 42,
        num_loops: int = 8,
        num_sampling_steps: int = 32,
        candidate_sites: list[list[int]] | None = None,
    ) -> CascadeResponse:
        """Convenience: run full cascade for one compound."""
        return self.run(
            protein_sequence,
            [smiles],
            seed=seed,
            num_loops=num_loops,
            num_sampling_steps=num_sampling_steps,
            candidate_sites=candidate_sites,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Internal helpers
    # ═══════════════════════════════════════════════════════════════════════

    def _complex_prep(self, fold_result, wd: Path):
        """Split co-fold complex into receptor & ligand files. No scoring, no protonation."""
        from app.cascade_prep import split_complex

        cfg = self.cfg
        rec_raw, lig_sdf = split_complex(
            fold_result.complex_path,
            wd,
            smiles=fold_result.smiles,
            protonation_tool=cfg.protonation_tool,
            ph=cfg.protonation_pH,
        )
        return rec_raw, lig_sdf

    def _gnina_score(self, leg: LegResult, fold_result, wd: Path) -> LegResult:
        """Run complex prep + Gnina scoring on a leg result."""
        from app.cascade_prep import protonate_receptor
        from app.gnina_scorer import gnina_rescore

        cfg = self.cfg
        rec_raw, lig_sdf = self._complex_prep(fold_result, wd)
        rec_h = protonate_receptor(rec_raw, wd, ph=cfg.protonation_pH)
        gnina = gnina_rescore(
            fold_result.complex_path,
            wd,
            smiles=fold_result.smiles,
            protonation_tool=cfg.protonation_tool,
            ph=cfg.protonation_pH,
        )
        rec_h = protonate_receptor(rec_raw, wd, ph=cfg.protonation_pH)
        gnina = gnina_rescore(
            rec_h,
            lig_sdf,
            gnina_bin=cfg.gnina_bin,
            cnn_model=cfg.gnina_cnn_model,
            cnn_scoring=cfg.gnina_cnn_scoring,
            use_gpu=cfg.gnina_use_gpu,
            local_minimize=cfg.gnina_local_minimize,
            seed=cfg.gnina_seed,
            workdir=wd,
            timeout_s=cfg.subprocess_timeout_s,
        )
        leg.gnina = GninaScoreDetail(
            cnn_affinity=gnina.cnn_affinity,
            cnn_score=gnina.cnn_score,
            vina_affinity=gnina.vina_affinity,
            raw_output=gnina.raw,
        )
        leg._rec_raw = str(rec_raw)
        leg._lig_sdf = str(lig_sdf)
        return leg

    def _boltz_score(
        self, leg: LegResult, fold_result, protein_sequence: str, smiles: str, wd: Path
    ) -> LegResult:
        """Run Boltz-2 affinity scoring on a co-folded complex."""
        from app.boltz_scorer import score_single
        from app.models import BoltzScoreDetail

        cid = _short_id(smiles)
        logger.info("[%s] Boltz-2 affinity scoring on %s ...", cid, leg.leg)

        try:
            mmcif_text = Path(fold_result.complex_path).read_text()
            result = score_single(
                protein_sequence=protein_sequence,
                smiles=smiles,
                mmcif_text=mmcif_text,
                recycling_steps=self.cfg.boltz_recycling_steps,
                use_msa_server=self.cfg.boltz_use_msa,
                timeout=int(__import__('os').environ.get('NESSO_SCORE_TIMEOUT', '120')),
            )
            leg.boltz = BoltzScoreDetail(
                affinity_binary=result["affinity_binary"],
                affinity_value=result["affinity_value"],
                error=result.get("error"),
            )
            logger.info(
                "[%s] Boltz-2 affinity_binary=%.4f, affinity_value=%.4f",
                cid,
                leg.boltz.affinity_binary,
                leg.boltz.affinity_value,
            )
        except Exception as e:
            logger.exception("[%s] Boltz-2 scoring failed for %s", cid, leg.leg)
            leg.boltz = BoltzScoreDetail(
                affinity_binary=0.0,
                affinity_value=0.0,
                error=f"{type(e).__name__}: {e}",
            )
        return leg

    def _run_mmgbsa_on_merge(self, mr: CompoundMergeResult):
        """Run MM-GBSA on the best-scoring leg result from merge."""
        from app.mmgbsa import mmgbsa_refine

        cfg = self.cfg
        # Find best-scoring leg result
        best_leg = None
        best_score = -float("inf")
        for leg_list in mr.legs.values():
            for lr in leg_list:
                if lr.gnina and lr.gnina.cnn_affinity > best_score:
                    best_score = lr.gnina.cnn_affinity
                    best_leg = lr
                elif lr.boltz and lr.boltz.affinity_binary > best_score:
                    best_score = lr.boltz.affinity_binary
                    best_leg = lr

        if best_leg is None or not hasattr(best_leg, "_rec_raw"):
            return

        try:
            rec_raw = Path(best_leg._rec_raw)
            lig_sdf = Path(best_leg._lig_sdf)
            if rec_raw.exists() and lig_sdf.exists():
                cid = mr.compound_id
                wd = self._workdir_for(cid, "mmgbsa")
                # Use short_md for divergent compounds
                mode = cfg.mmgbsa_mode
                if (
                    mr.confidence_tier == "divergent"
                    and cfg.divergence_triggers_short_md
                ):
                    mode = "short_md"

                mm = mmgbsa_refine(
                    rec_raw,
                    lig_sdf,
                    wd,
                    igb=cfg.igb,
                    saltcon=cfg.saltcon_M,
                    protein_ff=cfg.protein_ff,
                    ligand_ff=cfg.ligand_ff,
                    charge_method=cfg.ligand_charge_method,
                    nagl_model=cfg.nagl_model,
                    min_engine=cfg.min_engine,
                    openmm_platform=cfg.openmm_platform,
                    sander_min_steps=cfg.sander_min_steps,
                    mmgbsa_mode=mode,
                    timeout_s=cfg.subprocess_timeout_s,
                )
                mr.best_mmgbsa = MMGBSADetail(
                    dg_gb=mm.dg_gb,
                    dg_std=mm.dg_std,
                    raw_output=mm.raw,
                )
                mr.best_dg_gb = mm.dg_gb
        except Exception as e:
            logger.warning("[%s] MM-GBSA failed: %s", mr.compound_id, e)

    def _auto_detect_sites(
        self,
        protein_sequence: str,
        seed: int,
        num_loops: int,
        num_sampling_steps: int,
    ) -> list[list[int]]:
        """Auto-detect candidate pocket sites via fpocket + Leg 1 first-compound clustering."""
        logger.info("Auto-detecting pocket sites...")
        from app.pocket_detector import detect_pocket_residues

        # Fold apo once
        mmcif = self.cofolder.fold_apo_protein(
            protein_sequence,
            seed=seed,
            num_loops=num_loops,
            num_sampling_steps=num_sampling_steps,
        )
        pockets = detect_pocket_residues(
            mmcif,
            chain_id="A",
            top_n=5,
            filter_by_quality=True,
            min_quality=0.15,
            min_volume=100.0,
            min_residues=5,
        )
        sites = []
        for p in pockets:
            residues = p.get("residues", [])
            # Extract residue numbers from (chain, resnum) tuples
            site_res = [r[1] for r in residues[:25]]
            if len(site_res) >= 5:
                sites.append(site_res)

        logger.info("Auto-detected %d pocket sites (fpocket)", len(sites))
        return sites

    def _collect_sites_from_legs(
        self, leg1: LegResult, leg2_results: list[LegResult]
    ) -> list[list[int]]:
        """Collect pocket residue lists from all legs for Leg 3 rescue."""
        sites = []
        if leg1.gate and leg1.gate.pocket_residues:
            sites.append(leg1.gate.pocket_residues)
        for lr in leg2_results:
            if lr.gate and lr.gate.pocket_residues:
                sites.append(lr.gate.pocket_residues)
        return sites if sites else [[]]  # empty → blind rescue

    def _workdir_for(self, cid: str, stage: str) -> Path:
        wd = self.cfg.workdir / cid / stage
        wd.mkdir(parents=True, exist_ok=True)
        return wd

    # ── Validation export hooks ──────────────────────────────────────────

    def export_calibration_csv(
        self,
        response: CascadeResponse,
        labels: dict[str, int],
        output_path: str | Path,
    ) -> Path:
        """Export Leg-1 (pocket_off) gate signals as calibration CSV.

        HARD CONSTRAINT: ONLY reads from pocket_off leg.
        Leg-2 PAE is contaminated by PocketConditioning constraint.
        Leg-3 is orthogonal and has no ESMFold2 gate signals.

        Columns: compound_id,label,interface_pae,ligand_plddt,pocket_plddt,
                 ligand_iptm,consistency_rmsd,posebusters_pass
        """
        import pandas as pd

        rows: list[dict] = []
        for mr in response.results + response.parked:
            cid = mr.compound_id
            leg1_results = mr.legs.get("pocket_off", [])
            if not leg1_results:
                continue
            gate = leg1_results[0].gate
            if gate is None:
                continue
            rows.append(
                {
                    "compound_id": cid,
                    "label": labels.get(cid, 0),
                    "interface_pae": gate.interface_pae,
                    "ligand_plddt": gate.ligand_plddt,
                    "pocket_plddt": gate.pocket_plddt,
                    "ligand_iptm": gate.ligand_iptm,
                    "consistency_rmsd": gate.pose_rmsd_max,
                    "posebusters_pass": gate.posebusters_passed,
                }
            )

        df = pd.DataFrame(rows)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info(
            "Exported calibration CSV (%d rows, Leg-1 only) to %s", len(df), out
        )
        return out

    def export_pose_manifest_csv(
        self,
        response: CascadeResponse,
        output_path: str | Path,
        ref_sdfs: dict[str, str | Path] | None = None,
        ref_receptor_pdbs: dict[str, str | Path] | None = None,
        similarities: dict[str, float] | None = None,
        ood_flags: dict[str, int] | None = None,
        method: str = "cascade",
        pocket_radius: float = 10.0,
    ) -> tuple[Path, list[dict]]:
        """Export pose validation manifest CSV with coordinate alignment.

        pred_sdf is auto-discovered from the best-scoring leg's split_complex
        output and ALIGNED to the reference receptor frame.

        Uses pocket-local superposition (Ca near reference ligand) when
        ref_ligand_sdf is available — gives meaningful ligand pose RMSD.

        Per-compound fault tolerance: ValueError (insufficient pocket Ca,
        low sequence identity) skips the compound. Returns (csv_path, skipped).

        Columns: complex_id,pred_sdf,ref_sdf,receptor_pdb,method,pocket_similarity,ood
        """
        from .structure_align import align_ligand_to_reference

        if ref_sdfs is None:
            ref_sdfs = {}
        if ref_receptor_pdbs is None:
            ref_receptor_pdbs = {}
        if similarities is None:
            similarities = {}
        if ood_flags is None:
            ood_flags = {}

        rows: list[dict] = []
        skipped: list[dict] = []
        for mr in response.results + response.parked:
            cid = mr.compound_id

            # Find best-scoring leg that has split paths (_lig_sdf, _rec_raw)
            pred_ligand = None
            pred_receptor = None
            for leg_key in ("pocket_off", "pocket_on", "orthogonal"):
                for lr in mr.legs.get(leg_key, []):
                    if hasattr(lr, "_lig_sdf") and lr._lig_sdf:
                        pred_ligand = lr._lig_sdf
                        pred_receptor = getattr(lr, "_rec_raw_pdb", None) or getattr(
                            lr, "_rec_pdb", None
                        )
                        break
                if pred_ligand:
                    break

            if pred_ligand is None:
                rows.append(
                    {
                        "complex_id": cid,
                        "pred_sdf": "",
                        "ref_sdf": str(ref_sdfs.get(cid, "")),
                        "receptor_pdb": str(ref_receptor_pdbs.get(cid, "")),
                        "method": method,
                        "pocket_similarity": similarities.get(cid),
                        "ood": ood_flags.get(cid, 0),
                    }
                )
                continue

            # Coordinate alignment with pocket-local superposition
            ref_pdb = ref_receptor_pdbs.get(cid)
            ref_lig = ref_sdfs.get(cid) if ref_sdfs else None
            aligned_sdf = pred_ligand  # default: no alignment
            if ref_pdb and pred_receptor:
                try:
                    import tempfile

                    wd = self.cfg.workdir / cid / "align"
                    wd.mkdir(parents=True, exist_ok=True)
                    aligned_path = wd / f"{cid}_aligned.sdf"
                    aligned_sdf = align_ligand_to_reference(
                        pred_receptor,
                        pred_ligand,
                        ref_pdb,
                        aligned_path,
                        pocket_radius=pocket_radius,
                        ref_ligand_sdf=ref_lig,
                    )
                except ValueError as exc:
                    # Insufficient pocket Ca / low sequence identity
                    reason = str(exc)
                    logger.warning(
                        "Alignment skipped for %s: %s",
                        cid,
                        reason,
                    )
                    skipped.append({"compound_id": cid, "reason": reason})
                    continue
                except Exception as exc:
                    logger.warning(
                        "Alignment failed for %s: %s — using unaligned pose",
                        cid,
                        exc,
                    )

            rows.append(
                {
                    "complex_id": cid,
                    "pred_sdf": str(aligned_sdf),
                    "ref_sdf": str(ref_sdfs.get(cid, "")),
                    "receptor_pdb": str(ref_pdb or ""),
                    "method": method,
                    "pocket_similarity": similarities.get(cid),
                    "ood": ood_flags.get(cid, 0),
                }
            )

        df = pd.DataFrame(rows)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info(
            "Exported pose manifest CSV (%d rows, %d skipped) to %s",
            len(df),
            len(skipped),
            out,
        )
        return out, skipped

    def export_enrichment_csv(
        self,
        response: CascadeResponse,
        output_path: str | Path,
        labels: dict[str, int],
        target_name: str = "target",
        include_vina_baseline: bool = True,
        additional_scores: list[dict] | None = None,
    ) -> Path:
        """Export enrichment scores CSV with parked compounds included.

        CRITICAL (revision #2): parked compounds are INCLUDED with survived=False,
        so funnel_recall reflects compounds dropped by the gate.

        Cascade score is single-direction (revision #4): best_dg_gb (lower better).
        Compounds without dg_gb (parked / didn't reach MM-GBSA) use a sentinel
        value to sort last: WORST = max(all_dg_gb) + 10.0.

        Columns: target,method,compound_id,score,label,survived
        """
        import math

        import pandas as pd

        rows: list[dict] = []

        # Collect all dg_gb values to compute sentinel
        all_dg: list[float] = []
        for mr in response.results + response.parked:
            if mr.best_dg_gb is not None and not math.isnan(mr.best_dg_gb):
                all_dg.append(mr.best_dg_gb)
        sentinel = (max(all_dg) + 10.0) if all_dg else 1e9

        for mr in response.results + response.parked:
            cid = mr.compound_id
            label = labels.get(cid, 0)
            survived = mr.status == "kept"

            # Cascade score: single direction, lower is better (revision #4)
            if mr.best_dg_gb is not None and not math.isnan(mr.best_dg_gb):
                cascade_score = float(mr.best_dg_gb)
            elif mr.best_gnina is not None:
                # Parked / didn't reach MM-GBSA — use sentinel (NOT cnn_affinity)
                cascade_score = sentinel
            else:
                cascade_score = sentinel

            rows.append(
                {
                    "target": target_name,
                    "method": "cascade",
                    "compound_id": cid,
                    "score": cascade_score,
                    "label": label,
                    "survived": survived,
                }
            )

            # Vina baseline from gnina output
            if include_vina_baseline and mr.best_gnina is not None:
                rows.append(
                    {
                        "target": target_name,
                        "method": "vina",
                        "compound_id": cid,
                        "score": float(mr.best_gnina.vina_affinity)
                        if not (
                            math.isnan(mr.best_gnina.vina_affinity)
                            if isinstance(mr.best_gnina.vina_affinity, float)
                            else False
                        )
                        else sentinel,
                        "label": label,
                        "survived": None,
                    }
                )

            # Boltz-2 baseline when gnina is not available
            if mr.best_boltz is not None and mr.best_gnina is None:
                rows.append(
                    {
                        "target": target_name,
                        "method": "boltz2",
                        "compound_id": cid,
                        "score": float(mr.best_boltz.affinity_binary)
                        if not (
                            math.isnan(mr.best_boltz.affinity_binary)
                            if isinstance(mr.best_boltz.affinity_binary, float)
                            else False
                        )
                        else sentinel,
                        "label": label,
                        "survived": None,
                    }
                )

        # Additional baseline methods (boltz2, etc.)
        if additional_scores:
            for row in additional_scores:
                r = dict(row)
                r.setdefault("target", target_name)
                r.setdefault("survived", None)
                rows.append(r)

        df = pd.DataFrame(rows)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info(
            "Exported enrichment CSV (%d rows, methods: %s) to %s",
            len(df),
            sorted(df["method"].unique()),
            out,
        )
        return out

    def cleanup(self):
        if not self.cfg.keep_intermediates:
            import shutil

            wd = self.cfg.workdir
            if wd.exists():
                shutil.rmtree(wd, ignore_errors=True)


# ── Singleton ─────────────────────────────────────────────────────────────

_cascade: ScreeningCascade | None = None


def get_cascade(config: CascadeConfig | None = None) -> ScreeningCascade:
    global _cascade
    if _cascade is None:
        _cascade = ScreeningCascade(config=config)
    elif config is not None:
        _cascade.cfg = config
    return _cascade


def _short_id(smiles: str) -> str:
    bad = "()[]=@#/\\"
    s = smiles[:20]
    for c in bad:
        s = s.replace(c, "")
    s = s.strip()
    if not s:
        s = "lig"
    return s


def _is_nan(v) -> bool:
    import math

    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


# ── Two-phase MM-GBSA batch processing ────────────────────────────────────


def _mmgbsa_batch(merged_results: list, cfg) -> None:
    """Two-phase MM-GBSA: Phase 1 CPU-parallel, Phase 2 GPU-serial.

    Phase 1: antechamber + tleap for ALL compounds in parallel (ThreadPool, CPU)
    Phase 2: OpenMM minimize + MMPBSA serially (GPU)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Collect work items
    phase1_items = []
    mr_indices = []
    for idx, mr in enumerate(merged_results):
        best_leg = _find_best_leg(mr)
        if best_leg is None or not hasattr(best_leg, "_rec_raw"):
            continue
        rec_raw = getattr(best_leg, "_rec_raw", None)
        lig_sdf = getattr(best_leg, "_lig_sdf", None)
        if not rec_raw or not lig_sdf:
            continue
        wd = Path(cfg.workdir) / mr.compound_id / "mmgbsa"
        netq = _ligand_net_charge_from_leg(mr)
        phase1_items.append(
            (
                str(rec_raw),
                str(lig_sdf),
                str(wd),
                netq,
                cfg.ligand_charge_method,
                cfg.nagl_model,
                cfg.protein_ff,
                cfg.ligand_ff,
            )
        )
        mr_indices.append(idx)

    if not phase1_items:
        return

    n_cpu = min(cfg.mmgbsa_cpu_workers, len(phase1_items))
    logger.info(
        "MM-GBSA Phase 1: %d compounds on %d CPU threads...", len(phase1_items), n_cpu
    )

    from app.mmgbsa import mmgbsa_prepare_one

    t0 = time.time()
    phase1_results = {}
    with ThreadPoolExecutor(max_workers=n_cpu) as pool:
        futures = {
            pool.submit(mmgbsa_prepare_one, item): i
            for i, item in enumerate(phase1_items)
        }
        for f in as_completed(futures):
            i = futures[f]
            try:
                wd_str, _, _, _, ok, err = f.result()
                phase1_results[i] = (ok, err)
            except Exception as e:
                phase1_results[i] = (False, str(e))

    ok_count = sum(1 for ok, _ in phase1_results.values() if ok)
    logger.info(
        "MM-GBSA Phase 1 done: %d/%d OK in %.0fs",
        ok_count,
        len(phase1_items),
        time.time() - t0,
    )

    # Phase 2: GPU parallel (ProcessPool, spawn context)
    phase2_items = []
    for i, mr_idx in enumerate(mr_indices):
        if i in phase1_results and phase1_results[i][0]:
            wd = Path(cfg.workdir) / merged_results[mr_idx].compound_id / "mmgbsa"
            mode = cfg.mmgbsa_mode
            if (
                merged_results[mr_idx].confidence_tier == "divergent"
                and cfg.divergence_triggers_short_md
            ):
                mode = "short_md"
            phase2_items.append(
                (
                    mr_idx,
                    (
                        str(wd),
                        cfg.igb,
                        cfg.saltcon_M,
                        cfg.min_engine,
                        cfg.openmm_platform,
                        cfg.sander_min_steps,
                        mode,
                        cfg.subprocess_timeout_s,
                    ),
                )
            )

    n_gpu = cfg.mmgbsa_workers
    logger.info(
        "MM-GBSA Phase 2: %d compounds on %d GPU workers...", len(phase2_items), n_gpu
    )

    from app.mmgbsa import mmgbsa_finalize_one

    if n_gpu > 1 and len(phase2_items) > 1:
        import multiprocessing as _mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        _ctx = _mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_gpu, mp_context=_ctx) as pool:
            futures = {
                pool.submit(mmgbsa_finalize_one, item): (mr_idx, item)
                for mr_idx, item in phase2_items
            }
            for j, f in enumerate(as_completed(futures)):
                mr_idx, _ = futures[f]
                try:
                    result = f.result()
                    if result is not None:
                        merged_results[mr_idx].best_mmgbsa = MMGBSADetail(
                            dg_gb=result.dg_gb,
                            dg_std=result.dg_std,
                            raw_output="",
                        )
                        merged_results[mr_idx].best_dg_gb = result.dg_gb
                except Exception as e:
                    logger.warning(
                        "[%s] Phase2 worker failed: %s",
                        merged_results[mr_idx].compound_id,
                        e,
                    )
                if (j + 1) % 10 == 0:
                    logger.info("MM-GBSA Phase 2: %d/%d", j + 1, len(phase2_items))
    else:
        t0 = time.time()
        for idx, (mr_idx, item) in enumerate(phase2_items):
            result = mmgbsa_finalize_one(item)
            if result is not None:
                merged_results[mr_idx].best_mmgbsa = MMGBSADetail(
                    dg_gb=result.dg_gb,
                    dg_std=result.dg_std,
                    raw_output="",
                )
                merged_results[mr_idx].best_dg_gb = result.dg_gb
            if (idx + 1) % 5 == 0:
                logger.info("MM-GBSA Phase 2: %d/%d", idx + 1, len(phase2_items))


def _ligand_net_charge_from_leg(mr) -> int:
    """Extract net charge from merged result's ligand."""
    from rdkit import Chem

    for leg_list in mr.legs.values():
        for lr in leg_list:
            if hasattr(lr, "_lig_sdf") and lr._lig_sdf:
                try:
                    mol = next(Chem.SDMolSupplier(str(lr._lig_sdf), removeHs=False))
                    if mol:
                        return Chem.GetFormalCharge(mol)
                except Exception:
                    pass
    return 0


# ── Original parallel MM-GBSA worker (module-level for pickling) ──────────


def _find_best_leg(mr) -> object | None:
    """Find the best-scored leg result from a CompoundMergeResult.
    Prefers gnina score, falls back to boltz affinity_binary if gnina is unavailable.
    """
    best_leg = None
    best_score = -float("inf")
    for leg_list in mr.legs.values():
        for lr in leg_list:
            if lr.gnina and lr.gnina.cnn_affinity > best_score:
                best_score = lr.gnina.cnn_affinity
                best_leg = lr
            elif lr.boltz and lr.boltz.affinity_binary > best_score:
                best_score = lr.boltz.affinity_binary
                best_leg = lr
    return best_leg


def _mmgbsa_worker(args: tuple) -> dict:
    """Standalone MM-GBSA worker for ProcessPoolExecutor.

    Args: (rec_raw, lig_sdf, workdir, igb, saltcon, protein_ff, ligand_ff,
           charge_method, nagl_model, min_engine, openmm_platform,
           sander_min_steps, mmgbsa_mode, timeout_s)
    Returns: {"dg_gb": float | None, "error": str | None}
    """
    import logging

    logger = logging.getLogger("cascade.mmgbsa_worker")

    (
        rec_raw,
        lig_sdf,
        workdir,
        igb,
        saltcon,
        protein_ff,
        ligand_ff,
        charge_method,
        nagl_model,
        min_engine,
        openmm_platform,
        sander_min_steps,
        mmgbsa_mode,
        timeout_s,
    ) = args

    try:
        from app.mmgbsa import mmgbsa_refine

        mm = mmgbsa_refine(
            Path(rec_raw),
            Path(lig_sdf),
            Path(workdir),
            igb=igb,
            saltcon=saltcon,
            protein_ff=protein_ff,
            ligand_ff=ligand_ff,
            charge_method=charge_method,
            nagl_model=nagl_model,
            min_engine=min_engine,
            openmm_platform=openmm_platform,
            sander_min_steps=sander_min_steps,
            mmgbsa_mode=mmgbsa_mode,
            timeout_s=timeout_s,
        )
        return {"dg_gb": mm.dg_gb, "dg_std": mm.dg_std, "error": None}
    except Exception as e:
        logger.debug("MM-GBSA worker failed: %s", e)
        return {"dg_gb": None, "dg_std": None, "error": str(e)[:200]}
