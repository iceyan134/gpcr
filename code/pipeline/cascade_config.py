"""
CascadeConfig — all tunable thresholds for the recall-first screening cascade.

References
- docs/screening_cascade.py          (reference implementation)
- docs/PIPELINE_vFinal.md           (architecture specification)
- docs/REBUILD_PLAN_v2.md           (v2 rebuild, enrichment-first)
- docs/REBUILD_PLAN_addendum_A_*.md (recall-first: three-leg + triage)
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CascadeConfig:
    # ═══════════════════════════════════════════════════════════════════════
    # Top-level mode
    # ═══════════════════════════════════════════════════════════════════════
    # "triage" = recall-first: route, never kill; "filter" = enrichment-first
    gate_mode: str = "triage"

    # Which legs to run
    run_modes: list[str] = field(default_factory=lambda: ["pocket_off", "pocket_on"])

    # Never drop a compound — only park with full audit trail
    never_drop: bool = True

    # ═══════════════════════════════════════════════════════════════════════
    # L1 — Pre-screen (before L2 co-folding)
    #   Methods: "maccs" (2D similarity, zero-cost),
    #            "rf" (RandomForest on ECFP4, needs pre-trained model),
    #            "gnina" (Gnina docking, GPU-or-CPU),
    #            "drugclip" (DrugCLIP contrastive learning, GPU)
    # ═══════════════════════════════════════════════════════════════════════
    run_l1: bool = True                      # enable L1 pre-screening
    l1_method: str = "rf"                    # "maccs" | "rf" | "gnina" | "drugclip"
    l1_top_fraction: float = 0.20            # keep top 20% of library
    l1_reference_ligand: str = ""            # path to reference ligand MOL2/SDF for MACCS/RF
    l1_rf_model_path: str = ""               # path to pre-trained RF pickle (required for rf method)
    # drugclip-specific (only used when l1_method="drugclip")
    l1_drugclip_ensemble: str = "6_folds"    # "6_folds" | "8_folds" | "single"
    l1_drugclip_receptor_pdb: str = ""       # path to receptor PDB for DrugCLIP pocket detection
    l1_drugclip_pocket_pdb: str = ""         # path to pre-defined pocket PDB (optional, overrides fpocket)
    l1_drugclip_top_n_pockets: int = 3       # number of fpocket pockets to use
    l1_drugclip_min_quality: float = 0.20    # min pocket quality score
    # gnina-specific (only used when l1_method="gnina")
    l1_n_conformers: int = 3                 # apo ensemble size
    l1_exhaustiveness: int = 8               # gnina docking exhaustiveness
    l1_cnn_model: str = "fast"               # ~3× faster than crossdock, same ranking

    # ═══════════════════════════════════════════════════════════════════════
    # Leg 1 — Unconstrained co-fold (pocket-off)
    #   Purpose: unbiased site discovery + honest (uncontaminated) PAE signal
    # ═══════════════════════════════════════════════════════════════════════
    # Number of independent co-fold samples per compound
    consistency_samples: int = 8    # recall-first: 8–10
    max_pose_rmsd_A: float = 2.0

    # ═══════════════════════════════════════════════════════════════════════
    # Leg 2 — PocketConditioning co-fold (pocket-on)
    #   Purpose: guide ligand to candidate sites via ESMFold2 native constraint
    # ═══════════════════════════════════════════════════════════════════════
    # Candidate site source: "auto" = fpocket + Leg1 clustering union
    pocket_on_sites: str = "auto"
    # Or explicit: pocket_on_sites_manual: list[list[int]] | None = None

    # ═══════════════════════════════════════════════════════════════════════
    # Leg 3 — Orthogonal rescue (non-ESMFold2)
    #   Purpose: catch systemic co-fold failures with independent pose source
    # ═══════════════════════════════════════════════════════════════════════
    orthogonal_rescue: bool = True
    orthogonal_rescue_engine: str = "gnina"   # "gnina" | "boltz"
    # Gnina docking box size for rescue docking
    rescue_box_size: tuple = (25, 25, 25)
    rescue_exhaustiveness: int = 16

    # ═══════════════════════════════════════════════════════════════════════
    # Confidence gate thresholds (6 signals, used for TRIAGE not filtering)
    # HARD CONSTRAINT: gate signal ONLY from Leg 1 (unconstrained).
    # Leg 2 PAE is contaminated by the PocketConditioning constraint.
    # ═══════════════════════════════════════════════════════════════════════
    gate_signal_from: str = "pocket_off"     # HARD: only Leg 1

    pocket_contact_cutoff_A: float = 5.0
    # Calibrated on 10 RNP systems (6 actives, 4 decoys, recall=100%).
    # ⚠ Small sample — treat as suggested ranges, not precise cutoffs.
    #   95% bootstrap CI on each threshold in calibrate_gate.py output.
    min_pocket_plddt: float = 97.0           # best=96.9,  CI [96.9, 98.0]
    min_ligand_plddt: float = 87.0           # best=86.95, CI [86.95, 92.4]
    max_interface_pae: float = 2.9           # best=2.90,  CI [2.00, 2.90]
    min_ligand_iptm: float = 0.91            # best=0.908, CI [0.908, 0.955]

    # PoseBusters
    run_posebusters: bool = True
    posebusters_config: str = "dock"

    # ═══════════════════════════════════════════════════════════════════════
    # Triage routing rules
    # ═══════════════════════════════════════════════════════════════════════

    # Weak PAE / low iPTM → downgrade confidence, upgrade sampling
    weak_pae_upgrade_samples: int = 4     # extra co-fold samples on weak signal

    # Failed PoseBusters → mark pose invalid, resample (don't drop compound)
    pb_max_retries: int = 3

    # Cross-leg divergence → upgrade to short_md in MM-GBSA
    # Disabled: _short_md_openmm() is a stub (NotImplementedError).
    # All compounds use minimize_only for now.
    divergence_triggers_short_md: bool = False

    # Only park when ALL legs + ALL rescue attempts exhausted
    # parked → separate bucket, full audit trail, never deleted

    # ═══════════════════════════════════════════════════════════════════════
    # Gnina CNN rescoring (shared by Leg 1/2)
    # ═══════════════════════════════════════════════════════════════════════
    gnina_bin: str = "gnina"
    gnina_cnn_model: str = "crossdock_default2018"
    gnina_cnn_scoring: str = "rescore"
    gnina_use_gpu: bool = True
    gnina_seed: int = 42
    gnina_local_minimize: bool = False

    # ═══════════════════════════════════════════════════════════════════════
    # Boltz-2 affinity scoring (replaces Gnina when enabled)
    # ═══════════════════════════════════════════════════════════════════════
    use_boltz: bool = False
    boltz_recycling_steps: int = 1       # 1 = fastest (Boltzina Cycle=1)
    boltz_use_msa: bool = True           # requires --use_msa_server

    # ═══════════════════════════════════════════════════════════════════════
    # Protonation / charge
    # ═══════════════════════════════════════════════════════════════════════
    protonation_tool: str = "dimorphite"
    protonation_pH: float = 7.4
    ligand_charge_method: str = "xtb"  # GFN2-xTB, fast + accurate; fallback: sqm, nagl
    nagl_model: str = "openff-gnn-am1bcc-1.0.0.pt"

    # ═══════════════════════════════════════════════════════════════════════
    # MM-GBSA — ALL gate survivors (recall-first); no top-k cutoff
    # ═══════════════════════════════════════════════════════════════════════
    mmgbsa_selection: str = "all_gate_survivors"  # NOT "top_k"
    igb: int = 5
    saltcon_M: float = 0.150
    protein_ff: str = "leaprc.protein.ff19SB"
    ligand_ff: str = "leaprc.gaff2"
    min_engine: str = "openmm"  # conda-forge openmm includes CUDA
    openmm_platform: str = "CUDA"  # conda-forge openmm has CUDA
    sander_min_steps: int = 2000
    mmgbsa_workers: int = 3                 # Phase 2 GPU-parallel workers (ProcessPool spawn)
    mmgbsa_cpu_workers: int = 16            # Phase 1 CPU-parallel antechamber threads

    # Ablation flags
    gate_penalty_enabled: bool = True       # False → remove gate penalty from consensus
    consensus_equal_weights: bool = False   # True → use 0.25 each instead of optimized
    mmgbsa_mode: str = "minimize_only"
    md_ns: float = 1.0
    md_timestep_fs: float = 4.0
    md_frames: int = 50

    # ═══════════════════════════════════════════════════════════════════════
    # General
    # ═══════════════════════════════════════════════════════════════════════
    workdir: Path = field(default_factory=lambda: Path(tempfile.mkdtemp(prefix="cascade_")))
    keep_intermediates: bool = False
    subprocess_timeout_s: int = 60 * 60

    def __post_init__(self):
        self.workdir = Path(self.workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)

    def apply_calibration(
        self, thresholds: dict, *, explicit_confirm: bool = False,
    ) -> None:
        """Apply gate thresholds from calibrate_gate.py output.

        HARD CONSTRAINTS:
        - gate_mode NEVER changes (stays "triage")
        - never_drop NEVER changes
        - Parked bucket is never touched
        """
        import logging

        _log = logging.getLogger("cascade_config")

        if not explicit_confirm:
            raise RuntimeError(
                "apply_calibration() requires explicit_confirm=True. "
                "This prevents silent overwrite of gate thresholds."
            )

        attr_map = {
            "interface_pae": "max_interface_pae",
            "ligand_plddt": "min_ligand_plddt",
            "pocket_plddt": "min_pocket_plddt",
            "ligand_iptm": "min_ligand_iptm",
            "consistency_rmsd": "max_pose_rmsd_A",
        }

        for sig, val in thresholds.items():
            if sig in ("gate_mode", "never_drop"):
                _log.warning("Refusing to modify %s (stays triage / never_drop)", sig)
                continue
            if sig == "posebusters_pass":
                _log.info("Ignoring posebusters_pass: triage signal, not a threshold")
                continue

            attr = attr_map.get(sig)
            if attr is None:
                _log.warning("Unknown calibration signal '%s', skipped", sig)
                continue

            old = getattr(self, attr)
            setattr(self, attr, val)
            _log.info("calibration: %s = %s -> %s", attr, old, val)


# ── Multi-signal consensus score ───────────────────────────────────────
# Normalizes gate + gnina + MM-GBSA signals into a single rankable score.
# Higher = better compound. Weights from 10-system RNP data analysis.

# P2 revision: dG_GB downgraded from ranker to sanity filter.
# Gate signals lead the ranking; dG_GB provides a binary bonus/penalty
# (plausibly bindable if dG < -20 kcal/mol) rather than continuous scoring.
# This avoids false positives like 8aop (RMSD 4.89, dG_GB=-38.78).
_CONSENSUS_WEIGHTS = {
    "interface_pae": 0.30,
    "ligand_plddt": 0.30,
    "ligand_iptm": 0.25,
    "cnn_affinity": 0.15,
    # dg_gb removed from continuous scoring — used as sanity bonus below
}


def compute_consensus_score(
    interface_pae: float | None,
    ligand_plddt: float | None,
    ligand_iptm: float | None,
    cnn_affinity: float | None,
    dg_gb: float | None,
    *,
    max_iPAE: float = 10.0,
    max_CNNaff: float = 5.0,
    gate_penalty: bool = True,
    equal_weights: bool = False,
) -> float:
    """Compute a weighted consensus score in [0, 1].

    Signals are normalized so 1.0 = best, then weighted per _CONSENSUS_WEIGHTS.

    - interface_pae: lower better, 0→max_iPAE mapped to 1→0
    - ligand_plddt: higher better, scaled by /100
    - ligand_iptm: higher better, already in [0,1]
    - cnn_affinity: higher better, /max_CNNaff
    - dg_gb: BINARY sanity check — +0.10 bonus if dG < -20 (plausibly bindable),
              -0.15 penalty if dG > 0 (likely non-binder). NaN → neutral.
    """
    score = 0.0
    total_w = 0.0

    if interface_pae is not None:
        w = 0.25 if equal_weights else _CONSENSUS_WEIGHTS["interface_pae"]
        s = max(0.0, 1.0 - interface_pae / max_iPAE)
        score += w * s
        total_w += w

    if ligand_plddt is not None:
        w = 0.25 if equal_weights else _CONSENSUS_WEIGHTS["ligand_plddt"]
        s = min(1.0, ligand_plddt / 100.0)
        score += w * s
        total_w += w

    if ligand_iptm is not None:
        w = 0.25 if equal_weights else _CONSENSUS_WEIGHTS["ligand_iptm"]
        s = max(0.0, min(1.0, ligand_iptm))
        score += w * s
        total_w += w

    if cnn_affinity is not None:
        w = 0.25 if equal_weights else _CONSENSUS_WEIGHTS["cnn_affinity"]
        s = min(1.0, cnn_affinity / max_CNNaff)
        score += w * s
        total_w += w

    if total_w == 0:
        return 0.0

    base = score / total_w

    # dG_GB as binary sanity check (not in continuous scoring)
    import math
    if dg_gb is not None and not math.isnan(dg_gb):
        if dg_gb < -20.0:
            base = min(1.0, base + 0.10)   # plausible binder bonus
        elif dg_gb > 0.0:
            base = max(0.0, base - 0.15)   # likely non-binder penalty
    # NaN dG_GB → neutral, no adjustment

    # Gate penalty: if any signal is below calibrated threshold, reduce
    # the final score. This catches cases like 8aop (good dG_GB but bad gate).
    if gate_penalty:
        penalty = 1.0
        if ligand_plddt is not None and ligand_plddt < 87.0:
            penalty *= max(0.5, ligand_plddt / 87.0)
        if ligand_iptm is not None and ligand_iptm < 0.91:
            penalty *= max(0.5, ligand_iptm / 0.91)
        if interface_pae is not None and interface_pae > 3.0:
            penalty *= max(0.5, 3.0 / interface_pae)
        return base * penalty
    return base
