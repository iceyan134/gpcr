"""
ESMFold Cascade Benchmark Engine
==================================
Design: checkpoint/resume + ...

...
- BenchmarkEngine: ...
- Experiment: ... + ...）
- CheckpointManager: Checkpoint/resume: writes checkpoint after each molecule
- ResultCollector: ... AUROC/AUPRC/EF

...
  output/benchmark/
    └── <experiment_name>/
        ├── config.json           # ...
        ├── checkpoint.json       # completion state (checkpoint/resume)
        ├── results.json          # ...
        └── logs/
            ├── <target>.log      # ...
            └── summary.log       # ...

...
  python -m app.benchmark_engine --experiment dude_cascade_full
  python -m app.benchmark_engine --experiment dude_cascade_full --resume
  python -m app.benchmark_engine --experiment dude_drugclip_only
  python -m app.benchmark_engine --list
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import gc
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

# ── Config ────────────────────────────────────────────────────────────────

BENCHMARK_ROOT = Path("/workspace/output/benchmark")
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

# Experiment definitions
EXPERIMENTS = {
    "dude_drugclip_l0": {
        "description": "DrugCLIP L0 on all 99 DUD-E targets (full library)",
        "targets": "all",  # "all" or list of target names
        "variant": "drugclip_l0",  # scoring variant
        "n_sample": 0,  # 0 = all molecules
        "n_actives": 0,  # 0 = all
        "n_decoys": 0,  # 0 = all
        "ensemble": "6_folds",
        "run_order": 0,  # run first
    },
    "dude_cascade_full": {
        "description": "Full cascade (L0+L1+L2) on 60 DUD-E targets, 80 mol/target",
        "targets": "selected_60",
        "variant": "cascade_full",
        "n_sample": 80,
        "n_actives": 40,
        "n_decoys": 40,
        "ensemble": "6_folds",
        "run_order": 1,
    },
    "dude_gnina_baseline": {
        "description": "Gnina-only baseline on 60 targets, 80 mol/target",
        "targets": "selected_60",
        "variant": "gnina_baseline",
        "n_sample": 80,
        "n_actives": 40,
        "n_decoys": 40,
        "ensemble": "none",
        "run_order": 2,
    },
    "dude_ablation_no_l0": {
        "description": "Ablation: skip L0 DrugCLIP, L1→L2 on 15 targets, 40 mol/target",
        "targets": "selected_15",
        "variant": "ablation_no_l0",
        "n_sample": 40,
        "n_actives": 20,
        "n_decoys": 20,
        "ensemble": "6_folds",
        "run_order": 3,
    },
    "dude_boltz_quicktest": {
        "description": "Quick test: Boltz-2 on 3 targets, 10 mol/target",
        "targets": ["aa2ar", "abl1", "ace"],
        "variant": "cascade_full",
        "n_sample": 10,
        "n_actives": 5,
        "n_decoys": 5,
        "ensemble": "6_folds",
        "run_order": 99,
    },
    "dude_ablation_no_l2": {
        "description": "Ablation: skip L2 ESMFold2, L0→L1→L4 on 15 targets, 40 mol/target",
        "targets": "selected_15",
        "variant": "ablation_no_l2",
        "n_sample": 40,
        "n_actives": 20,
        "n_decoys": 20,
        "ensemble": "6_folds",
        "run_order": 4,
    },
    "dude_ablation_no_l4": {
        "description": "Ablation: skip L4 MM-GBSA, L0→L1→L2 on 15 targets, 40 mol/target",
        "targets": "selected_15",
        "variant": "ablation_no_l4",
        "n_sample": 40,
        "n_actives": 20,
        "n_decoys": 20,
        "ensemble": "6_folds",
        "run_order": 5,
    },
    "dude_mmgbsa_global": {
        "description": "MM-GBSA scoring on 60 targets, 80 mol/target (from cascade_full results)",
        "targets": "selected_60",
        "variant": "mmgbsa_only",
        "n_sample": 80,
        "n_actives": 40,
        "n_decoys": 40,
        "ensemble": "none",
        "run_order": 6,
    },
    "dude_speed_benchmark": {
        "description": "Speed benchmark: 5 targets × 1000 mol, measure throughput per stage",
        "targets": "speed_5",
        "variant": "speed_benchmark",
        "n_sample": 1000,
        "n_actives": 500,
        "n_decoys": 500,
        "ensemble": "6_folds",
        "run_order": 7,
    },
}

# Selected 60 targets (covering all protein families)
SELECTED_60 = [
    "aa2ar",
    "abl1",
    "ace",
    "aces",
    "ada",
    "ada17",
    "adrb1",
    "adrb2",
    "akt1",
    "aldr",
    "ampc",
    "andr",
    "aofb",
    "bace1",
    "braf",
    "cah2",
    "casp3",
    "cdk2",
    "comt",
    "cp2c9",
    "cp3a4",
    "csf1r",
    "cxcr4",
    "def",
    "dhi1",
    "dpp4",
    "drd3",
    "dyr",
    "egfr",
    "esr1",
    "esr2",
    "fa10",
    "fa7",
    "fabp4",
    "fak1",
    "fgfr1",
    "fkb1a",
    "fnta",
    "fpps",
    "gcr",
    "glcm",
    "gria2",
    "grik1",
    "hdac2",
    "hdac8",
    "hivint",
    "hivpr",
    "hivrt",
    "hmdh",
    "hs90a",
    "hxk4",
    "igf1r",
    "inha",
    "ital",
    "jak2",
    "kif11",
    "kit",
    "kith",
    "kpcb",
    "lck",
]

# Selected 15 targets for ablation
SELECTED_15 = [
    "aa2ar",
    "abl1",
    "ace",
    "cxcr4",
    "egfr",
    "fak1",
    "hivpr",
    "kith",
    "mk01",
    "parp1",
    "ptn1",
    "pur2",
    "src",
    "try1",
    "vgfr2",
]

# Speed benchmark: 5 targets of varying sizes
SPEED_5 = ["aa2ar", "abl1", "ace", "dpp4", "fnta"]


# ── Data structures ───────────────────────────────────────────────────────


@dataclass
class TargetResult:
    """Results for a single target."""

    def __init__(self, target: str = "", variant: str = ""):
        self.target = target
        self.variant = variant
        self.n_mols = 0
        self.n_actives = 0
        self.n_decoys = 0
        self.n_completed = 0
        self.n_failed = 0
        self.auroc = None
        self.auprc = None
        self.ef1 = None
        self.ef5 = None
        self.ef10 = None
        self.wall_time_s = 0.0
        self.per_mol_time_s = 0.0
        self.error = None
        self.status = "pending"
        self.scores = []
        self.labels = []
        self.smiles = []
        self._score_sources = []

    def to_dict(self):
        return {
            "target": self.target,
            "variant": self.variant,
            "n_mols": self.n_mols,
            "n_actives": self.n_actives,
            "n_decoys": self.n_decoys,
            "n_completed": self.n_completed,
            "n_failed": self.n_failed,
            "auroc": self.auroc,
            "auprc": self.auprc,
            "ef1": self.ef1,
            "ef5": self.ef5,
            "ef10": self.ef10,
            "wall_time_s": self.wall_time_s,
            "per_mol_time_s": self.per_mol_time_s,
            "error": self.error,
            "status": self.status,
            # ── Per-molecule audit trail (M1 fix) ─────────────────────
            # Persist raw scores/labels/smiles so AUROC/EF distributions and
            # score-label scatter are reproducible/auditable post-hoc, and so
            # failure-rate × label cross-tables can be computed. Required for
            # Nature Methods-style reproducibility.
            "scores": self.scores if self.scores else None,
            "labels": self.labels if self.labels else None,
            "smiles": self.smiles if self.smiles else None,
            "_score_sources": self._score_sources if self._score_sources else None,
        }


@dataclass
class ExperimentState:
    """Full experiment state for checkpoint/resume."""

    name: str
    config: dict
    start_time: str = ""
    last_update: str = ""
    total_targets: int = 0
    completed_targets: int = 0
    failed_targets: int = 0
    total_mols: int = 0
    completed_mols: int = 0
    targets: dict[str, TargetResult] = field(default_factory=dict)
    summary: dict = field(default_factory=dict)


# ── Checkpoint Manager ────────────────────────────────────────────────────


class CheckpointManager:
    """Manages checkpoint files for resumable experiments."""

    def __init__(self, exp_dir: Path):
        self.exp_dir = exp_dir
        self.checkpoint_path = exp_dir / "checkpoint.json"
        self.results_path = exp_dir / "results.json"
        self.config_path = exp_dir / "config.json"
        self.log_dir = exp_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, state: ExperimentState):
        """Write checkpoint atomically."""
        state.last_update = datetime.now().isoformat()
        tmp = self.checkpoint_path.with_suffix(".tmp")

        # Build serializable dict manually (TargetResult is not a dataclass)
        data = {
            "name": state.name,
            "config": state.config,
            "start_time": state.start_time,
            "last_update": state.last_update,
            "total_targets": state.total_targets,
            "completed_targets": state.completed_targets,
            "failed_targets": state.failed_targets,
            "total_mols": state.total_mols,
            "completed_mols": state.completed_mols,
            "targets": {t: r.to_dict() for t, r in state.targets.items()},
            "summary": state.summary,
        }
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        tmp.replace(self.checkpoint_path)

    def load_checkpoint(self) -> Optional[ExperimentState]:
        """Load checkpoint if exists."""
        if not self.checkpoint_path.exists():
            return None
        with open(self.checkpoint_path) as f:
            data = json.load(f)
        state = ExperimentState(name=data["name"], config=data["config"])
        for k, v in data.items():
            if k == "targets":
                for t, td in v.items():
                    tr = TargetResult(target=t, variant=state.config.get("variant", ""))
                    for fk, fv in td.items():
                        if hasattr(tr, fk):
                            setattr(tr, fk, fv)
                    state.targets[t] = tr
            elif hasattr(state, k):
                setattr(state, k, v)
        return state

    def save_results(self, state: ExperimentState):
        """Write final results (summary only, no per-molecule arrays)."""
        output = {
            "experiment": state.name,
            "config": state.config,
            "start_time": state.start_time,
            "end_time": datetime.now().isoformat(),
            "total_targets": state.total_targets,
            "completed_targets": state.completed_targets,
            "failed_targets": state.failed_targets,
            "total_mols": state.total_mols,
            "completed_mols": state.completed_mols,
            "summary": state.summary,
            "targets": {t: r.to_dict() for t, r in state.targets.items()},
        }
        tmp = self.results_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(output, f, indent=2, default=str)
        tmp.replace(self.results_path)

    def get_target_logger(self, target: str) -> logging.Logger:
        """Get per-target logger."""
        log_path = self.log_dir / f"{target}.log"
        logger = logging.getLogger(f"bench.{target}")
        logger.setLevel(logging.DEBUG)
        if not logger.handlers:
            fh = logging.FileHandler(str(log_path))
            fh.setFormatter(logging.Formatter(LOG_FORMAT))
            logger.addHandler(fh)
        return logger


# ── Scoring Functions ────────────────────────────────────────────────────


def compute_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Compute AUROC, AUPRC, EF1%, EF5%, EF10%."""
    valid = np.isfinite(scores)
    s, l = scores[valid], labels[valid]
    n, n_act = len(s), int(l.sum())
    if n_act == 0 or n_act == n:
        return {"auroc": None, "auprc": None, "ef1": None, "ef5": None, "ef10": None}

    auroc = float(roc_auc_score(l, s))
    auprc = float(average_precision_score(l, s))

    order = np.argsort(s)[::-1]

    def ef(pct):
        k = max(1, int(n * pct / 100))
        return float((l[order][:k].sum() / k) / (n_act / n))

    return {
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "ef1": round(ef(1), 2),
        "ef5": round(ef(5), 2),
        "ef10": round(ef(10), 2),
    }


# ── Health Checks ─────────────────────────────────────────────────────────


def check_target_health(
    target: str, tr: TargetResult, tlog: logging.Logger
) -> list[str]:
    """Single-target quality gate. Returns list of issues found."""
    issues = []
    try:
        if tr.status == "completed":
            if tr.n_completed == 0:
                issues.append("ALL_MOLS_FAILED: n_completed=0")
            if tr.auroc is None:
                issues.append("METRICS_EMPTY: auroc=None")
            if tr.n_mols > 0:
                fail_ratio = tr.n_failed / tr.n_mols
                if fail_ratio > 0.8:
                    issues.append(f"HIGH_FAIL_RATE: {fail_ratio:.0%}")
        if tr.wall_time_s > 7200:
            issues.append(f"SLOW: {tr.wall_time_s:.0f}s")
        for issue in issues:
            tlog.warning(f"HEALTH: {issue}")
    except Exception as e:
        tlog.error(f"HEALTH_CHECK_ERROR: {e}")
    return issues


def check_experiment_health(
    state: ExperimentState,
    recently_completed: list[TargetResult] | None = None,
    consecutive_fail_threshold: int = 3,
    consecutive_zero_threshold: int = 3,
) -> tuple[bool, str]:
    """Global trend check. Returns (should_abort, reason)."""
    try:
        recent = (
            recently_completed
            or list(state.targets.values())[-consecutive_fail_threshold:]
        )
        if len(recent) >= consecutive_fail_threshold and all(
            t.status == "failed" for t in recent[-consecutive_fail_threshold:]
        ):
            return (
                True,
                f"CONSECUTIVE_FAILURES: last {consecutive_fail_threshold} targets all failed",
            )
        if len(recent) >= consecutive_zero_threshold and all(
            t.n_completed == 0 for t in recent[-consecutive_zero_threshold:]
        ):
            return (
                True,
                f"CONSECUTIVE_ZERO: last {consecutive_zero_threshold} targets all 0 completed",
            )
    except Exception as e:
        logging.getLogger("bench").error(f"HEALTH_EXPERIMENT_ERROR: {e}")
    return False, ""


# ── Target resolvers ──────────────────────────────────────────────────────


def resolve_targets(target_spec: str) -> list[str]:
    """Resolve target spec to list of DUD-E target names."""
    import lmdb, pickle
    from pathlib import Path

    DUD_E_DIR = Path("/workspace/data/dude")
    all_targets = sorted([d.name for d in DUD_E_DIR.iterdir() if d.is_dir()])

    if target_spec == "all":
        return all_targets
    elif target_spec == "selected_60":
        return [t for t in SELECTED_60 if t in all_targets]
    elif target_spec == "selected_15":
        return [t for t in SELECTED_15 if t in all_targets]
    elif target_spec == "speed_5":
        return [t for t in SPEED_5 if t in all_targets]
    elif isinstance(target_spec, list):
        return [t for t in target_spec if t in all_targets]
    else:
        # Assume it's a comma-separated list
        return [t.strip() for t in target_spec.split(",") if t.strip() in all_targets]


def sample_molecules(
    target: str, n_actives: int, n_decoys: int, seed: int = 42
) -> tuple[list[str], np.ndarray, list[int]]:
    """Load and sample molecules from a DUD-E target."""
    import lmdb, pickle
    from pathlib import Path
    import numpy as np

    DUD_E_DIR = Path("/workspace/data/dude")
    td = DUD_E_DIR / target

    env = lmdb.open(str(td / "mols.lmdb"), readonly=True, lock=False, subdir=False)
    all_smiles, all_labels = [], []
    with env.begin() as txn:
        for key, value in txn.cursor():
            data = pickle.loads(value)
            if isinstance(data, dict):
                all_smiles.append(data.get("smi", str(key)))
                all_labels.append(data.get("label", 0))
    env.close()

    labels = np.array(all_labels)
    rng = np.random.default_rng(seed)

    act_idx = rng.choice(
        np.where(labels == 1)[0], min(n_actives, int(labels.sum())), replace=False
    )
    dec_idx = rng.choice(
        np.where(labels == 0)[0],
        min(n_decoys, len(labels) - int(labels.sum())),
        replace=False,
    )

    idx = np.concatenate([act_idx, dec_idx])
    rng.shuffle(idx)

    return [all_smiles[i] for i in idx], labels[idx], idx.tolist()


def get_receptor_seq(target: str) -> str:
    """Extract protein sequence from DUD-E receptor PDB."""
    from Bio.PDB import PDBParser
    from Bio.SeqUtils import seq1

    DUD_E_DIR = Path("/workspace/data/dude")
    td = DUD_E_DIR / target
    for name in ["AF2_receptor.pdb", "receptor.pdb", "receptor_noX.pdb"]:
        rp = td / name
        if rp.exists():
            parser = PDBParser(QUIET=True)
            struct = parser.get_structure(target, str(rp))
            for model in struct:
                for chain in model:
                    return seq1("".join(r.get_resname() for r in chain))
    return ""


# ── Experiment Runners ────────────────────────────────────────────────────


def run_drugclip_l0(
    state: ExperimentState, ckpt: CheckpointManager, engine: logging.Logger
):
    """Run DrugCLIP L0 on all targets."""
    from app.drugclip_scorer import DrugCLIPScorer

    logger = logging.getLogger("bench.drugclip_l0")
    logger.info("Loading DrugCLIP 6-fold ensemble...")
    scorer = DrugCLIPScorer(use_ensemble=True, ensemble_dir="6_folds")
    _ = scorer.model

    _recent = []
    for target in state.targets:
        tr = state.targets[target]
        if tr.status == "completed":
            logger.info(f"  SKIP {target}: already completed")
            continue

        tr.status = "running"
        t0 = time.time()
        tlog = ckpt.get_target_logger(target)
        tlog.info(f"Starting DrugCLIP L0 for {target}")

        try:
            # Load data
            import lmdb, pickle

            DUD_E_DIR = Path("/workspace/data/dude")
            td = DUD_E_DIR / target

            # Load pocket
            env = lmdb.open(
                str(td / "pocket.lmdb"), readonly=True, lock=False, subdir=False
            )
            pocket_data = None
            with env.begin() as txn:
                for key, value in txn.cursor():
                    pocket_data = pickle.loads(value)
                    break
            env.close()

            # Create pocket PDB
            pocket_atoms = pocket_data.get("pocket_atoms", [])
            pocket_coords = pocket_data.get("pocket_coordinates", [])
            pocket_pdb = ckpt.log_dir / f"{target}_pocket.pdb"
            with open(pocket_pdb, "w") as f:
                f.write("REMARK DrugCLIP pocket\n")
                for i, (atom, coord) in enumerate(zip(pocket_atoms, pocket_coords)):
                    resseq = (i // 20) + 1
                    atom_name = atom if len(atom) <= 2 else atom[:2]
                    f.write(
                        f"HETATM{i + 1:5d} {atom_name:<2s}  LIG A {resseq:4d}    "
                        f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00  0.00          {atom_name:>2s}\n"
                    )

            # Encode pocket once
            pocket_emb = scorer.encode_pocket(str(pocket_pdb))

            # ── BUG FIX: use encode_molecules_from_lmdb instead of encode_molecules ──
            # encode_molecules() calls RDKit conformer generation for each SMILES
            # (CPU-bound, GPU idle). encode_molecules_from_lmdb() reads pre-computed
            # atom coordinates from DUD-E's mols.lmdb directly — 10-50x faster.
            env = lmdb.open(
                str(td / "mols.lmdb"), readonly=True, lock=False, subdir=False
            )
            batch_smiles, batch_labels = [], []
            with env.begin() as txn:
                for key, value in txn.cursor():
                    data = pickle.loads(value)
                    if isinstance(data, dict):
                        smi = data.get("smi", "")
                        label = data.get("label", 0)
                        batch_smiles.append(smi)
                        batch_labels.append(label)
            env.close()

            tlog.info(f"Loaded {len(batch_smiles)} molecules, encoding via LMDB...")
            mol_embs = scorer.encode_molecules_from_lmdb(str(td / "mols.lmdb"))
            tlog.info(f"Encoded {len(mol_embs)} molecules")

            scores = np.full(len(batch_smiles), np.nan)
            for i, smi in enumerate(batch_smiles):
                if smi in mol_embs:
                    scores[i] = float(np.dot(pocket_emb, mol_embs[smi]))

            tr.scores = scores.tolist()
            tr.labels = batch_labels
            tr.smiles = batch_smiles
            tr.n_mols = len(batch_smiles)
            tr.n_actives = int(sum(batch_labels))
            tr.n_completed = len(batch_smiles)

            metrics = compute_metrics(np.array(scores), np.array(batch_labels))
            for k, v in metrics.items():
                setattr(tr, k, v)

            tr.wall_time_s = round(time.time() - t0, 1)
            tr.per_mol_time_s = round(tr.wall_time_s / max(1, tr.n_mols), 3)
            tr.status = "completed"
            state.completed_targets += 1
            state.completed_mols += tr.n_mols

            tlog.info(
                f"AUROC={'N/A' if tr.auroc is None else f'{tr.auroc:.4f}'} EF1={'N/A' if tr.ef1 is None else f'{tr.ef1}'} ({tr.wall_time_s:.0f}s)"
            )
            logger.info(
                f"  {target}: AUROC={'N/A' if tr.auroc is None else f'{tr.auroc:.4f}'} ({tr.wall_time_s:.0f}s)"
            )

            check_target_health(target, tr, tlog)
            _recent.append(tr)

        except Exception as e:
            tr.status = "failed"
            tr.error = str(e)
            state.failed_targets += 1
            tlog.error(f"FAILED: {e}")
            logger.error(f"  {target}: FAILED: {e}")
            _recent.append(tr)

        ckpt.save_checkpoint(state)

        should_abort, reason = check_experiment_health(state, _recent)
        if should_abort:
            logger.critical(f"ABORT: {reason}")
            break

    # Release GPU
    scorer = None
    gc.collect()
    import torch

    torch.cuda.empty_cache()

    # Compute summary
    compute_summary(state)


def run_cascade_variant(state: ExperimentState, ckpt: CheckpointManager, variant: str):
    """Run cascade for a given variant."""
    from app.cascade import ScreeningCascade
    from app.cascade_config import CascadeConfig
    import torch

    # Determine config based on variant
    cfg = CascadeConfig()
    cfg.workdir = BENCHMARK_ROOT / state.name / "cascade_workdir"

    if variant == "cascade_full":
        cfg.run_l1 = True
        cfg.l1_method = "drugclip"
        cfg.l1_drugclip_ensemble = "6_folds"
        cfg.l1_top_fraction = 0.5
        cfg.never_drop = True
        cfg.mmgbsa_selection = "none"  # MM-GBSA handled separately
        cfg.consistency_samples = 4
        cfg.use_boltz = True
    elif variant == "ablation_no_l0":
        cfg.run_l1 = False
        cfg.never_drop = True
    elif variant == "ablation_no_l2":
        cfg.run_modes = ["pocket_off"]
        cfg.never_drop = True
    elif variant == "ablation_no_l4":
        cfg.mmgbsa_selection = "none"
        cfg.never_drop = True
    else:
        raise ValueError(f"Unknown variant: {variant}")

    logger = logging.getLogger(f"bench.{variant}")
    logger.info(f"Starting variant={variant}")

    _recent = []
    for target in state.targets:
        tr = state.targets[target]
        if tr.status == "completed":
            logger.info(f"  SKIP {target}: already completed")
            continue

        tr.status = "running"
        t0 = time.time()
        tlog = ckpt.get_target_logger(target)
        tlog.info(f"Starting {variant} for {target}")

        try:
            n_act = state.config.get("n_actives", tr.n_actives)
            n_dec = state.config.get("n_decoys", n_act)
            smiles, labels, _ = sample_molecules(target, n_act, n_dec, seed=42)
            prot_seq = get_receptor_seq(target)

            # ── C2 fix: explicit pocket-pdb handling (no silent fallback) ──
            # A pre-computed pocket PDB from a separate L0 experiment may not
            # exist in THIS experiment's log dir. Rather than pass a dangling
            # path (which silently triggers an undocumented 2-loop fold+fpocket
            # fallback inside L1), resolve existence here and log the chosen
            # path explicitly. When absent, the cascade self-bootstraps the
            # pocket via ESMFold2 fold + fpocket (same co-folding engine as
            # Leg 1/2 → same source, disclosed in Methods).
            _pocket_pdb = ckpt.log_dir / f"{target}_pocket.pdb"
            if _pocket_pdb.exists():
                cfg.l1_drugclip_pocket_pdb = str(_pocket_pdb)
                logger.info(
                    "[%s] L1 DrugCLIP using pre-computed pocket: %s",
                    target,
                    _pocket_pdb,
                )
            else:
                cfg.l1_drugclip_pocket_pdb = ""
                logger.info(
                    "[%s] No pre-computed pocket PDB found (%s); "
                    "L1 DrugCLIP will self-bootstrap pocket via "
                    "ESMFold2 fold + fpocket (disclosed path).",
                    target,
                    _pocket_pdb,
                )

            cascade = ScreeningCascade(config=cfg)
            result = cascade.run(
                smiles_list=smiles,
                protein_sequence=prot_seq,
                seed=42,
                num_loops=8,
            )

            # Extract scores (single monotonic direction: HIGHER = better).
            # - Boltz affinity_binary / Gnina cnn_affinity: higher = better
            # - dG_GB: LOWER (more negative) = better → negate to unify direction
            # Provenance per molecule is recorded for auditability (M1/C3 fix).
            scores = np.full(len(smiles), np.nan)
            score_sources = [""] * len(smiles)
            for r in result.results:
                if r.smiles in smiles:
                    idx = smiles.index(r.smiles)
                    if r.best_boltz and r.best_boltz.affinity_binary > 0:
                        scores[idx] = r.best_boltz.affinity_binary
                        score_sources[idx] = "boltz"
                    elif r.best_gnina and r.best_gnina.cnn_affinity is not None:
                        scores[idx] = r.best_gnina.cnn_affinity
                        score_sources[idx] = "gnina"
                    elif r.best_dg_gb is not None:
                        scores[idx] = -float(r.best_dg_gb)  # unify direction
                        score_sources[idx] = "mmgbsa"

            # ── Failure-rate × label cross-table (C3 fix) ────────────────
            # Silently dropping NaN-scored molecules biases AUROC/EF if failure
            # correlates with label (e.g. large active molecules dock poorly).
            # Report the joint distribution instead.
            valid = np.isfinite(scores)
            if valid.any():
                _labs = np.asarray(labels)
                for lab_name, lab_val in (("active", 1), ("decoy", 0)):
                    n_tot = int((_labs == lab_val).sum())
                    n_fail = int((~valid & (_labs == lab_val)).sum())
                    if n_tot:
                        logger.info(
                            "[%s] %s: %d/%d failed to score (%.1f%%)",
                            target,
                            lab_name,
                            n_fail,
                            n_tot,
                            100 * n_fail / n_tot,
                        )

            tr.scores = scores.tolist()
            tr.labels = labels.tolist()
            tr.smiles = smiles
            tr.n_mols = len(smiles)
            tr.n_actives = int(labels.sum())

            valid = np.isfinite(scores)
            tr.n_completed = int(valid.sum())
            tr.n_failed = len(smiles) - tr.n_completed

            # M1: persist score source provenance alongside raw scores
            if hasattr(tr, "_score_sources"):
                tr._score_sources = score_sources

            metrics = compute_metrics(scores, labels)
            for k, v in metrics.items():
                setattr(tr, k, v)

            tr.wall_time_s = round(time.time() - t0, 1)
            tr.per_mol_time_s = round(tr.wall_time_s / max(1, tr.n_completed), 1)
            tr.status = "completed"
            state.completed_targets += 1
            state.completed_mols += tr.n_completed

            tlog.info(
                f"AUROC={'N/A' if tr.auroc is None else f'{tr.auroc:.4f}'} EF1={'N/A' if tr.ef1 is None else f'{tr.ef1}'} ({tr.wall_time_s:.0f}s)"
            )
            logger.info(
                f"  {target}: AUROC={'N/A' if tr.auroc is None else f'{tr.auroc:.4f}'} ({tr.wall_time_s:.0f}s)"
            )

            check_target_health(target, tr, tlog)
            _recent.append(tr)

        except Exception as e:
            tr.status = "failed"
            tr.error = str(e)
            state.failed_targets += 1
            tlog.error(f"FAILED: {e}")
            logger.error(f"  {target}: FAILED: {e}")
            _recent.append(tr)
            import traceback

            tlog.error(traceback.format_exc())

        ckpt.save_checkpoint(state)

        should_abort, reason = check_experiment_health(state, _recent)
        if should_abort:
            logger.critical(f"ABORT: {reason}")
            break

        # GPU memory cleanup between targets
        import gc, torch

        cascade = None
        result = None
        gc.collect()
        torch.cuda.empty_cache()

    compute_summary(state)


def run_gnina_baseline(state: ExperimentState, ckpt: CheckpointManager):
    """Run gnina-only baseline."""
    import subprocess, tempfile
    from rdkit import Chem

    logger = logging.getLogger("bench.gnina_baseline")

    _recent = []
    for target in state.targets:
        tr = state.targets[target]
        if tr.status == "completed":
            logger.info(f"  SKIP {target}: already completed")
            continue

        tr.status = "running"
        t0 = time.time()
        tlog = ckpt.get_target_logger(target)
        tlog.info(f"Starting gnina baseline for {target}")

        try:
            smiles, labels, _ = sample_molecules(
                target, tr.n_actives, tr.n_decoys, seed=42
            )
            scores = np.full(len(smiles), np.nan)

            for i, smi in enumerate(smiles):
                wd = Path(tempfile.mkdtemp(prefix=f"gnina_{target}_"))
                try:
                    mol = Chem.MolFromSmiles(smi)
                    sdf = wd / "lig.sdf"
                    Chem.MolToMolFile(mol, str(sdf))

                    out = wd / "out.sdf"
                    cmd = [
                        "gnina",
                        "-r",
                        str(ckpt.log_dir / f"{target}_pocket.pdb"),
                        "-l",
                        str(sdf),
                        "--autobox_ligand",
                        str(sdf),
                        "--cnn_scoring",
                        "rescore",
                        "-o",
                        str(out),
                        "--exhaustiveness",
                        "4",
                        "--seed",
                        "42",
                    ]
                    result = subprocess.run(
                        cmd, cwd=str(wd), capture_output=True, text=True, timeout=300
                    )
                    if result.returncode == 0:
                        docked = Chem.SDMolSupplier(str(out))[0]
                        if docked and docked.HasProp("CNNaffinity"):
                            scores[i] = float(docked.GetProp("CNNaffinity"))
                except:
                    pass
                finally:
                    import shutil

                    shutil.rmtree(wd, ignore_errors=True)

            tr.scores = scores.tolist()
            tr.labels = labels.tolist()
            tr.smiles = smiles
            tr.n_mols = len(smiles)
            tr.n_actives = int(labels.sum())
            tr.n_completed = int(np.isfinite(scores).sum())
            tr.n_failed = len(smiles) - tr.n_completed

            metrics = compute_metrics(scores, labels)
            for k, v in metrics.items():
                setattr(tr, k, v)

            tr.wall_time_s = round(time.time() - t0, 1)
            tr.per_mol_time_s = round(tr.wall_time_s / max(1, tr.n_completed), 1)
            tr.status = "completed"
            state.completed_targets += 1
            state.completed_mols += tr.n_completed

            tlog.info(
                f"AUROC={'N/A' if tr.auroc is None else f'{tr.auroc:.4f}'} EF1={'N/A' if tr.ef1 is None else f'{tr.ef1}'} ({tr.wall_time_s:.0f}s)"
            )
            logger.info(
                f"  {target}: AUROC={'N/A' if tr.auroc is None else f'{tr.auroc:.4f}'} ({tr.wall_time_s:.0f}s)"
            )

            check_target_health(target, tr, tlog)

        except Exception as e:
            tr.status = "failed"
            tr.error = str(e)
            state.failed_targets += 1
            tlog.error(f"FAILED: {e}")
            _recent.append(tr)

        ckpt.save_checkpoint(state)

        should_abort, reason = check_experiment_health(state, _recent)
        if should_abort:
            logger.critical(f"ABORT: {reason}")
            break

    compute_summary(state)


def compute_summary(state: ExperimentState):
    """Compute cross-target summary statistics."""
    valid = [
        t
        for t in state.targets.values()
        if t.status == "completed" and t.auroc is not None
    ]
    if not valid:
        state.summary = {"error": "no completed targets"}
        return

    aucs = [t.auroc for t in valid]
    auprcs = [t.auprc for t in valid if t.auprc]
    ef1s = [t.ef1 for t in valid if t.ef1]
    ef5s = [t.ef5 for t in valid if t.ef5]

    state.summary = {
        "n_targets": len(valid),
        "mean_auroc": round(float(np.mean(aucs)), 4),
        "median_auroc": round(float(np.median(aucs)), 4),
        "std_auroc": round(float(np.std(aucs)), 4),
        "min_auroc": round(float(np.min(aucs)), 4),
        "max_auroc": round(float(np.max(aucs)), 4),
        "mean_auprc": round(float(np.mean(auprcs)), 4) if auprcs else None,
        "mean_ef1": round(float(np.mean(ef1s)), 2) if ef1s else None,
        "mean_ef5": round(float(np.mean(ef5s)), 2) if ef5s else None,
        "total_wall_time_h": round(sum(t.wall_time_s for t in valid) / 3600, 1),
    }


# ── Main ──────────────────────────────────────────────────────────────────


def setup_logging(exp_dir: Path):
    """Configure root logger."""
    log_path = exp_dir / "logs" / "summary.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fh = logging.FileHandler(str(log_path))
    fh.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    ch.setLevel(logging.INFO)
    root.addHandler(ch)

    return logging.getLogger("bench")


def main():
    parser = argparse.ArgumentParser(description="ESMFold Cascade Benchmark Engine")
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument(
        "--list", action="store_true", help="List available experiments"
    )
    args = parser.parse_args()

    if args.list:
        print("Available experiments:")
        for name, cfg in sorted(
            EXPERIMENTS.items(), key=lambda x: x[1].get("run_order", 0)
        ):
            print(f"  {name:30s} {cfg['description']}")
        return

    if args.experiment not in EXPERIMENTS:
        print(f"Unknown experiment: {args.experiment}")
        print("Use --list to see available experiments")
        return

    config = EXPERIMENTS[args.experiment]
    exp_dir = BENCHMARK_ROOT / args.experiment
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt = CheckpointManager(exp_dir)

    # Save config
    if not (ckpt.config_path).exists():
        with open(ckpt.config_path, "w") as f:
            json.dump(config, f, indent=2)

    # Setup logging
    logger = setup_logging(exp_dir)

    # Load or create state
    state = None
    if args.resume:
        state = ckpt.load_checkpoint()
        if state is None:
            logger.warning("No checkpoint found, starting fresh")
            state = None

    if state is None:
        targets = resolve_targets(config["targets"])
        state = ExperimentState(
            name=args.experiment,
            config=config,
            start_time=datetime.now().isoformat(),
            total_targets=len(targets),
        )
        for target in targets:
            tr = TargetResult(target=target, variant=config["variant"])
            tr.n_actives = config["n_actives"]
            state.targets[target] = tr
        ckpt.save_checkpoint(state)

    logger.info(f"Starting experiment: {args.experiment}")
    logger.info(
        f"  Targets: {state.total_targets} ({state.completed_targets} completed)"
    )
    logger.info(f"  Variant: {config['variant']}")

    variant = config["variant"]

    if variant == "drugclip_l0":
        run_drugclip_l0(state, ckpt, logger)
    elif variant in (
        "cascade_full",
        "ablation_no_l0",
        "ablation_no_l2",
        "ablation_no_l4",
    ):
        run_cascade_variant(state, ckpt, variant)
    elif variant == "gnina_baseline":
        run_gnina_baseline(state, ckpt)
    else:
        logger.error(f"Unknown variant: {variant}")
        return

    # Save final results
    ckpt.save_results(state)
    logger.info(f"Experiment complete!")
    logger.info(f"Summary: {json.dumps(state.summary, indent=2)}")


if __name__ == "__main__":
    main()
