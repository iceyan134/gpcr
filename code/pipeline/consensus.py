"""Calibrated consensus scoring (GPCR-Bench 24-target, 5-fold target-level CV).

Weights and signal transforms follow notes/consensus-calibration-results.md
and tmp/calibration_optimize.py exactly:
    consensus = (w_aff*aff + w_pae*(1-pae/10) + w_lp*plddt/100 + w_ip*iptm)
                / sum(weights of available signals)
with each signal clipped to [0,1]; NaN/None signals drop out of both
numerator and denominator. Calibrated weights (mean over folds):
    affinity 0.65, iPAE 0.18, ligand pLDDT 0.11, iPTM 0.06.
Calibrated mean AUROC 0.9103 vs Nesso-only 0.9035 (fixed 15% prior 0.8544).
Sensitivity to ±20% weight perturbation < 0.001 AUROC (plateau solution).

Design note (audit 2026-08-31): this ADDS a consensus_score alongside
best_boltz.affinity_binary; it does not replace the affinity-only ranking
used by historical runs, preserving comparability of past results.
"""
from dataclasses import dataclass

CALIBRATED_WEIGHTS = {
    "affinity": 0.65,
    "ipae": 0.18,
    "ligand_plddt": 0.11,
    "iptm": 0.06,
}
MAX_IPAE = 10.0  # iPAE at/above this scores 0 (gate threshold scale)


def _f(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


def consensus_score(
    affinity: float | None,
    interface_pae: float | None,
    ligand_plddt: float | None,
    ligand_iptm: float | None,
    weights: dict[str, float] | None = None,
    max_ipae: float = MAX_IPAE,
) -> float | None:
    """Weighted consensus; None when no signal is available."""
    w = weights or CALIBRATED_WEIGHTS
    signals = {
        "affinity": min(max(_f(affinity) or 0.0, 0.0), 1.0) if _f(affinity) is not None else None,
        "ipae": (
            min(max(1.0 - _f(interface_pae) / max_ipae, 0.0), 1.0)
            if _f(interface_pae) is not None else None
        ),
        "ligand_plddt": (
            min(max(_f(ligand_plddt) / 100.0, 0.0), 1.0)
            if _f(ligand_plddt) is not None else None
        ),
        "iptm": (
            min(max(_f(ligand_iptm), 0.0), 1.0)
            if _f(ligand_iptm) is not None else None
        ),
    }
    total = 0.0
    wsum = 0.0
    for name, s in signals.items():
        if s is not None:
            total += w[name] * s
            wsum += w[name]
    if wsum <= 0:
        return None
    return total / wsum


@dataclass
class ConsensusInputs:
    affinity: float | None
    interface_pae: float | None
    ligand_plddt: float | None
    ligand_iptm: float | None


def inputs_from_merge_result(mr) -> ConsensusInputs | None:
    """Extract consensus signals from a CompoundMergeResult-like object.

    Gate signals follow cascade_config.gate_signal_from ("pocket_off", the
    HARD rule): signals come from the leg used for gating, while affinity
    comes from best_boltz (any leg).
    """
    gate = None
    legs = getattr(mr, "legs", {}) or {}
    for lr in legs.get("pocket_off", []):
        if lr.gate is not None:
            gate = lr.gate
            break
    bb = getattr(mr, "best_boltz", None)
    affinity = float(bb.affinity_binary) if bb is not None and bb.affinity_binary > 0 else None
    if gate is None and affinity is None:
        return None
    return ConsensusInputs(
        affinity=affinity,
        interface_pae=getattr(gate, "interface_pae", None) if gate else None,
        ligand_plddt=getattr(gate, "ligand_plddt", None) if gate else None,
        ligand_iptm=getattr(gate, "ligand_iptm", None) if gate else None,
    )


def rank_by_consensus(results: list) -> list[tuple[str, float]]:
    """Rank compounds by consensus score (descending, None-scores last).

    Returns [(smiles, consensus_score), ...]; affinity-only ranking remains
    available via best_boltz for comparability.
    """
    scored = []
    for mr in results:
        inp = inputs_from_merge_result(mr)
        s = consensus_score(inp.affinity, inp.interface_pae,
                            inp.ligand_plddt, inp.ligand_iptm) if inp else None
        scored.append((mr.smiles, s if s is not None else float("-inf")))
    scored.sort(key=lambda kv: -kv[1])
    return scored
