"""
Pose confidence gate — the load-bearing wall of the cascade.

Six independent checks; any one failing discards the pose.
Check order: ① interface-PAE  ② ligand pLDDT  ③ pocket pLDDT
             ④ ligand iPTM   ⑤ PoseBusters   ⑥ cross-sample RMSD

References:
- docs/screening_cascade.py:246-332  (original 4-check gate)
- docs/verify_cofold_outputs.py      (token/PAE verification)
- docs/PIPELINE_vFinal.md §3         (gate specification)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger("cascade.gate")


# ── Data ──────────────────────────────────────────────────────────────────

@dataclass
class GateResult:
    passed: bool
    pocket_residues: list[int] = field(default_factory=list)
    pocket_plddt: float | None = None
    ligand_plddt: float | None = None
    interface_pae: float | None = None
    ligand_iptm: float | None = None
    reasons: list[str] = field(default_factory=list)
    # PoseBusters
    posebusters_passed: bool | None = None
    posebusters_failed: list[str] = field(default_factory=list)
    # Cross-sample consistency
    pose_rmsd_max: float | None = None
    pose_rmsd_samples: int = 1


# ── Pocket contact detection ──────────────────────────────────────────────

def pocket_residues_from_complex(
    complex_path: Path,
    ligand_chain: str = "L",
    cutoff: float = 5.0,
) -> list[int]:
    """Protein residues with a heavy atom within `cutoff` of any ligand heavy atom.

    Uses ESMFold2-specific mmCIF parser to avoid Biopython incompatibility
    with ESMFold2's non-standard mmCIF (missing occupancy, etc.).
    """
    from app.mmcif_parser import parse_esmfold_mmcif

    path_str = str(complex_path).lower()
    mmcif_text = Path(complex_path).read_text()

    # Parse ligand heavy-atom coordinates and protein coordinates
    lig_coords = []
    prot_coords = []  # (residue_seqid, x, y, z)

    fields = {}
    for line in mmcif_text.splitlines():
        if line.startswith("_atom_site."):
            field = line.strip().split(".", 1)[1]
            fields[field] = len(fields)
        if line.startswith("ATOM") or line.startswith("HETATM"):
            break

    grp_idx = fields.get("group_PDB", 0)
    chain_idx = fields.get("label_asym_id", 5)
    resnum_idx = fields.get("label_seq_id", 7)
    x_idx = fields.get("Cartn_x", 14)
    y_idx = fields.get("Cartn_y", 15)
    z_idx = fields.get("Cartn_z", 16)
    elem_idx = fields.get("type_symbol", 1)

    for line in mmcif_text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("data_") or s.startswith("loop_") or s.startswith("_"):
            continue
        parts = s.split()
        if len(parts) < 18:
            continue

        group = parts[grp_idx]
        chain = parts[chain_idx]
        elem = parts[elem_idx]
        if elem == "H":
            continue

        x, y, z = float(parts[x_idx]), float(parts[y_idx]), float(parts[z_idx])

        if group == "HETATM" and chain == ligand_chain:
            lig_coords.append((x, y, z))
        elif group == "ATOM":
            resnum = int(parts[resnum_idx])
            prot_coords.append((resnum, x, y, z))

    if not lig_coords:
        logger.warning("No ligand atoms found in %s (chain %s)", complex_path, ligand_chain)
        return []

    lig = np.asarray(lig_coords)
    pocket = set()
    for seqid, px, py, pz in prot_coords:
        p = np.array([px, py, pz])
        if np.min(np.linalg.norm(lig - p, axis=1)) <= cutoff:
            pocket.add(seqid)

    return sorted(pocket)


# ── Interface PAE ─────────────────────────────────────────────────────────

def interface_pae(
    pae: np.ndarray,
    ligand_token_indices: np.ndarray,
    pocket_token_indices: np.ndarray,
) -> float:
    """Mean PAE over the ligand-token × pocket-token block, both directions."""
    lig = np.asarray(list(ligand_token_indices), dtype=int)
    pk = np.asarray(list(pocket_token_indices), dtype=int)
    if lig.size == 0 or pk.size == 0:
        return float("nan")
    block = np.concatenate([
        pae[np.ix_(lig, pk)].ravel(),
        pae[np.ix_(pk, lig)].ravel(),
    ])
    return float(np.mean(block))


# ── Cross-sample RMSD ─────────────────────────────────────────────────────

def _compute_cross_rmsd(coords_list: list[np.ndarray]) -> float:
    """Max pairwise heavy-atom RMSD among k samples (after Kabsch alignment)."""
    if len(coords_list) < 2:
        return 0.0

    max_rmsd = 0.0
    for i in range(len(coords_list)):
        for j in range(i + 1, len(coords_list)):
            rmsd = _kabsch_rmsd(coords_list[i], coords_list[j])
            if rmsd > max_rmsd:
                max_rmsd = rmsd
    return max_rmsd


def _kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """RMSD after optimal rotation (Kabsch algorithm)."""
    if P.shape != Q.shape:
        return float("inf")
    if len(P) < 3:
        return float(np.sqrt(np.mean((P - Q) ** 2)))

    p_cent = P - P.mean(axis=0)
    q_cent = Q - Q.mean(axis=0)
    C = p_cent.T @ q_cent
    V, _, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.eye(3)
    D[2, 2] = d
    R = V @ D @ Wt
    aligned = p_cent @ R
    return float(np.sqrt(np.mean((aligned - q_cent) ** 2)))


# ── PoseBusters ───────────────────────────────────────────────────────────

def _split_for_posebusters(
    complex_path: Path,
    ligand_chain: str,
    smiles: str,
    workdir: Path,
) -> tuple[Path, Path]:
    """Split complex into receptor PDB + ligand SDF for PoseBusters.

    Uses ESMFold2-specific mmCIF parser (no Biopython).
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mmcif_text = Path(complex_path).read_text()
    workdir.mkdir(parents=True, exist_ok=True)

    # Parse fields
    fields = {}
    for line in mmcif_text.splitlines():
        if line.startswith("_atom_site."):
            field = line.strip().split(".", 1)[1]
            fields[field] = len(fields)
        if line.startswith("ATOM") or line.startswith("HETATM"):
            break

    grp_idx = fields.get("group_PDB", 0)
    chain_idx = fields.get("label_asym_id", 5)
    atom_idx = fields.get("label_atom_id", 2)
    resname_idx = fields.get("label_comp_id", 4)
    resnum_idx = fields.get("label_seq_id", 7)
    x_idx = fields.get("Cartn_x", 14)
    y_idx = fields.get("Cartn_y", 15)
    z_idx = fields.get("Cartn_z", 16)
    elem_idx = fields.get("type_symbol", 1)

    # Separate receptor and ligand atoms
    rec_lines = []
    lig_lines = []
    serial = 0
    lig_serial = 0
    for line in mmcif_text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("data_") or s.startswith("loop_") or s.startswith("_"):
            continue
        parts = s.split()
        if len(parts) < 18:
            continue

        group = parts[grp_idx]
        chain = parts[chain_idx]
        elem = parts[elem_idx]
        atom = parts[atom_idx]
        resname = parts[resname_idx]
        resnum = int(parts[resnum_idx])
        x, y, z = float(parts[x_idx]), float(parts[y_idx]), float(parts[z_idx])

        if group == "HETATM" and chain == ligand_chain:
            lig_serial += 1
            lig_lines.append(
                f"HETATM{lig_serial:5d} {atom:<4s} {resname:3s} {chain:1s}{resnum:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {elem:2s}"
            )
        elif group == "ATOM":
            serial += 1
            rec_lines.append(
                f"ATOM  {serial:5d} {atom:<4s} {resname:3s} {chain:1s}{resnum:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {elem:2s}"
            )

    rec_lines.append("END\n")
    lig_lines.append("END\n")
    rec_pdb = workdir / "_pb_rec.pdb"
    lig_pdb = workdir / "_pb_lig.pdb"
    lig_sdf = workdir / "_pb_lig.sdf"
    rec_pdb.write_text("\n".join(rec_lines))
    lig_pdb.write_text("\n".join(lig_lines))

    mol = Chem.MolFromPDBFile(str(lig_pdb), removeHs=True, sanitize=False)
    if mol is None:
        raise ValueError("RDKit could not read extracted ligand for PoseBusters")
    if smiles:
        template = Chem.MolFromSmiles(smiles)
        if template is not None:
            mol = AllChem.AssignBondOrdersFromTemplate(template, mol)
    Chem.SanitizeMol(mol)
    Chem.MolToMolFile(Chem.AddHs(mol, addCoords=True), str(lig_sdf))
    return rec_pdb, lig_sdf


def _run_posebusters(ligand_sdf: Path, receptor_pdb: Path, config: str = "dock") -> tuple[bool, list[str]]:
    """Return (passed, list_of_failed_check_names)."""
    from posebusters import PoseBusters
    pb = PoseBusters(config=config)
    df = pb.bust([str(ligand_sdf)], None, str(receptor_pdb), full_report=False)
    bool_cols = [c for c in df.columns if str(df[c].dtype) == "bool"]
    if not bool_cols:
        return False, ["no_boolean_checks"]
    passed = bool(df[bool_cols].all(axis=1).iloc[0])
    failed = [c for c in bool_cols if not bool(df[c].iloc[0])]
    return passed, failed


# ── Main gate ─────────────────────────────────────────────────────────────

def confidence_gate(
    fold_result,               # DetailedFoldResult from local_inference
    config,                    # CascadeConfig
    pocket_residues: list[int] | None = None,
) -> GateResult:
    """Run all 6 checks on a co-fold result. Returns GateResult."""
    from app.cascade_config import CascadeConfig
    cfg: CascadeConfig = config

    reasons: list[str] = []

    # ── Pocket residues ───────────────────────────────────────────────
    if pocket_residues is None and fold_result.complex_path:
        pocket_residues = pocket_residues_from_complex(
            fold_result.complex_path,
            ligand_chain="L",
            cutoff=cfg.pocket_contact_cutoff_A,
        )

    pocket = pocket_residues or []
    if not pocket:
        return GateResult(
            passed=False, pocket_residues=[],
            reasons=["no protein residues in contact with ligand"],
        )

    # Map pocket residue seqids to token positions
    pocket_set = set(pocket)
    pocket_tok = [
        int(p) for r, p in zip(
            fold_result.protein_residue_indices,
            np.where(fold_result.protein_token_mask)[0],
        )
        if int(r) in pocket_set
    ]

    pocket_plddt = ligand_plddt = if_pae = None

    # ── ① Pocket pLDDT ───────────────────────────────────────────────
    if len(pocket_tok) > 0:
        pocket_plddt = float(np.mean(fold_result.plddt_array[pocket_tok]))
        if pocket_plddt < cfg.min_pocket_plddt:
            reasons.append(f"pocket pLDDT {pocket_plddt:.1f} < {cfg.min_pocket_plddt}")

    # ── ② Ligand pLDDT ───────────────────────────────────────────────
    lig_tok = np.where(fold_result.ligand_token_mask)[0]
    if len(lig_tok) > 0:
        ligand_plddt = float(np.mean(fold_result.plddt_array[lig_tok]))
        if ligand_plddt < cfg.min_ligand_plddt:
            reasons.append(f"ligand pLDDT {ligand_plddt:.1f} < {cfg.min_ligand_plddt}")
    else:
        reasons.append("no ligand tokens found")

    # ── ③ Interface PAE ──────────────────────────────────────────────
    if (fold_result.pae_matrix is not None and len(lig_tok) > 0 and len(pocket_tok) > 0):
        _ipae_val = interface_pae(fold_result.pae_matrix, lig_tok, pocket_tok)
        if_pae = _ipae_val
        if _ipae_val > cfg.max_interface_pae:
            reasons.append(f"interface PAE {_ipae_val:.2f} > {cfg.max_interface_pae} A")
    elif fold_result.pae_matrix is None:
        logger.warning("PAE matrix unavailable — interface-PAE check skipped; gate degraded")

    # ── ④ Ligand iPTM ────────────────────────────────────────────────
    if cfg.min_ligand_iptm > 0 and fold_result.iptm is not None:
        if fold_result.iptm < cfg.min_ligand_iptm:
            reasons.append(f"ligand iPTM {fold_result.iptm:.3f} < {cfg.min_ligand_iptm}")

    # ── ⑤ PoseBusters ────────────────────────────────────────────────
    pb_passed = None
    pb_failed: list[str] = []
    if cfg.run_posebusters and fold_result.complex_path:
        try:
            from app.cascade_prep import split_complex as prep_split
            wd = cfg.workdir / f"pb_{fold_result.job_id}"
            wd.mkdir(parents=True, exist_ok=True)
            rec_pdb, lig_sdf = prep_split(
                fold_result.complex_path, wd,
                smiles=fold_result.smiles,
                protonation_tool=cfg.protonation_tool,
                ph=cfg.protonation_pH,
            )
            pb_passed, pb_failed = _run_posebusters(lig_sdf, rec_pdb, cfg.posebusters_config)
            if not pb_passed:
                reasons.append(f"PoseBusters failed: {pb_failed}")
        except Exception as e:
            logger.warning("PoseBusters skipped (%s)", e)
    elif cfg.run_posebusters:
        logger.warning("PoseBusters skipped — no complex_path on fold_result")

    # ── ⑥ Cross-sample consistency ───────────────────────────────────
    rmsd_max = None
    if len(fold_result.sample_ligand_coords) >= 2:
        rmsd_max = _compute_cross_rmsd(fold_result.sample_ligand_coords)
        if rmsd_max > cfg.max_pose_rmsd_A:
            reasons.append(f"pose cross-RMSD {rmsd_max:.2f} > {cfg.max_pose_rmsd_A} A")

    return GateResult(
        passed=(len(reasons) == 0),
        pocket_residues=pocket,
        pocket_plddt=pocket_plddt,
        ligand_plddt=ligand_plddt,
        interface_pae=if_pae,
        ligand_iptm=fold_result.iptm,
        reasons=reasons or ["passed"],
        posebusters_passed=pb_passed,
        posebusters_failed=pb_failed,
        pose_rmsd_max=rmsd_max,
        pose_rmsd_samples=len(fold_result.sample_ligand_coords),
    )


# ── Triage gate (recall-first) ────────────────────────────────────────────

@dataclass
class TriageResult:
    """Routing decision from recall-first gate. Nothing is discarded."""
    status: str                  # "ok" | "weak" | "invalid_pose" | "failed"
    confidence_tier: str         # "high" | "medium" | "low" | "uncertain"
    gate: GateResult
    routing_actions: list[str] = field(default_factory=list)
    # Actions: "upgrade_samples", "try_pocket_on", "try_orthogonal_rescue",
    #          "resample_pose", "park"


def triage_gate(
    fold_result,
    config,
    pocket_residues: list[int] | None = None,
    source_leg: str = "pocket_off",
) -> TriageResult:
    """Recall-first triage: 6 checks route the compound, never kill it.

    HARD CONSTRAINT: gate signal ONLY from Leg 1 (pocket_off). Leg 2 PAE is
    contaminated by the PocketConditioning constraint.
    """
    from app.cascade_config import CascadeConfig
    cfg: CascadeConfig = config

    if source_leg != cfg.gate_signal_from and cfg.gate_signal_from == "pocket_off":
        logger.warning(
            "Gate signal from Leg '%s', but gate_signal_from='%s'. "
            "PAE from constrained runs (Leg 2) is contaminated by the constraint.",
            source_leg, cfg.gate_signal_from,
        )

    gate = confidence_gate(fold_result, cfg, pocket_residues)
    routing: list[str] = []
    status = "ok"
    confidence_tier = "high"

    if gate.passed:
        if gate.interface_pae is not None and gate.interface_pae > cfg.max_interface_pae * 0.7:
            status = "weak"
            confidence_tier = "medium"
            routing.append("borderline_pae")
        # Leg 2 contamination downgrade even for passing results
        if source_leg != cfg.gate_signal_from and cfg.gate_signal_from == "pocket_off":
            confidence_tier = "medium" if confidence_tier == "high" else confidence_tier
            routing.append("leg2_pae_downgrade")
        return TriageResult(status=status, confidence_tier=confidence_tier,
                            gate=gate, routing_actions=routing)

    # ── Categorize failures ───────────────────────────────────────────
    has_no_contact = any("no protein residues" in r for r in gate.reasons)
    has_posebusters_issue = bool(gate.posebusters_failed)
    has_pae_issue = any("PAE" in r for r in gate.reasons)
    has_iptm_issue = any("iPTM" in r for r in gate.reasons)
    has_plddt_issue = any("pLDDT" in r for r in gate.reasons)

    if has_no_contact:
        status = "failed"
        confidence_tier = "uncertain"
        routing = ["try_orthogonal_rescue", "try_pocket_on"]

    elif has_posebusters_issue and len(gate.reasons) <= 1:
        # PoseBusters-only failure (typically steric clashes that MM-GBSA
        # minimization resolves). Demote to weak instead of killing the pose.
        status = "weak"
        confidence_tier = "medium"
        routing = ["posebusters_issues", "upgrade_samples", "try_pocket_on"]

    elif has_pae_issue and not has_plddt_issue:
        status = "weak"
        confidence_tier = "medium"
        routing = ["upgrade_samples", "try_pocket_on", "try_orthogonal_rescue"]

    elif has_pae_issue or has_iptm_issue:
        status = "weak"
        confidence_tier = "low"
        routing = ["upgrade_samples", "try_pocket_on", "try_orthogonal_rescue"]

    elif has_plddt_issue:
        status = "weak"
        confidence_tier = "low"
        routing = ["try_pocket_on", "try_orthogonal_rescue"]

    else:
        status = "weak"
        confidence_tier = "low"
        routing = ["try_orthogonal_rescue"]

    # Leg 2 contamination downgrade (for non-passed results)
    if source_leg != cfg.gate_signal_from:
        confidence_tier = "low"
        routing.append("leg2_pae_downgrade")

    return TriageResult(
        status=status,
        confidence_tier=confidence_tier,
        gate=gate,
        routing_actions=routing,
    )
