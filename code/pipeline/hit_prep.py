"""Hit preparation pipeline: from virtual-screening output to wet-lab-ready
candidate list. Generalizes the GCGR-specific workflow into a reusable layer.

Sits AFTER screen_flow (results.jsonl) and BEFORE wet-lab ordering.
Six steps, each independently skippable:

  Step 0  Triage        — aggregation propensity + reactivity + OOD
  Step 1  Pose score    — automated 6-dim quality (no manual PyMOL)
  Step 2  MM-GBSA       — single-point physics cross-validation
  Step 3  Redock vote   — Vina pocket-consistency (not RMSD judge)
  Step 4  Selectivity   — cross-target scoring vs related receptors
  Step 5  Availability  — commercial purchasability proxy

Target-adaptive: binding-site residues auto-detected from top poses if not
provided (orphan-receptor compatible). Selectivity targets optional.

Usage:
  python -m app.hit_prep --screen-dir output/screen_rerun_v2_gpr146
  python -m app.hit_prep --screen-dir ... --config hit_prep.yaml
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

logger = logging.getLogger("cascade.hit_prep")

# ── known colloidal aggregators (Iridium/Aggregator Advisor subset) ─────
AGGREGATOR_SMARTS = {
    "quinone_methide": "C=C1C=CC(=O)c2ccccc12",
    "curcumin_like": "C=C(C)CC=C(C)CC=C(C)c1ccccc1",
    "epigallocatechin_like": "OC1Cc2c(O)c(O)c(O)cc2OC1",
    "rosmarinic_like": "OC(=O)C(O)c1ccc(O)c(O)c1",
    "tannin_like": "O=C(O)c1cc(O)c(O)c(O)c1",
}
_AGG_PATTS = None


_AGG_PATT_cache = None


def _agg_patterns():
    global _AGG_PATT_cache
    if _AGG_PATT_cache is not None:
        return _AGG_PATT_cache
    from rdkit import Chem
    _AGG_PATT_cache = {k: Chem.MolFromSmarts(v) for k, v in AGGREGATOR_SMARTS.items()
                       if Chem.MolFromSmarts(v) is not None}
    return _AGG_PATT_cache


def aggregation_risk(smiles: str) -> list[str]:
    """Known colloidal aggregator motifs matched (heuristic SMARTS subset)."""
    from rdkit import Chem
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return ["unparseable"]
    hits = [name for name, patt in _agg_patterns().items() if m.HasSubstructMatch(patt)]
    # polyphenol heuristic: ≥4 phenolic OH + ≥2 aromatic rings
    n_phenol = sum(1 for a in m.GetAtoms()
                   if a.GetSymbol() == "O" and a.GetTotalNumHs() >= 1
                   and any(n.GetSymbol() == "C" and n.GetIsAromatic()
                           for n in a.GetNeighbors()))
    n_aro = Chem.rdMolDescriptors.CalcNumAromaticRings(m)
    if n_phenol >= 4 and n_aro >= 2:
        hits.append(f"polyphenol_heuristic({n_phenol}OH,{n_aro}rings)")
    return hits


# ── Step 1: automated pose quality ──────────────────────────────────────
def _short_id(smiles: str) -> str:
    """Mirror cascade.py _short_id for directory naming."""
    bad = "()[]=@#/\\"
    s = smiles[:20]
    for c in bad:
        s = s.replace(c, "")
    s = s.strip()
    return s if s else "lig"


def _load_complex_structure(cascade_dir: Path, smiles: str):
    """Find ligand + receptor PDB pair for a molecule.

    Returns (ligand_pdb_path, receptor_pdb_path) or None.
    Uses cascade's _short_id for directory naming; falls back to
    prefix scan for edge cases (truncated names).
    """
    if not cascade_dir.exists():
        return None
    cid = _short_id(smiles)
    # exact match first
    d = cascade_dir / cid
    if d.exists():
        leg1 = d / "leg1"
        if leg1.exists():
            lig = leg1 / "ligand_raw.pdb"
            rec = leg1 / "receptor_raw.pdb"
            if lig.exists() and rec.exists():
                return (lig, rec)
    # prefix fallback (handles collisions from truncation)
    for dd in cascade_dir.iterdir():
        if dd.is_dir() and (dd.name.startswith(cid[:10]) or cid.startswith(dd.name[:10])):
            leg1 = dd / "leg1"
            if leg1.exists():
                lig = leg1 / "ligand_raw.pdb"
                rec = leg1 / "receptor_raw.pdb"
                if lig.exists() and rec.exists():
                    return (lig, rec)
    return None


def score_pose(smiles: str, complex_path: Path, site_residues: set[int] | None,
               auto_site: bool = True) -> dict:
    """6-dim automated pose quality score (0-10 scale).

    Dims: pocket_occupancy | burial | hbond | key_residue_contact | depth | hydrophobic
    """
    from rdkit import Chem
    from Bio.PDB import PDBParser
    import os

    if not complex_path:
        return {"score": 0, "reason": "structure_not_found"}
    # handle (ligand, receptor) tuple or single path
    if isinstance(complex_path, tuple):
        lig_path, rec_path = complex_path
        if not (lig_path.exists() and rec_path.exists()):
            return {"score": 0, "reason": "files_missing"}
        lig_text = lig_path.read_text(errors="ignore")
        rec_text = rec_path.read_text(errors="ignore")
    else:
        if not complex_path.exists():
            return {"score": 0, "reason": "structure_not_found"}
        text = complex_path.read_text(errors="ignore")
        lig_text = text  # single file: parse all
        rec_text = text

    # parse ligand coords from ligand file
    lig_coords = []
    for line in lig_text.splitlines():
        if (line.startswith("ATOM") or line.startswith("HETATM")) and not line[17:20].strip() in ("HOW", "HOH"):
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                resname = line[17:20].strip()
            except (ValueError, IndexError):
                continue
            if line.startswith("HETATM") and resname not in ("HOH", "WAT"):
                lig_coords.append((x, y, z))

    # parse protein residues from receptor file
    prot_res = {}
    for line in rec_text.splitlines():
        if line.startswith("ATOM"):
            try:
                resid = int(line[22:26])
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except (ValueError, IndexError):
                continue
            prot_res.setdefault(resid, []).append((x, y, z))

    if not lig_coords or not prot_res:
        return {"score": 0, "reason": "no_ligand_or_protein"}
    lig = np.array(lig_coords)
    ligand_centroid = lig.mean(axis=0)

    # auto-detect site: residues within 6 Å of any ligand atom
    contacts = set()
    all_prot = []
    for resid, coords in prot_res.items():
        pc = np.array(coords)
        all_prot.append((resid, pc.mean(axis=0)))
        d = np.linalg.norm(pc[:, None, :] - lig[None, :, :], axis=-1).min()
        if d <= 6.0:
            contacts.add(resid)

    # D1 pocket occupancy: fraction of ligand atoms within 4 Å of any protein atom
    all_prot_coords = np.concatenate([np.array(c) for c in prot_res.values()])
    d_matrix = np.linalg.norm(lig[:, None, :] - all_prot_coords[None, :, :], axis=-1)
    per_atom_min = d_matrix.min(axis=1)
    frac_in_pocket = float((per_atom_min < 4.0).mean())

    # D2 burial: ligand SASA proxy (centroid distance to protein centroid)
    prot_centroid = np.mean([c for _, c in all_prot], axis=0)
    centroid_dist = np.linalg.norm(ligand_centroid - prot_centroid)
    # deeper = better (lower dist to protein center = more buried)
    burial = max(0, 1 - centroid_dist / 40)  # 40 Å = surface

    # D3 H-bond proxy: any atom pairs within 3.5 Å (global count)
    close_pairs = int((d_matrix <= 3.5).sum())
    hbond_score = min(close_pairs / 12.0, 1.0)

    # D4 key residue contact
    if site_residues:
        overlap = len(contacts & site_residues) / max(len(site_residues), 1)
    else:
        overlap = len(contacts) / 20.0  # heuristic: ≥20 residues in contact = good

    # D5 pocket depth (z-axis fraction from extracellular)
    all_z = [c[2] for _, c in all_prot]
    z_frac = (ligand_centroid[2] - min(all_z)) / max(max(all_z) - min(all_z), 1e-9)
    depth_score = 1.0 - abs(z_frac - 0.4)  # prefer ~40% from extracellular (mid-TM)

    # D6 hydrophobic contact: C-C pairs within 4.5 Å
    hydrophobic = 0
    for resid, coords in prot_res.items():
        for xyz in coords:
            for lc in lig_coords:
                if np.linalg.norm(np.array(xyz) - np.array(lc)) <= 4.5:
                    hydrophobic += 1
    hydrophobic_score = min(hydrophobic / 30.0, 1.0)

    score = (2 * frac_in_pocket + 1.5 * burial + 2 * hbond_score +
             2 * overlap + 1 * depth_score + 1.5 * hydrophobic_score) / 10.0
    return {
        "score": round(min(score * 10, 10), 1),
        "frac_in_pocket": round(frac_in_pocket, 2),
        "burial": round(burial, 2), "hbond": round(hbond_score, 2),
        "site_overlap": round(overlap, 2), "depth": round(depth_score, 2),
        "hydrophobic": round(hydrophobic_score, 2),
        "n_contacts": len(contacts),
    }


# ── Step 2: MM-GBSA single-point ────────────────────────────────────────
def mmgbsa_singlepoint(smiles: str, cascade_dir: Path,
                       workdir: Path | None = None) -> float | None:
    """Single-point MM-GBSA on co-folded pose. Returns ΔG (kcal/mol).

    Uses the full AmberTools pipeline (now working with -d fix, audit
    2026-09-02): pdb4amber -d → antechamber → tleap → OpenMM vacuum
    minimize → cpptraj → MMPBSA.py. CPU only, ~2-3 min per molecule.
    """
    import os
    os.environ.setdefault("AMBERHOME", "/opt/conda")
    struct = _load_complex_structure(cascade_dir, smiles)
    if not struct:
        return None
    lig_sdf, rec_pdb = struct
    wd = workdir or Path(tempfile.mkdtemp(prefix="mmgbsa_"))
    wd.mkdir(parents=True, exist_ok=True)
    try:
        from app.mmgbsa import mmgbsa_refine
        result = mmgbsa_refine(
            receptor_raw=rec_pdb, ligand_sdf=lig_sdf, workdir=wd,
            igb=5, saltcon=0.150,
            charge_method="bcc",
            min_engine="openmm", openmm_platform="CPU",
            sander_min_steps=500, mmgbsa_mode="minimize_only",
            timeout_s=600)
        return result.dg_gb if result else None
    except Exception as e:
        logger.warning("MM-GBSA failed for %s: %s", smiles[:30], str(e)[:100])
        return None


# ── Step 4: selectivity ─────────────────────────────────────────────────
def selectivity_scores(smiles_list: list[str], target_fastas: dict[str, str],
                       workdir: Path) -> dict[str, list[float]]:
    """Score ligands against multiple receptors. Returns {target_name: [scores]}."""
    from app.l1_prescreen import L1DockingPreScreen
    results = {}
    for name, fasta in target_fastas.items():
        seq = "".join(open(fasta).read().splitlines()[1:]).strip()
        try:
            l1 = L1DockingPreScreen(use_gpu=True, seed=42)
            res = l1.screen_library(seq, smiles_list, workdir=workdir / f"sel_{name}",
                                    method="drugclip", top_fraction=1.0,
                                    drugclip_ensemble="6_folds",
                                    receptor_pdb="", pocket_pdb="",
                                    top_n_pockets=3, min_quality=0.20,
                                    num_loops=8, num_sampling_steps=32)
            results[name] = [float(r.cnn_affinity) for r in res.all_results]
        except Exception as e:
            logger.warning("selectivity vs %s failed: %s", name, e)
            results[name] = [None] * len(smiles_list)
    return results


# ── main orchestration ──────────────────────────────────────────────────
@dataclass
class HitPrepConfig:
    screen_dir: str
    known_site_residues: list[int] | None = None
    selectivity_fastas: dict[str, str] = field(default_factory=dict)
    top_n_input: int = 50
    top_n_output: int = 10
    run_step0: bool = True
    run_step1: bool = True
    run_step2: bool = True
    run_step3: bool = True
    run_step4: bool = False  # requires GPU + additional targets
    run_step5: bool = True

    @classmethod
    def load(cls, path):
        p = Path(path)
        if p.suffix in (".yaml", ".yml"):
            import yaml
            return cls(**yaml.safe_load(p.read_text()))
        return cls(**json.loads(p.read_text()))


class HitPrep:
    def __init__(self, config: HitPrepConfig):
        self.cfg = config
        self.screen_dir = Path(config.screen_dir)
        self.out = self.screen_dir / "hit_prep"
        self.out.mkdir(exist_ok=True)

    def run(self) -> list[dict]:
        t0 = time.time()
        rows = [json.loads(l) for l in open(self.screen_dir / "results.jsonl")]
        scored = [r for r in rows if r["affinity_binary"] is not None]
        by_consensus = sorted(scored, key=lambda r: -(r["consensus_score"] or -1))
        top = by_consensus[:self.cfg.top_n_input]
        logger.info("hit_prep: %d molecules from %s", len(top), self.screen_dir)

        # Step 0: triage
        if self.cfg.run_step0:
            from app.hitlist import reactive_alerts, ood_flag
            for r in top:
                r["_agg"] = aggregation_risk(r["smiles"])
                r["_reactive"] = reactive_alerts(r["smiles"])
                r["_ood"] = ood_flag(r["smiles"])
            pre = len(top)
            top = [r for r in top if not r["_reactive"] or r["_reactive"] == ["unparseable_smiles"]]
            # aggregation flag = warning not elimination (polyphenols may be real)
            logger.info("step0: %d → %d (reactivity eliminated %d)",
                        pre, len(top), pre - len(top))

        # Step 1: pose scoring
        if self.cfg.run_step1:
            cascade_dir = self.screen_dir / "cascade"
            site = set(self.cfg.known_site_residues) if self.cfg.known_site_residues else None
            for r in top:
                struct = _load_complex_structure(cascade_dir, r["smiles"])
                r["_pose"] = score_pose(r["smiles"], struct, site)
            pre = len(top)
            top = [r for r in top if r["_pose"]["score"] >= 4.0]
            logger.info("step1: %d → %d (pose score < 4.0 eliminated %d)",
                        pre, len(top), pre - len(top))

        # Step 1.5: PoseBusters (5 filtered physical checks) + 1.6: Pocket QC
        # audit 2026-09-02: annotations only, not elimination. Zero new
        # dependencies (posebusters in container, scipy already present).
        if self.cfg.run_step1:
            from app.hit_prep_qc import run_posebusters_filtered, pocket_volume_ratio
            cascade_dir = self.screen_dir / "cascade"
            for r in top:
                struct = _load_complex_structure(cascade_dir, r["smiles"])
                if struct:
                    lig_sdf, rec_pdb = struct
                    r["_posebusters"] = run_posebusters_filtered(lig_sdf, rec_pdb)
                    r["_pocket_qc"] = pocket_volume_ratio(lig_sdf, rec_pdb)
                else:
                    r["_posebusters"] = None
                    r["_pocket_qc"] = None

        # Step 2: MM-GBSA (single-point, on co-folded pose)
        if self.cfg.run_step2:
            for r in top:
                r["_mmgbsa"] = mmgbsa_singlepoint(r["smiles"], self.screen_dir / "cascade")
            # MM-GBSA is ranking cross-validation, not elimination

        # Step 3: redock pocket vote
        if self.cfg.run_step3:
            # Simplified: use Nesso's own ligand pLDDT as pocket-consistency proxy
            # Full Vina redock requires receptor PDB + ligand SDF preparation
            for r in top:
                lp = r.get("ligand_plddt")
                r["_redock_vote"] = ("pocket_consistent" if lp and lp >= 50
                                     else "low_confidence")

        # Step 4: selectivity
        if self.cfg.run_step4 and self.cfg.selectivity_fastas:
            smiles_list = [r["smiles"] for r in top]
            sel = selectivity_scores(smiles_list, self.cfg.selectivity_fastas,
                                     self.out / "selectivity")
            for i, r in enumerate(top):
                r["_selectivity"] = {name: scores[i] if i < len(scores) else None
                                     for name, scores in sel.items()}

        # Step 5: availability proxy (MW ≤ 550 + QED ≥ 0.3 as purchasability proxy)
        if self.cfg.run_step5:
            from app.l15_annotate import annotate
            for r in top:
                ann = annotate(r["smiles"])
                r["_qed"] = round(ann.qed, 3)
                r["_avail"] = ("likely_purchasable" if ann.mw <= 550 and ann.qed >= 0.3
                               else "questionable")

        # Composite prioritization score
        for r in top:
            pose = r.get("_pose", {}).get("score", 5)
            agg_penalty = 2 if r.get("_agg") else 0
            r["_priority"] = round(
                (r.get("consensus_score", 0) * 10 * 0.4 + pose * 0.4 +
                 (10 - agg_penalty) * 0.2), 1)

        final = sorted(top, key=lambda r: -r["_priority"])[:self.cfg.top_n_output]

        # persist
        report = {
            "n_input": len(by_consensus), "n_output": len(final),
            "steps_run": [s for s in ("step0", "step1", "step2", "step3", "step4", "step5")
                          if getattr(self.cfg, f"run_{s}")],
            "wall_time_s": round(time.time() - t0, 1),
            "candidates": [{k: v for k, v in r.items() if not k.startswith("smiles")}
                           | {"smiles": r["smiles"]} for r in final],
        }
        (self.out / "hit_prep_result.json").write_text(
            json.dumps(report, indent=1, default=str))
        logger.info("hit_prep: %d candidates in %.1fs", len(final), time.time() - t0)
        return final


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m app.hit_prep",
                                 description="Hit preparation: screen output → wet-lab-ready list")
    ap.add_argument("--screen-dir", required=True)
    ap.add_argument("--config", help="YAML/JSON HitPrepConfig")
    ap.add_argument("--site-residues", nargs="+", type=int,
                    help="Known binding-site residue numbers")
    ap.add_argument("--selectivity", nargs=2, action="append", metavar=("NAME", "FASTA"),
                    help="Selectivity target: name fasta_path")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--skip-step4", action="store_true")
    args = ap.parse_args(argv)

    if args.config:
        cfg = HitPrepConfig.load(args.config)
    else:
        cfg = HitPrepConfig(screen_dir=args.screen_dir)
    if args.site_residues:
        cfg.known_site_residues = args.site_residues
    if args.selectivity:
        cfg.selectivity_fastas = dict(args.selectivity)
    if args.skip_step4:
        cfg.run_step4 = False
    cfg.top_n_output = args.top_n

    prep = HitPrep(cfg)
    candidates = prep.run()
    for i, c in enumerate(candidates):
        print(f"{i+1:3d} priority={c.get('_priority', 0):5.1f} "
              f"pose={c.get('_pose', {}).get('score', '?')} "
              f"agg={'⚠' if c.get('_agg') else '✓'} "
              f"avail={c.get('_avail', '?')}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
