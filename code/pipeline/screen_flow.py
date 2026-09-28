"""Canonical screening flow — consolidates the tmp-script paradigm into one
audited, fully-persisting pipeline stage set (W3, industrialization P0-3/P0-4).

Every run produces:
  manifest.json        — config/code/model fingerprint (app/run_manifest.py)
  heartbeat.jsonl      — stage/progress/utime/GPU samples + stall warnings
  l1_scores.json       — full-library L1 scores (audit iron rule)
  l1_ranked.json       — ranking
  topk_decision.json   — QC gate audit
  results.jsonl        — PER-MOLECULE full signals: six gate signals raw +
                         affinity + consensus + failure class + is_control
  summary.json         — rankings (consensus primary), positive ranks, Z',
                         hit-list annotations, diversity picks

Codified lessons: metal filter + strictParsing=False (UFFTYPER/GBK hang),
L1 receptor cache pinning (S3), positive force-injection, consensus default
ranking (G3), Z' assay window (E1.2), annotate-never-filter hit lists (S4).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

ORGANIC = {"C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "H", "B", "Si"}


# ── failure classification (P0-4) ────────────────────────────────────────
def classify_failure(r) -> str | None:
    """None when the molecule produced a score; otherwise a failure class.

    Classes observed in production runs:
      gate_blocked — no usable leg (all parked/failed), no score
      scorer_error — fold ok but Nesso/pose step failed (e.g. Bad Conformer Id)
      no_pose      — legs ran but produced no scored pose
    """
    if getattr(r, "best_boltz", None) is not None:
        return None
    err = (getattr(r, "error", None) or "").lower()
    if "conformer" in err:
        return "scorer_error"
    if getattr(r, "status", "") == "parked":
        return "gate_blocked"
    if err:
        return "scorer_error"
    return "no_pose"


@dataclass
class ScreenConfig:
    run_id: str
    target_name: str
    fasta: str                       # path to receptor FASTA
    library_sdfs: list[str] = field(default_factory=list)  # SDF bypass path
    library_ids: list[str] | None = None  # registered ids ("L5610", "L5610@v2")
    positives: dict[str, str] = field(default_factory=dict)   # name -> SMILES
    positive_history: dict[str, list[float]] = field(default_factory=dict)
    library_limit: int = 0           # 0 = all (demo runs may subsample)
    cofold_batch_k: int = 20         # top-K molecules sent to co-fold
    n_null_control: int = 6          # assumed-non-binder sample for Z'
    seed: int = 42
    num_loops: int = 8
    min_quality: float = 0.20
    # cascade gate thresholds (relaxed production set, USER_GUIDE §4.6)
    min_pocket_plddt: float = 70.0
    min_ligand_plddt: float = 60.0
    max_interface_pae: float = 10.0
    min_ligand_iptm: float = 0.5

    @classmethod
    def load(cls, path: str | Path) -> "ScreenConfig":
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if p.suffix in (".yaml", ".yml"):
            import yaml
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return cls(**data)

    def resolved(self) -> dict:
        d = asdict(self)
        d["fasta"] = str(self.fasta)
        return d


def load_library(sdf_paths: list[str], limit: int = 0) -> tuple[list[str], dict]:
    """Codified loader: strictParsing=False + metal filter + prop extraction."""
    from rdkit import Chem
    from rdkit.Chem import MolToSmiles
    smiles, meta = [], {"n_files": 0, "n_fail": 0, "n_metal": 0}
    seen = set()
    for sp in sdf_paths:
        meta["n_files"] += 1
        suppl = Chem.SDMolSupplier(sp, sanitize=False, strictParsing=False)
        for m in suppl:
            if m is None:
                meta["n_fail"] += 1
                continue
            if m.HasProp("SMILES"):
                smi = m.GetProp("SMILES")
            else:
                # New libraries carry no SMILES prop: sanitize a copy before
                # writing SMILES or aromatic rings fail re-parse (kekulize).
                mm = Chem.Mol(m)
                try:
                    Chem.SanitizeMol(mm)
                    smi = MolToSmiles(mm)
                except Exception:
                    smi = MolToSmiles(m)
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                meta["n_fail"] += 1
                continue
            if any(a.GetSymbol() not in ORGANIC for a in mol.GetAtoms()):
                meta["n_metal"] += 1
                continue
            if smi in seen:
                continue
            seen.add(smi)
            smiles.append(smi)
            if limit and len(smiles) >= limit:
                return smiles, meta
    return smiles, meta


class ScreenFlow:
    def __init__(self, config: ScreenConfig, out_root: str | Path = "/workspace/output"):
        self.cfg = config
        self.out = Path(out_root) / f"screen_{config.run_id}"
        self.out.mkdir(parents=True, exist_ok=True)

    # -- helpers -----------------------------------------------------------
    def _append_jsonl(self, name: str, record: dict):
        with open(self.out / name, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def _gate_signals(self, r) -> dict:
        for legs in getattr(r, "legs", {}).values():
            for lr in legs:
                if lr.gate:
                    g = lr.gate
                    return {"gate_passed": g.passed,
                            "pocket_plddt": g.pocket_plddt,
                            "ligand_plddt": g.ligand_plddt,
                            "interface_pae": g.interface_pae,
                            "ligand_iptm": g.ligand_iptm,
                            "pose_rmsd_max": g.pose_rmsd_max}
        return {}

    # -- main ---------------------------------------------------------------
    def run(self) -> dict:
        from app.run_manifest import write_manifest
        from app.run_monitor import RunMonitor

        cfg = self.cfg
        seq = "".join(open(cfg.fasta).read().splitlines()[1:]).strip()
        write_manifest(self.out / "manifest.json", run_id=cfg.run_id,
                       config=cfg.resolved(),
                       inputs={"fasta": cfg.fasta, **{f"lib{i}": p for i, p in enumerate(cfg.library_sdfs)}})
        mon = RunMonitor(self.out, interval_s=30, stall_min=10)
        mon.start(stage="library_load")

        # 1. library + controls (registry path preferred; SDF bypass kept)
        lib_of: dict[str, str] = {}
        if cfg.library_ids:
            from app.library_registry import LibraryRegistry, load_cached
            paths, metas = LibraryRegistry().resolve_ids(cfg.library_ids)
            smiles, lib_of = load_cached(paths)
            lib_meta = {"mode": "registry", "n_libs": len(paths),
                        "libs": [f"{m['lib_id']}@{m['version']}:{m['n_molecules']}"
                                 for m in metas]}
        else:
            smiles, lib_meta = load_library(cfg.library_sdfs, cfg.library_limit)
        pos_by_smiles = {smi: name for name, smi in cfg.positives.items()}
        import numpy as np
        rng = np.random.default_rng(cfg.seed)
        null_idx = list(rng.choice(len(smiles), min(cfg.n_null_control, len(smiles)), replace=False))
        null_smiles = [smiles[i] for i in null_idx]
        mon.log_event("library_loaded", **lib_meta, n_unique=len(smiles))

        # 2. L1 (receptor cache pinned via workdir reuse, S3; results cached
        #    for pause/resume — audit 2026-09-01: resume without this redid
        #    the full-library encode for ~45 min)
        l1_scores = l1_ranked = None
        if (self.out / "l1_scores.json").exists() and (self.out / "l1_ranked.json").exists():
            l1_scores = json.loads((self.out / "l1_scores.json").read_text())
            l1_ranked = json.loads((self.out / "l1_ranked.json").read_text())
            mon.log_event("L1_reused_from_cache", n=len(l1_scores))
            t_l1 = 0.0
        if l1_scores is None:
            mon.start(stage="L1_drugclip")
            from app.l1_prescreen import L1DockingPreScreen
            t0 = time.time()
            l1 = L1DockingPreScreen(use_gpu=True, seed=cfg.seed)
            l1_res = l1.screen_library(
                seq, smiles + list(cfg.positives.values()) + null_smiles,
                workdir=self.out / "l1",
                method="drugclip", top_fraction=1.0, drugclip_ensemble="6_folds",
                receptor_pdb="", pocket_pdb="", top_n_pockets=3,
                min_quality=cfg.min_quality, num_loops=cfg.num_loops,
                num_sampling_steps=32)
            l1_scores = {r.smiles: float(r.cnn_affinity) for r in l1_res.all_results}
            (self.out / "l1_scores.json").write_text(json.dumps(l1_scores, indent=1))
            ranked = [r.smiles for r in sorted(l1_res.all_results, key=lambda r: -r.cnn_affinity)]
            (self.out / "l1_ranked.json").write_text(json.dumps(ranked, indent=1))
            t_l1 = time.time() - t0
        else:
            ranked = l1_ranked
        mon.log_event("L1_done", n=len(l1_scores), time_s=round(t_l1, 1))

        # 3. adaptive top-K (audit record; co-fold batch capped by config)
        from app.topk_selector import TopKSelector
        pos_rank = next((ranked.index(s) for s in cfg.positives.values() if s in ranked), None)
        dec = TopKSelector().select([l1_scores[s] for s in ranked if s in l1_scores], pos_rank=pos_rank)
        (self.out / "topk_decision.json").write_text(json.dumps(
            {"method": dec.method, "top_k": dec.top_k, "qc_gates": dec.qc_gates,
             "pos_rank": pos_rank}, indent=2, default=str))

        # 4. co-fold batch: top-K' + positives + null controls
        batch, tags = [], {}
        for s in ranked:
            if len([b for b in batch if tags.get(b) == "hit"]) >= cfg.cofold_batch_k:
                break
            if s in pos_by_smiles or s in null_smiles:
                continue
            batch.append(s); tags[s] = "hit"
        for name, s in cfg.positives.items():
            if s not in batch:
                batch.append(s)
            tags[s] = f"positive:{name}"
        for s in null_smiles:
            if s not in batch:
                batch.append(s)
            tags.setdefault(s, "null_control")
        mon.start(stage="cofold_nesso", n_batch=len(batch))

        from app.cascade import ScreeningCascade
        from app.cascade_config import CascadeConfig
        ccfg = CascadeConfig()
        ccfg.use_boltz = True; ccfg.run_l1 = False
        ccfg.run_modes = ["pocket_off"]; ccfg.mmgbsa_selection = "none"
        ccfg.workdir = self.out / "cascade"
        ccfg.min_pocket_plddt = cfg.min_pocket_plddt
        ccfg.min_ligand_plddt = cfg.min_ligand_plddt
        ccfg.max_interface_pae = cfg.max_interface_pae
        ccfg.min_ligand_iptm = cfg.min_ligand_iptm
        ccfg.run_posebusters = False
        # streaming per-molecule co-fold with pause/resume skip
        # (audit 2026-09-01: cascade has NO fold-level resume despite
        # USER_GUIDE §5.3 — this makes screen-level resume real and gives
        # live progress instead of end-of-run-only persistence)
        done = set()
        res_path = self.out / "results.jsonl"
        if res_path.exists():
            for line in open(res_path, encoding="utf-8"):
                try:
                    done.add(json.loads(line)["smiles"])
                except Exception:
                    continue
            mon.log_event("cofold_resume", n_done=len(done))
        cascade = ScreeningCascade(config=ccfg)
        rows = []
        t0 = time.time()
        n_done_now = 0
        for smi in batch:
            if smi in done:
                continue
            mol_res = cascade.run(smiles_list=[smi], protein_sequence=seq,
                                  seed=cfg.seed, num_loops=cfg.num_loops)
            r = next((x for x in mol_res.results if x.smiles == smi), None)
            if r is None:
                continue
            row = {"smiles": r.smiles, "tag": tags.get(r.smiles, "hit"),
                   "source_lib": lib_of.get(r.smiles, "sdf"),
                   "status": r.status,
                   "failure_class": classify_failure(r),
                   **self._gate_signals(r),
                   "affinity_binary": float(r.best_boltz.affinity_binary) if r.best_boltz else None,
                   "affinity_value": float(r.best_boltz.affinity_value) if r.best_boltz else None,
                   "consensus_score": r.consensus_score}
            rows.append(row)
            self._append_jsonl("results.jsonl", row)
            n_done_now += 1
            if n_done_now % 10 == 0:
                mon.log_event("cofold_progress", done=n_done_now + len(done),
                              total=len(batch))
        t_fold = time.time() - t0
        mon.log_event("cofold_done", n=len(batch), time_s=round(t_fold, 1))

        # rebuilt full rows view from jsonl (covers resumed molecules)
        rows = []
        if res_path.exists():
            for line in open(res_path, encoding="utf-8"):
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue

        scored = [x for x in rows if x["affinity_binary"] is not None]
        by_consensus = sorted(scored, key=lambda x: -(x["consensus_score"] or -1))
        by_binary = sorted(scored, key=lambda x: -x["affinity_binary"])

        # 6. Z' assay window: positives (this run + history) vs null controls
        z = None
        pos_scores = [x["affinity_binary"] for x in scored if str(x["tag"]).startswith("positive")]
        for name, hist in cfg.positive_history.items():
            pos_scores.extend(hist)
        null_scores = [x["affinity_binary"] for x in scored if x["tag"] == "null_control"]
        if len(pos_scores) >= 2 and len(null_scores) >= 2:
            from app.hitlist import z_prime
            z = round(z_prime(pos_scores, null_scores), 3)

        # 7. hit-list annotations + diversity pick (S4)
        from app.hitlist import annotate_hits, diversity_pick
        hits = [{"smiles": x["smiles"], "binary": x["affinity_binary"]} for x in scored if x["tag"] == "hit"]
        annotate_hits(hits)
        picks, skipped = diversity_pick([(h["smiles"], h["binary"] or 0.0) for h in hits],
                                        k=min(20, len(hits)))

        summary = {
            "run_id": cfg.run_id, "target": cfg.target_name,
            "library_meta": lib_meta,
            "timings": {"l1_s": round(t_l1, 1), "cofold_s": round(t_fold, 1)},
            "n_batch": len(batch), "n_scored": len(scored),
            "failure_classes": {c: sum(1 for x in rows if x["failure_class"] == c)
                                for c in {x["failure_class"] for x in rows if x["failure_class"]}},
            "ranking_consensus": [{"tag": x["tag"], "consensus": x["consensus_score"],
                                   "binary": x["affinity_binary"]} for x in by_consensus[:20]],
            "ranking_binary_top20": [{"tag": x["tag"], "binary": x["affinity_binary"]}
                                     for x in by_binary[:20]],
            "positive_final_rank": {x["tag"].split(":", 1)[1]: i + 1
                                    for i, x in enumerate(by_consensus)
                                    if str(x["tag"]).startswith("positive")},
            "z_prime": z,
            "diversity_picks": [p[0][:60] for p in picks],
            "diversity_skipped_reasons": [s[1][:60] for s in skipped],
            "hit_annotations": {h["smiles"][:60]: {"alerts": h["reactive_alerts"],
                                                   "mw": h["mw"], "ood": h["ood_risk"]}
                                for h in hits[:20]},
        }
        (self.out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        mon.start(stage="done")
        mon.stop()
        print(json.dumps({k: summary[k] for k in
                          ("run_id", "n_batch", "n_scored", "failure_classes", "z_prime")},
                         indent=1, default=str))
        return summary
