"""
DrugCLIP scoring via LMDB + DataLoader inference.

Uses task.load_mols_dataset_new / load_pockets_dataset for proper
atom tokenization, distance/edge-type computation — not the earlier
hand-written encoder that produced incorrect EF@1%=0 on DUD-E.

Reference: Gao et al., NeurIPS 2023 / Jia et al., Science 2026.
"""
from __future__ import annotations

import logging
import os
import pickle
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger("cascade.drugclip")

_DRUGCLIP_HOME = Path(__file__).resolve().parent.parent / "drugclip"
_CHECKPOINT = str(_DRUGCLIP_HOME / "checkpoints" / "ProFSADB" / "checkpoint_best.pt")


@dataclass
class DrugCLIPResult:
    smiles: str
    score: float


class DrugCLIPScorer:
    """Load DrugCLIP model(s), expose LMDB-based scoring API."""

    _instance: Optional["DrugCLIPScorer"] = None

    def __init__(self, checkpoint=None, device="cuda", use_ensemble=True, 
                 ensemble_dir: str = "8_folds"):
        self.use_ensemble = use_ensemble
        self.ensemble_dir = ensemble_dir
        if use_ensemble:
            fold_dir = _DRUGCLIP_HOME / "checkpoints" / ensemble_dir
            self._fold_paths = sorted(fold_dir.glob("fold_*.pt"))
            if not self._fold_paths:
                logger.warning("No %s ckpts, trying 8_folds", ensemble_dir)
                fold_dir = _DRUGCLIP_HOME / "checkpoints" / "8_folds"
                self._fold_paths = sorted(fold_dir.glob("fold_*.pt"))
            if not self._fold_paths:
                logger.warning("No 8-fold ckpts either, using single"); self.use_ensemble = False
        else:
            self._fold_paths = []
        self.checkpoint = checkpoint or _CHECKPOINT
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self._models = []
        self._tasks = []
        self._model = None
        self._task = None

    @property
    def model(self):
        if self._model is None: self._load()
        return self._model

    @property
    def task(self):
        if self._task is None: self._load()
        return self._task

    @property
    def n_folds(self):
        return len(self._models) if self._models else 1

    def _load(self):
        if self.use_ensemble:
            self._load_ensemble()
        else:
            self._load_single(self.checkpoint)

    def _load_ensemble(self):
        logger.info("Loading %d-fold ensemble", len(self._fold_paths))
        for i, p in enumerate(self._fold_paths):
            m, t = self._load_single(str(p), verbose=(i == 0))
            self._models.append(m); self._tasks.append(t)
            if i == 0: self._model, self._task = m, t
        logger.info("%d-fold loaded on %s", len(self._models), self.device)

    def _load_single(self, ckpt_path, verbose=True):
        if verbose: logger.info("Loading %s", ckpt_path)
        parent = str(_DRUGCLIP_HOME)
        if parent in sys.path: sys.path.remove(parent)
        sys.path.append(parent)
        from unicore import checkpoint_utils, tasks
        import unimol.tasks, unimol.models  # noqa
        state = checkpoint_utils.load_checkpoint_to_cpu(ckpt_path)
        cfg = state["args"]
        cfg.task = "drugclip"; cfg.data = str(_DRUGCLIP_HOME / "dict")
        # Map checkpoint arch names to registered model names
        _ARCH_MAP = {"binding_affinity": "drugclip", "unimol": "drugclip"}
        if cfg.arch in _ARCH_MAP:
            cfg.arch = _ARCH_MAP[cfg.arch]
        elif cfg.arch not in ("drugclip",):
            cfg.arch = "drugclip"
        cfg.finetune_mol_model = None; cfg.finetune_pocket_model = None
        task = tasks.setup_task(cfg)
        model = task.build_model(cfg)
        model.load_state_dict(state["model"], strict=False); model.eval()
        if self.device == "cuda": model.cuda()
        if not self.use_ensemble: self._model, self._task = model, task
        return model, task

    # ── LMDB builders ──────────────────────────────────────────────



    def _mol_smiles_to_lmdb(self, smiles_list, lmdb_path):
        import lmdb
        from rdkit import Chem
        from rdkit.Chem import AllChem
        # RDKit UFFTYPER cannot parameterize metals (Co/Ca/Fe...) and
        # EmbedMolecule hangs in an infinite C++ loop on them. Exclude.
        ORGANIC = {"C","N","O","S","P","F","Cl","Br","I","H","B","Si"}
        env = lmdb.open(lmdb_path, subdir=False, map_size=10*1024*1024*1024)
        txn = env.begin(write=True)
        n = len(smiles_list)
        if n >= 256:
            # parallel embed (audit 2026-08-31: 23.7k-mol run used 1 of 32
            # threads for ~50 min; pool cuts this ~10x, CPU-only, zero VRAM)
            import multiprocessing as mp
            results = {}
            with mp.Pool(min(mp.cpu_count(), 16)) as pool:
                for idx, data in pool.imap_unordered(
                        _embed_worker, enumerate(smiles_list), chunksize=64):
                    results[idx] = data
            serial_iter = results.items()
        else:
            serial_iter = ((_embed_worker((i, s))) for i, s in enumerate(smiles_list))
        for idx, data in serial_iter:
            smi = smiles_list[idx]
            if data is None: continue
            txn.put(str(idx).encode(), pickle.dumps(data))
        txn.commit(); env.close()

    def _pocket_pdb_to_lmdb(self, pocket_pdb, lmdb_path):
        import lmdb
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True)
        s = parser.get_structure("p", pocket_pdb)
        atoms, coords = [], []
        for a in s.get_atoms():
            e = a.element.strip()
            if not e or e == "H": continue
            atoms.append(e); coords.append(list(a.get_vector()))
        if not atoms: raise ValueError("no atoms in " + str(pocket_pdb))
        env = lmdb.open(lmdb_path, subdir=False, map_size=1024*1024*1024)
        txn = env.begin(write=True)
        txn.put(b"0", pickle.dumps({
            "pocket": Path(pocket_pdb).stem,
            "pocket_atoms": np.array(atoms, dtype=object),
            "pocket_coordinates": np.array(coords, dtype=np.float32)}))
        txn.commit(); env.close()

    # ── Inference ──────────────────────────────────────────────────

    @torch.no_grad()
    def encode_pocket(self, pocket_pdb: str | Path) -> np.ndarray:
        import sys
        _dp = str(Path(__file__).resolve().parent.parent / "drugclip")
        if _dp not in sys.path: sys.path.append(_dp)
        import unicore
        models = self._models if self.use_ensemble and self._models else [self.model]
        tasks_list = self._tasks if self.use_ensemble and self._tasks else [self.task]
        fold_embs = []
        for model, task in zip(models, tasks_list):
            with tempfile.TemporaryDirectory() as td:
                lp = os.path.join(td, "p.lmdb")
                self._pocket_pdb_to_lmdb(str(pocket_pdb), lp)
                ds = task.load_pockets_dataset(lp)
                dl = torch.utils.data.DataLoader(ds, batch_size=1, collate_fn=ds.collater)
                for sample in dl:
                    if self.device == "cuda": sample = unicore.utils.move_to_cuda(sample)
                    st = sample["net_input"]["pocket_src_tokens"]
                    dist = sample["net_input"]["pocket_src_distance"]
                    et = sample["net_input"]["pocket_src_edge_type"]
                    mask = st.eq(model.pocket_model.padding_idx)
                    x = model.pocket_model.embed_tokens(st)
                    n = dist.size(-1)
                    g = model.pocket_model.gbf(dist, et)
                    go = model.pocket_model.gbf_proj(g)
                    attn = go.permute(0,3,1,2).contiguous().view(-1,n,n)
                    out = model.pocket_model.encoder(x, padding_mask=mask, attn_mask=attn)
                    e = model.pocket_project(out[0][:,0,:])
                    e = e / e.norm(dim=-1, keepdim=True)
                    fold_embs.append(e.cpu().numpy())
        if not fold_embs: return np.zeros(128, dtype=np.float32)
        r = np.mean(np.concatenate(fold_embs), axis=0)
        n = np.linalg.norm(r); return r / n if n else r

    @torch.no_grad()
    def encode_molecules_from_lmdb(self, lmdb_path: str) -> dict[str, np.ndarray]:
        """Encode molecules directly from a pre-built DUD-E mols.lmdb.
        
        Skips RDKit conformer generation by loading pre-computed atom coordinates
        from LMDB. ~100x faster than encode_molecules() for large libraries.
        """
        import lmdb, pickle
        import sys
        _dp = str(Path(__file__).resolve().parent.parent / "drugclip")
        if _dp not in sys.path: sys.path.append(_dp)
        import unicore
        
        # Load all molecules from LMDB
        env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False)
        all_data = []
        all_smiles = []
        with env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                data = pickle.loads(value)
                if isinstance(data, dict):
                    all_data.append(data)
                    all_smiles.append(data.get("smi", str(key)))
        env.close()
        
        if not all_data:
            return {}
        
        models = self._models if self.use_ensemble and self._models else [self.model]
        tasks_list = self._tasks if self.use_ensemble and self._tasks else [self.task]
        fold_accum = {}
        
        for fi, (model, task) in enumerate(zip(models, tasks_list)):
            with tempfile.TemporaryDirectory() as td:
                lp = os.path.join(td, "m.lmdb")
                # Write LMDB from pre-computed data
                self._write_lmdb_from_dicts(all_data, all_smiles, lp)
                ds = task.load_mols_dataset_new(lp, "atoms", "coordinates")
                dl = torch.utils.data.DataLoader(ds, batch_size=64, collate_fn=ds.collater)
                for sample in dl:
                    if self.device == "cuda": sample = unicore.utils.move_to_cuda(sample)
                    batch_smis = sample.get("smi_name", [])
                    st = sample["net_input"]["mol_src_tokens"]
                    dist = sample["net_input"]["mol_src_distance"]
                    et = sample["net_input"]["mol_src_edge_type"]
                    mask = st.eq(model.mol_model.padding_idx)
                    x = model.mol_model.embed_tokens(st)
                    n = dist.size(-1)
                    g = model.mol_model.gbf(dist, et)
                    go = model.mol_model.gbf_proj(g)
                    attn = go.permute(0,3,1,2).contiguous().view(-1,n,n)
                    out = model.mol_model.encoder(x, padding_mask=mask, attn_mask=attn)
                    e = model.mol_project(out[0][:,0,:])
                    e = e / e.norm(dim=-1, keepdim=True)
                    en = e.cpu().numpy()
                    for j, s in enumerate(batch_smis):
                        if isinstance(s, bytes): s = s.decode()
                        if fi == 0: fold_accum[s] = [en[j]]
                        elif s in fold_accum: fold_accum[s].append(en[j])
        return {k: np.mean(v, axis=0) for k, v in fold_accum.items()}
    
    def _write_lmdb_from_dicts(self, data_list, smiles_list, lmdb_path):
        """Write LMDB from pre-computed molecule dicts (no RDKit)."""
        import lmdb, pickle
        env = lmdb.open(lmdb_path, subdir=False, map_size=10*1024*1024*1024)
        txn = env.begin(write=True)
        for idx, data in enumerate(data_list):
            smi = smiles_list[idx]
            if "atoms" not in data or "coordinates" not in data:
                continue
            record = {
                "smi": smi, "atoms": np.array(data["atoms"], dtype=object),
                "coordinates": np.array(data["coordinates"], dtype=np.float32),
                "name": smi[:50], "IDs": smi, "subset": "test",
            }
            txn.put(str(idx).encode(), pickle.dumps(record))
        txn.commit(); env.close()

    @torch.no_grad()
    def encode_molecules(self, smiles_list: list[str], batch_size: int = 64) -> dict[str, np.ndarray]:
        import sys
        _dp = str(Path(__file__).resolve().parent.parent / "drugclip")
        if _dp not in sys.path: sys.path.append(_dp)
        import unicore
        models = self._models if self.use_ensemble and self._models else [self.model]
        tasks_list = self._tasks if self.use_ensemble and self._tasks else [self.task]
        fold_accum = {}
        for fi, (model, task) in enumerate(zip(models, tasks_list)):
            with tempfile.TemporaryDirectory() as td:
                lp = os.path.join(td, "m.lmdb")
                self._mol_smiles_to_lmdb(smiles_list, lp)
                ds = task.load_mols_dataset_new(lp, "atoms", "coordinates")
                dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, collate_fn=ds.collater)
                for sample in dl:
                    if self.device == "cuda": sample = unicore.utils.move_to_cuda(sample)
                    batch_smis = sample.get("smi_name", [])
                    st = sample["net_input"]["mol_src_tokens"]
                    dist = sample["net_input"]["mol_src_distance"]
                    et = sample["net_input"]["mol_src_edge_type"]
                    mask = st.eq(model.mol_model.padding_idx)
                    x = model.mol_model.embed_tokens(st)
                    n = dist.size(-1)
                    g = model.mol_model.gbf(dist, et)
                    go = model.mol_model.gbf_proj(g)
                    attn = go.permute(0,3,1,2).contiguous().view(-1,n,n)
                    out = model.mol_model.encoder(x, padding_mask=mask, attn_mask=attn)
                    e = model.mol_project(out[0][:,0,:])
                    e = e / e.norm(dim=-1, keepdim=True)
                    en = e.cpu().numpy()
                    for j, s in enumerate(batch_smis):
                        if isinstance(s, bytes): s = s.decode()
                        key = s
                        if fi == 0: fold_accum[key] = [en[j]]
                        elif key in fold_accum: fold_accum[key].append(en[j])
        return {k: np.mean(v, axis=0) for k, v in fold_accum.items()}

    def score_library(self, pocket_pdb, smiles_list):
        pe = self.encode_pocket(pocket_pdb)
        me = self.encode_molecules(smiles_list)
        results = []
        for smi, emb in me.items():
            results.append(DrugCLIPResult(smiles=smi, score=float(np.dot(pe, emb))))
        results.sort(key=lambda r: r.score, reverse=True)
        return results


    def screen_with_fpocket(
        self, receptor_pdb: str | Path, smiles_list: list[str],
        sphere_radius: float = 10.0, top_n_pockets: int = 5,
        min_quality: float = 0.20,
    ) -> list[DrugCLIPResult]:
        """Full screening pipeline: fpocket → quality filter → DrugCLIP multi-pocket scoring.

        Uses PQR vertex analysis (6A residue extraction) + structural quality filtering,
        then max-pooling across top pockets. Equivalent to DrugCLIP paper's fpocket+GenPack
        level when structure quality is adequate.

        1. Run fpocket on receptor PDB → detect pockets via PQR
        2. Extract 6A pocket residues, compute quality score
        3. Filter by min_quality/min_volume/min_residues
        4. Encode each quality pocket + all SMILES (8-fold ensemble)
        5. For each compound, take MAX score across all pockets (max pooling)
        6. Return ranked list
        """
        from app.drugclip_pocket import extract_pockets_from_pdb

        pocket_pdbs = extract_pockets_from_pdb(
            receptor_pdb, sphere_radius=sphere_radius, top_n=top_n_pockets,
            min_quality=min_quality,
        )
        if not pocket_pdbs:
            logger.warning("No fpocket pockets found; using full receptor as fallback")
            pocket_pdbs = [Path(receptor_pdb)]

        logger.info("DrugCLIP screening with %d pocket(s)", len(pocket_pdbs))

        # Encode all pockets (8-fold ensemble mean pooling)
        pocket_embs = {}
        for pp in pocket_pdbs:
            pocket_embs[str(pp)] = self.encode_pocket(str(pp))

        # Encode all molecules once (8-fold ensemble mean pooling)
        mol_embs = self.encode_molecules(smiles_list)

        # Score: max pooling across all pockets (per DrugCLIP paper)
        results = {}
        for smi, mol_emb in mol_embs.items():
            best_score = -float("inf")
            for pe in pocket_embs.values():
                score = float(np.dot(pe, mol_emb))
                if score > best_score:
                    best_score = score
            results[smi] = best_score

        ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)
        return [DrugCLIPResult(smiles=smi, score=score) for smi, score in ranked]


def get_drugclip_scorer() -> DrugCLIPScorer:
    if DrugCLIPScorer._instance is None:
        DrugCLIPScorer._instance = DrugCLIPScorer()
    return DrugCLIPScorer._instance


def _embed_worker(args):
    """Conformer embed + heavy-atom extraction for one SMILES (pool worker).

    Deterministic (randomSeed=42); MMFF deliberately skipped (UFFTYPER hang
    lesson). Returns (idx, data_dict | None)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import numpy as np
    idx, smi = args
    ORGANIC = {"C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "H", "B", "Si"}
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return idx, None
    if any(a.GetSymbol() not in ORGANIC for a in mol.GetAtoms()):
        return idx, None
    mol = Chem.AddHs(mol)
    try:
        if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
            AllChem.Compute2DCoords(mol)
    except Exception:
        try:
            AllChem.Compute2DCoords(mol)
        except Exception:
            return idx, None
    conf = mol.GetConformer()
    atoms, coords_3d = [], []
    for a in mol.GetAtoms():
        s = a.GetSymbol()
        if s == "H":
            continue
        atoms.append(s)
        p = conf.GetAtomPosition(a.GetIdx())
        coords_3d.append([p.x, p.y, p.z])
    if not atoms:
        return idx, None
    return idx, {"smi": smi, "atoms": np.array(atoms, dtype=object),
                 "coordinates": np.array(coords_3d, dtype=np.float32),
                 "name": smi[:50], "IDs": smi, "subset": "test"}

