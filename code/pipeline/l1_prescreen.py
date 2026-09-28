"""
L1 — Compound library pre-screen.

Methods:
  - maccs: 2D MACCS fingerprint similarity to reference ligand (zero-cost)
  - rf:    RandomForest on ECFP4 fingerprints (needs pre-trained model)
  - gnina: Gnina docking against ESMFold2 receptor ensemble

Speed: maccs/rf = instant; gnina = ~1s/compound on GPU.
"""

from __future__ import annotations

import logging
import math
import pickle
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger("cascade.l1")


@dataclass
class L1Result:
    smiles: str
    cnn_affinity: float  # pK, higher = predicted stronger binding
    cnn_score: float  # pose quality 0-1
    vina_affinity: float  # kcal/mol
    receptor_idx: int  # which receptor conformer scored best


@dataclass
class L1ScreenResult:
    survivors: list[str]  # SMILES passing the pre-screen
    all_results: list[L1Result]  # full ranked list
    n_input: int
    n_survivors: int
    wall_time_s: float
    pocket_sites: list[list[int]] | None = (
        None  # pocket residue lists for L2 PocketConditioning
    )
    pocket_pdb: str | None = None  # path to refined pocket PDB (GenPack/fpocket)


def resolve_receptor_pdb(workdir: Path, pocket_pdb: str, receptor_pdb: str) -> tuple[str, str]:
    """Decide the structure source for DrugCLIP pocket detection.

    Returns (mode, path):
      ("pocket", p)  explicit pocket PDB — use directly
      ("receptor", p) explicit receptor PDB
      ("cached", p)  workdir/receptor.pdb from a previous run — PIN the fold
                     (audit 2026-08-31: re-folding per run made L1 scores
                      non-reproducible, Spearman 0.389 across runs; reusing
                      the cached fold pins pocket identity)
      ("fold", "")   no structure available — caller must fold the sequence
    """
    if pocket_pdb and Path(pocket_pdb).exists():
        return "pocket", pocket_pdb
    if receptor_pdb and Path(receptor_pdb).exists():
        return "receptor", receptor_pdb
    cached = Path(workdir) / "receptor.pdb"
    if cached.exists():
        return "cached", str(cached)
    return "fold", ""


class L1DockingPreScreen:
    """Ultra-fast structure-based pre-screening via Gnina docking."""

    def __init__(
        self,
        gnina_bin: str = "gnina",
        cnn_model: str = "fast",  # distilled CNN for speed
        exhaustiveness: int = 8,
        box_size: tuple = (22, 22, 22),
        use_gpu: bool = True,
        seed: int = 42,
    ):
        self.gnina_bin = gnina_bin
        self.cnn_model = cnn_model
        self.exhaustiveness = exhaustiveness
        self.box_size = box_size
        self.use_gpu = use_gpu
        self.seed = seed

    def prepare_receptor_ensemble(
        self,
        protein_sequence: str,
        workdir: Path,
        *,
        n_conformers: int = 3,
        num_loops: int = 2,
        num_sampling_steps: int = 32,
    ) -> list[dict]:
        """Fold apo protein N times → detect pockets → return receptor ensemble.

        Each entry: {pdb_path, pocket_centers [(x,y,z), ...], pocket_residues}
        """
        from app.models import (
            ChainInput,
            FoldingConfig,
            MoleculeType,
            PredictionRequest,
        )
        from app.local_inference import get_engine

        engine = get_engine()
        workdir.mkdir(parents=True, exist_ok=True)
        ensemble = []

        chains = [
            ChainInput(id="A", sequence=protein_sequence, type=MoleculeType.protein)
        ]
        config = FoldingConfig(
            num_loops=num_loops,
            num_sampling_steps=num_sampling_steps,
            num_diffusion_samples=1,
            seed=self.seed,
        )
        request = PredictionRequest(name="l1_apo", chains=chains, config=config)

        for i in range(n_conformers):
            logger.info(f"L1 apo fold {i + 1}/{n_conformers}")
            result = engine.predict(request)
            mmcif_text = result.mmcif

            # Write mmCIF, then convert to PDB for gnina compatibility
            apo_cif = workdir / f"apo_{i}.cif"
            apo_cif.write_text(mmcif_text)
            apo_pdb = workdir / f"apo_{i}.pdb"
            from app.mmcif_parser import write_mmcif_to_pdb

            write_mmcif_to_pdb(mmcif_text, apo_pdb)

            # Detect pockets via fpocket on the PDB (not mmCIF)
            pockets = self._detect_pockets_fpocket(apo_pdb, workdir)
            if pockets:
                ensemble.append(
                    {
                        "pdb_path": str(apo_pdb),
                        "pocket_centers": [p["center"] for p in pockets],
                        "n_atoms": [p["n_atoms"] for p in pockets],
                    }
                )

        logger.info(f"L1 receptor ensemble: {len(ensemble)} conformers with pockets")
        return ensemble

    def _detect_pockets_fpocket(self, pdb_path: Path, workdir: Path) -> list[dict]:
        """Run fpocket on a receptor PDB, return pocket centers from atom coordinates.

        Uses fpocket's pocketN_atm.pdb files to compute accurate geometric centers.
        """
        import subprocess

        fp_dir = workdir / f"{pdb_path.stem}_out"
        fp_dir.mkdir(parents=True, exist_ok=True)

        # Copy PDB to workdir and run fpocket
        input_copy = workdir / pdb_path.name
        if not input_copy.exists():
            import shutil

            shutil.copy(str(pdb_path), str(input_copy))

        try:
            result = subprocess.run(
                ["fpocket", "-f", str(input_copy.name)],
                cwd=str(workdir),
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                logger.warning("fpocket failed: %s", result.stderr[-200:])
                return []
        except Exception as e:
            logger.warning("fpocket error: %s", e)
            return []

        pockets_dir = fp_dir / "pockets"
        if not pockets_dir.exists():
            return []

        atm_files = sorted(
            pockets_dir.glob("pocket*_atm.pdb"),
            key=lambda p: int(p.name.replace("pocket", "").replace("_atm.pdb", "")),
        )
        if not atm_files:
            return []

        results = []
        for pf in atm_files:
            coords = []
            for line in pf.read_text().splitlines():
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        coords.append([x, y, z])
                    except (ValueError, IndexError):
                        continue
            if not coords:
                continue
            coords = np.array(coords)
            center = coords.mean(axis=0)
            results.append(
                {
                    "center": (float(center[0]), float(center[1]), float(center[2])),
                    "n_atoms": len(coords),
                }
            )
        return results[:5]  # top 5 pockets

    def screen_library(
        self,
        protein_sequence: str,
        smiles_list: list[str],
        workdir: Path | None = None,
        *,
        method: str = "rf",
        top_fraction: float = 0.20,
        reference_ligand_path: str = "",
        rf_model_path: str = "",
        # gnina-specific
        n_conformers: int = 3,
        num_loops: int = 2,
        num_sampling_steps: int = 32,
        # drugclip-specific (via kwargs)
        **kwargs,
    ) -> L1ScreenResult:
        """Screen a compound library against the protein.

        Parameters
        ----------
        method : str
            "maccs" | "rf" | "gnina" | "drugclip"
        top_fraction : float
            Fraction of compounds to keep (e.g. 0.20 = top 20%).
        reference_ligand_path : str
            Path to reference ligand MOL2/SDF for MACCS (or RF feature alignment).
        rf_model_path : str
            Path to pre-trained RandomForest pickle (required for rf method).
        drugclip_ensemble : str
            Ensemble dir for DrugCLIP ("6_folds" | "8_folds" | "single").
        receptor_pdb : str
            Path to receptor PDB for DrugCLIP fpocket pocket detection.
        pocket_pdb : str
            Path to pre-defined pocket PDB (overrides fpocket detection).
        """
        import time

        t0 = time.time()

        if workdir is None:
            workdir = Path(tempfile.mkdtemp(prefix="l1_"))
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

        if method == "maccs":
            result = self._screen_maccs(
                smiles_list, reference_ligand_path, top_fraction, t0
            )
        elif method == "rf":
            result = self._screen_rf(smiles_list, rf_model_path, top_fraction, t0)
        elif method == "gnina":
            result = self._screen_gnina(
                protein_sequence,
                smiles_list,
                workdir,
                top_fraction=top_fraction,
                n_conformers=n_conformers,
                num_loops=num_loops,
                num_sampling_steps=num_sampling_steps,
                t0=t0,
            )
        elif method == "drugclip":
            result = self._screen_drugclip(
                protein_sequence,
                smiles_list,
                workdir,
                top_fraction=top_fraction,
                receptor_pdb=kwargs.get("receptor_pdb", ""),
                pocket_pdb=kwargs.get("pocket_pdb", ""),
                ensemble_dir=kwargs.get("drugclip_ensemble", "6_folds"),
                top_n_pockets=kwargs.get("top_n_pockets", 3),
                min_quality=kwargs.get("min_quality", 0.20),
                num_loops=num_loops,
                num_sampling_steps=num_sampling_steps,
                t0=t0,
            )
        else:
            raise ValueError(f"Unknown L1 method: {method}")

        return result

    def _screen_maccs(
        self,
        smiles_list: list[str],
        reference_ligand_path: str,
        top_fraction: float,
        t0: float,
    ) -> L1ScreenResult:
        """MACCS fingerprint similarity to reference ligand."""
        import time
        from rdkit import Chem, RDLogger
        from rdkit.Chem import MACCSkeys, DataStructs

        RDLogger.DisableLog("rdApp.*")

        if not reference_ligand_path or not Path(reference_ligand_path).exists():
            raise FileNotFoundError(
                f"Reference ligand not found: {reference_ligand_path}"
            )

        # Load reference
        ref = Chem.MolFromMol2File(reference_ligand_path, removeHs=False)
        if ref is None:
            ref = Chem.SDMolSupplier(reference_ligand_path)[0]
        ref = Chem.RemoveHs(ref)
        fp_ref = MACCSkeys.GenMACCSKeys(ref)

        scores = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                scores.append(
                    DataStructs.TanimotoSimilarity(fp_ref, MACCSkeys.GenMACCSKeys(mol))
                )
            else:
                scores.append(0.0)

        # Sort and filter
        idx_sorted = np.argsort(scores)[::-1]
        n_keep = max(1, int(len(smiles_list) * top_fraction))
        survivors = [smiles_list[i] for i in idx_sorted[:n_keep]]

        all_results = [
            L1Result(
                smiles=smiles_list[i],
                cnn_affinity=scores[i],
                cnn_score=0,
                vina_affinity=0,
                receptor_idx=0,
            )
            for i in idx_sorted
        ]

        elapsed = time.time() - t0
        logger.info(
            f"L1 MACCS: {len(smiles_list)} → {len(survivors)} ({top_fraction:.0%}) in {elapsed:.1f}s"
        )
        return L1ScreenResult(
            survivors=survivors,
            all_results=all_results,
            n_input=len(smiles_list),
            n_survivors=len(survivors),
            wall_time_s=elapsed,
        )

    def _screen_rf(
        self,
        smiles_list: list[str],
        rf_model_path: str,
        top_fraction: float,
        t0: float,
    ) -> L1ScreenResult:
        """RandomForest on ECFP4 fingerprints."""
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem, DataStructs

        RDLogger.DisableLog("rdApp.*")

        if not rf_model_path or not Path(rf_model_path).exists():
            raise FileNotFoundError(
                f"RF model not found: {rf_model_path}. "
                "Train with _analysis_stat_ml.py first."
            )

        # Load pre-trained RF model
        with open(rf_model_path, "rb") as f:
            rf_model = pickle.load(f)

        # Generate ECFP4 fingerprints
        X = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                X.append(list(fp))
            else:
                X.append([0] * 2048)
        X = np.array(X)

        # Predict active probability
        scores = rf_model.predict_proba(X)[:, 1]

        idx_sorted = np.argsort(scores)[::-1]
        n_keep = max(1, int(len(smiles_list) * top_fraction))
        survivors = [smiles_list[i] for i in idx_sorted[:n_keep]]

        all_results = [
            L1Result(
                smiles=smiles_list[i],
                cnn_affinity=float(scores[i]),
                cnn_score=0,
                vina_affinity=0,
                receptor_idx=0,
            )
            for i in idx_sorted
        ]

        import time

        elapsed = time.time() - t0
        logger.info(
            f"L1 RF: {len(smiles_list)} → {len(survivors)} ({top_fraction:.0%}) in {elapsed:.1f}s"
        )
        return L1ScreenResult(
            survivors=survivors,
            all_results=all_results,
            n_input=len(smiles_list),
            n_survivors=len(survivors),
            wall_time_s=elapsed,
        )

    def _screen_gnina(
        self,
        protein_sequence: str,
        smiles_list: list[str],
        workdir: Path,
        top_fraction: float,
        n_conformers: int,
        num_loops: int,
        num_sampling_steps: int,
        t0: float,
    ) -> L1ScreenResult:
        """Gnina docking pre-screen (original method)."""
        import time

        if shutil.which(self.gnina_bin) is None:
            raise FileNotFoundError(f"{self.gnina_bin} not found on PATH")

        # Step 1: Build receptor ensemble
        ensemble = self.prepare_receptor_ensemble(
            protein_sequence,
            workdir,
            n_conformers=n_conformers,
            num_loops=num_loops,
            num_sampling_steps=num_sampling_steps,
        )

        if not ensemble:
            logger.warning(
                "L1: no pockets found in any apo conformer; returning all compounds"
            )
            return L1ScreenResult(
                survivors=smiles_list,
                all_results=[],
                n_input=len(smiles_list),
                n_survivors=len(smiles_list),
                wall_time_s=time.time() - t0,
            )

        # Step 2: Dock each compound sequentially
        all_results: list[L1Result] = []
        dock_dir = workdir / "docking"
        dock_dir.mkdir(exist_ok=True)

        for idx, smi in enumerate(smiles_list):
            if idx % max(1, len(smiles_list) // 10) == 0:
                logger.info(f"L1 docking: {idx}/{len(smiles_list)}")

            best_cnn = -float("inf")
            best_result = None

            for ri, rec in enumerate(ensemble):
                if not rec["pocket_centers"]:
                    continue
                for pi, (cx, cy, cz) in enumerate(rec["pocket_centers"]):
                    result = self._dock_single(
                        smi,
                        str(rec["pdb_path"]),
                        cx,
                        cy,
                        cz,
                        idx * 100 + pi,
                        dock_dir,
                    )
                    if result and result.cnn_affinity > best_cnn:
                        best_cnn = result.cnn_affinity
                        best_result = result
                        best_result.receptor_idx = ri

            if best_result:
                all_results.append(best_result)

        # Step 3: Sort and filter
        all_results.sort(key=lambda r: r.cnn_affinity, reverse=True)
        n_keep = max(1, int(len(smiles_list) * top_fraction))
        survivors = [r.smiles for r in all_results[:n_keep]]

        elapsed = time.time() - t0
        logger.info(
            f"L1 gnina: {len(smiles_list)} → {len(survivors)} compounds "
            f"({top_fraction:.1%}) in {elapsed:.1f}s"
        )

        return L1ScreenResult(
            survivors=survivors,
            all_results=all_results,
            n_input=len(smiles_list),
            n_survivors=len(survivors),
            wall_time_s=elapsed,
        )

    def _dock_single(
        self,
        smiles: str,
        receptor_path: str,
        cx: float,
        cy: float,
        cz: float,
        compound_idx: int,
        workdir: Path,
    ) -> L1Result | None:
        """Dock one compound to one receptor conformer."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        # Generate 3D conformer from SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        try:
            AllChem.EmbedMolecule(mol, randomSeed=self.seed)
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            try:
                AllChem.Compute2DCoords(mol)
            except Exception:
                return None

        lig_sdf = workdir / f"lig_{compound_idx}.sdf"
        try:
            Chem.MolToMolFile(mol, str(lig_sdf))
        except Exception:
            return None

        out_sdf = workdir / f"out_{compound_idx}.sdf"
        cmd = [
            self.gnina_bin,
            "-r",
            receptor_path,
            "-l",
            str(lig_sdf),
            "-o",
            str(out_sdf),
            "--cnn",
            self.cnn_model,
            "--cnn_scoring",
            "rescore",
            "--center_x",
            str(cx),
            "--center_y",
            str(cy),
            "--center_z",
            str(cz),
            "--size_x",
            str(self.box_size[0]),
            "--size_y",
            str(self.box_size[1]),
            "--size_z",
            str(self.box_size[2]),
            "--exhaustiveness",
            str(self.exhaustiveness),
            "--seed",
            str(self.seed),
        ]
        if not self.use_gpu:
            cmd.append("--no_gpu")

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(workdir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            return None
        except Exception as e:
            logger.debug(f"gnina dock failed for {smiles[:20]}: {e}")
            return None

        stdout = proc.stdout
        if proc.returncode != 0:
            return None

        # Parse gnina docking output (table format, not --score_only)
        cnn_aff = cnn_score = vina_aff = math.nan
        in_table = False
        for line in stdout.splitlines():
            s = line.strip()
            # --score_only format
            if s.startswith("CNNaffinity:"):
                cnn_aff = float(s.split()[-1])
            elif s.startswith("CNNscore:"):
                cnn_score = float(s.split()[-1])
            elif s.startswith("Affinity:"):
                vina_aff = float(s.split()[1])
            # Docking table format: "    N    -X.XX    -X.XX    0.XXXX    X.XXX"
            if s.startswith("-----+") and "-----" in s:
                in_table = True
                continue
            if in_table and s and s[0].isdigit():
                parts = s.split()
                if len(parts) >= 5:
                    try:
                        this_aff = float(parts[1])
                        this_cnn_score = float(parts[3])
                        this_cnn_aff = float(parts[4])
                        # Keep the pose with best CNN affinity
                        if math.isnan(cnn_aff) or this_cnn_aff > cnn_aff:
                            cnn_aff = this_cnn_aff
                            cnn_score = this_cnn_score
                            vina_aff = this_aff
                    except ValueError:
                        pass

        if math.isnan(cnn_aff):
            return None

        return L1Result(
            smiles=smiles,
            cnn_affinity=cnn_aff,
            cnn_score=cnn_score,
            vina_affinity=vina_aff,
            receptor_idx=0,
        )

    # ── DrugCLIP L0 prescreen ──────────────────────────────────────────

    def _screen_drugclip(
        self,
        protein_sequence: str,
        smiles_list: list[str],
        workdir: Path,
        *,
        top_fraction: float,
        receptor_pdb: str,
        pocket_pdb: str,
        ensemble_dir: str,
        top_n_pockets: int,
        min_quality: float,
        num_loops: int,
        num_sampling_steps: int,
        t0: float,
    ) -> L1ScreenResult:
        """DrugCLIP contrastive learning screening.

        If pocket_pdb is provided, uses it directly.
        Otherwise, folds protein with ESMFold2, runs fpocket, and uses
        best quality pocket. 6-fold ensemble + multi-pocket max pooling.
        """
        from app.drugclip_scorer import DrugCLIPScorer

        # Determine pocket (explicit > cached fold > fold sequence)
        mode, pdb_to_use = resolve_receptor_pdb(workdir, pocket_pdb, receptor_pdb)
        if mode == "fold":
            if protein_sequence:
                # Fold protein with ESMFold2.
                # C2 fix: use full co-folding loops (same engine as Leg 1/2)
                # so the self-bootstrapped L1 pocket is source-consistent with
                # L2 PocketConditioning. Previously used min(2, num_loops),
                # a low-quality fold that produced pockets mismatched with the
                # final complex structure. Disclosed in Methods.
                from app.models import (
                    ChainInput,
                    FoldingConfig,
                    MoleculeType,
                    PredictionRequest,
                )
                from app.local_inference import get_engine

                engine = get_engine()
                chains = [
                    ChainInput(
                        id="A", sequence=protein_sequence, type=MoleculeType.protein
                    )
                ]
                config = FoldingConfig(
                    num_loops=num_loops, num_sampling_steps=num_sampling_steps, seed=42
                )
                result = engine.predict(
                    PredictionRequest(name="apo_l1", chains=chains, config=config)
                )
                from app.mmcif_parser import write_mmcif_to_pdb

                pdb_path = workdir / "receptor.pdb"
                write_mmcif_to_pdb(result.mmcif, pdb_path)
                pdb_to_use = str(pdb_path)
                # free ESMFold2 weights before the DrugCLIP ensemble loads —
                # double residency pushed a shared desktop GPU to the OOM
                # edge on the 23.7k rerun (audit 2026-08-31)
                del engine, result
                from app.local_inference import unload_engine
                unload_engine()
                logger.info(
                    "DrugCLIP L1 self-bootstrapped pocket via ESMFold2 "
                    "co-folding fold (%d loops) + fpocket -> %s",
                    num_loops,
                    pdb_path,
                )
            else:
                raise ValueError("No receptor PDB or protein sequence for DrugCLIP")

        # Load DrugCLIP scorer
        scorer = DrugCLIPScorer(use_ensemble=True, ensemble_dir=ensemble_dir)
        _ = scorer.model
        logger.info(f"DrugCLIP loaded ({scorer.n_folds}-fold)")

        # Score via fpocket + multi-pocket max pooling
        if Path(pdb_to_use).exists() and not pocket_pdb:
            from app.drugclip_pocket import extract_pockets_from_pdb

            pocket_pdbs = extract_pockets_from_pdb(
                pdb_to_use,
                workdir=workdir / "fpocket",
                top_n=top_n_pockets,
                min_quality=min_quality,
            )
            if pocket_pdbs:
                mol_embs = scorer.encode_molecules(smiles_list)
                all_pocket_scores = []
                for pp in pocket_pdbs:
                    p_emb = scorer.encode_pocket(str(pp))
                    scores = np.array(
                        [
                            float(np.dot(p_emb, mol_embs.get(smi, np.zeros(128))))
                            for smi in smiles_list
                        ]
                    )
                    # MAD-Z normalize
                    med = np.median(scores)
                    mad = np.median(np.abs(scores - med))
                    if mad > 1e-10:
                        scores = 0.6745 * (scores - med) / mad
                    all_pocket_scores.append(scores)
                if all_pocket_scores:
                    final_scores = np.max(all_pocket_scores, axis=0)
                else:
                    final_scores = np.full(len(smiles_list), -1.0)
            else:
                # Fallback: use full receptor as pocket
                p_emb = scorer.encode_pocket(pdb_to_use)
                final_scores = np.array(
                    [
                        float(np.dot(p_emb, mol_embs.get(smi, np.zeros(128))))
                        for smi in smiles_list
                    ]
                )
        else:
            # Direct pocket PDB scoring
            mol_embs = scorer.encode_molecules(smiles_list)
            p_emb = scorer.encode_pocket(pdb_to_use)
            final_scores = np.array(
                [
                    float(np.dot(p_emb, mol_embs.get(smi, np.zeros(128))))
                    for smi in smiles_list
                ]
            )

        # Sort and filter
        idx_sorted = np.argsort(final_scores)[::-1]
        n_keep = max(1, int(len(smiles_list) * top_fraction))
        survivors = [smiles_list[i] for i in idx_sorted[:n_keep]]

        all_results = [
            L1Result(
                smiles=smiles_list[i],
                cnn_affinity=float(final_scores[i]),
                cnn_score=0,
                vina_affinity=0,
                receptor_idx=0,
            )
            for i in idx_sorted
        ]

        # Extract pocket residue info for L2 PocketConditioning
        # Use fpocket + GenPack-Lite refinement on the receptor structure.
        pocket_sites = None
        pocket_pdb_path = (
            pocket_pdb if (pocket_pdb and Path(pocket_pdb).exists()) else None
        )
        try:
            from app.genpack_lite import run_fpocket_with_genpack

            pocket_sites = run_fpocket_with_genpack(
                pdb_to_use,
                workdir=workdir / "fpocket",
                top_n=top_n_pockets,
                min_quality=min_quality,
            )
            if pocket_sites:
                logger.info(
                    "GenPack-Lite: %d site(s), %d total residues for L2 PocketConditioning",
                    len(pocket_sites),
                    sum(len(s) for s in pocket_sites),
                )
            else:
                logger.warning("GenPack-Lite returned no pockets for L2")
        except Exception as e:
            logger.warning(f"GenPack-Lite failed, fallback to fpocket: {e}")
            try:
                from app.drugclip_pocket import _get_pocket_residues_from_pqr

                pocket_sites = _get_pocket_residues_from_pqr(
                    pdb_to_use,
                    workdir=workdir / "fpocket",
                    top_n=top_n_pockets,
                    min_quality=min_quality,
                )
            except Exception:
                pass

        elapsed = time.time() - t0
        logger.info(
            f"L1 DrugCLIP: {len(smiles_list)} -> {len(survivors)} ({top_fraction:.0%}) in {elapsed:.1f}s"
        )
        return L1ScreenResult(
            survivors=survivors,
            all_results=all_results,
            n_input=len(smiles_list),
            n_survivors=len(survivors),
            wall_time_s=elapsed,
            pocket_sites=pocket_sites,
            pocket_pdb=pocket_pdb_path,
        )


def get_l1_prescreen() -> L1DockingPreScreen:
    return L1DockingPreScreen()
