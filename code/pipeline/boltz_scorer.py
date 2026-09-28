"""
Nesso-1 affinity scoring for ESMFold2 co-fold poses.
Direct Python API (no subprocess), caches model between calls.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger("cascade.boltz")


def _cif_to_sdf(cif_text: str, smiles: str, workdir: Path) -> Path:
    """Convert ESMFold2 mmCIF to clean ligand SDF for Nesso-1.

    Uses obabel to directly convert CIF → SDF (multi-molecule), then
    picks the last molecule (ligand). Overlays SMILES-based coords.
    """
    import subprocess
    from rdkit import Chem
    from rdkit.Chem import AllChem

    workdir.mkdir(parents=True, exist_ok=True)
    cif_path = workdir / "complex.cif"
    cif_path.write_text(cif_text)

    # obabel CIF → SDF (multi-molecule: protein + ligand)
    raw_sdf = workdir / "raw.sdf"
    subprocess.run(
        ["obabel", str(cif_path), "-O", str(raw_sdf)],
        capture_output=True,
        text=True,
        timeout=30,
    )

    # Read last molecule from multi-mol SDF (ligand)
    lig_sdf = workdir / "ligand.sdf"
    suppl = Chem.SDMolSupplier(str(raw_sdf), sanitize=False)
    mol_from_cif = None
    for m in suppl:
        if m is not None:
            mol_from_cif = m

    if mol_from_cif is not None:
        # Overlay SMILES-based mol onto ESMFold2 coords from CIF
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        conf = mol.GetConformer()

        ref_conf = mol_from_cif.GetConformer()
        n = min(conf.GetNumAtoms(), ref_conf.GetNumAtoms())
        for i in range(n):
            pt = ref_conf.GetAtomPosition(i)
            conf.SetAtomPosition(i, pt)

        mol = Chem.RemoveHs(mol)
        w = Chem.SDWriter(str(lig_sdf))
        w.write(mol)
        w.close()
        # nesso deadlocks on chemically invalid inputs; never hand one over
        chk = Chem.SDMolSupplier(str(lig_sdf))
        if not chk or chk[0] is None:
            logger.warning("cif-derived SDF unsanitizable, smiles fallback: %s", smiles)
        else:
            return lig_sdf

    # Fallback: SMILES-only, no structure coords
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    mol = Chem.RemoveHs(mol)
    w = Chem.SDWriter(str(lig_sdf))
    w.write(mol)
    w.close()

    return lig_sdf


def score_single(
    protein_sequence: str,
    smiles: str,
    mmcif_text: str = "",
    recycling_steps: int = 1,
    use_msa_server: bool = False,
    timeout: int = 600,
) -> dict:
    """Score one complex via nesso CLI subprocess.
    Uses a persistent cache dir so ESM embeddings are reused per protein.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        if mmcif_text:
            sdf_path = _cif_to_sdf(mmcif_text, smiles, tmpdir)
            lig_block = f"      sdf: {sdf_path}"
        else:
            lig_block = f"      smiles: '{smiles}'"

        (tmpdir / "input.yaml").write_text(f"""sequences:
  - protein:
      id: A
      sequence: {protein_sequence}
  - ligand:
      id: B
{lig_block}
properties:
  - affinity:
      binder: B
""")

        cache_dir = Path(os.environ.get("NESSO_CACHE", "/tmp/nesso_cache"))
        cache_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "/opt/nesso-venv/bin/nesso",
            "predict",
            str(tmpdir / "input.yaml"),
            "--out_dir",
            str(tmpdir / "output"),
            "--accelerator",
            "gpu",
            "--devices",
            "1",
            "--recycling_steps",
            str(recycling_steps),
            "--no_kernels",
            "--cache",
            str(cache_dir),
        ]
        if os.environ.get("NESSO_CCD_PATH"):
            cmd += ["--ccd", os.environ["NESSO_CCD_PATH"]]
        if os.environ.get("NESSO_CKPT_DIR"):
            cmd += ["--checkpoint", os.environ["NESSO_CKPT_DIR"]]
        env = {
            **os.environ,
            "BOLTZ_CACHE": os.environ.get("BOLTZ_CACHE", "/cache/boltz"),
        }

        import fcntl as _fcntl
        _lock = open("/tmp/nesso_gpu.lock", "w")
        try:
            _fcntl.flock(_lock, _fcntl.LOCK_EX)
            try:
                import torch as _torch
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                r = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=env,
                )
            except subprocess.TimeoutExpired:
                logger.error("Nesso-1 timed out after %ds", timeout)
                try:
                    import shutil as _sh, time as _t
                    _dbg = Path("/tmp/nesso_debug")
                    _dbg.mkdir(exist_ok=True)
                    _tag = _t.strftime("%H%M%S")
                    for _f in (tmpdir / "input.yaml", tmpdir / "ligand.sdf", tmpdir / "complex.cif"):
                        if _f.exists():
                            _sh.copy(_f, _dbg / ("%s_%s" % (_tag, _f.name)))
                except Exception:
                    pass
                return {"affinity_binary": 0.0, "affinity_value": 0.0, "error": "timeout"}
            except subprocess.CalledProcessError as e:
                err = e.stderr[-1000:] if e.stderr else str(e)
                logger.error("Nesso-1 failed: %s", err)
                return {"affinity_binary": 0.0, "affinity_value": 0.0, "error": err}
        finally:
            _fcntl.flock(_lock, _fcntl.LOCK_UN)
            _lock.close()

        af = tmpdir / "output" / "predictions" / "input" / "affinity.json"
        if not af.exists():
            return {
                "affinity_binary": 0.0,
                "affinity_value": 0.0,
                "error": "no_affinity_json",
            }

        data = json.loads(af.read_text())
        return {
            "affinity_binary": float(data.get("affinity_probability_binary", 0.0)),
            "affinity_value": float(data.get("affinity_pred_value", 0.0)),
            "error": None,
        }


def _cif_to_sdf_batch(
    cif_texts: list[str], smiles_list: list[str], tmpdir: Path
) -> list[Path]:
    """Convert multiple mmCIFs to SDFs in parallel."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    futures = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, (cif, smi) in enumerate(zip(cif_texts, smiles_list)):
            wd = tmpdir / f"mol_{i}"
            wd.mkdir(parents=True, exist_ok=True)
            futures[pool.submit(_cif_to_sdf, cif, smi, wd)] = i
        results = [None] * len(cif_texts)
        for f in as_completed(futures):
            i = futures[f]
            results[i] = f.result()
    return results


def score_batch(
    protein_sequence: str,
    smiles_list: list[str],
    mmcif_texts: list[str] | None = None,
    recycling_steps: int = 1,
    use_msa_server: bool = False,
    max_workers: int = 4,
) -> list[dict]:
    """Score multiple complexes in one model call (no repeated model loading)."""
    mc_list = mmcif_texts or [""] * len(smiles_list)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        yaml_dir = tmpdir / "yaml"
        yaml_dir.mkdir()
        out_dir = tmpdir / "output"

        sdf_paths = [None] * len(smiles_list)
        cif_indices = [i for i, mc in enumerate(mc_list) if mc]
        cif_texts = [mc_list[i] for i in cif_indices]
        cif_smiles = [smiles_list[i] for i in cif_indices]
        if cif_texts:
            _cif_dir = tmpdir / "cif_work"
            _cif_dir.mkdir(exist_ok=True)
            sdf_results = _cif_to_sdf_batch(cif_texts, cif_smiles, _cif_dir)
            for idx, sdf_path in zip(cif_indices, sdf_results):
                sdf_paths[idx] = sdf_path

        for i, (smiles, mc_text) in enumerate(zip(smiles_list, mc_list)):
            if mc_text and sdf_paths[i] is not None:
                lig_block = f"      sdf: {sdf_paths[i]}"
            else:
                lig_block = f"      smiles: '{smiles}'"

            (yaml_dir / f"mol_{i:04d}.yaml").write_text(f"""sequences:
  - protein:
      id: A
      sequence: {protein_sequence}
  - ligand:
      id: B
{lig_block}
properties:
  - affinity:
      binder: B
""")
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            cmd = [
                "/opt/nesso-venv/bin/nesso",
                "predict",
                str(yaml_dir),
                "--out_dir",
                str(out_dir),
                "--accelerator",
                "gpu",
                "--devices",
                "1",
                "--recycling_steps",
                str(recycling_steps),
                "--no_kernels",
                "--num_workers",
                "0",
            ]
            if os.environ.get("NESSO_CCD_PATH"):
                cmd += ["--ccd", os.environ["NESSO_CCD_PATH"]]
            if os.environ.get("NESSO_CKPT_DIR"):
                cmd += ["--checkpoint", os.environ["NESSO_CKPT_DIR"]]

            batch_timeout = 1800  # 30 min for 160 complexes
            env = {
                **os.environ,
                "BOLTZ_CACHE": os.environ.get("BOLTZ_CACHE", "/cache/boltz"),
            }
            logger.info(
                "Running nesso predict on %d YAMLs (timeout=%ds) ...",
                len(smiles_list),
                batch_timeout,
            )
            with (
                open(out_dir / "nesso_stdout.log", "w") as _out,
                open(out_dir / "nesso_stderr.log", "w") as _err,
            ):
                r = subprocess.run(
                    cmd,
                    check=True,
                    stdout=_out,
                    stderr=_err,
                    timeout=batch_timeout,
                    env=env,
                )
        except subprocess.TimeoutExpired:
            stderr_snip = ""
            err_log = out_dir / "nesso_stderr.log"
            if err_log.exists():
                stderr_snip = err_log.read_text()[-300:]
            logger.error(
                "Nesso-1 batch timed out after %ds (stderr: %s)",
                batch_timeout,
                stderr_snip,
            )
            return [
                {"affinity_binary": 0.0, "affinity_value": 0.0, "error": "timeout"}
                for _ in smiles_list
            ]
        except subprocess.CalledProcessError as e:
            err = e.stderr[-1000:] if e.stderr else str(e)
            logger.error("Nesso-1 batch failed: %s", err)
            return [
                {"affinity_binary": 0.0, "affinity_value": 0.0, "error": err}
                for _ in smiles_list
            ]

        results = []
        for i in range(len(smiles_list)):
            af = out_dir / "predictions" / f"mol_{i:04d}" / "affinity.json"
            if af.exists():
                data = json.loads(af.read_text())
                results.append(
                    {
                        "affinity_binary": float(
                            data.get("affinity_probability_binary", 0.0)
                        ),
                        "affinity_value": float(data.get("affinity_pred_value", 0.0)),
                        "error": None,
                    }
                )
            else:
                results.append(
                    {
                        "affinity_binary": 0.0,
                        "affinity_value": 0.0,
                        "error": "no_output",
                    }
                )
        return results
