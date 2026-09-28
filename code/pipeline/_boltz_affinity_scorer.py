#!/usr/bin/env python
"""
Standalone Boltz-2 affinity scorer for ESMFold2 poses.
Pipeline: ESMFold2 mmCIF → obabel PDB → gemmi.read_pdb → StructureV2 → predict_affinity

Usage:
    python _boltz_affinity_scorer.py --cif complex.cif --sequence "MA..." --smiles "CCO"
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import subprocess
import tempfile
from pathlib import Path

import gemmi
import numpy as np
from pytorch_lightning import Trainer

from boltz.main import get_cache_path
from boltz.data.types import (
    StructureV2, StructureInfo, AtomV2, BondV2, Residue, Chain, Coords, Ensemble, Interface, Manifest,
)

# Patch numpy lib.utils.safe_eval to avoid recursion with object arrays
import numpy.lib.utils as _np_utils
_orig_safe_eval = _np_utils.safe_eval
_patched_called = [False]
def _patched_safe_eval(source):
    try:
        return _orig_safe_eval(source)
    except RecursionError:
        _patched_called[0] = True
        # Return a minimal valid numpy dtype descriptor
        return {"descr": "<f8", "fortran_order": False, "shape": ()}
_np_utils.safe_eval = _patched_safe_eval

from boltz.data.module.inferencev2 import Boltz2InferenceDataModule, PredictionDataset, load_input
from boltz.data.write.writer import BoltzAffinityWriter
from boltz.model.models.boltz2 import Boltz2

# Monkey-patch ResidueConstraints.load to avoid numpy recursion bug
from boltz.data.types import ResidueConstraints as _RC
import numpy as _np
_orig_load = _RC.load
def _safe_rc_load(path):
    d = _np.load(path, allow_pickle=True)
    return _RC(**{k: d[k] for k in d.files})
_RC.load = staticmethod(_safe_rc_load)


def structure_to_v2(structure: gemmi.Structure) -> tuple[StructureV2, StructureInfo, int, int]:
    """Convert a gemmi.Structure (from read_pdb) to StructureV2 arrays."""
    AD = np.dtype(AtomV2)
    BD = np.dtype(BondV2)
    RD = np.dtype(Residue)
    CD = np.dtype(Chain)
    CD_ens = np.dtype(Ensemble)
    CD_coords = np.dtype(Coords)
    ID = np.dtype(Interface)

    model = structure[0]

    # Count atoms and residues per chain
    chain_atoms = []
    chain_residues = []
    for chain in model:
        n_res = 0
        n_at = 0
        for res in chain:
            n_res += 1
            n_at += len(list(res))
        chain_atoms.append(n_at)
        chain_residues.append(n_res)

    N = sum(chain_atoms)
    NR = sum(chain_residues)
    NC = len(model)

    # Atoms
    atoms = np.zeros(N, dtype=AD)
    # Bonds
    bonds_list = []
    # Residues
    residues = np.zeros(NR, dtype=RD)
    # Chains
    chains = np.zeros(NC, dtype=CD)
    # Coords
    coords = np.zeros(N, dtype=CD_coords)
    # Ensemble
    ensemble = np.zeros(1, dtype=CD_ens)

    a_idx = 0
    r_idx = 0
    for ci, chain in enumerate(model):
        chain_name = chain.name
        chain_res_start = r_idx
        chain_atom_start = a_idx

        for ri, res in enumerate(chain):
            res_name = res.name
            seq_id = res.seqid.num
            res_atoms = list(res)
            n_res_atoms = len(res_atoms)

            residues[r_idx]["name"] = res_name[:5]
            residues[r_idx]["res_type"] = 0
            residues[r_idx]["res_idx"] = seq_id
            residues[r_idx]["atom_idx"] = a_idx
            residues[r_idx]["atom_num"] = n_res_atoms
            residues[r_idx]["atom_center"] = a_idx
            residues[r_idx]["atom_disto"] = a_idx + n_res_atoms - 1
            residues[r_idx]["is_standard"] = True
            residues[r_idx]["is_present"] = True

            for ai, atom in enumerate(res_atoms):
                name = atom.name.strip()
                nb = name.encode()[:4]
                atoms[a_idx]["name"] = name[:4]
                atoms[a_idx]["coords"] = [atom.pos.x, atom.pos.y, atom.pos.z]
                atoms[a_idx]["is_present"] = True
                atoms[a_idx]["bfactor"] = atom.b_iso
                atoms[a_idx]["plddt"] = 1.0

                coords[a_idx]["coords"] = (atom.pos.x, atom.pos.y, atom.pos.z)

                # Backbone bonds (C-N)
                if name == "C" and ai < n_res_atoms - 1:
                    next_atom = res_atoms[ai + 1]
                    if next_atom.name.strip() == "N":
                        bonds_list.append((ci, ci, r_idx, r_idx, a_idx, a_idx + 1, 1))

                a_idx += 1

            r_idx += 1

        # Peptide bonds between residues
        for ri in range(chain_res_start, r_idx - 1):
            r1 = ri
            r2 = ri + 1
            # Find C of r1, N of r2
            c_at = -1
            n_at = -1
            r1_s = int(residues[r1]["atom_idx"])
            r1_e = r1_s + int(residues[r1]["atom_num"])
            r2_s = int(residues[r2]["atom_idx"])
            r2_e = r2_s + int(residues[r2]["atom_num"])
            for a in range(r1_s, r1_e):
                if atoms[a]["name"][0] == ord('C') and atoms[a]["name"][1] == 0:
                    c_at = a
                    break
            for a in range(r2_s, r2_e):
                if atoms[a]["name"][0] == ord('N') and atoms[a]["name"][1] == 0:
                    n_at = a
                    break
            if c_at >= 0 and n_at >= 0:
                bonds_list.append((ci, ci, ri, ri + 1, c_at, n_at, 1))

        n_ch_atoms = a_idx - chain_atom_start
        n_ch_res = r_idx - chain_res_start
        chains[ci]["name"] = chain_name[:5]
        chains[ci]["mol_type"] = 0 if ci == 0 else 1  # first chain=protein, rest=ligand
        chains[ci]["entity_id"] = ci
        chains[ci]["sym_id"] = 0
        chains[ci]["asym_id"] = ci
        chains[ci]["atom_idx"] = chain_atom_start
        chains[ci]["atom_num"] = n_ch_atoms
        chains[ci]["res_idx"] = chain_res_start
        chains[ci]["res_num"] = n_ch_res
        chains[ci]["cyclic_period"] = 0

    # Bonds array
    bonds = np.zeros(len(bonds_list), dtype=BD)
    for i, (c1, c2, r1, r2, a1, a2, bt) in enumerate(bonds_list):
        bonds[i]["chain_1"] = c1
        bonds[i]["chain_2"] = c2
        bonds[i]["res_1"] = r1
        bonds[i]["res_2"] = r2
        bonds[i]["atom_1"] = a1
        bonds[i]["atom_2"] = a2
        bonds[i]["type"] = bt

    # Mask
    mask = np.ones(NC, dtype=bool)
    # Ensemble
    ensemble[0] = (0, N)

    # Interfaces (empty)
    interfaces = np.zeros(0, dtype=ID)

    sv2 = StructureV2(
        atoms=atoms, bonds=bonds, residues=residues, chains=chains,
        interfaces=interfaces, mask=mask, ensemble=ensemble, coords=coords,
        pocket=np.zeros(0, dtype=np.int32),
    )
    info = StructureInfo(
        resolution=None, method="other",
        deposited=None, released=None, revised=None,
        num_chains=NC, num_interfaces=0, pH=None, temperature=None,
    )
    return sv2, info, NC


def load_model(cache_dir: Path, recycling_steps: int) -> Boltz2:
    from dataclasses import asdict
    from boltz.main import Boltz2DiffusionParams, PairformerArgsV2, MSAModuleArgs
    ckpt = cache_dir / "boltz2_aff.ckpt"
    predict_args = {
        "recycling_steps": recycling_steps, "sampling_steps": 200,
        "diffusion_samples": 5, "max_parallel_samples": 1,
        "write_confidence_summary": False, "write_full_pae": False, "write_full_pde": False,
    }
    dp = dict(asdict(Boltz2DiffusionParams()))
    dp["step_scale"] = 1.5
    model = Boltz2.load_from_checkpoint(
        str(ckpt), strict=True, predict_args=predict_args, map_location="cpu",
        diffusion_process_args=dp, ema=False,
        pairformer_args=dict(asdict(PairformerArgsV2())),
        msa_args=dict(asdict(MSAModuleArgs(subsample_msa=True, num_subsampled_msa=1024, use_paired_feature=True))),
        steering_args={"fk_steering": False, "physical_guidance_update": False,
                       "contact_guidance_update": False, "guidance_update": False},
        affinity_mw_correction=False, skip_run_structure=True,
        run_trunk_and_structure=True, confidence_prediction=True, use_kernels=True,
    )
    model.confidence_prediction = True
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cif", required=True, help="ESMFold2 mmCIF file")
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--recycling_steps", type=int, default=1)
    args = parser.parse_args()

    cache_dir = Path(get_cache_path())
    cif_path = Path(args.cif)

    # 1. ESMFold2 CIF -> PDB -> gemmi Structure
    tmpdir = Path(tempfile.mkdtemp())
    pdb_path = tmpdir / "complex.pdb"
    subprocess.run(["obabel", str(cif_path), "-O", str(pdb_path)], check=True, capture_output=True, timeout=30)
    structure = gemmi.read_pdb(str(pdb_path))

    # 2. Convert to StructureV2
    sv2, info, NC = structure_to_v2(structure)

    # 3. Save NPZ and build manifest
    work_dir = Path(tempfile.mkdtemp())
    record_id = "complex"
    pred_dir = work_dir / "predictions"
    record_dir = pred_dir / record_id
    record_dir.mkdir(parents=True)
    npz_path = record_dir / f"pre_affinity_{record_id}.npz"
    sv2.dump(npz_path)

    proc_dir = work_dir / "processed"
    proc_dir.mkdir(parents=True)

    # TEST: verify NPZ loads correctly
    try:
        from boltz.data.types import StructureV2 as _SV2
        _test = _SV2.load(npz_path)
        print(f"[DEBUG] StructureV2.load OK: {len(_test.atoms)} atoms", file=sys.stderr)
    except Exception as _e:
        print(f"[DEBUG] StructureV2.load FAILED: {_e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)

    chain_info = [{"chain_id": i, "chain_name": "A" if i == 0 else "L",
                   "mol_type": 0 if i == 0 else 1,
                   "cluster_id": "", "msa_id": -1, "num_residues": 0,
                   "valid": True, "entity_id": i}
                  for i in range(NC)]

    json.dump([{
        "id": record_id,
        "structure": {"resolution": None, "method": "other",
                      "deposited": None, "released": None, "revised": None,
                      "num_chains": NC, "num_interfaces": 0, "pH": None, "temperature": None},
        "chains": chain_info, "interfaces": [],
        "affinity": {"chain_id": 1, "mw": 0.0},
        "structures": [str(npz_path)], "msa": [], "templates": [], "constraints": [],
    }], (proc_dir / "manifest.json").open("w"))

    for sub in ("structures", "msa", "constraints", "templates"):
        (proc_dir / sub).mkdir(parents=True)
    (record_dir / "mols").mkdir(parents=True)

    np.savez_compressed(proc_dir / "msa" / "empty.npz",
        sequences=np.zeros((1, 1), dtype=np.int32),
        deletions=np.zeros((1, 1), dtype=np.float32),
        residues=np.zeros(1, dtype=np.int32),
    )
    # Create empty constraints placeholder
    _c = proc_dir / "constraints" / "complex.npz"
    np.savez(_c,
        rdkit_bounds_constraints=np.zeros((0, 4), dtype=np.int32),
        chiral_atom_constraints=np.zeros((0, 5), dtype=np.int32),
        stereo_bond_constraints=np.zeros((0, 3), dtype=np.int32),
        planar_bond_constraints=np.zeros((0, 2), dtype=np.int32),
        planar_ring_5_constraints=np.zeros((0, 5), dtype=np.int32),
        planar_ring_6_constraints=np.zeros((0, 6), dtype=np.int32),
    )

    manifest_obj = Manifest.load(proc_dir / "manifest.json")

    # 4. Load model and predict
    model = load_model(cache_dir, args.recycling_steps)
    mol_dir = cache_dir / "mols"

    writer = BoltzAffinityWriter(data_dir=proc_dir / "structures", output_dir=record_dir)
    dm = Boltz2InferenceDataModule(
        manifest=manifest_obj, target_dir=pred_dir,
        msa_dir=proc_dir / "msa", mol_dir=mol_dir,
        num_workers=0, extra_mols_dir=record_dir / "mols",
        override_method="other", affinity=True,
    )

    trainer = Trainer(
        default_root_dir=str(work_dir), strategy="auto", callbacks=[writer],
        accelerator="gpu", devices=1, precision="bf16-mixed",
        enable_progress_bar=False, enable_model_summary=False, logger=False,
    )
    trainer.predict(model, datamodule=dm, return_predictions=False)

    # 5. Results
    af_files = list(record_dir.glob("affinity_*.json"))
    if not af_files:
        print(json.dumps({"affinity_binary": 0.0, "affinity_value": 0.0, "error": "no_affinity_json"}))
        sys.exit(0)

    with open(af_files[0]) as f:
        data = json.load(f)

    result = {
        "affinity_binary": float(data.get("affinity_probability_binary", 0.0)),
        "affinity_value": float(data.get("affinity_pred_value", 0.0)),
        "error": None,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()