#!/usr/bin/env python
"""
Standalone Boltz-2 affinity scorer for ESMFold2 mmCIF poses.
Completely bypasses parse_mmcif. Instead:
1. Parses ESMFold2 mmCIF directly with gemmi.cif
2. Builds StructureV2 numpy arrays manually from coordinates
3. Loads Boltz-2 affinity checkpoint
4. Runs predict_affinity via Boltz2InferenceDataModule + Trainer

Usage:
    python _boltz_affinity_runner_v2.py --cif complex.cif --sequence "MA..." --smiles "CCO"
"""
from __future__ import annotations

import argparse
import json
import sys
import os
import tempfile
from pathlib import Path

import gemmi
import numpy as np
from pytorch_lightning import Trainer

from boltz.main import get_cache_path, Boltz2DiffusionParams, PairformerArgsV2, MSAModuleArgs
from boltz.data.types import (
    StructureV2, StructureInfo, Atom, Bond, Residue, Chain, Interface, Manifest,
)
from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.write.writer import BoltzAffinityWriter
from boltz.model.models.boltz2 import Boltz2


ELEMENT_MAP = {
    "H": 0, "C": 1, "N": 2, "O": 3, "F": 4, "P": 5,
    "S": 6, "Cl": 7, "CL": 7, "Br": 8, "I": 9, "Fe": 10,
    "Zn": 11, "Mg": 12, "Ca": 13, "Na": 14, "K": 15,
}


def _el(element: str) -> int:
    return ELEMENT_MAP.get(element.strip().title(), 1)


def parse_esmfold2_cif(cif_path: Path) -> tuple[StructureV2, StructureInfo, int, int]:
    """Parse ESMFold2 mmCIF → StructureV2, completely bypassing parse_mmcif."""
    block = gemmi.cif.read(str(cif_path))[0]
    AD = np.dtype(Atom)
    BD = np.dtype(Bond)
    RD = np.dtype(Residue)
    CD = np.dtype(Chain)
    ID = np.dtype(Interface)

    # Read atom_site columns
    group_pdb = list(block.find_values("_atom_site.group_PDB"))
    atom_names = list(block.find_values("_atom_site.label_atom_id"))
    chain_ids = list(block.find_values("_atom_site.label_asym_id"))
    seq_ids = list(block.find_values("_atom_site.label_seq_id"))
    xs = list(block.find_values("_atom_site.Cartn_x"))
    ys = list(block.find_values("_atom_site.Cartn_y"))
    zs = list(block.find_values("_atom_site.Cartn_z"))
    elems = list(block.find_values("_atom_site.type_symbol"))
    N = len(xs)
    if N == 0:
        raise ValueError("No atoms")

    is_prot = [g == "ATOM" for g in group_pdb]
    prot_chains = sorted(set(c.strip() for i, c in enumerate(chain_ids) if is_prot[i]))
    lig_chains = sorted(set(c.strip() for i, c in enumerate(chain_ids) if not is_prot[i]))
    all_chains = prot_chains + lig_chains
    c2i = {c: i for i, c in enumerate(all_chains)}
    NC = len(all_chains)

    # Atoms
    atoms = np.zeros(N, dtype=AD)
    res_map = {}
    res_order = []
    for i in range(N):
        ci = c2i[chain_ids[i].strip()]
        si = int(seq_ids[i]) if seq_ids[i].strip() else 0
        name = atom_names[i].strip()
        nb = name.encode()[:4]
        x, y, z = float(xs[i]), float(ys[i]), float(zs[i])
        atoms[i]["name"] = list(nb.ljust(4, b'\x00'))
        atoms[i]["element"] = _el(elems[i])
        atoms[i]["charge"] = 0
        atoms[i]["coords"] = [x, y, z]
        atoms[i]["conformer"] = [x, y, z]
        atoms[i]["is_present"] = True
        atoms[i]["chirality"] = 0
        key = (ci, si)
        if key not in res_map:
            res_map[key] = len(res_order)
            res_order.append(key)

    NR = len(res_order)
    residues = np.zeros(NR, dtype=RD)
    for ri, (ci, si) in enumerate(res_order):
        a_start = None
        a_count = 0
        for j in range(N):
            if c2i[chain_ids[j].strip()] == ci and (int(seq_ids[j]) if seq_ids[j].strip() else 0) == si:
                if a_start is None:
                    a_start = j
                a_count += 1
        residues[ri]["name"] = "UNK"
        residues[ri]["res_type"] = 0
        residues[ri]["res_idx"] = si
        residues[ri]["atom_idx"] = a_start or 0
        residues[ri]["atom_num"] = a_count
        residues[ri]["atom_center"] = a_start or 0
        residues[ri]["atom_disto"] = (a_start or 0) + a_count - 1
        residues[ri]["is_standard"] = True
        residues[ri]["is_present"] = True

    # Chains
    chains = np.zeros(NC, dtype=CD)
    for ci in range(NC):
        res_idxs = [j for j, (cj, _) in enumerate(res_order) if cj == ci]
        if not res_idxs:
            continue
        rs = res_idxs[0]
        re = res_idxs[-1]
        a_start = int(residues[rs]["atom_idx"])
        a_end = int(residues[re]["atom_idx"]) + int(residues[re]["atom_num"])
        chains[ci]["name"] = all_chains[ci][:5]
        chains[ci]["mol_type"] = 1 if ci >= len(prot_chains) else 0
        chains[ci]["entity_id"] = ci
        chains[ci]["sym_id"] = ci
        chains[ci]["asym_id"] = ci
        chains[ci]["atom_idx"] = a_start
        chains[ci]["atom_num"] = a_end - a_start
        chains[ci]["res_idx"] = rs
        chains[ci]["res_num"] = len(res_idxs)
        chains[ci]["cyclic_period"] = 0

    # Backbone bonds (C-N between consecutive residues)
    bond_list = []
    for ci in range(NC):
        if ci >= len(prot_chains):
            continue
        res_idxs = [j for j, (cj, _) in enumerate(res_order) if cj == ci]
        for j in range(len(res_idxs) - 1):
            r1, r2 = res_idxs[j], res_idxs[j + 1]
            s1 = int(residues[r1]["atom_idx"])
            e1 = s1 + int(residues[r1]["atom_num"])
            s2 = int(residues[r2]["atom_idx"])
            e2 = s2 + int(residues[r2]["atom_num"])
            c_idx = n_idx = -1
            for a in range(s1, e1):
                if bytes(atoms[a]["name"]).decode("ascii", errors="ignore").strip() == "C":
                    c_idx = a; break
            for a in range(s2, e2):
                if bytes(atoms[a]["name"]).decode("ascii", errors="ignore").strip() == "N":
                    n_idx = a; break
            if c_idx >= 0 and n_idx >= 0:
                bond_list.append((c_idx, n_idx, 1))

    bonds = np.zeros(len(bond_list), dtype=BD)
    for i, (a1, a2, bt) in enumerate(bond_list):
        bonds[i]["atom_1"] = a1
        bonds[i]["atom_2"] = a2
        bonds[i]["type"] = bt

    coords = np.zeros((N, 3), dtype=np.float32)
    for i in range(N):
        coords[i] = [float(xs[i]), float(ys[i]), float(zs[i])]

    mask = np.ones(N, dtype=np.bool_)
    ensemble = np.zeros(N, dtype=np.int32)
    interfaces = np.zeros(0, dtype=ID)

    sv2 = StructureV2(
        atoms=atoms, bonds=bonds, residues=residues, chains=chains,
        interfaces=interfaces, mask=mask, coords=coords, ensemble=ensemble, pocket=None,
    )
    info = StructureInfo(
        resolution=None, method="other",
        deposited=None, released=None, revised=None,
        num_chains=NC, num_interfaces=0, pH=None, temperature=None,
    )
    return sv2, info, NC, len(prot_chains)


def load_model(cache_dir: Path, recycling_steps: int) -> Boltz2:
    from dataclasses import asdict
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
    parser.add_argument("--cif", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--recycling_steps", type=int, default=1)
    args = parser.parse_args()

    cache_dir = Path(get_cache_path())
    cif_path = Path(args.cif)

    structure, info, NC, n_prot = parse_esmfold2_cif(cif_path)

    work_dir = Path(tempfile.mkdtemp())
    record_id = "complex"
    pred_dir = work_dir / "predictions"
    record_dir = pred_dir / record_id
    record_dir.mkdir(parents=True)
    npz_path = record_dir / f"pre_affinity_{record_id}.npz"
    structure.dump(npz_path)

    proc_dir = work_dir / "processed"
    proc_dir.mkdir(parents=True)
    import json as _json
    _json.dump(
        [{
            "id": record_id,
            "structure": {
                "resolution": None, "method": "other",
                "deposited": None, "released": None, "revised": None,
                "num_chains": NC, "num_interfaces": 0,
                "pH": None, "temperature": None,
            },
            "chains": [{"chain_id": ci, "chain_name": "A" if ci == 0 else "L",
                        "mol_type": 1 if ci >= n_prot else 0,
                        "cluster_id": "", "msa_id": "empty", "num_residues": 0,
                        "valid": True, "entity_id": ci}
                       for ci in range(NC)],
            "interfaces": [],
            "structures": [str(npz_path)],
            "msa": [], "templates": [], "constraints": [],
        }],
        (proc_dir / "manifest.json").open("w"),
    )
    manifest_obj = Manifest.load(proc_dir / "manifest.json")

    for sub in ("structures", "msa", "constraints", "templates"):
        (proc_dir / sub).mkdir(parents=True)
    (record_dir / "mols").mkdir(parents=True)

    import numpy as _np
    _np.savez_compressed(proc_dir / "msa" / "empty.npz",
        sequences=_np.zeros((1, 1), dtype=_np.int32),
        deletions=_np.zeros((1, 1), dtype=_np.float32),
        residues=_np.zeros(1, dtype=_np.int32),
    )

    mol_dir = cache_dir / "mols"
    model = load_model(cache_dir, args.recycling_steps)

    writer = BoltzAffinityWriter(data_dir=proc_dir / "structures", output_dir=record_dir)
    dm = Boltz2InferenceDataModule(
        manifest=manifest_obj, target_dir=pred_dir,
        msa_dir=proc_dir / "msa", mol_dir=mol_dir,
        num_workers=0, constraints_dir=proc_dir / "constraints",
        template_dir=proc_dir / "templates", extra_mols_dir=record_dir / "mols",
        override_method="other", affinity=True,
    )

    trainer = Trainer(
        default_root_dir=str(work_dir), strategy="auto", callbacks=[writer],
        accelerator="gpu", devices=1, precision="bf16-mixed",
        enable_progress_bar=False, enable_model_summary=False, logger=False,
    )
    trainer.predict(model, datamodule=dm, return_predictions=False)

    af_files = list(pred_dir.glob("affinity_*.json"))
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