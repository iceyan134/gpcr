#!/usr/bin/env python
"""
Standalone Boltz-2 affinity scorer. Runs in boltz-venv, called via subprocess
from boltz_scorer.py. Takes an ESMFold2 mmCIF file, parses it with Boltz-2's
internal parser, and runs the affinity module.

Usage:
    python _boltz_affinity_runner.py --cif complex.cif --smiles "CCO" [--recycling_steps 1]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import os
from pathlib import Path

import torch

from boltz.main import get_cache_path, Boltz2DiffusionParams, PairformerArgsV2, MSAModuleArgs
from boltz.data.types import Manifest
from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.write.writer import BoltzAffinityWriter
from boltz.data.parse.mmcif import parse_mmcif
from boltz.model.models.boltz2 import Boltz2
from pytorch_lightning import Trainer


def load_ccd(cache_dir: Path) -> dict:
    ccd_path = cache_dir / "ccd.pkl"
    if ccd_path.exists():
        with open(ccd_path, "rb") as f:
            return pickle.load(f)
    return {}


def load_model(cache_dir: Path, recycling_steps: int) -> Boltz2:
    ckpt = cache_dir / "boltz2_aff.ckpt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Affinity checkpoint not found at {ckpt}")

    predict_args = {
        "recycling_steps": recycling_steps,
        "sampling_steps": 200,
        "diffusion_samples": 5,
        "max_parallel_samples": 1,
        "write_confidence_summary": False,
        "write_full_pae": False,
        "write_full_pde": False,
    }
    diffusion_params = dict(asdict(Boltz2DiffusionParams()))
    diffusion_params["step_scale"] = 1.5

    model = Boltz2.load_from_checkpoint(
        str(ckpt), strict=True,
        predict_args=predict_args,
        map_location="cpu",
        diffusion_process_args=diffusion_params,
        ema=False,
        pairformer_args=dict(asdict(PairformerArgsV2())),
        msa_args=dict(asdict(MSAModuleArgs(
            subsample_msa=True, num_subsampled_msa=1024, use_paired_feature=True,
        ))),
        steering_args={
            "fk_steering": False, "physical_guidance_update": False,
            "contact_guidance_update": False, "guidance_update": False,
        },
        affinity_mw_correction=False,
        skip_run_structure=True,
        run_trunk_and_structure=True,
        confidence_prediction=True,
        use_kernels=True,
    )
    model.confidence_prediction = True
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cif", required=True)
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--recycling_steps", type=int, default=1)
    parser.add_argument("--use_msa_server", action="store_true")
    args = parser.parse_args()

    cache_dir = Path(get_cache_path())
    cif_path = Path(args.cif)

    # 1. Parse mmCIF
    ccd = load_ccd(cache_dir)
    mol_dir = cache_dir / "mols"
    parsed = parse_mmcif(
        path=str(cif_path),
        mols=ccd,
        moldir=str(mol_dir),
    )

    # 2. Build output structure
    import tempfile
    work_dir = Path(tempfile.mkdtemp())
    record_id = "complex"
    predictions_dir = work_dir / "predictions" / record_id
    predictions_dir.mkdir(parents=True)

    npz_path = predictions_dir / f"pre_affinity_{record_id}.npz"
    parsed.data.dump(npz_path)

    processed_dir = work_dir / "processed"
    processed_dir.mkdir(parents=True)
    manifest = {
        "records": [{
            "id": record_id,
            "structures": [str(npz_path)],
            "msa": [], "templates": [], "constraints": [],
        }]
    }
    (processed_dir / "manifest.json").write_text(json.dumps(manifest))

    for sub in ("structures", "msa", "constraints", "templates"):
        (processed_dir / sub).mkdir(parents=True)
    (predictions_dir / "mols").mkdir(parents=True)

    # 3. Load model
    model = load_model(cache_dir, args.recycling_steps)

    # 4. Setup data module and predict
    writer = BoltzAffinityWriter(
        data_dir=str(processed_dir / "structures"),
        output_dir=str(predictions_dir),
    )
    manifest_obj = Manifest.load(str(processed_dir / "manifest.json"))
    data_module = Boltz2InferenceDataModule(
        manifest=manifest_obj,
        target_dir=str(predictions_dir),
        msa_dir=str(processed_dir / "msa"),
        mol_dir=str(mol_dir),
        num_workers=0,
        constraints_dir=str(processed_dir / "constraints"),
        template_dir=str(processed_dir / "templates"),
        extra_mols_dir=str(predictions_dir / "mols"),
        override_method="other",
        affinity=True,
        batch_size=1,
    )

    dataset = data_module.predict_dataset()
    if len(dataset) == 0:
        print(json.dumps({"affinity_binary": 0.0, "affinity_value": 0.0, "error": "empty_dataset"}))
        sys.exit(0)

    trainer = Trainer(
        default_root_dir=str(work_dir),
        strategy="auto",
        callbacks=[writer],
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    trainer.predict(model, datamodule=data_module, return_predictions=False)

    # 5. Parse result
    affinity_files = list(predictions_dir.glob("affinity_*.json"))
    if not affinity_files:
        print(json.dumps({"affinity_binary": 0.0, "affinity_value": 0.0, "error": "no_affinity_json"}))
        sys.exit(0)

    with open(affinity_files[0]) as f:
        data = json.load(f)

    result = {
        "affinity_binary": float(data.get("affinity_probability_binary", 0.0)),
        "affinity_value": float(data.get("affinity_pred_value", 0.0)),
        "error": None,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()