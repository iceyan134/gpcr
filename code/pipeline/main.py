"""
ESMFold2 Service — FastAPI application
Provides local GPU inference and Biohub API-based prediction endpoints.
"""
from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from app.models import (
    CascadeRequest,
    CcdSearchResult,
    PredictionRequest,
    PredictionResult,
    SmilesBindingRequest,
)


# ── Lifespan ──────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm: load model on startup if local mode
    mode = os.environ.get("ESMFOLD2_MODE", "local")
    if mode == "local" and os.environ.get("PRELOAD_MODEL", "1") == "1":
        try:
            from app.local_inference import get_engine
            get_engine()  # triggers model load
        except Exception as e:
            print(f"[WARN] Model preload failed: {e}")
    yield


app = FastAPI(
    title="ESMFold2 Service",
    description="All-atom biomolecular structure prediction — protein, DNA, RNA, ligands",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Helpers ───────────────────────────────────────────────────────────
def _get_mode() -> str:
    return os.environ.get("ESMFOLD2_MODE", "local")


def _save_result(result: PredictionResult, output_dir: str = "/output"):
    path = Path(output_dir) / f"{result.name}_{result.job_id}.cif"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.mmcif)
    return str(path)


# ── Endpoints ─────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    import torch
    mode = _get_mode()
    return {
        "status": "ok",
        "mode": mode,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.post("/fold", response_model=PredictionResult)
async def fold_protein(request: PredictionRequest):
    """Predict 3D structure for a protein or biomolecular complex."""
    mode = _get_mode()

    try:
        if mode == "local":
            from app.local_inference import get_engine
            engine = get_engine()
            result = engine.predict(request)
        elif mode == "api":
            from app.api_client import get_api_client
            client = get_api_client()
            result = client.predict(request)
        else:
            raise HTTPException(400, f"Unknown mode: {mode}")
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")

    # Save to output directory
    saved_path = _save_result(result)
    result.mmcif = f"Saved to {saved_path}"

    return result


@app.post("/fold/raw", response_class=PlainTextResponse)
async def fold_raw(request: PredictionRequest):
    """Predict and return raw mmCIF output."""
    mode = _get_mode()

    if mode == "local":
        from app.local_inference import get_engine
        result = get_engine().predict(request)
    elif mode == "api":
        from app.api_client import get_api_client
        result = get_api_client().predict(request)
    else:
        raise HTTPException(400, f"Unknown mode: {mode}")

    _save_result(result)
    return result.mmcif


@app.get("/fold/job/{job_id}")
async def get_job_status(job_id: str):
    """Get batch job status (requires Redis + worker)."""
    import redis
    r = redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    data = r.get(f"job:{job_id}")
    if data is None:
        raise HTTPException(404, f"Job {job_id} not found")
    import json
    return json.loads(data)


# ── SMILES / Compound endpoints ──────────────────────────────────────
@app.post("/ccd/search")
async def search_ccd(smiles: str):
    """Search for CCD code matching a SMILES string (online PubChem/PDBe + local fallback)."""
    from app.ccd_online import search_compound
    from app.ccd_database import get_ccd_db
    result = search_compound(smiles)
    db = get_ccd_db()
    canonical = db._canonicalize(smiles)
    return CcdSearchResult(
        smiles=smiles,
        canonical_smiles=canonical or result.get("canonical_smiles"),
        match_type=result["match_type"],
        ccd_code=result.get("ccd_code"),
        candidates=result.get("candidates", []),
    )


@app.post("/bind")
async def predict_binding(request: SmilesBindingRequest):
    """Predict protein-ligand binding. SMILES directly (no CCD needed) or CCD codes."""
    if not request.smiles and not request.ccd:
        raise HTTPException(400, "Either smiles or ccd must be provided")

    from app.pipeline import get_pipeline
    pipe = get_pipeline()
    result = pipe.predict_binding(
        protein_sequence=request.protein_sequence,
        smiles=request.smiles or None,
        ccd=request.ccd or None,
        protein_id=request.protein_id,
        ligand_id=request.ligand_id,
        num_loops=request.num_loops,
        num_sampling_steps=request.num_sampling_steps,
        seed=request.seed,
        pocket_residues=[tuple(r) for r in request.pocket_residues] if request.pocket_residues else None,
        covalent_bonds=request.covalent_bonds,
    )
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@app.post("/bind/auto")
async def predict_binding_auto(request: SmilesBindingRequest):
    """Auto-detect pockets on folded protein (quality-filtered), then co-fold with ligand."""
    if not request.smiles and not request.ccd:
        raise HTTPException(400, "Either smiles or ccd must be provided")

    from app.pipeline import get_pipeline
    pipe = get_pipeline()
    result = pipe.predict_binding_auto(
        protein_sequence=request.protein_sequence,
        smiles=request.smiles or None,
        ccd=request.ccd or None,
        protein_id=request.protein_id,
        ligand_id=request.ligand_id,
        num_loops=request.num_loops,
        num_sampling_steps=request.num_sampling_steps,
        seed=request.seed,
        min_quality=request.min_pocket_quality,
        min_volume=request.min_pocket_volume,
        min_residues=request.min_pocket_residues,
    )
    return result


# ── Vina Docking endpoint (deprecated — use /cascade) ─────────────────
@app.post("/dock")
async def dock_compound(request: SmilesBindingRequest):
    """[DEPRECATED] ESMFold2 fold + fpocket + Vina docking.

    This endpoint uses the legacy fpocket→Vina→composite_score pipeline.
    Prefer POST /cascade for the new Gnina + MM-GBSA cascade.
    """
    import warnings
    warnings.warn(
        "VinaDockingPipeline (/dock) is deprecated. Use /cascade for the "
        "new Gnina + MM-GBSA pipeline, or /cascade/batch for multi-compound screening.",
        DeprecationWarning,
    )

    if not request.smiles:
        raise HTTPException(400, "smiles is required for docking")

    from app.docking import get_docking
    d = get_docking()
    result = d.dock(
        protein_sequence=request.protein_sequence,
        smiles=request.smiles,
        num_loops=request.num_loops,
        num_sampling_steps=request.num_sampling_steps,
        seed=request.seed,
        min_quality=request.min_pocket_quality,
        min_volume=request.min_pocket_volume,
        min_residues=request.min_pocket_residues,
    )
    if "error" in result and result.get("best_affinity") is None:
        raise HTTPException(500, result["error"])
    return result


# ── Cascade screening endpoints (recall-first, three-leg) ────────────

@app.post("/cascade")
async def cascade_single(request: CascadeRequest):
    """Single-compound recall-first cascade (three legs → merge).

    Leg 1 (pocket_off): unconstrained co-fold → honest PAE signal.
    Leg 2 (pocket_on): PocketConditioning co-fold per candidate site.
    Leg 3 (orthogonal): Gnina docking rescue (only if Leg1+2 fail/weak).

    Returns CompoundMergeResult with confidence_tier and routing_log.
    """
    if not request.smiles_list:
        raise HTTPException(400, "smiles_list is required")

    from app.cascade import get_cascade
    cascade = get_cascade()
    result = cascade.run_single(
        protein_sequence=request.protein_sequence,
        smiles=request.smiles_list[0],
        seed=request.seed,
        candidate_sites=request.pocket_sites,
    )
    return result


@app.post("/cascade/batch")
async def cascade_batch(request: CascadeRequest):
    """Multi-compound recall-first cascade.

    All compounds run through all configured legs. Results include:
    - kept: ranked by MM-GBSA ΔG_GB
    - parked: compounds where ALL legs + rescue failed (never deleted)
    - Each compound has confidence_tier (convergent/divergent/rescue/parked)
    """
    import json

    job_id = uuid.uuid4().hex[:12]

    # Try persist as pending
    try:
        import redis
        r = redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
        r.set(f"cascade_job:{job_id}", json.dumps({
            "job_id": job_id, "status": "running",
            "n_compounds": len(request.smiles_list),
        }))
    except Exception:
        pass

    from app.cascade import get_cascade
    cascade = get_cascade()
    result = cascade.run(
        protein_sequence=request.protein_sequence,
        smiles_list=request.smiles_list,
        seed=request.seed,
        candidate_sites=request.pocket_sites,
    )
    return result


@app.get("/cascade/jobs/{job_id}")
async def cascade_job_status(job_id: str):
    """Poll cascade job status (kept + parked counts)."""
    try:
        import redis, json
        r = redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
        data = r.get(f"cascade_job:{job_id}")
        if data is None:
            raise HTTPException(404, f"Job {job_id} not found")
        return json.loads(data)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(503, "Job status unavailable (Redis not configured)")


@app.get("/cascade/parked/{job_id}")
async def cascade_parked(job_id: str):
    """Retrieve parked compounds for a completed cascade job.

    Parked compounds have failed ALL legs and ALL rescue attempts.
    They are never deleted — full audit trail preserved.
    """
    raise HTTPException(501, "Parked bucket retrieval requires persistent storage. "
                             "Use the parked list in the /cascade/batch response.")


# ── Batch endpoints ───────────────────────────────────────────────────
@app.post("/batch/submit")
async def batch_submit(requests: list[PredictionRequest]):
    """Submit a batch of predictions (async processing with Redis queue)."""
    import json
    import redis
    r = redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))

    job_ids = []
    for req in requests:
        job_id = uuid.uuid4().hex[:12]
        job = {
            "job_id": job_id,
            "status": "pending",
            "request": req.model_dump(),
        }
        r.set(f"job:{job_id}", json.dumps(job))
        r.lpush("queue:pending", job_id)
        job_ids.append(job_id)

    return {"submitted": len(job_ids), "job_ids": job_ids}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
