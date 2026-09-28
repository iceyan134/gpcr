from __future__ import annotations

import enum
from typing import Optional

from pydantic import BaseModel, Field


class MoleculeType(str, enum.Enum):
    protein = "protein"
    dna = "dna"
    rna = "rna"
    ligand = "ligand"


class Modification(BaseModel):
    position: int = Field(..., ge=1, description="1-based residue position")
    ccd: str = Field(..., description="CCD code for the chemical modification")


class ChainInput(BaseModel):
    id: str = Field(..., description="Chain identifier, e.g. 'A', 'B'")
    sequence: str = Field(default="", description="Sequence (empty for ligand with SMILES/CCD)")
    type: MoleculeType = MoleculeType.protein
    modifications: list[Modification] = Field(default_factory=list)
    smiles: Optional[str] = Field(default=None, description="SMILES string for ligand (primary method)")
    ccd: list[str] = Field(default_factory=list, description="CCD codes for known PDB ligands (fallback)")


class FoldingConfig(BaseModel):
    num_loops: int = Field(default=3, ge=1, le=20, description="Recycling iterations")
    num_sampling_steps: int = Field(default=32, ge=1, le=200, description="Diffusion sampling steps")
    num_diffusion_samples: int = Field(default=1, ge=1, le=10, description="Number of diffusion samples")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    include_pae: bool = Field(default=True, description="Include PAE output")
    include_distogram: bool = Field(default=False, description="Include distogram prediction")


class PredictionRequest(BaseModel):
    name: str = Field(default="prediction", description="Job name for output file naming")
    chains: list[ChainInput] = Field(..., min_length=1, description="List of chains in the complex")
    config: FoldingConfig = Field(default_factory=FoldingConfig)


class PredictionResult(BaseModel):
    job_id: str
    name: str
    mmcif: str = Field(..., description="mmCIF format structure data")
    plddt_mean: float
    ptm: float
    iptm: Optional[float] = None
    num_chains: int
    num_residues: int
    wall_time_s: float


class SmilesBindingRequest(BaseModel):
    protein_sequence: str = Field(..., min_length=1, description="Amino acid sequence of the protein")
    smiles: str = Field(default="", description="SMILES string of the compound (primary method)")
    ccd: list[str] = Field(default_factory=list, description="CCD codes for known PDB ligands (alternative)")
    protein_id: str = Field(default="A")
    ligand_id: str = Field(default="L")
    num_loops: int = Field(default=3, ge=1, le=20)
    num_sampling_steps: int = Field(default=32, ge=1, le=200)
    seed: int = Field(default=42)
    pocket_residues: Optional[list[list]] = Field(default=None, description="Pocket contacts: [[chain_id, res_idx], ...]")
    covalent_bonds: Optional[list[dict]] = Field(default=None, description="Covalent bonds for userCCD mode")
    # Pocket quality filtering (for /dock endpoint)
    min_pocket_quality: float = Field(default=0.20, ge=0, le=1, description="Min composite structural quality score")
    min_pocket_volume: float = Field(default=100.0, ge=0, description="Min pocket volume in Å³")
    min_pocket_residues: int = Field(default=5, ge=1, description="Min pocket-lining residues")


class CcdSearchResult(BaseModel):
    smiles: str
    canonical_smiles: Optional[str] = None
    match_type: str  # exact, similar, none
    ccd_code: Optional[str] = None
    candidates: list[dict] = Field(default_factory=list)


class BatchJob(BaseModel):
    job_id: str
    status: str  # pending, running, completed, failed
    request: PredictionRequest
    result: Optional[PredictionResult] = None
    error: Optional[str] = None


# ── Cascade screening models (recall-first, multi-leg) ────────────────────

class CascadeRequest(BaseModel):
    """Request for the recall-first cascade screening pipeline."""
    protein_sequence: str = Field(..., min_length=1, description="Amino acid sequence of the protein")
    smiles_list: list[str] = Field(..., min_length=1, description="SMILES strings to screen")
    seed: int = Field(default=42, description="Base random seed")
    run_modes: list[str] = Field(default_factory=lambda: ["pocket_off", "pocket_on"],
                                 description="Legs to run")
    pocket_sites: Optional[list[list[int]]] = Field(default=None, description="Manual pocket residue list for Leg 2")
    config_overrides: dict = Field(default_factory=dict, description="Optional CascadeConfig overrides")


class GateSignalDetail(BaseModel):
    """Confidence gate signal from ONE leg (used for triage routing)."""
    source_leg: str = "pocket_off"        # "pocket_off" | "pocket_on" | "orthogonal"
    passed: bool
    pocket_residues: list[int] = Field(default_factory=list)
    pocket_plddt: Optional[float] = None
    ligand_plddt: Optional[float] = None
    interface_pae: Optional[float] = None
    ligand_iptm: Optional[float] = None
    reasons: list[str] = Field(default_factory=list)
    posebusters_passed: Optional[bool] = None
    posebusters_failed: list[str] = Field(default_factory=list)
    pose_rmsd_max: Optional[float] = None
    pose_rmsd_samples: int = 1


class GninaScoreDetail(BaseModel):
    """Gnina CNN --score_only result."""
    cnn_affinity: float
    cnn_score: float
    vina_affinity: float
    raw_output: Optional[str] = None


class BoltzScoreDetail(BaseModel):
    """Boltz-2 affinity prediction result."""
    affinity_binary: float
    affinity_value: float
    error: Optional[str] = None


class MMGBSADetail(BaseModel):
    """MM-GBSA result."""
    dg_gb: float                 # ΔG_bind (kcal/mol), more negative = stronger
    dg_std: Optional[float] = None
    raw_output: Optional[str] = None


class LegResult(BaseModel):
    """Result from one leg (pocket_off / pocket_on / orthogonal) for one compound."""
    leg: str                     # "pocket_off" | "pocket_on" | "orthogonal"
    site_label: str = ""         # pocket site label for pocket_on (e.g. "site_0", "site_1")
    status: str = "pending"      # "pending" | "running" | "ok" | "weak" | "invalid_pose" | "failed"
    gate: Optional[GateSignalDetail] = None
    gnina: Optional[GninaScoreDetail] = None
    boltz: Optional[BoltzScoreDetail] = None
    mmgbsa: Optional[MMGBSADetail] = None
    # For multiple poses from same leg/site (cross-sample or multi-site)
    pose_coords: list[list[list[float]]] = Field(default_factory=list)
    routing_log: list[str] = Field(default_factory=list)
    error: Optional[str] = None
    wall_time_s: Optional[float] = None


class CompoundMergeResult(BaseModel):
    """Merged result for one compound across all legs."""
    compound_id: str
    smiles: str
    status: str = "kept"         # "kept" | "parked"
    confidence_tier: str = ""    # "convergent" | "divergent" | "rescue" | "parked"

    # Per-leg results
    legs: dict[str, list[LegResult]] = Field(default_factory=dict)
    #   {"pocket_off": [LegResult], "pocket_on": [LegResult(site_0), ...], "orthogonal": [LegResult]}

    # Best available scores (from any leg)
    best_gnina: Optional[GninaScoreDetail] = None
    best_boltz: Optional[BoltzScoreDetail] = None
    best_mmgbsa: Optional[MMGBSADetail] = None
    best_dg_gb: Optional[float] = None     # primary ranking key

    # Calibrated consensus (affinity 0.65 + gate signals; app/consensus.py).
    # Additive alongside affinity-only ranking for comparability.
    consensus_score: Optional[float] = None

    # Merged gate summary
    gate_summary: dict = Field(default_factory=dict)

    # Routing audit
    routing_log: list[str] = Field(default_factory=list)
    agreement: str = ""          # "convergent" | "partial" | "divergent" | "single_source"

    # Park reasons (only when status="parked")
    park_reasons: list[str] = Field(default_factory=list)

    error: Optional[str] = None
    wall_time_s: Optional[float] = None


class CascadeResponse(BaseModel):
    """Response from the recall-first cascade pipeline."""
    job_id: str
    status: str = "completed"
    results: list[CompoundMergeResult] = Field(default_factory=list)
    parked: list[CompoundMergeResult] = Field(default_factory=list)
    funnel_summary: dict = Field(default_factory=dict)
    wall_time_s: float = 0.0


class CascadeJobStatus(BaseModel):
    """Status of an async multi-leg cascade job or merge job."""
    job_id: str
    job_type: str = "leg"        # "leg" | "merge"
    leg: str = ""                # "pocket_off" | "pocket_on" | "orthogonal" | "merge"
    status: str = "pending"      # "pending" | "running" | "completed" | "failed"
    progress: dict = Field(default_factory=dict)
    result: Optional[CascadeResponse] = None
    error: Optional[str] = None


# ═══════════════════════════════════════════════════════════
# Validation export rows (schema contracts)
# ═══════════════════════════════════════════════════════════

class CalibrationExportRow(BaseModel):
    """One row in calibration CSV (Leg-1 gate signals ONLY)."""
    compound_id: str
    label: int = 0
    interface_pae: Optional[float] = None
    ligand_plddt: Optional[float] = None
    pocket_plddt: Optional[float] = None
    ligand_iptm: Optional[float] = None
    consistency_rmsd: Optional[float] = None  # = gate.pose_rmsd_max
    posebusters_pass: Optional[bool] = None


class PoseManifestRow(BaseModel):
    """One row in pose validation manifest (aligned to reference frame)."""
    complex_id: str
    pred_sdf: str                    # cascade predicted ligand SDF, aligned to ref receptor
    ref_sdf: str                     # crystal reference ligand SDF
    receptor_pdb: str                # reference (crystal) receptor PDB
    method: str = "cascade"          # cascade / af3 / boltz2
    pocket_similarity: Optional[float] = None  # RNP/PLINDER training similarity in [0,1]
    ood: int = 0


class EnrichmentScoreRow(BaseModel):
    """One row in enrichment scores CSV."""
    target: str
    method: str                      # "cascade", "vina", "boltz2", etc.
    compound_id: str
    score: float
    label: int                       # 1=active, 0=inactive
    survived: Optional[bool] = None  # True if compound reached MM-GBSA tier
