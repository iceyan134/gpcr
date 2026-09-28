"""
Local ESMFold2 model inference engine.
Wraps the esm package for protein/complex structure prediction on local GPU.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from app.models import (
    ChainInput,
    FoldingConfig,
    MoleculeType,
    PredictionRequest,
    PredictionResult,
)

logger = logging.getLogger(__name__)


# ── Detailed co-fold result for cascade ──────────────────────────────────


@dataclass
class DetailedFoldResult:
    """Co-fold output with per-token confidence arrays for the cascade gate.

    Mirrors PredictionResult summary fields plus the full pLDDT/PAE arrays
    and token-to-chain mapping needed by confidence_gate.
    """

    job_id: str
    mmcif: str
    plddt_mean: float
    ptm: float
    iptm: float | None
    num_chains: int
    num_residues: int
    wall_time_s: float

    # Per-token arrays
    plddt_array: np.ndarray  # (n_tokens,)  per-token pLDDT
    pae_matrix: np.ndarray | None  # (n_tokens, n_tokens)  PAE in Angstrom

    # Token mapping
    token_chain_ids: list[str]  # chain id per token
    protein_token_mask: np.ndarray  # boolean mask for protein tokens
    ligand_token_mask: np.ndarray  # boolean mask for ligand tokens
    protein_residue_indices: np.ndarray  # residue seqid per protein token

    # Per-sample ligand coordinates for cross-RMSD consistency check
    sample_ligand_coords: list[np.ndarray] = field(default_factory=list)

    # File paths for downstream tools
    complex_path: Path | None = None
    smiles: str = ""

    @property
    def n_tokens(self) -> int:
        return len(self.plddt_array)

    @property
    def n_ligand_atoms(self) -> int:
        return int(self.ligand_token_mask.sum())


# ── Engine ────────────────────────────────────────────────────────────────


class LocalInferenceEngine:
    def __init__(self, model_id: str = "biohub/ESMFold2", device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self._model = None
        self._builder = None

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def builder(self):
        if self._builder is None:
            from esm.models.esmfold2 import ESMFold2InputBuilder

            self._builder = ESMFold2InputBuilder()
        return self._builder

    def _load_model(self):
        logger.info(f"Loading ESMFold2 model: {self.model_id}")
        t0 = time.time()

        from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

        self._model = ESMFold2Model.from_pretrained(
            self.model_id,
        )
        self._model = self._model.to(self.device)
        self._model.eval()

        elapsed = time.time() - t0
        logger.info(f"Model loaded in {elapsed:.1f}s on {self.device}")

    def _build_prediction_input(self, chains: list[ChainInput]):
        from esm.models.esmfold2 import (
            DNAInput,
            LigandInput,
            Modification,
            ProteinInput,
            RNAInput,
            StructurePredictionInput,
        )

        sequences = []
        for chain in chains:
            mods = [
                Modification(position=m.position, ccd=m.ccd)
                for m in chain.modifications
            ]

            if chain.type == MoleculeType.protein:
                sequences.append(
                    ProteinInput(
                        id=chain.id, sequence=chain.sequence, modifications=mods
                    )
                )
            elif chain.type == MoleculeType.dna:
                sequences.append(
                    DNAInput(id=chain.id, sequence=chain.sequence, modifications=mods)
                )
            elif chain.type == MoleculeType.rna:
                sequences.append(
                    RNAInput(id=chain.id, sequence=chain.sequence, modifications=mods)
                )
            elif chain.type == MoleculeType.ligand:
                if chain.smiles:
                    sequences.append(LigandInput(id=chain.id, smiles=chain.smiles))
                else:
                    sequences.append(
                        LigandInput(id=chain.id, ccd=chain.ccd if chain.ccd else None)
                    )
            else:
                raise ValueError(f"Unknown molecule type: {chain.type}")

        return StructurePredictionInput(sequences=sequences)

    # ── Standard predict (backward-compatible) ────────────────────────────

    def predict(
        self, request: PredictionRequest, pocket=None, covalent_bonds=None
    ) -> PredictionResult:
        t0 = time.time()
        job_id = uuid.uuid4().hex[:12]

        model = self.model
        spi = self._build_prediction_input(request.chains)

        if pocket is not None:
            spi.pocket = pocket
        if covalent_bonds is not None:
            spi.covalent_bonds = covalent_bonds

        cfg = request.config

        result = self.builder.fold(
            model,
            spi,
            num_loops=cfg.num_loops,
            num_sampling_steps=cfg.num_sampling_steps,
            num_diffusion_samples=cfg.num_diffusion_samples,
            seed=cfg.seed,
        )

        mmcif_str = result.complex.to_mmcif()
        plddt_mean = float(result.plddt.mean())
        ptm = float(result.ptm)
        iptm = float(result.iptm) if result.iptm is not None else None
        num_residues = int(result.plddt.shape[0])

        elapsed = time.time() - t0

        return PredictionResult(
            job_id=job_id,
            name=request.name,
            mmcif=mmcif_str,
            plddt_mean=round(plddt_mean, 4),
            ptm=round(ptm, 4),
            iptm=round(iptm, 4) if iptm else None,
            num_chains=len(request.chains),
            num_residues=num_residues,
            wall_time_s=round(elapsed, 1),
        )

    # ── Detailed predict for cascade ──────────────────────────────────────

    def predict_with_details(
        self,
        request: PredictionRequest,
        pocket=None,
        covalent_bonds=None,
        n_samples: int = 1,
        smiles: str = "",
        output_dir: str | Path = "/output",
        pocket_residues: list[int] | None = None,
        binder_chain_id: str = "L",
    ) -> DetailedFoldResult:
        """Co-fold and return per-token pLDDT, PAE, and token-to-chain mapping.

        Parameters
        ----------
        n_samples : int
            Number of independent co-fold samples for cross-RMSD consistency.
        smiles : str
            Reference SMILES for downstream bond-order perception.
        pocket_residues : list[int] | None
            If provided, construct PocketConditioning to constrain ligand to
            these residues (sequence-position 1-indexed). Used by Leg 2
            (pocket-on). Residue numbering is verified against protein sequence
            length before use.
        binder_chain_id : str
            Chain ID of the ligand being constrained (default "L").
        """
        t0 = time.time()
        job_id = uuid.uuid4().hex[:12]
        output_dir = Path(output_dir)

        model = self.model
        spi = self._build_prediction_input(request.chains)

        # ── PocketConditioning (Leg 2 / pocket-on) ─────────────────────
        if pocket_residues and not pocket:
            pocket = _build_pocket_conditioning(
                pocket_residues,
                binder_chain_id,
                request.chains[0].sequence if request.chains else "",
            )

        if pocket is not None:
            spi.pocket = pocket
        if covalent_bonds is not None:
            spi.covalent_bonds = covalent_bonds

        cfg = request.config

        # ── Primary sample ────────────────────────────────────────────
        result = self.builder.fold(
            model,
            spi,
            num_loops=cfg.num_loops,
            num_sampling_steps=cfg.num_sampling_steps,
            num_diffusion_samples=cfg.num_diffusion_samples,
            seed=cfg.seed,
        )

        mmcif_str = result.complex.to_mmcif()
        # pLDDT in [0,100] scale (match mmCIF B-factors and gate thresholds)
        plddt_array = (
            result.plddt.cpu().numpy()
            if hasattr(result.plddt, "cpu")
            else np.asarray(result.plddt)
        )
        if plddt_array.max() < 1.1:  # detected [0,1] scale → convert to [0,100]
            plddt_array = plddt_array * 100.0

        # Extract PAE matrix if available
        pae_matrix = None
        try:
            if hasattr(result, "pae") and result.pae is not None:
                pae_matrix = (
                    result.pae.cpu().numpy()
                    if hasattr(result.pae, "cpu")
                    else np.asarray(result.pae)
                )
        except Exception:
            logger.debug("PAE not available from ESMFold2 result; gate will degrade")

        # ── Token mapping from mmCIF ──────────────────────────────────
        tok = _reconstruct_tokens_from_mmcif(
            mmcif_str,
            ligand_chain=request.chains[-1].id if request.chains else "L",
        )

        # Build masks aligned with plddt_array order
        n_tok = len(plddt_array)
        protein_mask = np.zeros(n_tok, dtype=bool)
        ligand_mask = np.zeros(n_tok, dtype=bool)
        chain_ids = [""] * n_tok
        residue_indices = np.full(n_tok, -1, dtype=int)

        for i, pos in enumerate(tok.protein_token_positions):
            if pos < n_tok:
                protein_mask[pos] = True
                chain_ids[pos] = (
                    tok.protein_token_chain[i]
                    if i < len(tok.protein_token_chain)
                    else "A"
                )
                residue_indices[pos] = tok.protein_token_residue[i]

        for pos in tok.ligand_token_indices:
            if pos < n_tok:
                ligand_mask[pos] = True
                chain_ids[pos] = request.chains[-1].id if request.chains else "L"

        # ── Additional samples for cross-RMSD ─────────────────────────
        sample_coords = []
        primary_lig_coords = _extract_ligand_coords(result)
        if primary_lig_coords is not None:
            sample_coords.append(primary_lig_coords)

        for si in range(1, n_samples):
            try:
                r = self.builder.fold(
                    model,
                    spi,
                    num_loops=cfg.num_loops,
                    num_sampling_steps=cfg.num_sampling_steps,
                    num_diffusion_samples=cfg.num_diffusion_samples,
                    seed=cfg.seed + si,
                )
                coords = _extract_ligand_coords(r)
                if coords is not None:
                    sample_coords.append(coords)
            except Exception:
                logger.warning("Cross-sample %d/%d failed; skipping", si + 1, n_samples)

        # ── Save complex mmCIF ────────────────────────────────────────
        complex_path = output_dir / f"{job_id}.cif"
        complex_path.parent.mkdir(parents=True, exist_ok=True)
        complex_path.write_text(mmcif_str)

        elapsed = time.time() - t0

        return DetailedFoldResult(
            job_id=job_id,
            mmcif=mmcif_str,
            plddt_mean=round(float(plddt_array.mean()), 1),
            ptm=float(result.ptm),
            iptm=float(result.iptm) if result.iptm is not None else None,
            num_chains=len(request.chains),
            num_residues=int(plddt_array.shape[0]),
            wall_time_s=round(elapsed, 1),
            plddt_array=plddt_array,
            pae_matrix=pae_matrix,
            token_chain_ids=chain_ids,
            protein_token_mask=protein_mask,
            ligand_token_mask=ligand_mask,
            protein_residue_indices=residue_indices,
            sample_ligand_coords=sample_coords,
            complex_path=complex_path,
            smiles=smiles,
        )


# ── Token reconstruction from mmCIF ───────────────────────────────────────


@dataclass
class _TokenMap:
    n_protein: int
    n_ligand: int
    protein_token_residue: np.ndarray
    protein_token_positions: np.ndarray
    protein_token_chain: list[str]
    ligand_token_indices: np.ndarray


def _reconstruct_tokens_from_mmcif(
    mmcif_text: str,
    ligand_chain: str = "L",
) -> _TokenMap:
    """Walk mmCIF in file order to map tokens to protein residues / ligand atoms.

    Uses ESMFold2-specific mmCIF parser (no Biopython dependency) that handles
    ESMFold2's numeric chain labels and omitted standard fields.

    Assumption (verified by Step 0): protein per-residue, ligand per-atom tokens,
    in mmCIF file order, matching pLDDT/PAE array order.
    """
    from app.mmcif_parser import parse_esmfold_mmcif

    tok = parse_esmfold_mmcif(mmcif_text, polymer_chain="A", ligand_chain=ligand_chain)

    return _TokenMap(
        n_protein=tok.n_protein_residues,
        n_ligand=tok.n_ligand_atoms,
        protein_token_residue=tok.protein_residue_indices,
        protein_token_positions=tok.protein_token_positions,
        protein_token_chain=tok.protein_chain_ids,
        ligand_token_indices=tok.ligand_token_positions,
    )


def _extract_ligand_coords(fold_result, ligand_chain: str = "L") -> np.ndarray | None:
    """Extract ligand heavy-atom coordinates from a fold result's mmCIF."""
    mmcif = fold_result.complex.to_mmcif()

    # Parse mmCIF manually (ESMFold2 mmCIF is missing _atom_site.occupancy,
    # which breaks Bio.PDB.MMCIFParser).
    field_idx: dict[str, int] = {}
    coords = []
    for line in mmcif.splitlines():
        if line.startswith("_atom_site."):
            field_idx[line.strip().split(".", 1)[1]] = len(field_idx)
        elif line.startswith(("ATOM", "HETATM")):
            break

    grp = field_idx.get("group_PDB")
    chain_f = field_idx.get("label_asym_id")
    x_f, y_f, z_f = (
        field_idx.get("Cartn_x"),
        field_idx.get("Cartn_y"),
        field_idx.get("Cartn_z"),
    )
    elem_f = field_idx.get("type_symbol")
    min_cols = (
        max(grp, chain_f, x_f, y_f, z_f, elem_f) + 1
        if None not in (grp, chain_f, x_f, y_f, z_f, elem_f)
        else 18
    )

    for line in mmcif.splitlines():
        s = line.strip()
        if not s or s[0] == "#" or s.startswith(("data_", "loop_", "_")):
            continue
        parts = s.split()
        if len(parts) < min_cols:
            continue
        if parts[grp] == "HETATM" and parts[chain_f] == ligand_chain:
            if elem_f is not None and parts[elem_f] == "H":
                continue
            coords.append(
                [
                    float(parts[x_f]),
                    float(parts[y_f]),
                    float(parts[z_f]),
                ]
            )

    return np.asarray(coords) if coords else None


def _build_pocket_conditioning(
    pocket_residues: list[int],
    binder_chain_id: str = "L",
    protein_sequence: str = "",
    mmcif_source: str | None = None,
):
    """Build PocketConditioning for Leg 2 (pocket-on) co-folding.

    pocket_residues may come from:
      - ESMFold2-generated mmCIF → label_seq_id == 1-indexed sequence position → OK
      - External PDB (fpocket) → residue numbers may differ from sequence positions

    If mmcif_source is provided, attempts to build a mapping from PDB residue
    numbers to 1-indexed sequence positions.
    """
    # NOTE: the heavy esm.utils import is deferred until after boundary checks
    # so that invalid pocket_residues fail fast (and the checks are unit-testable
    # in environments without the esm package).

    # Build a mapping: PDB residue number → 1-indexed sequence position.
    # pocket_residues may carry external-PDB numbering (e.g. GPCR crystal
    # templates with insertion codes like "123A"); these MUST be remapped to
    # sequence positions before building contacts, or PocketConditioning will
    # silently constrain the wrong residue.
    mapped_contacts: list[tuple[str, int]] | None = None
    if mmcif_source and protein_sequence:
        seq_positions = _extract_label_seq_ids(mmcif_source)
        if seq_positions:
            mapped_contacts = []
            unmapped = []
            for r in pocket_residues:
                m = seq_positions.get(r)
                if m is None:
                    unmapped.append(r)
                else:
                    mapped_contacts.append(("A", int(m)))
            if unmapped:
                logger.warning(
                    "PocketConditioning: %d residue(s) not found in seq mapping "
                    "(will use raw numbers): %s",
                    len(unmapped),
                    unmapped[:10],
                )
                for r in unmapped:
                    mapped_contacts.append(("A", int(r)))
            logger.info(
                "PocketConditioning: mapped %d/%d residues via label_seq_id",
                len(mapped_contacts) - len(unmapped),
                len(pocket_residues),
            )
        # else: seq_positions empty → fall through to raw numbers (below)

    if mapped_contacts is None:
        mapped_contacts = [("A", int(r)) for r in pocket_residues]

    # ── HARD boundary check (was soft warning) ─────────────────────────────
    # External-PDB residue numbers that fall outside the protein sequence are
    # a silent-failure hazard: they would constrain a non-existent residue and
    # produce garbage poses. Fail loudly instead of propagating bad poses.
    if protein_sequence:
        seq_len = len(protein_sequence)
        bad = [r for _c, r in mapped_contacts if r < 1 or r > seq_len]
        if bad:
            raise ValueError(
                f"PocketConditioning contact(s) {bad} outside protein sequence "
                f"(length={seq_len}). Remap external PDB residue numbers "
                f"(incl. insertion codes) to sequence positions before calling. "
                f"pocket_residues={pocket_residues}"
            )

    logger.info(
        "PocketConditioning: binder=%s, contacts=%d residues",
        binder_chain_id,
        len(mapped_contacts),
    )
    from esm.utils.structure.input_builder import PocketConditioning

    return PocketConditioning(binder_chain_id=binder_chain_id, contacts=mapped_contacts)


def _extract_label_seq_ids(mmcif_text: str) -> dict[int, int]:
    """Extract a mapping from PDB residue number to 1-indexed sequence position
    from an mmCIF file.

    For ESMFold2-generated mmCIF, label_seq_id IS the 1-indexed sequence
    position. For external PDBs, auth_seq_id may carry insertion codes
    (e.g. "123A"); the integer prefix is used for lookup.
    """
    mapping: dict[int, int] = {}
    in_loop = False
    col_map: dict[str, int] = {}
    for line in mmcif_text.splitlines():
        if line.startswith("loop_"):
            in_loop = True
            col_map = {}
            continue
        if in_loop and line.startswith("_"):
            field = line.strip().split(".", 1)[1] if "." in line else ""
            col_map[field] = len(col_map)
            continue
        if in_loop and (line.startswith("ATOM") or line.startswith("HETATM")):
            parts = line.strip().split()
            auth_idx = col_map.get("auth_seq_id")
            label_idx = col_map.get("label_seq_id")
            if auth_idx is None or label_idx is None:
                continue
            if auth_idx >= len(parts) or label_idx >= len(parts):
                continue
            auth_raw = parts[auth_idx]
            try:
                # Strip insertion code suffix if present ("123A" → 123)
                auth = _parse_residue_number(auth_raw)
            except ValueError:
                logger.debug(
                    "Skipping residue with non-numeric auth_seq_id %r", auth_raw
                )
                continue
            try:
                label = int(parts[label_idx])
            except ValueError:
                continue
            mapping.setdefault(auth, label)
    return mapping


def _parse_residue_number(raw: str) -> int:
    """Parse a PDB residue number, tolerating insertion codes ('123A' → 123).
    Raises ValueError if no leading integer is present."""
    s = raw.strip()
    num = ""
    for ch in s:
        if ch.isdigit() or ch == "-":
            num += ch
        else:
            break
    if not num:
        raise ValueError(f"no leading integer in residue id {raw!r}")
    return int(num)


# ── Singleton ─────────────────────────────────────────────────────────────

_engine: Optional[LocalInferenceEngine] = None


def get_engine() -> LocalInferenceEngine:
    global _engine
    if _engine is None:
        import os

        model_id = os.environ.get("ESMFOLD2_MODEL_ID", "biohub/ESMFold2")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _engine = LocalInferenceEngine(model_id=model_id, device=device)
    return _engine


def unload_engine():
    """Free ESMFold2 model from GPU memory so OpenMM can use it."""
    global _engine
    if _engine is not None and _engine._model is not None:
        logger.info("Unloading ESMFold2 model from GPU...")
        _engine._model = None
        _engine._builder = None
        _engine = None
        torch.cuda.empty_cache()
        logger.info("ESMFold2 unloaded, GPU memory freed.")
