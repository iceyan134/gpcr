"""
Direct mmCIF parser for ESMFold2 output — no Biopython dependency.

ESMFold2 mmCIF uses numeric chain labels and may omit standard fields
(occupancy, formal_charge) that Biopython's MMCIFParser expects.
This parser reads the _atom_site loop directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ESMFoldTokenMap:
    """Token mapping from ESMFold2 mmCIF file-order traversal."""
    n_protein_residues: int
    n_ligand_atoms: int
    n_total_tokens: int
    protein_residue_indices: np.ndarray   # (n_protein,) residue seqids per token position
    protein_chain_ids: list[str]          # chain per protein token
    protein_token_positions: np.ndarray   # (n_protein,) token-axis positions
    ligand_token_positions: np.ndarray    # (n_ligand,) token-axis positions
    per_token_plddt: np.ndarray           # (n_tokens,) B-factor values
    ligand_centroid: np.ndarray | None    # (3,) mean ligand heavy-atom coordinates

    @property
    def n_tokens(self) -> int:
        return self.n_total_tokens


def parse_esmfold_mmcif(
    mmcif_text: str,
    polymer_chain: str = "A",
    ligand_chain: str = "L",
) -> ESMFoldTokenMap:
    """Parse ESMFold2 mmCIF and build token map in file order.

    ESMFold2 tokenises protein per-RESIDUE and ligand per-HEAVY-ATOM.
    Token order in pLDDT/PAE arrays = file order of atoms in mmCIF.

    Parameters
    ----------
    polymer_chain : str
        Chain label for protein in the mmCIF (ESMFold2 typically uses "A").
    ligand_chain : str
        Chain label for ligand (ESMFold2 typically uses "L").
    """
    # ── Parse field indices from header ───────────────────────────────
    fields = {}
    for line in mmcif_text.splitlines():
        if line.startswith("_atom_site."):
            field = line.strip().split(".", 1)[1]
            fields[field] = len(fields)
        if line.startswith("ATOM") or line.startswith("HETATM"):
            break

    required = ["group_PDB", "label_asym_id", "label_seq_id", "B_iso_or_equiv",
                "Cartn_x", "Cartn_y", "Cartn_z", "type_symbol"]
    missing = [f for f in required if f not in fields]
    if missing:
        raise KeyError(f"mmCIF missing required fields: {missing}. Found: {list(fields)}")

    grp_idx = fields["group_PDB"]
    chain_idx = fields["label_asym_id"]
    resnum_idx = fields["label_seq_id"]
    bfac_idx = fields["B_iso_or_equiv"]
    x_idx, y_idx, z_idx = fields["Cartn_x"], fields["Cartn_y"], fields["Cartn_z"]
    elem_idx = fields["type_symbol"]

    # ── Walk atoms in file order ──────────────────────────────────────
    seen_protein = set()        # (chain, resnum) already tokenised
    protein_residues = []       # residue seqid per protein token
    protein_chains = []         # chain per protein token
    protein_positions = []      # token-axis positions
    ligand_positions = []       # token-axis positions
    plddt = []                  # per-token B-factor
    lig_coords = []             # ligand heavy-atom coords
    t = 0

    for line in mmcif_text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("data_") or s.startswith("loop_"):
            continue
        if s.startswith("_"):
            continue
        parts = s.split()
        if len(parts) < 18:
            continue

        group = parts[grp_idx]
        chain = parts[chain_idx]
        elem = parts[elem_idx]

        # Skip hydrogens
        if elem == "H":
            continue

        if group == "ATOM" and chain == polymer_chain:
            resnum = int(parts[resnum_idx])
            key = (chain, resnum)
            if key not in seen_protein:
                seen_protein.add(key)
                protein_residues.append(resnum)
                protein_chains.append(chain)
                protein_positions.append(t)
                plddt.append(float(parts[bfac_idx]))
                t += 1

        elif group == "HETATM" and chain == ligand_chain:
            ligand_positions.append(t)
            plddt.append(float(parts[bfac_idx]))
            lig_coords.append([float(parts[x_idx]), float(parts[y_idx]), float(parts[z_idx])])
            t += 1

    centroid = np.mean(lig_coords, axis=0) if lig_coords else None

    return ESMFoldTokenMap(
        n_protein_residues=len(protein_residues),
        n_ligand_atoms=len(ligand_positions),
        n_total_tokens=t,
        protein_residue_indices=np.asarray(protein_residues, dtype=int),
        protein_chain_ids=protein_chains,
        protein_token_positions=np.asarray(protein_positions, dtype=int),
        ligand_token_positions=np.asarray(ligand_positions, dtype=int),
        per_token_plddt=np.asarray(plddt, dtype=float),
        ligand_centroid=centroid,
    )


def write_mmcif_to_pdb(mmcif_text: str, output_path: Path, chain_map: dict | None = None):
    """Convert ESMFold2 mmCIF to PDB with all heavy atoms (protein + ligand).

    chain_map: optional dict mapping mmCIF chain labels → PDB chain letters.
               Default: {"A": "A", "L": "L"} for standard ESMFold2 output.
    """
    if chain_map is None:
        chain_map = {}

    fields = {}
    for line in mmcif_text.splitlines():
        if line.startswith("_atom_site."):
            field = line.strip().split(".", 1)[1]
            fields[field] = len(fields)
        if line.startswith("ATOM") or line.startswith("HETATM"):
            break

    grp_idx = fields.get("group_PDB", 0)
    chain_idx = fields.get("label_asym_id", 5)
    atom_idx = fields.get("label_atom_id", 2)
    resname_idx = fields.get("label_comp_id", 4)
    resnum_idx = fields.get("label_seq_id", 7)
    x_idx, y_idx, z_idx = fields["Cartn_x"], fields["Cartn_y"], fields["Cartn_z"]
    elem_idx = fields.get("type_symbol", 1)
    bfac_idx = fields.get("B_iso_or_equiv", 13)

    lines_out = []
    atom_serial = 0
    for line in mmcif_text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("data_") or s.startswith("loop_") or s.startswith("_"):
            continue
        parts = s.split()
        if len(parts) < 18:
            continue

        group = parts[grp_idx]
        chain_label = parts[chain_idx]
        elem = parts[elem_idx]
        atom_name = parts[atom_idx]

        if elem == "H":
            continue

        pdb_chain = chain_map.get(chain_label, chain_label)
        atom_serial += 1
        resname = parts[resname_idx]
        resnum = int(parts[resnum_idx])
        x, y, z = float(parts[x_idx]), float(parts[y_idx]), float(parts[z_idx])
        bfac = float(parts[bfac_idx])

        lines_out.append(
            f"ATOM  {atom_serial:5d} {atom_name:4s} {resname:3s} {pdb_chain:1s}{resnum:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{bfac:6.2f}      {pdb_chain:1s}  "
        )

    lines_out.append("END\n")
    output_path.write_text("\n".join(lines_out))
