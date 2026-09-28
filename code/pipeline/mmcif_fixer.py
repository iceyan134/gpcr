"""
Fix ESMFold2 mmCIF to be compatible with Boltz-2's parse_mmcif.

ESMFold2 mmCIF is missing:
  - _entity_poly (entity_id, type, pdbx_seq_one_letter_code)
  - _entity_poly_seq (entity_id, mon_id, seq_id)

We add these tables to the mmCIF string before feeding to parse_mmcif.
"""
from __future__ import annotations

import re


def fix_mmcif_for_boltz(mmcif_text: str, protein_sequence: str) -> str:
    """Add missing entity_poly and entity_poly_seq tables to ESMFold2 mmCIF.

    Boltz-2's parse_mmcif requires these tables for polymer parsing.
    The mmCIF must already have _entity and _struct_asym tables.

    Parameters
    ----------
    mmcif_text : str
        Raw mmCIF text from ESMFold2.
    protein_sequence : str
        One-letter amino acid sequence of the protein chain.

    Returns
    -------
    str
        Patched mmCIF text compatible with Boltz-2 parse_mmcif.
    """
    # Find the entity IDs for polymer chains
    # ESMFold2 format: entity 1 = polymer, entity 2 = ligand
    lines = mmcif_text.splitlines()

    # Remove trailing "#" if present
    while lines and lines[-1].strip() == "#":
        lines = lines[:-1]

    # Build _entity_poly table
    poly_lines = [
        "loop_",
        "_entity_poly.entity_id",
        "_entity_poly.type",
        "_entity_poly.pdbx_seq_one_letter_code",
        "_entity_poly.pdbx_seq_one_letter_code_can",
        "1 polypeptide(L) " + protein_sequence + " " + protein_sequence,
    ]

    # Build _entity_poly_seq table
    poly_seq_lines = [
        "loop_",
        "_entity_poly_seq.entity_id",
        "_entity_poly_seq.mon_id",
        "_entity_poly_seq.seq_id",
    ]
    one_letter_to_three = {
        'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
        'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
        'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
        'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
        'U': 'SEC', 'O': 'PYL',
    }
    for i, aa in enumerate(protein_sequence, start=1):
        three = one_letter_to_three.get(aa, 'UNK')
        poly_seq_lines.append(f"1 {three} {i}")

    # Add trailing #
    poly_lines.append("#")
    poly_seq_lines.append("#")

    result = lines + [""] + poly_lines + [""] + poly_seq_lines + [""] + ["#"]

    return "\n".join(result)