"""
Binding pocket detection.
Uses DTWG-integrated fpocket PQR analysis for accurate pocket identification
with structural quality scoring and filtering.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def detect_pocket_residues(
    mmcif_text: str,
    chain_id: str = "A",
    top_n: int = 3,
    use_template: bool = False,
    template_pdb: Optional[str] = None,
    template_ligand_sdf: Optional[str] = None,
    min_quality: float = 0.25,
    min_volume: float = 150.0,
    min_residues: int = 5,
    filter_by_quality: bool = True,
) -> list[dict]:
    """
    Detect binding pockets using fpocket PQR-based analysis with quality filtering.

    Args:
        mmcif_text: mmCIF format protein structure
        chain_id: Chain identifier for pocket residues
        top_n: Max number of pockets to return
        filter_by_quality: If True, filter out low-quality pockets before ranking
        min_quality: Minimum composite quality score (0-1)
        min_volume: Minimum pocket volume in Å³
        min_residues: Minimum number of pocket-lining residues

    Returns list of pocket dicts with:
        - rank, score, volume, quality_score
        - residue_ids, residues, atom_count, alpha_sphere_count
    """
    from app.dtwg_pocket import (
        detect_pockets_fpocket_pqr,
        detect_pocket_by_template,
        filter_quality_pockets,
    )

    # Convert mmCIF to PDB-like format for fpocket
    pdb_text = _mmcif_to_pdb_fragment(mmcif_text)

    # Detect all pockets
    pockets = detect_pockets_fpocket_pqr(pdb_text, top_n=max(top_n * 2, 10))

    if not pockets:
        return []

    # Apply structural quality filtering
    if filter_by_quality:
        pockets = filter_quality_pockets(
            pockets,
            min_quality=min_quality,
            min_volume=min_volume,
            min_residues=min_residues,
        )
        pockets = pockets[:top_n]
    else:
        pockets = pockets[:top_n]

    if pockets:
        logger.info(
            f"Selected {len(pockets)} pockets. "
            f"Top: quality={pockets[0].get('quality_score', 'N/A')}, "
            f"fpocket_score={pockets[0]['score']:.1f}, "
            f"volume={pockets[0]['volume']:.0f}Å³, "
            f"residues={len(pockets[0].get('residue_ids', []))}"
        )

    # Optional template matching
    if use_template and template_pdb:
        tm_result = detect_pocket_by_template(pdb_text, template_pdb, template_ligand_sdf)
        if tm_result and tm_result.get("residue_ids"):
            tm_result["rank"] = 0
            tm_result["score"] = tm_result.get("TMscore", 0)
            tm_result["volume"] = 0
            tm_result["quality_score"] = 1.0
            tm_result["residues"] = [("A", r) for r in tm_result["residue_ids"]]
            pockets.insert(0, tm_result)
            logger.info(
                f"Template pocket: TM-score={tm_result.get('TMscore', 0):.4f}, "
                f"{len(tm_result['residue_ids'])} residues"
            )

    return pockets


def _mmcif_to_pdb_fragment(mmcif_text: str) -> str:
    """Convert ESMFold2 mmCIF to PDB with all heavy protein atoms for fpocket."""
    lines = []
    atom_count = 0

    in_loop = False
    col_map = {}
    data_lines = []

    for line in mmcif_text.split("\n"):
        line = line.strip()
        if not in_loop and "_atom_site.group_PDB" in line:
            in_loop = True
        if in_loop and line.startswith("_atom_site."):
            name = line.split(".")[-1]
            col_map[name] = len(col_map)
            continue
        if in_loop and line and not line.startswith("_") and not line.startswith("#"):
            data_lines.append(line)
        elif in_loop and line.startswith("#"):
            break

    if not col_map:
        logger = logging.getLogger("cascade.pocket")
        logger.warning("No _atom_site columns found; using lossy PDB fallback (all atoms -> ALA A 1)")
        for line in mmcif_text.split("\n"):
            if line.startswith("ATOM") or line.startswith("HETATM"):
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        x, y, z = float(parts[-3]), float(parts[-2]), float(parts[-1])
                        elem = parts[1] if len(parts) > 1 else "C"
                        atom_count += 1
                        lines.append(
                            f"ATOM  {atom_count:5d}  CA  ALA A   1    "
                            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {elem:2s}"
                        )
                    except (ValueError, IndexError):
                        continue
        if lines:
            return "\n".join(lines)
        return mmcif_text

    x_col = col_map.get("Cartn_x", 14)
    y_col = col_map.get("Cartn_y", 15)
    z_col = col_map.get("Cartn_z", 16)
    elem_col = col_map.get("type_symbol", 1)
    atom_col = col_map.get("label_atom_id", 2)
    res_col = col_map.get("label_comp_id", 5)
    chain_col = col_map.get("label_asym_id", 6)
    seq_col = col_map.get("label_seq_id", 8)

    for parts_str in data_lines:
        parts = parts_str.split()
        try:
            atom_name = parts[atom_col] if atom_col < len(parts) else "CA"
            elem = parts[elem_col] if elem_col < len(parts) else "C"
            if elem == "H":
                continue

            x, y, z = float(parts[x_col]), float(parts[y_col]), float(parts[z_col])
            res_name = parts[res_col] if res_col < len(parts) else "ALA"
            chain = parts[chain_col] if chain_col < len(parts) else "A"
            seq_id = int(parts[seq_col]) if seq_col < len(parts) and parts[seq_col].isdigit() else 1

            atom_count += 1
            lines.append(
                f"ATOM  {atom_count:5d} {atom_name:4s} {res_name:3s} {chain}{seq_id:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {elem:2s}"
            )
        except (ValueError, IndexError, KeyError):
            continue

    return "\n".join(lines) if lines else mmcif_text
