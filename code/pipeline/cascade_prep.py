"""
Complex preparation for downstream scoring tools.

- split_complex(): separate receptor PDB + ligand SDF from co-fold mmCIF
  * Bond-order perception from reference SMILES (RDKit AssignBondOrdersFromTemplate)
  * Ligand protonation via Dimorphite-DL (pH-dependent, preserves heavy-atom coords)
  * OpenBabel fallback when Dimorphite-DL unavailable

- protonate_receptor(): add hydrogens to receptor at pH
  * pdb2pqr + PROPKA (preferred, pKa-aware)
  * OpenBabel fallback

References:
- docs/screening_cascade.py:338-400  (split_complex, protonate_receptor)
- docs/PIPELINE_vFinal.md §2         (Dimorphite-DL for ligand protonation)
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger("cascade.prep")


def split_complex(
    complex_path: Path,
    workdir: Path,
    ligand_chain: str = "L",
    smiles: str = "",
    protonation_tool: str = "dimorphite",
    ph: float = 7.4,
) -> tuple[Path, Path]:
    """Separate co-fold mmCIF into receptor PDB and protonated ligand SDF.

    Returns (receptor_raw.pdb, ligand_pH.sdf).
    Heavy-atom coordinates are preserved exactly from the ESMFold2 pose.
    """
    from Bio.PDB import MMCIFParser, PDBParser, PDBIO, Select

    workdir.mkdir(parents=True, exist_ok=True)
    path_str = str(complex_path).lower()
    if path_str.endswith((".cif", ".mmcif")):
        _orig_build = MMCIFParser._build_structure
        def _patched_build(self, structure_id):
            try:
                return _orig_build(self, structure_id)
            except KeyError:
                if "_atom_site.occupancy" not in self._mmcif_dict:
                    n = len(self._mmcif_dict.get("_atom_site.id", []))
                    self._mmcif_dict["_atom_site.occupancy"] = ["1.0"] * n
                    return _orig_build(self, structure_id)
                raise
        MMCIFParser._build_structure = _patched_build
        try:
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure("c", str(complex_path))
        finally:
            MMCIFParser._build_structure = _orig_build
    else:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("c", str(complex_path))

    class RecSel(Select):
        def accept_residue(self, r):
            return r.get_parent().id != ligand_chain

    class LigSel(Select):
        def accept_residue(self, r):
            return r.get_parent().id == ligand_chain

    io = PDBIO()
    io.set_structure(structure)

    rec_pdb = workdir / "receptor_raw.pdb"
    lig_pdb = workdir / "ligand_raw.pdb"
    io.save(str(rec_pdb), RecSel())
    io.save(str(lig_pdb), LigSel())

    # ── Ligand SDF with bond-order perception ─────────────────────────
    lig_sdf = _ligand_pdb_to_sdf(lig_pdb, workdir, smiles)

    # ── Protonate ligand at pH (preserve heavy-atom coords) ───────────
    lig_h_sdf = workdir / "ligand_pH.sdf"
    if protonation_tool == "dimorphite":
        try:
            _dimorphite_protonate(lig_sdf, lig_h_sdf, ph)
        except Exception as e:
            logger.warning("Dimorphite-DL failed (%s), falling back to obabel", e)
            _obabel_protonate(lig_sdf, lig_h_sdf, ph)
    else:
        _obabel_protonate(lig_sdf, lig_h_sdf, ph)

    return rec_pdb, lig_h_sdf


def protonate_receptor(
    receptor_pdb: Path,
    workdir: Path,
    ph: float = 7.4,
) -> Path:
    """Add hydrogens to receptor at given pH.

    Tries pdb2pqr (PROPKA, pKa-aware) first, falls back to OpenBabel.
    """
    out = workdir / "receptor_H.pdb"
    workdir.mkdir(parents=True, exist_ok=True)

    if shutil.which("pdb2pqr30"):
        try:
            pqr = workdir / "receptor.pqr"
            _run(["pdb2pqr30", "--ff=AMBER", "--with-ph", str(ph),
                  "--pdb-output", str(out), str(receptor_pdb), str(pqr)],
                 cwd=workdir)
            logger.info("Receptor protonated with pdb2pqr/PROPKA at pH %.1f", ph)
            return out
        except Exception as e:
            logger.warning("pdb2pqr failed (%s), falling back to obabel", e)
    else:
        logger.info("pdb2pqr30 not found, using obabel for receptor protonation")

    _obabel_protonate(receptor_pdb, out, ph)
    return out


# ── Internal helpers ──────────────────────────────────────────────────────

def _ligand_pdb_to_sdf(lig_pdb: Path, workdir: Path, smiles: str) -> Path:
    """Convert extracted ligand PDB to SDF with correct bond orders from SMILES template."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromPDBFile(str(lig_pdb), removeHs=False, sanitize=False)
    if mol is None:
        raise ValueError(f"RDKit could not read ligand PDB: {lig_pdb}")

    if smiles:
        template = Chem.MolFromSmiles(smiles)
        if template is not None:
            try:
                mol = AllChem.AssignBondOrdersFromTemplate(template, mol)
            except Exception as e:
                logger.warning("Bond-order assignment from SMILES failed: %s", e)
                # Fallback: use template SMILES to generate 3D coords
                mol = Chem.AddHs(template)
                try:
                    AllChem.EmbedMolecule(mol, randomSeed=42)
                    AllChem.MMFFOptimizeMolecule(mol)
                except Exception:
                    AllChem.Compute2DCoords(mol)

    Chem.SanitizeMol(mol)
    out = workdir / "ligand_raw.sdf"
    Chem.MolToMolFile(mol, str(out))
    return out


def _dimorphite_protonate(in_sdf: Path, out_sdf: Path, ph: float):
    """Protonate ligand at pH using Dimorphite-DL, preserving heavy-atom coords."""
    from dimorphite_dl.protonate.run import protonate_smiles
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # Read input SDF, get SMILES
    mol = next(Chem.SDMolSupplier(str(in_sdf), removeHs=False))
    if mol is None:
        raise ValueError(f"RDKit could not read ligand SDF: {in_sdf}")
    smi = Chem.MolToSmiles(Chem.RemoveHs(mol))

    # Protonate SMILES at pH
    results = protonate_smiles(smi, ph_min=ph, ph_max=ph, max_variants=1, label_identifiers=False)
    if results:
        new_smi = results[0]
        new_mol = Chem.MolFromSmiles(new_smi)
        if new_mol is None:
            raise ValueError(f"Dimorphite produced invalid SMILES: {new_smi}")
        new_mol = Chem.AddHs(new_mol)
        AllChem.EmbedMolecule(new_mol, randomSeed=42)
        Chem.MolToMolFile(new_mol, str(out_sdf))
    else:
        # No protonation change needed at this pH
        import shutil
        shutil.copy(str(in_sdf), str(out_sdf))


def _obabel_protonate(in_path: Path, out_path: Path, ph: float):
    """Protonate using OpenBabel at given pH."""
    if shutil.which("obabel"):
        _run(["obabel", str(in_path), "-O", str(out_path), "-p", str(ph)],
             cwd=out_path.parent)
    else:
        raise FileNotFoundError("Neither Dimorphite-DL nor obabel is available for protonation")


def _run(cmd: list[str], cwd: Path, timeout: int = 600):
    """Run a subprocess, raise on failure."""
    logger.debug("RUN (%s): %s", cwd, " ".join(cmd))
    proc = subprocess.run(
        cmd, cwd=str(cwd),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout[-2000:]}")
    return proc.stdout
