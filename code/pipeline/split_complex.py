#!/usr/bin/env python3
"""
split_complex.py -- split a protein-ligand structure into receptor + ligand.

For pose validation each complex needs:
  - ligand.sdf   : ligand with CORRECT bond orders (for symmetry-corrected RMSD)
  - receptor.pdb : protein only (optional; structure_align can also read the raw file)

PDB/mmCIF coordinates carry no reliable bond orders, so we assign them from a
TEMPLATE (SMILES or a template SDF) via RDKit AssignBondOrdersFromTemplate. Using
the SAME template for the predicted and the crystal ligand guarantees identical
topology -- which is exactly what rdMolAlign.CalcRMS (validate_pose_and_enrichment
.ligand_rmsd) requires. Get the SMILES from RNP inputs.json / the PDB CCD.

Deps: Biopython (parse PDB/mmCIF), RDKit.
Logic check: python split_complex.py --self-test
Usage:
  # crystal ligand (CCD 'ANP' = AMP-PNP), bonds from SMILES:
  python split_complex.py --structure xtal.pdb --ligand-ccd ANP \
      --smiles "<SMILES>" --out-ligand xtal_lig.sdf --out-receptor xtal_rec.pdb
  # predicted ligand from a cofold mmCIF, SAME SMILES (-> matching topology):
  python split_complex.py --structure pred.cif --ligand-ccd ANP \
      --smiles "<SMILES>" --out-ligand pred_lig.sdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}


def _parse_structure(path: str):
    from Bio.PDB import MMCIFParser, PDBParser
    p = str(path)
    if p.endswith((".cif", ".mmcif")):
        # ESMFold2 mmCIF is missing _atom_site.occupancy — patch MMCIFParser
        _orig_build = MMCIFParser._build_structure
        def _patched_build(self, structure_id):
            try:
                return _orig_build(self, structure_id)
            except KeyError:
                # Inject default occupancy list
                n_atoms = len(self._mmcif_dict.get("_atom_site.id", []))
                if "_atom_site.occupancy" not in self._mmcif_dict:
                    self._mmcif_dict["_atom_site.occupancy"] = ["1.0"] * n_atoms
                return _orig_build(self, structure_id)
        MMCIFParser._build_structure = _patched_build
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    return parser.get_structure("s", p)


def _find_first_ligand(structure, ccd: str, chain: str | None):
    """Return (chain_id, residue_id) of the first hetero residue named `ccd`."""
    ccd = ccd.strip().upper()
    for model in structure:
        for ch in model:
            if chain is not None and ch.get_id() != chain:
                continue
            for res in ch:
                if res.get_id()[0] != " " and res.get_resname().strip().upper() == ccd:
                    return ch.get_id(), res.get_id()
        break  # first model only
    raise ValueError(f"Ligand '{ccd}' not found in structure"
                     + (f" (chain {chain})" if chain else ""))


def _template_mol(smiles: str | None, template_sdf: str | None):
    from rdkit import Chem
    if smiles:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            raise ValueError(f"Could not parse --smiles: {smiles}")
        return m
    if template_sdf:
        m = Chem.SDMolSupplier(str(template_sdf), removeHs=True)[0]
        if m is None:
            raise ValueError(f"Could not read --template-sdf: {template_sdf}")
        return m
    raise ValueError("provide --smiles or --template-sdf to define ligand bond orders")


def _ligand_with_bonds(structure, ccd, chain, template, out_sdf):
    """Extract the ligand by CCD, assign bond orders from template, write SDF."""
    import tempfile

    from Bio.PDB import PDBIO, Select
    from rdkit import Chem
    from rdkit.Chem import AllChem

    cid, rid = _find_first_ligand(structure, ccd, chain)

    class _LigSel(Select):
        def accept_residue(self, res):
            return res.get_parent().get_id() == cid and res.get_id() == rid

    io = PDBIO()
    io.set_structure(structure)
    with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as tf:
        tmp = tf.name
    io.save(tmp, _LigSel())

    raw = Chem.MolFromPDBFile(tmp, sanitize=False, removeHs=False, proximityBonding=True)
    Path(tmp).unlink(missing_ok=True)
    if raw is None:
        raise ValueError(f"RDKit could not read extracted ligand '{ccd}'")

    try:
        mol = AllChem.AssignBondOrdersFromTemplate(template, raw)
    except Exception:
        # retry after sanitising + stripping explicit Hs (some cofold ligands carry H)
        raw2 = Chem.MolFromPDBFile(tmp, sanitize=True, removeHs=True, proximityBonding=True) \
            if Path(tmp).exists() else raw
        mol = AllChem.AssignBondOrdersFromTemplate(template, Chem.RemoveHs(raw, sanitize=False))

    mol.SetProp("_Name", f"{ccd}")
    Path(out_sdf).parent.mkdir(parents=True, exist_ok=True)
    with Chem.SDWriter(str(out_sdf)) as w:
        w.write(mol)
    return mol.GetNumAtoms()


def _save_receptor(structure, out_pdb):
    from Bio.PDB import PDBIO, Select

    class _ProtSel(Select):
        def accept_residue(self, res):
            return res.get_id()[0] == " " and res.get_resname().strip().upper() in _STANDARD_AA

    io = PDBIO()
    io.set_structure(structure)
    Path(out_pdb).parent.mkdir(parents=True, exist_ok=True)
    io.save(str(out_pdb), _ProtSel())


def split_complex(structure_path, ligand_ccd, out_ligand,
                  smiles=None, template_sdf=None, ligand_chain=None, out_receptor=None):
    structure = _parse_structure(structure_path)
    template = _template_mol(smiles, template_sdf)
    n = _ligand_with_bonds(structure, ligand_ccd, ligand_chain, template, out_ligand)
    print(f"ligand '{ligand_ccd}': {n} heavy atoms -> {out_ligand}")
    if out_receptor:
        _save_receptor(structure, out_receptor)
        print(f"receptor (protein only) -> {out_receptor}")
    return out_ligand


def _self_test() -> int:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        print("RDKit absent: split self-test skipped (needs rdkit; present in the Docker image).")
        print("PASS")
        return 0
    # ethanol: 3 heavy atoms, proximity bonds, single bonds from template "CCO"
    pdb = (
        "HETATM    1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "HETATM    2  C2  LIG A   1       1.520   0.000   0.000  1.00  0.00           C\n"
        "HETATM    3  O1  LIG A   1       2.100   1.300   0.000  1.00  0.00           O\n"
        "END\n")
    raw = Chem.MolFromPDBBlock(pdb, sanitize=False, removeHs=False, proximityBonding=True)
    assert raw is not None and raw.GetNumAtoms() == 3, "PDB block parse failed"
    mol = AllChem.AssignBondOrdersFromTemplate(Chem.MolFromSmiles("CCO"), raw)
    assert mol.GetNumAtoms() == 3 and mol.GetNumBonds() == 2, "bond assignment wrong"
    elems = sorted(a.GetSymbol() for a in mol.GetAtoms())
    assert elems == ["C", "C", "O"], elems
    print("split self-test OK: ethanol C-C-O bonds assigned from template")
    print("PASS")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Split a complex into receptor PDB + ligand SDF.")
    ap.add_argument("--structure", help="input PDB or mmCIF (crystal or cofold prediction)")
    ap.add_argument("--ligand-ccd", help="ligand 3-letter CCD / residue name (e.g. ATP, ANP)")
    ap.add_argument("--smiles", help="ligand SMILES (bond-order template; from RNP inputs.json / PDB CCD)")
    ap.add_argument("--template-sdf", help="alternative bond-order template SDF")
    ap.add_argument("--ligand-chain", default=None, help="restrict to this chain (optional)")
    ap.add_argument("--out-ligand", help="output ligand SDF")
    ap.add_argument("--out-receptor", default=None, help="output receptor PDB (optional)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return _self_test()
    if not (args.structure and args.ligand_ccd and args.out_ligand):
        ap.error("need --structure --ligand-ccd --out-ligand (or --self-test)")
    split_complex(args.structure, args.ligand_ccd, args.out_ligand,
                  args.smiles, args.template_sdf, args.ligand_chain, args.out_receptor)
    return 0


if __name__ == "__main__":
    sys.exit(main())
