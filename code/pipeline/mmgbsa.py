"""
MM-GBSA physics-based refinement (Stage 3 of the cascade).

Fully open-source toolchain:
  Ligand charges:  NAGL (GNN, ~2000× faster than sqm) or EspalomaCharge
                   with sqm (AM1-BCC) fallback
  Minimization:    OpenMM (GPU, MIT-licensed), sander (CPU, fallback)
  GB model:        igb=5 (OBC2), aligned with OpenMM OBC2 implicit solvent
  Binding energy:  MMPBSA.py single-trajectory ΔG_bind

PAID tools deliberately avoided: no pmemd.cuda, no Schrödinger FEP+.

References:
- docs/screening_cascade.py:444-581  (base implementation, igb=2/sqm/sander)
- docs/PIPELINE_vFinal.md §4         (NAGL/OpenMM/igb=5 upgrades)
"""
from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("cascade.mmgbsa")


@dataclass
class MMGBSAResult:
    dg_gb: float              # ΔG_bind (kcal/mol), more negative = stronger
    dg_std: float | None = None
    raw: str = ""


# ── AmberTools templates ──────────────────────────────────────────────────

_TLEAP_TEMPLATE = """\
source {protein_ff}
source {ligand_ff}
LIG = loadmol2 {lig_mol2}
loadamberparams {lig_frcmod}
REC = loadpdb {rec_amber_pdb}
COM = combine {{ REC LIG }}
saveamberparm LIG {p}/lig.prmtop {p}/lig.inpcrd
saveamberparm REC {p}/rec.prmtop {p}/rec.inpcrd
saveamberparm COM {p}/com.prmtop {p}/com.inpcrd
quit
"""

_SANDER_MIN_IN = """\
implicit-solvent minimisation of the co-fold complex
 &cntrl
  imin=1, maxcyc={maxcyc}, ncyc={ncyc},
  igb={igb}, saltcon={saltcon}, gbsa=0,
  cut=999.0, ntb=0, ntr=0,
 /
"""

_MMPBSA_IN = """\
single-snapshot MM-GBSA
&general
 startframe=1, endframe=1, interval=1, verbose=1, keep_files=0,
/
&gb
 igb={igb}, saltcon={saltcon},
/
"""


# ── Ligand net charge ────────────────────────────────────────────────────

def _ligand_net_charge(ligand_sdf: Path) -> int:
    from rdkit import Chem
    mol = next(Chem.SDMolSupplier(str(ligand_sdf), removeHs=False))
    if mol is None:
        raise ValueError(f"RDKit could not read ligand SDF for charge: {ligand_sdf}")
    return Chem.GetFormalCharge(mol)


# ── Charge computation ────────────────────────────────────────────────────

def compute_ligand_charges(
    ligand_sdf: Path,
    workdir: Path,
    net_charge: int,
    method: str = "nagl",
    nagl_model: str = "openff-gnn-am1bcc-1.0.0.pt",
) -> Path:
    """Compute partial charges for the ligand and write a mol2 file.

    Parameters
    ----------
    method : "nagl" | "espaloma" | "sqm"
    nagl_model : model filename for NAGL (searched via openff-nagl-models)

    Returns path to ligand mol2 with GAFF2 atom types + charges.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    mol2 = workdir / "lig_gaff2.mol2"

    if method == "nagl":
        try:
            return _charges_nagl(ligand_sdf, mol2, net_charge, nagl_model, workdir)
        except Exception as e:
            logger.warning("NAGL charge assignment failed (%s), falling back to sqm", e)
            return _charges_sqm(ligand_sdf, mol2, net_charge, workdir)

    elif method == "espaloma":
        try:
            return _charges_espaloma(ligand_sdf, mol2, net_charge, workdir)
        except Exception as e:
            logger.warning("Espaloma charge assignment failed (%s), falling back to sqm", e)
            return _charges_sqm(ligand_sdf, mol2, net_charge, workdir)

    elif method == "xtb":
        try:
            return _charges_xtb(ligand_sdf, mol2, net_charge, workdir)
        except Exception as e:
            logger.warning("xtb charge assignment failed (%s), falling back to sqm", e)
            return _charges_sqm(ligand_sdf, mol2, net_charge, workdir)

    else:  # "sqm"
        try:
            return _charges_sqm(ligand_sdf, mol2, net_charge, workdir)
        except Exception as e:
            logger.warning("sqm/antechamber failed (%s) — check for unusual elements (B, Si, metals)", e)
            raise


def _charges_nagl(
    ligand_sdf: Path, out_mol2: Path, net_charge: int,
    nagl_model: str, workdir: Path,
) -> Path:
    """Compute AM1-BCC charges via NAGL GNN, then assign GAFF2 types via antechamber -c rc."""
    from openff.toolkit import Molecule
    from openff.nagl import NAGLChargeModel
    from openff.nagl_models import get_model_paths
    from openff.units import unit
    import numpy as np

    # Load the NAGL model
    model_paths = get_model_paths()
    model_path = None
    for p in model_paths:
        if nagl_model in str(p) or Path(p).name == nagl_model:
            model_path = p
            break
    if model_path is None:
        raise FileNotFoundError(f"NAGL model '{nagl_model}' not found in openff-nagl-models")

    charge_model = NAGLChargeModel.load(model_path)

    # Load ligand with OpenFF
    off_mol = Molecule.from_file(str(ligand_sdf))
    off_mol.assign_partial_charges(
        charge_model,
        partial_charge_method="am1bcc",
        use_conformers=[off_mol.conformers[0]],
    )
    charges = off_mol.partial_charges.m_as(unit.elementary_charge)

    # Write a mol2 with these charges, then run antechamber -c rc to assign GAFF2 types
    # First, run antechamber with -c bcc -s 0 to generate the mol2 skeleton
    _run(["antechamber", "-i", str(ligand_sdf), "-fi", "sdf",
          "-o", str(workdir / "_tmp.mol2"), "-fo", "mol2",
          "-c", "bcc", "-s", "0", "-at", "gaff2",
          "-nc", str(net_charge)], cwd=workdir)

    # Patch the charges into the mol2, then run antechamber -c rc to reuse them
    _patch_mol2_charges(workdir / "_tmp.mol2", out_mol2, charges, net_charge)

    # Now run antechamber -c rc: reuse charges, only assign GAFF2 types
    _run(["antechamber", "-i", str(out_mol2), "-fi", "mol2",
          "-o", str(out_mol2), "-fo", "mol2",
          "-c", "rc", "-cf", str(out_mol2.with_suffix(".dat")),
          "-at", "gaff2", "-nc", str(net_charge)], cwd=workdir)

    return out_mol2


def _charges_espaloma(ligand_sdf: Path, out_mol2: Path, net_charge: int, workdir: Path) -> Path:
    """Compute charges via EspalomaCharge GNN, then assign GAFF2 types."""
    from openff.toolkit import Molecule
    from openff.units import unit
    import numpy as np

    off_mol = Molecule.from_file(str(ligand_sdf))
    off_mol.assign_partial_charges("espaloma-am1bcc", use_conformers=[off_mol.conformers[0]])
    charges = off_mol.partial_charges.m_as(unit.elementary_charge)

    _run(["antechamber", "-i", str(ligand_sdf), "-fi", "sdf",
          "-o", str(workdir / "_tmp.mol2"), "-fo", "mol2",
          "-c", "bcc", "-s", "0", "-at", "gaff2",
          "-nc", str(net_charge)], cwd=workdir)

    _patch_mol2_charges(workdir / "_tmp.mol2", out_mol2, charges, net_charge)

    _run(["antechamber", "-i", str(out_mol2), "-fi", "mol2",
          "-o", str(out_mol2), "-fo", "mol2",
          "-c", "rc", "-cf", str(out_mol2.with_suffix(".dat")),
          "-at", "gaff2", "-nc", str(net_charge)], cwd=workdir)

    return out_mol2


def _charges_xtb(ligand_sdf: Path, out_mol2: Path, net_charge: int, workdir: Path) -> Path:
    """Compute GFN2-xTB charges (semi-empirical, ~2000× faster than sqm).

    Workflow: SDF → XYZ → xtb --gfn 2 → parse charges → write mol2 → antechamber -c rc.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # 1. Convert SDF to XYZ with coordinate validation
    mol = Chem.SDMolSupplier(str(ligand_sdf), removeHs=False)[0]
    if mol is None:
        raise ValueError(f"RDKit could not read {ligand_sdf}")
    
    # Check for overlapping atoms (xtb will crash on zero-distance coordinates)
    conf = mol.GetConformer()
    coords = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append((pos.x, pos.y, pos.z))
    
    # Detect overlapping atoms: if any pair has distance < 0.001, fallback to sqm
    min_dist = float("inf")
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            d = ((coords[i][0]-coords[j][0])**2 + (coords[i][1]-coords[j][1])**2 + (coords[i][2]-coords[j][2])**2)**0.5
            if d < min_dist:
                min_dist = d
    
    if min_dist < 0.001:
        logger.warning("xtb: overlapping atoms detected (min_dist=%g), falling back to sqm", min_dist)
        from app.mmgbsa import _charges_sqm
        return _charges_sqm(ligand_sdf, out_mol2, net_charge, workdir)
    
    xyz = workdir / "lig.xyz"
    xyz_lines = [str(mol.GetNumAtoms()), ""]
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        xyz_lines.append(f"{atom.GetSymbol():2s}  {pos.x:12.6f}  {pos.y:12.6f}  {pos.z:12.6f}")
    xyz.write_text("\n".join(xyz_lines))

    # 2. Run xtb GFN2
    _run(["xtb", str(xyz), "--gfn", "2", "--chrg", str(net_charge),
          "--namespace", "xtb"], cwd=workdir, timeout=600)

    # 3. Parse charges from xtb output (one float per line)
    charges_path = workdir / "xtb.charges"
    charges = []
    with open(charges_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    charges.append(float(line))
                except ValueError:
                    pass
    if len(charges) != mol.GetNumAtoms():
        raise ValueError(f"xtb charge count {len(charges)} != atom count {mol.GetNumAtoms()}")

    # 4. Generate mol2 with antechamber -c bcc -s 0 (skeleton only, skip sqm)
    tmp_mol2 = workdir / "_tmp.mol2"
    _run(["antechamber", "-i", str(ligand_sdf), "-fi", "sdf",
          "-o", str(tmp_mol2), "-fo", "mol2",
          "-c", "bcc", "-s", "0", "-at", "gaff2",
          "-nc", str(net_charge)], cwd=workdir)

    # 5. Patch xtb charges into mol2
    _patch_mol2_charges(tmp_mol2, out_mol2, charges, net_charge)

    return out_mol2


def _charges_sqm(ligand_sdf: Path, out_mol2: Path, net_charge: int, workdir: Path) -> Path:
    """Compute AM1-BCC charges via sqm (pure open-source, slower)."""
    _run(["antechamber", "-i", str(ligand_sdf), "-fi", "sdf",
          "-o", str(out_mol2), "-fo", "mol2",
          "-c", "bcc", "-s", "2", "-at", "gaff2",
          "-nc", str(net_charge)], cwd=workdir)
    return out_mol2


def _patch_mol2_charges(in_mol2: Path, out_mol2: Path, charges, net_charge: int = None):
    """Replace charges in a mol2 file with the given array while keeping atom types."""
    import numpy as np
    lines = in_mol2.read_text().splitlines()
    charge_idx = 0
    in_atom_section = False
    new_lines = []
    for line in lines:
        if line.strip().startswith("@<TRIPOS>ATOM"):
            in_atom_section = True
            new_lines.append(line)
            continue
        if in_atom_section and line.strip().startswith("@<TRIPOS>"):
            in_atom_section = False
        if in_atom_section and len(line.strip().split()) >= 9 and charge_idx < len(charges):
            parts = line.split()
            parts[-1] = f"{charges[charge_idx]:.6f}"
            new_lines.append(" ".join(parts))
            charge_idx += 1
        else:
            new_lines.append(line)
    out_mol2.write_text("\n".join(new_lines))

    # Charge sum check — catches atom reordering bugs silently producing wrong MM-GBSA ΔG
    if net_charge is not None and len(charges) > 0:
        charge_sum = np.sum(charges)
        # Allow ±0.15 e tolerance (rounding in GNN charges)
        if abs(charge_sum - net_charge) > 0.15:
            logger.error(
                "Charge sum %.3f != formal charge %d (diff=%.3f e). "
                "Atom reordering or charge assignment bug likely — MM-GBSA ΔG will be garbage.",
                charge_sum, net_charge, charge_sum - net_charge)
        else:
            logger.debug("Charge sum check OK: %.3f ≈ %d", charge_sum, net_charge)


# ── OpenMM minimization ───────────────────────────────────────────────────

def _minimize_openmm(
    com_prmtop: Path, com_inpcrd: Path,
    out_rst7: Path, workdir: Path,
    igb: int = 5, saltcon: float = 0.150,
    platform: str = "CUDA",
    max_steps: int = 2000,
) -> Path:
    """Minimize the complex in GB implicit solvent using OpenMM (GPU)."""
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit

    # Load Amber topology
    prmtop = app.AmberPrmtopFile(str(com_prmtop))
    inpcrd = app.AmberInpcrdFile(str(com_inpcrd))

    # GB implicit solvent (OBC2 = igb=5)
    system = prmtop.createSystem(
        implicitSolvent=app.OBC2,
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
        soluteDielectric=1.0,
        solventDielectric=78.5,
    )

    # Set platform — enumerate available, fall back gracefully
    available = [mm.Platform.getPlatform(i).getName()
                 for i in range(mm.Platform.getNumPlatforms())]
    logger.info("OpenMM available platforms: %s", available)

    if platform in available:
        p = mm.Platform.getPlatformByName(platform)
        props = {"CudaPrecision": "mixed"} if platform == "CUDA" else {}
    elif "CUDA" in available:
        logger.warning("'%s' not available, falling back to CUDA", platform)
        p = mm.Platform.getPlatformByName("CUDA")
        props = {"CudaPrecision": "mixed"}
    elif "CPU" in available:
        logger.warning("'%s' not available, falling back to CPU", platform)
        p = mm.Platform.getPlatformByName("CPU")
        props = {}
    else:
        logger.warning("'%s' not available, falling back to Reference", platform)
        p = mm.Platform.getPlatformByName("Reference")
        props = {}

    integrator = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtosecond)
    simulation = app.Simulation(prmtop.topology, system, integrator, p, props)
    simulation.context.setPositions(inpcrd.positions)

    # Minimize
    simulation.minimizeEnergy(maxIterations=max_steps)

    # Save minimized coordinates
    state = simulation.context.getState(getPositions=True)
    positions = state.getPositions()

    # Write as Amber restart (7-column format)
    import numpy as np
    coords = positions.value_in_unit(unit.angstrom)
    coords_arr = np.array([list(v) for v in coords])
    with open(out_rst7, "w") as f:
        f.write(f"{'':80s}{coords_arr.shape[0]:6d}{0.0:12.7f}\n")
        for i in range(0, len(coords_arr), 2):
            vals = coords_arr[i:i + 2].flatten()
            line = "".join(f"{v:12.7f}" for v in vals)
            f.write(line + "\n")
        f.write(f"{90.0:12.7f}{90.0:12.7f}{90.0:12.7f}\n")

    return out_rst7


def _minimize_sander(
    com_prmtop: Path, com_inpcrd: Path,
    out_rst7: Path, workdir: Path,
    igb: int = 5, saltcon: float = 0.150,
    max_steps: int = 2000,
) -> Path:
    """Minimize with sander (CPU fallback)."""
    min_in = workdir / "min.in"
    min_in.write_text(_SANDER_MIN_IN.format(
        maxcyc=max_steps, ncyc=max_steps // 2,
        igb=igb, saltcon=saltcon,
    ))
    _run(["sander", "-O", "-i", str(min_in),
          "-p", str(com_prmtop), "-c", str(com_inpcrd),
          "-r", str(out_rst7), "-o", str(workdir / "min.out"),
          "-ref", str(com_inpcrd)], cwd=workdir)
    return out_rst7


# ── Short MD (stub) ───────────────────────────────────────────────────────

def _short_md_openmm(
    com_prmtop: Path, out_traj: Path, workdir: Path,
    igb: int = 5, saltcon: float = 0.150,
    platform: str = "CUDA", ns: float = 1.0,
    timestep_fs: float = 4.0, n_frames: int = 50,
):
    """Run brief implicit-solvent MD and write multi-frame Amber NetCDF trajectory.

    This is a documented stub; a proper MD protocol is target-dependent.
    """
    raise NotImplementedError(
        "mmgbsa_mode='short_md' is intentionally a stub. "
        "Supply a target-specific equilibration+production protocol."
    )


# ── Main MM-GBSA workflow ─────────────────────────────────────────────────

def mmgbsa_refine(
    receptor_raw: Path,
    ligand_sdf: Path,
    workdir: Path,
    igb: int = 5,
    saltcon: float = 0.150,
    protein_ff: str = "leaprc.protein.ff19SB",
    ligand_ff: str = "leaprc.gaff2",
    charge_method: str = "nagl",
    nagl_model: str = "openff-gnn-am1bcc-1.0.0.pt",
    min_engine: str = "openmm",
    openmm_platform: str = "CUDA",
    sander_min_steps: int = 2000,
    mmgbsa_mode: str = "minimize_only",
    timeout_s: int = 3600,
) -> MMGBSAResult:
    """Run the full MM-GBSA pipeline on a co-fold complex.

    Returns MMGBSAResult with ΔG_bind (kcal/mol, more negative = stronger binding).
    """
    _require_ambertools()
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)

    netq = _ligand_net_charge(ligand_sdf)

    # ── 1. Ligand parameterization ────────────────────────────────────
    lig_mol2 = compute_ligand_charges(ligand_sdf, wd, netq, charge_method, nagl_model)
    lig_frcmod = wd / "lig.frcmod"
    _run(["parmchk2", "-i", str(lig_mol2), "-f", "mol2",
          "-o", str(lig_frcmod), "-s", "gaff2"], cwd=wd)

    # ── 2. Receptor cleanup ───────────────────────────────────────────
    rec_amber = wd / "rec_amber.pdb"
    # audit 2026-09-02: ESMFold2 co-folded receptors need -d (add missing
    # atoms) — without it cpptraj fails downstream (rebuild incomplete
    # sidechains that ESMFold2 didn't fully resolve)
    _run(["pdb4amber", "-i", str(receptor_raw), "-o", str(rec_amber),
          "--reduce", "--dry", "-d"], cwd=wd)

    # ── 3. Build topologies ───────────────────────────────────────────
    tleap_in = wd / "tleap.in"
    tleap_in.write_text(_TLEAP_TEMPLATE.format(
        protein_ff=protein_ff, ligand_ff=ligand_ff,
        lig_mol2=lig_mol2, lig_frcmod=lig_frcmod,
        rec_amber_pdb=rec_amber, p=wd))
    _run(["tleap", "-f", str(tleap_in)], cwd=wd)

    # ── 4. Minimize ───────────────────────────────────────────────────
    rst7 = wd / "com_min.rst7"
    if min_engine == "openmm" and shutil.which("python"):  # OpenMM is always importable if installed
        try:
            _minimize_openmm(
                wd / "com.prmtop", wd / "com.inpcrd", rst7, wd,
                igb=igb, saltcon=saltcon, platform=openmm_platform,
                max_steps=sander_min_steps,
            )
        except Exception as e:
            logger.warning("OpenMM minimization failed (%s), falling back to sander", e)
            _minimize_sander(wd / "com.prmtop", wd / "com.inpcrd", rst7, wd,
                             igb=igb, saltcon=saltcon, max_steps=sander_min_steps)
    else:
        _minimize_sander(wd / "com.prmtop", wd / "com.inpcrd", rst7, wd,
                         igb=igb, saltcon=saltcon, max_steps=sander_min_steps)

    # ── 5. Trajectory ─────────────────────────────────────────────────
    if mmgbsa_mode == "short_md":
        traj = _short_md_openmm(
            wd / "com.prmtop", wd / "traj.nc", wd,
            igb=igb, saltcon=saltcon, platform=openmm_platform)
    else:
        # Single-frame trajectory from minimized coordinates
        cpptraj_in = wd / "mktraj.in"
        cpptraj_in.write_text(
            f"parm {wd / 'com.prmtop'}\ntrajin {rst7}\n"
            f"trajout {wd / 'traj.nc'}\nrun\n")
        _run(["cpptraj", "-i", str(cpptraj_in)], cwd=wd)
        traj = wd / "traj.nc"

    # ── 6. MMPBSA.py ──────────────────────────────────────────────────
    mmpbsa_in = wd / "mmpbsa.in"
    mmpbsa_in.write_text(_MMPBSA_IN.format(igb=igb, saltcon=saltcon))
    _run(["MMPBSA.py", "-O", "-i", str(mmpbsa_in),
          "-o", str(wd / "FINAL_RESULTS.dat"),
          "-cp", str(wd / "com.prmtop"),
          "-rp", str(wd / "rec.prmtop"),
          "-lp", str(wd / "lig.prmtop"),
          "-y", str(traj)], cwd=wd)

    return _parse_mmpbsa(wd / "FINAL_RESULTS.dat")


# ── Output parsing ────────────────────────────────────────────────────────

def _parse_mmpbsa(dat: Path) -> MMGBSAResult:
    text = dat.read_text()
    in_delta = False
    for line in text.splitlines():
        if "Delta (Complex - Receptor - Ligand)" in line or line.strip().startswith("DELTA"):
            in_delta = True
        if in_delta and line.strip().startswith("DELTA TOTAL"):
            parts = line.split()
            nums = [p for p in parts if _isfloat(p)]
            if nums:
                avg = float(nums[0])
                std = float(nums[1]) if len(nums) > 1 else None
                return MMGBSAResult(dg_gb=avg, dg_std=std, raw=text)
    raise RuntimeError(f"Could not parse 'DELTA TOTAL' from {dat}")


def _isfloat(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _require_ambertools():
    missing = [b for b in ["antechamber", "parmchk2", "tleap", "sander", "cpptraj", "pdb4amber", "MMPBSA.py"]
               if shutil.which(b) is None]
    if missing:
        raise FileNotFoundError(
            f"Required AmberTools binaries not on PATH: {', '.join(missing)}. "
            "Install: conda install -c conda-forge ambertools"
        )


# ── Batch MM-GBSA: Phase 1 CPU-parallel, Phase 2 GPU-serial ──────────────

def mmgbsa_prepare_one(args: tuple) -> tuple:
    """Phase 1: CPU-heavy ligand prep + topology (runnable in parallel).

    Returns (workdir, lig_mol2, lig_frcmod, rec_amber, tleap_ok, error)
    """
    (receptor_raw, ligand_sdf, workdir, netq, charge_method, nagl_model,
     protein_ff, ligand_ff) = args

    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)

    try:
        # Charge computation
        lig_mol2 = compute_ligand_charges(Path(ligand_sdf), wd, netq, charge_method, nagl_model)

        # Force field params
        lig_frcmod = wd / "lig.frcmod"
        _run(["parmchk2", "-i", str(lig_mol2), "-f", "mol2",
              "-o", str(lig_frcmod), "-s", "gaff2"], cwd=wd)

        # Receptor cleanup
        rec_amber = wd / "rec_amber.pdb"
        _run(["pdb4amber", "-i", str(receptor_raw), "-o", str(rec_amber),
              "--reduce", "--dry", "-d"], cwd=wd)

        # Build topologies
        tleap_in = wd / "tleap.in"
        tleap_in.write_text(_TLEAP_TEMPLATE.format(
            protein_ff=protein_ff, ligand_ff=ligand_ff,
            lig_mol2=lig_mol2, lig_frcmod=lig_frcmod,
            rec_amber_pdb=rec_amber, p=wd))
        _run(["tleap", "-f", str(tleap_in)], cwd=wd)

        return (str(wd), str(lig_mol2), str(lig_frcmod), str(rec_amber), True, None)
    except Exception as e:
        logger.debug("Phase 1 failed for %s: %s", workdir, e)
        return (str(wd), "", "", "", False, str(e)[:200])


def mmgbsa_finalize_one(args: tuple) -> MMGBSAResult | None:
    """Phase 2: GPU-heavy minimization + scoring (serial recommended)."""
    (workdir, igb, saltcon, min_engine, openmm_platform, sander_min_steps,
     mmgbsa_mode, timeout_s) = args

    wd = Path(workdir)
    prmtop = wd / "com.prmtop"
    inpcrd = wd / "com.inpcrd"
    rst7 = wd / "com_min.rst7"

    if not prmtop.exists():
        return None

    try:
        # Minimize
        if min_engine == "openmm":
            try:
                _minimize_openmm(prmtop, inpcrd, rst7, wd, igb=igb, saltcon=saltcon,
                                platform=openmm_platform, max_steps=sander_min_steps)
            except Exception as e:
                logger.warning("OpenMM minimize failed (%s), falling back to sander", e)
                _minimize_sander(prmtop, inpcrd, rst7, wd, igb=igb, saltcon=saltcon,
                                max_steps=sander_min_steps)
        else:
            _minimize_sander(prmtop, inpcrd, rst7, wd, igb=igb, saltcon=saltcon,
                            max_steps=sander_min_steps)

        # Trajectory
        cpptraj_in = wd / "mktraj.in"
        cpptraj_in.write_text(
            f"parm {prmtop}\ntrajin {rst7}\ntrajout {wd / 'traj.nc'}\nrun\n")
        _run(["cpptraj", "-i", str(cpptraj_in)], cwd=wd)

        # MMPBSA
        mmpbsa_in = wd / "mmpbsa.in"
        mmpbsa_in.write_text(_MMPBSA_IN.format(igb=igb, saltcon=saltcon))
        _run(["MMPBSA.py", "-O", "-i", str(mmpbsa_in),
              "-o", str(wd / "FINAL_RESULTS.dat"),
              "-cp", str(prmtop), "-rp", str(wd / "rec.prmtop"),
              "-lp", str(wd / "lig.prmtop"), "-y", str(wd / "traj.nc")], cwd=wd)

        return _parse_mmpbsa(wd / "FINAL_RESULTS.dat")
    except Exception as e:
        logger.debug("Phase 2 failed for %s: %s", workdir, e)
        return None


def _run(cmd: list[str], cwd: Path, timeout: int = 3600):
    env = os.environ.copy()
    env.setdefault("AMBERHOME", "/opt/conda")
    logger.debug("RUN (%s): %s", cwd, " ".join(str(c) for c in cmd))
    proc = subprocess.run(
        cmd, cwd=str(cwd),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=timeout, env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(str(c) for c in cmd)}\n{proc.stdout[-2000:]}")
    return proc.stdout
