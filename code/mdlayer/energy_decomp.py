"""Force-group energy decomposition vs total (audit tool)."""
import sys
import numpy as np
from openmm import unit, CustomExternalForce, Platform, LangevinMiddleIntegrator
import openmm.app as app
from openmm import (HarmonicBondForce, HarmonicAngleForce, PeriodicTorsionForce,
                    NonbondedForce, MonteCarloBarostat)

PRM, INP, TAG = sys.argv[1:4]
STD = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS HID HIE HIP ILE LEU LYS MET "
          "PHE PRO SER THR TRP TYR VAL CYX ASH GLH LYN ACE NME NHE".split())
prmtop = app.AmberPrmtopFile(PRM)
inpcrd = app.AmberInpcrdFile(INP)
system = prmtop.createSystem(nonbondedMethod=app.PME,
                             nonbondedCutoff=1.0 * unit.nanometer,
                             constraints=app.HBonds, rigidWater=True)
names = {}
for i, f in enumerate(system.getForces()):
    if isinstance(f, (HarmonicBondForce, HarmonicAngleForce, PeriodicTorsionForce)):
        f.setForceGroup(1)
        names[1] = "bonded"
    elif isinstance(f, NonbondedForce):
        f.setForceGroup(2)
        names[2] = "nonbonded(incl. exceptions)"
    else:
        f.setForceGroup(3)
        names[3] = "other"
integ = LangevinMiddleIntegrator(310 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds)
sim = app.Simulation(prmtop.topology, system, integ, Platform.getPlatformByName("CPU"))
sim.context.setPositions(inpcrd.positions)
if inpcrd.boxVectors is not None:
    sim.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
tot = 0.0
print("== %s" % TAG)
for g in sorted(names):
    e = sim.context.getState(getEnergy=True, groups={g}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    tot += e
    print("   group %d %-30s = %20.4g kJ/mol" % (g, names[g], e))
print("   TOTAL = %.4g kJ/mol" % tot)
atoms = list(prmtop.topology.atoms())
lb = []
for f, i in zip(system.getForces(), range(system.getNumForces())):
    if isinstance(f, HarmonicBondForce):
        for k in range(f.getNumBonds()):
            bp = f.getBondParameters(k)
            p1, p2, r0 = bp[0], bp[1], bp[2]
            r0 = r0.value_in_unit(unit.nanometer)
            if r0 > 0.5:
                lb.append((r0, atoms[p1].residue.name, atoms[p1].residue.id,
                           atoms[p1].name, atoms[p2].residue.name, atoms[p2].residue.id, atoms[p2].name))
lb.sort(reverse=True)
if lb:
    print("   IMPLAUSIBLE BONDS (equilibrium length > 5 A) -- residues wrongly joined: %d" % len(lb))
    for r0, rn1, ri1, a1, rn2, ri2, a2 in lb[:8]:
        print("      r0=%.2f nm  %s%s:%s  <->  %s%s:%s" % (r0 * 10, rn1, ri1, a1, rn2, ri2, a2))
else:
    print("   no implausible bonds: residues are correctly segmented")
print("DECOMP_DONE")
