"""Refine negative-control poses by restrained minimisation."""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mdboxutil import as_box, min_image

from openmm import unit, CustomExternalForce, LangevinMiddleIntegrator, Platform
import openmm.app as app

PRM, INP, OUT, OUTJ = sys.argv[1:5]
cav_pick = int(sys.argv[sys.argv.index("--cavity") + 1]) if "--cavity" in sys.argv else 0
seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0

STD = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS HID HIE HIP ILE LEU LYS MET "
          "PHE PRO SER THR TRP TYR VAL CYX ASH GLH LYN ACE NME NHE".split())
JAC_MAX, DMIN_MIN = 0.30, 2.0

prmtop = app.AmberPrmtopFile(PRM)
inpcrd = app.AmberInpcrdFile(INP)
atoms = list(prmtop.topology.atoms())
X0 = np.array(inpcrd.positions.value_in_unit(unit.angstrom))
box = np.array(inpcrd.boxVectors.value_in_unit(unit.angstrom))
lig = [i for i, a in enumerate(atoms) if a.residue.name == "MOL"
       and a.element is not None and a.element.symbol != "H"]
lig_all = [i for i, a in enumerate(atoms) if a.residue.name == "MOL"]
prot = [i for i, a in enumerate(atoms) if a.residue.name in STD
        and a.element is not None and a.element.symbol != "H"]
prot_res = np.array([a.residue.name + a.residue.id for a in atoms])[prot]
P = X0[prot]
Lref = X0[lig]
anchor0 = Lref.mean(0)
cryst_contacts = {prot_res[i] for i in np.where(
    np.sqrt((min_image(Lref[:, None, :] - P[None, :, :], as_box(box)) ** 2).sum(-1)).min(0) < 4.5)[0]}

# candidate cavities (same rule as place_decoy, for the record)
from scipy.spatial import cKDTree
tree = cKDTree(P)
cand = []
lo, hi, step = P.min(0) - 2.0, P.max(0) + 2.0, 2.0
for x in np.arange(lo[0], hi[0] + step, step):
    for y in np.arange(lo[1], hi[1] + step, step):
        for z in np.arange(lo[2], hi[2] + step, step):
            pt = np.array([x, y, z])
            dmin_pt, _ = tree.query(pt)
            if dmin_pt < 3.0 or np.linalg.norm(pt - anchor0) < 10.0:
                continue
            n_near = len(tree.query_ball_point(pt, 8.0))
            if n_near >= 8:
                cand.append((n_near, float(np.linalg.norm(pt - anchor0)), pt))
cand.sort(key=lambda t: (-t[0], t[1]))
if not cand:
    print(json.dumps({"pass": False, "reason": "no cavity"}))
    sys.exit(1)
anchor = cand[cav_pick][2]
print("cavity %d: %s (enclosure %d, %.1f A from the crystal centroid, %d candidates)"
      % (cav_pick, np.round(anchor, 1), cand[cav_pick][0], cand[cav_pick][1], len(cand)))

rng = np.random.default_rng(seed)
A = rng.normal(size=(3, 3))
Q, Rm = np.linalg.qr(A)
Q = Q @ np.diag(np.sign(np.diag(Rm)))
if np.linalg.det(Q) < 0:
    Q[:, 0] *= -1
X = X0.copy()
newlig = (X0[lig] - X0[lig].mean(0)) @ Q.T + anchor
for k, i in enumerate(lig):
    X[i] = newlig[k]
# hydrogens ride along rigidly with their heavy atoms
off = {}
for i in lig_all:
    if i not in lig:
        pass
X[lig_all] = np.array([X0[i] for i in lig_all])  # reset; map H via nearest heavy below
heavy_map = {i: k for k, i in enumerate(lig)}
for i in lig_all:
    if i in heavy_map:
        X[i] = newlig[heavy_map[i]]
    else:
        d = np.linalg.norm(X0[lig] - X0[i], axis=1)
        j = heavy_map[lig[int(np.argmin(d))]]
        X[i] = newlig[j] + (X0[i] - X0[lig[int(np.argmin(d))]])

system = prmtop.createSystem(nonbondedMethod=app.PME,
                             nonbondedCutoff=1.0 * unit.nanometer,
                             constraints=app.HBonds, rigidWater=True)
hold = CustomExternalForce("0.5*k*periodicdistance(x,y,z,x0,y0,z0)^2")
hold.addGlobalParameter("k", 1000.0)
for p in ("x0", "y0", "z0"):
    hold.addPerParticleParameter(p)
for i in prot:                      # protein held, ligand FREE to fit
    q = X0[i] / 10.0                # nm
    hold.addParticle(i, [q[0], q[1], q[2]])
system.addForce(hold)
integ = LangevinMiddleIntegrator(310 * unit.kelvin, 1.0 / unit.picosecond,
                                0.002 * unit.picoseconds)
integ.setRandomNumberSeed(seed)
sim = app.Simulation(prmtop.topology, system, integ, Platform.getPlatformByName("CUDA"))
sim.context.setPositions(X * 0.1)   # angstrom -> nm for setPositions (Quantity-free)
sim.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
sim.minimizeEnergy(maxIterations=5000)
Y = np.array(sim.context.getState(getPositions=True).getPositions(asNumpy=True)
              .value_in_unit(unit.angstrom))

d = min_image(Y[lig][:, None, :] - Y[prot][None, :, :], as_box(box))
dm = np.sqrt((d ** 2).sum(-1))
dmin = float(dm.min())
contacts = {prot_res[i] for i in np.where(dm.min(0) < 4.5)[0]}
jac = len(contacts & cryst_contacts) / max(1, len(contacts | cryst_contacts))
disp = float(np.linalg.norm(Y[lig].mean(0) - anchor))
ok = dmin >= DMIN_MIN and len(contacts) > 0 and jac <= JAC_MAX and disp <= 6.0
rep = {"pass": ok, "cavity": cav_pick, "anchor": [round(float(v), 1) for v in anchor],
       "dmin": round(dmin, 2), "n_contacts": len(contacts),
       "jaccard_vs_crystal": round(jac, 3), "centroid_drift_A": round(disp, 2),
       "crystal_contacts": sorted(cryst_contacts), "decoy_contacts": sorted(contacts)}
print(json.dumps(rep, indent=1))
json.dump(rep, open(OUTJ, "w"), indent=1)
if not ok:
    sys.exit(1)

from scipy.io import netcdf_file
nc = netcdf_file(OUT, "w")
nc.Conventions = "AMBER"; nc.ConventionVersion = "1.0"
nc.program = "negctl_fit.py"
for nm, sz in (("spatial", 3), ("atom", len(Y)), ("cell", 3), ("label", 20), ("scalar", 1)):
    nc.createDimension(nm, sz)
v = nc.createVariable("coordinates", "f", ("atom", "spatial")); v.scale_factor = 1.0; v.add_offset = 0.0
v[:] = Y.astype("f")
v = nc.createVariable("velocities", "f", ("atom", "spatial")); v.scale_factor = 1.0; v.add_offset = 0.0
v[:] = np.zeros_like(Y, dtype="f")
cl = nc.createVariable("cell_lengths", "d", ("cell",)); cl[:] = np.diag(as_box(box))
ca = nc.createVariable("cell_angles", "d", ("cell",)); ca[:] = np.array([90.0, 90.0, 90.0])
t = nc.createVariable("time", "d", ("scalar",)); t[:] = 0.0
nc.close()
print("NEGCTL_FIT_OK -> %s" % OUT)
