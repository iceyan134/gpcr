"""Iterating element-aware separation of non-bonded close contacts (converges to zero)."""
import sys, json
import numpy as np
from openmm import unit
import openmm.app as app
from scipy.spatial import cKDTree
from scipy.io import netcdf_file

PRM, INP, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
ROUNDS_MAX = int(sys.argv[4]) if len(sys.argv) > 4 else 12

OVERLAP = {("H", "H"): 1.4, ("H", "X"): 1.5, ("X", "H"): 1.5, ("X", "X"): 2.0}
TARGET = {k: v + 0.15 for k, v in OVERLAP.items()}
MAXSTEP = 0.5  # A per atom per round; the iteration handles the rest

prmtop = app.AmberPrmtopFile(PRM)
inpcrd = app.AmberInpcrdFile(INP)
X = np.array(inpcrd.positions.value_in_unit(unit.angstrom), dtype=float)
box = np.array(inpcrd.boxVectors.value_in_unit(unit.angstrom))
L = np.diag(box)
n = len(X)
print("atoms=%d box=%s" % (n, np.round(L, 1)), flush=True)

top = prmtop.topology
res_idx = np.array([a.residue.index for a in top.atoms()])
el = np.array([(a.element.symbol if a.element is not None
                else a.name.strip()[0]).upper() for a in top.atoms()])
isH = el == "H"

bonds12, nbr = set(), {}
for b in top.bonds():
    i, j = b.atom1.index, b.atom2.index
    bonds12.add((min(i, j), max(i, j)))
    nbr.setdefault(i, []).append(j)
    nbr.setdefault(j, []).append(i)
bonds13 = set()
for c, ns in nbr.items():
    for a in ns:
        for b in ns:
            if a < b and (min(a, c), max(a, c)) in bonds12 and (min(b, c), max(b, c)) in bonds12:
                bonds13.add((a, b))
print("bonds12=%d bonds13=%d" % (len(bonds12), len(bonds13)), flush=True)

def cat_pair(i, j):
    if isH[i] and isH[j]:
        return ("H", "H")
    if not isH[i] and not isH[j]:
        return ("X", "X")
    return ("H", "X")

frac = X / L; frac -= np.floor(frac)
rmax = max(TARGET.values())
trail = []
for rnd in range(1, ROUNDS_MAX + 1):
    tree = cKDTree(frac, boxsize=1.0)
    pairs = tree.query_pairs(rmax / min(L), output_type="ndarray")
    bad_now, n_push = 0, 0
    for fi, fj in pairs:
        i, j = int(fi), int(fj)
        if res_idx[i] == res_idx[j]:
            continue
        if (min(i, j), max(i, j)) in bonds12 or (min(i, j), max(i, j)) in bonds13:
            continue
        cat = cat_pair(i, j)
        df = frac[j] - frac[i]; df -= np.round(df); dv = df * L
        d = float(np.linalg.norm(dv))
        if d >= OVERLAP[cat]:
            continue
        bad_now += 1
        dr = dv / (d + 1e-12)
        push = min((TARGET[cat] - d) / 2.0 + 0.05, MAXSTEP)
        X[i] -= dr * push; X[j] += dr * push
        frac[i] = X[i] / L; frac[i] -= np.floor(frac[i])
        frac[j] = X[j] / L; frac[j] -= np.floor(frac[j])
        n_push += 1
    # rescan with a fresh tree for the honest count
    tree2 = cKDTree(frac, boxsize=1.0)
    pairs2 = tree2.query_pairs(rmax / min(L), output_type="ndarray")
    residual = 0
    worst = 9e9
    for fi, fj in pairs2:
        i, j = int(fi), int(fj)
        if res_idx[i] == res_idx[j]:
            continue
        if (min(i, j), max(i, j)) in bonds12 or (min(i, j), max(i, j)) in bonds13:
            continue
        df = frac[j] - frac[i]; df -= np.round(df); dv = df * L
        d = float(np.linalg.norm(dv))
        cat = cat_pair(i, j)
        if d < OVERLAP[cat]:
            residual += 1
            worst = min(worst, d)
    trail.append({"round": rnd, "pushed": n_push, "residual_bad": residual,
                  "worst_A": (round(worst, 2) if residual else None)})
    print("round %d: pushed %d, residual bad %d (worst %s A)"
          % (rnd, n_push, residual, ("%.2f" % worst) if residual else "-"), flush=True)
    if residual == 0:
        break

nc = netcdf_file(OUT, "w")
nc.Conventions = "AMBER"; nc.ConventionVersion = "1.0"; nc.program = "omfix_elem.py"
for nm, sz in (("spatial", 3), ("atom", n), ("cell", 3), ("label", 20), ("scalar", 1)):
    nc.createDimension(nm, sz)
v = nc.createVariable("coordinates", "f", ("atom", "spatial")); v.scale_factor = 1.0; v.add_offset = 0.0
v[:] = X.astype("f")
v = nc.createVariable("velocities", "f", ("atom", "spatial")); v.scale_factor = 1.0; v.add_offset = 0.0
v[:] = np.zeros_like(X, dtype="f")
cl = nc.createVariable("cell_lengths", "d", ("cell",)); cl[:] = L
ca = nc.createVariable("cell_angles", "d", ("cell",)); ca[:] = [90.0, 90.0, 90.0]
t = nc.createVariable("time", "d", ("scalar",)); t[:] = 0.0
nc.close()
json.dump({"rounds": trail, "final_residual": trail[-1]["residual_bad"]},
          open(OUT + "_fix.json", "w"), indent=1)
print("OMFIX_ELEM_DONE -> %s (residual=%d)" % (OUT, trail[-1]["residual_bad"]))
