"""Strip buried waters from the bilayer core (parm-ed based)."""
import sys, json
import numpy as np
import parmed as pmd

PRM, INP, PREFIX, OUTJSON = sys.argv[1:5]
CORE = float(sys.argv[5]) if len(sys.argv) > 5 else 12.0
LIPIDS = {"POP","POPC","POPE","POPS","POPG","DPPC","DOPC","PC","PE","PS","PA","PGR","PH-","OL","OLA","OLC","PLM","MYR","STE","CHL","CHOL","CLR","POPE","POPA","POPG","DOPG","DPPE","DSPC","SAPI","SAPL","TLCL","CHL1"}
WATER = {"WAT", "HOH", "TIP3", "HO4"}

p = pmd.load_file(PRM, INP)
pz = [a.xz for a in p.atoms if a.residue.name in LIPIDS and a.name == "P"]  # parmed: xz is z
if len(pz) < 20:
    print("WARN: few lipid P atoms (%d)" % len(pz))
mid = float(np.median(pz)) if pz else 0.0

remove_res = []
for r in p.residues:
    if r.name not in WATER:
        continue
    o = next((a for a in r.atoms if a.name == "O"), None)
    if o is not None and abs(o.xz - mid) <= CORE:
        remove_res.append(r.idx)

# build an Amber residue-index mask (contiguous ranges)
remove_res = sorted(remove_res)
parts, start, prev = [], None, None
for i in remove_res:
    if start is None:
        start = prev = i
    elif i == prev + 1:
        prev = i
    else:
        parts.append((start, prev)); start = prev = i
if start is not None:
    parts.append((start, prev))
mask = ":" + ",".join("%d-%d" % (a, b) if a != b else "%d" % a for a, b in parts) if parts else None

n_before = len(p.atoms)
if mask:
    p.strip(mask)
n_after = len(p.atoms)

p.save(PREFIX + ".prmtop", overwrite=True)
p.save(PREFIX + ".inpcrd", overwrite=True)
p.save(PREFIX + ".pdb", overwrite=True)

info = {"midplane_z": round(mid, 2), "core_half_A": CORE,
        "n_atoms_before": n_before, "n_atoms_after": n_after,
        "n_water_residues_removed": len(remove_res),
        "atoms_removed": n_before - n_after,
        "method": "parmed strip on prmtop/inpcrd (no PDB round-trip; system exceeds PDB serial limit)",
        "note": "R0.6 verifies the core is clean"}
json.dump(info, open(OUTJSON, "w"), indent=1)
print(json.dumps(info, indent=1))
print("PARMED_STRIP_DONE")
