"""Remove terminal OXT atoms that clash with the next chain segment."""
import sys, json
import numpy as np
import parmed as pmd

PRM, INP, OUT = sys.argv[1:4]
p = pmd.load_file(PRM, xyz=INP)
atoms = list(p.atoms)

# find all N atoms and OXT atoms with residue IDs
n_atoms = [(i, a) for i, a in enumerate(atoms) if a.name.strip() == "N"]
oxt_atoms = [(i, a) for i, a in enumerate(atoms) if a.name.strip() == "OXT"]
X = np.array([[a.xx, a.xy, a.xz] for a in atoms])

to_strip = []
for oi, oa in oxt_atoms:
    for ni, na in n_atoms:
        if na.residue.idx == oa.residue.idx:
            continue
        d = float(np.linalg.norm(X[oi] - X[ni]))
        if d < 2.0:
            to_strip.append((oi, oa.residue.name, oa.residue.number, d))
            break

print("OXT atoms clashing with another residue's N (<2.0 A): %d" % len(to_strip))
for oi, rn, rid, d in to_strip:
    print("   %s%d OXT -> nearest N at %.2f A" % (rn, rid, d))

if to_strip:
    mask = np.zeros(len(atoms), dtype=bool)
    for oi, _, _, _ in to_strip:
        mask[oi] = True
    p.strip(mask)
    print("stripped %d atoms; total %d -> %d" % (len(to_strip), len(atoms), len(p.atoms)))

p.save(OUT + ".prmtop", overwrite=True)
p.save(OUT + ".inpcrd", overwrite=True)
p.save(OUT + ".pdb", overwrite=True)
json.dump({"n_oxt_removed": len(to_strip), "details": [
    {"residue": rn + str(rid), "dist_to_N": round(d, 2)} for _, rn, rid, d in to_strip
]}, open(OUT + "_oxt.json", "w"), indent=1)
print("STRIP_OXT_DONE -> %s" % OUT)
