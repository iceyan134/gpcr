"""Hexagonal grid packing of a lipid bilayer at a target area-per-lipid."""
import sys, os, json
import numpy as np

args = sys.argv[1:]
prot_pdb, popc_pdb, out_pdb = args[0], args[1], args[2]
APL = float(args[args.index("--aplon") + 1]) if "--aplon" in args else 63.0
Z_UP = float(args[args.index("--zupper") + 1]) if "--zupper" in args else 15.0
Z_LO = float(args[args.index("--zlower") + 1]) if "--zlower" in args else -15.0
SEED = int(args[args.index("--seed") + 1]) if "--seed" in args else 0
PAD = float(args[args.index("--pad") + 1]) if "--pad" in args else 8.0

STD = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS HID HIE HIP ILE LEU LYS MET "
          "PHE PRO SER THR TRP TYR VAL CYX ASH GLH LYN ACE NME NHE".split())

def read_pdb(path, std_only=False):
    out = []
    for ln in open(path, errors="ignore"):
        if ln[:6] not in ("ATOM  ", "HETATM"):
            continue
        rn = ln[17:20].strip()
        if std_only and rn not in STD:
            continue
        el = (ln[76:78].strip() or ln[12:16].strip()[:1]).upper()
        out.append((ln, rn, ln[22:26].strip(), ln[12:16].strip(), el,
                    np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])))
    return out

prot = read_pdb(prot_pdb, std_only=True)
prot_xyz = np.array([a[5] for a in prot])
popc_lines = read_pdb(popc_pdb)
popc_xyz = np.array([a[5] for a in popc_lines])
popc_centroid = popc_xyz.mean(0)

# grid spacing from APL
d = np.sqrt(2 * APL / np.sqrt(3))
print("grid spacing: %.2f A (APL=%.1f)" % (d, APL))

xy_lo = prot_xyz[:, :2].min(0) - 22
xy_hi = prot_xyz[:, :2].max(0) + 22
nx = int((xy_hi[0] - xy_lo[0]) / d) + 1
ny = int((xy_hi[1] - xy_lo[1]) / (d * np.sqrt(3) / 2)) + 1
rng = np.random.default_rng(SEED)

def rot_z(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

out_lines = []
n_upper = n_lower = n_rejected = 0
placed_heads = []

# protein first, preserving the template's residue numbering
for tmpl_ln, rn, rid, nm, el, xyz in prot:
    out_lines.append(tmpl_ln)
res_serial = len(prot) + 1   # lipids start after the protein

for leaflet, z0 in [("UPPER", Z_UP), ("LOWER", Z_LO)]:
    placed_heads.append([])
    for ix in range(nx):
        for iy in range(ny):
            x = xy_lo[0] + ix * d + (iy % 2) * d / 2
            y = xy_lo[1] + iy * d * np.sqrt(3) / 2
            head = np.array([x, y, z0])
            dd = np.sqrt(((prot_xyz - head[None, :]) ** 2).sum(1)).min()
            if dd < PAD:
                n_rejected += 1
                continue
            if placed_heads[-1]:
                dp = min(np.linalg.norm(head[:2] - p[:2]) for p in placed_heads[-1])
                if dp < d * 0.85:
                    n_rejected += 1
                    continue
            placed_heads[-1].append(head)
            theta = rng.uniform(0, 2 * np.pi)
            R = rot_z(theta)
            flip = -1 if leaflet == "LOWER" else 1
            for tmpl_ln, rn, rid, nm, el, xyz in popc_lines:
                v = xyz - popc_centroid
                v = v @ R.T
                v[2] *= flip
                p = head + v
                # copy template line, replace coords and residue number
                new_ln = (tmpl_ln[:22] + "%4d" % res_serial + tmpl_ln[26:30]
                          + "%8.3f%8.3f%8.3f" % (p[0], p[1], p[2]) + tmpl_ln[54:])
                out_lines.append(new_ln)
            res_serial += 1
            if leaflet == "UPPER":
                n_upper += 1
            else:
                n_lower += 1



with open(out_pdb, "w") as f:
    f.writelines(out_lines)
    f.write("END\n")

info = {"upper": n_upper, "lower": n_lower, "total": n_upper + n_lower,
        "rejected": n_rejected, "spacing_A": round(d, 2), "apl": APL, "seed": SEED}
print(json.dumps(info))
print("GRID_PACK_DONE -> %s (%d lipid residues, %d protein residues)"
      % (out_pdb, n_upper + n_lower, len(prot)))
