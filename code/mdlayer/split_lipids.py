"""Split a packed PDB into lipid and non-lipid parts for charmmlipid2amber."""
import sys

src, prefix = sys.argv[1], sys.argv[2]
LIPIDS = {"POP","POPC","POPE","POPS","POPG","DPPC","DOPC","PC","PE","PS","PA","PGR","PH-","OL","OLA","OLC","PLM","MYR","STE","CHL","CHOL","CLR","POPE","POPA","POPG","DOPG","DPPE","DSPC","SAPI","SAPL","TLCL","CHL1"}
non, lip = [], []
for ln in open(src):
    if ln.startswith(("ATOM", "HETATM")):
        (lip if ln[17:20].strip() in LIPIDS else non).append(ln)
with open(prefix + "_nonlipid.pdb", "w") as f:
    f.writelines(non); f.write("TER\nEND\n")
with open(prefix + "_lipid.pdb", "w") as f:
    f.writelines(lip); f.write("TER\nEND\n")
print("split: nonlipid atoms %d | lipid atoms %d" % (len(non), len(lip)))
