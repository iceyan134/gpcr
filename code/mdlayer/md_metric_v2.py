"""Pocket-anchored, PBC-safe pose metrics (minimum-image contacts, local Kabsch RMSD)."""
import json
import sys
import numpy as np

CUTOFF = 4.5  # A, heavy-atom contact


def min_image(delta, box):
    if box is None:
        return delta
    return delta - box * np.round(delta / box)


def per_ligand_min(lig, prot, box):
    out = np.empty(len(lig))
    for i in range(len(lig)):
        d = min_image(lig[i] - prot, box)
        out[i] = np.sqrt((d ** 2).sum(-1)).min()
    return out


def contact_residues(lig, prot, prot_res, box, cutoff=CUTOFF):
    hit = set()
    for i in range(len(lig)):
        d = min_image(lig[i] - prot, box)
        idx = np.where(np.sqrt((d ** 2).sum(-1)) <= cutoff)[0]
        hit.update(prot_res[j] for j in idx)
    return hit


def build_pocket(lig, prot, prot_res, box, cutoff=CUTOFF):
    """Pocket residue set and touching protein-atom indices (starting pose)."""
    touch = np.zeros(len(prot), bool)
    for i in range(len(lig)):
        d = min_image(lig[i] - prot, box)
        touch |= np.sqrt((d ** 2).sum(-1)) <= cutoff
    pocket = {prot_res[j] for j in np.where(touch)[0]}
    return pocket, np.where(touch)[0]


def contact_retention(pocket, current):
    if not pocket:
        return 0.0
    return len(pocket & current) / len(pocket)


def touch_imaged(lig_centroid, prot, touch_idx, box):
    """Touching protein atoms imaged into the ligand's neighbourhood."""
    return min_image(prot[touch_idx] - lig_centroid, box)


def frame_metrics(lig, prot, prot_res, pocket, touch_idx, box, cutoff=CUTOFF):
    """All per-frame quantities, PBC-safe and locally anchored."""
    pa = per_ligand_min(lig, prot, box)
    cur = contact_residues(lig, prot, prot_res, box, cutoff)
    c = lig.mean(0)
    ti = touch_imaged(c, prot, touch_idx, box)
    dpo = float(np.linalg.norm(ti.mean(0)))
    return {
        "retention": contact_retention(pocket, cur),
        "d_pocket": dpo,
        "d_min": float(pa.min()),
        "d_min_p5": float(np.percentile(pa, 5)),
        "d_min_p50": float(np.percentile(pa, 50)),
        "d_min_p95": float(np.percentile(pa, 95)),
        "n_contact_res": len(cur),
        "_touch_img": ti,
        "_lig_centroid": c,
    }


def ligand_rmsd_local(lig_ref, touch_ref_img, lig_cur, touch_cur_img):
    """Ligand RMSD after superposing the locally-imaged touch atoms."""
    A = touch_ref_img - touch_ref_img.mean(0)
    B = touch_cur_img - touch_cur_img.mean(0)
    U, S, Vt = np.linalg.svd(A.T @ B)
    R = (U @ Vt).T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = (U @ Vt).T
    moved = (lig_cur - lig_cur.mean(0)) @ R.T + touch_ref_img.mean(0)
    ref = lig_ref - lig_ref.mean(0) + touch_ref_img.mean(0)
    return float(np.sqrt(((moved - ref) ** 2).sum(1).mean()))


def _lattice_directions():
    dirs = []
    for x in (-1, 0, 1):
        for y in (-1, 0, 1):
            for z in (-1, 0, 1):
                if x == y == z == 0:
                    continue
                dirs.append(np.array([x, y, z], float))
    dirs.sort(key=lambda v: (abs(v).sum(), tuple(-abs(v))))
    return dirs


# ---------------------------------------------------------------- selftest --
def _read_pdb(path, lig_resname):
    prot, prot_res, lig = [], [], []
    seen = set()
    for ln in open(path):
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        alt = ln[16]
        key = (ln[21], ln[22:27], ln[12:16])
        if alt not in (" ", "A") or key in seen:
            continue
        seen.add(key)
        resn = ln[17:20].strip()
        el = (ln[76:78].strip() or ln[12:16].strip()[0]).upper()
        if el.startswith("H"):
            continue
        xyz = [float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]
        rid = (ln[21], ln[22:27].strip())
        if ln.startswith("ATOM"):
            prot.append(xyz)
            prot_res.append(rid)
        elif resn == lig_resname:
            lig.append(xyz)
    return np.array(prot), prot_res, np.array(lig)


def selftest(pdb_path, lig_resname, out_json):
    prot, prot_res, lig = _read_pdb(pdb_path, lig_resname)
    assert len(prot) > 100 and len(lig) >= 15, "parse failure"
    res = {}

    pocket, touch = build_pocket(lig, prot, prot_res, None)
    m = frame_metrics(lig, prot, prot_res, pocket, touch, None)
    res["C-a"] = {
        "contact_retention": m["retention"],
        "d_pocket_A": m["d_pocket"],
        "n_pocket_residues": len(pocket),
        "d_min_A": m["d_min"],
    }
    res["C-a"]["pass"] = (res["C-a"]["contact_retention"] == 1.0
                          and res["C-a"]["d_pocket_A"] < 6.0
                          and res["C-a"]["n_pocket_residues"] >= 8)

    lig_b, shift_used = None, None
    for shift in (15.0, 20.0, 25.0, 30.0, 40.0):
        for d in _lattice_directions():
            cand = lig + shift * d
            if per_ligand_min(cand, prot, None).min() >= 8.0:
                lig_b, shift_used = cand, shift
                break
        if lig_b is not None:
            break
    assert lig_b is not None, "no detached placement found"
    mb = frame_metrics(lig_b, prot, prot_res, pocket, touch, None)
    res["C-b"] = {
        "contact_retention": mb["retention"],
        "d_pocket_A": mb["d_pocket"],
        "shift_A": shift_used,
        "clearance_A": mb["d_min"],
    }
    res["C-b"]["pass"] = (res["C-b"]["contact_retention"] == 0.0
                          and res["C-b"]["d_pocket_A"] >= 12.0)

    box = np.array([150.0, 150.0, 150.0])
    pb, tb = build_pocket(lig, prot, prot_res, box)
    m1 = frame_metrics(lig, prot, prot_res, pb, tb, box)
    m2 = frame_metrics(lig + np.array([box[0], 0, 0]), prot, prot_res, pb, tb, box)
    res["C-c"] = {
        "retention_plain": m1["retention"], "retention_shifted": m2["retention"],
        "d_pocket_plain_A": m1["d_pocket"], "d_pocket_shifted_A": m2["d_pocket"],
        "d_min_plain_A": m1["d_min"], "d_min_shifted_A": m2["d_min"],
    }
    res["C-c"]["pass"] = (abs(m1["retention"] - m2["retention"]) < 1e-9
                          and abs(m1["d_pocket"] - m2["d_pocket"]) < 0.05
                          and abs(m1["d_min"] - m2["d_min"]) < 0.05)

    res["R1_pass"] = all(res[k]["pass"] for k in ("C-a", "C-b", "C-c"))
    json.dump(res, open(out_json, "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))
    print("\nR1_PASS" if res["R1_pass"] else "\nR1_FAIL")
    return res["R1_pass"]


if __name__ == "__main__":
    if sys.argv[1] == "--selftest":
        ok = selftest(sys.argv[2], sys.argv[3], sys.argv[4])
        sys.exit(0 if ok else 1)
