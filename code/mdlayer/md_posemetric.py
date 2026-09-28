"""Receptor-frame ligand pose RMSD retaining translation (5-case selftest)."""
import numpy as np

from mdboxutil import as_box, min_image


def _ligand_in_pocket_image(lig, prot, box):
    """Shift the ligand into the periodic image nearest the protein, as a rigid body.

    A ligand that has diffused across a box face must be unwrapped molecule-wise;
    shifting the whole ligand by a lattice vector keeps its internal geometry.
    """
    b = as_box(box)
    d = lig.mean(0) - prot.mean(0)
    n = np.round(d @ np.linalg.inv(b))
    return lig - n @ b


def kabsch(P, Q):
    """Rotation R and translation t mapping P onto Q (P @ R.T + t ~ Q)."""
    pc, qc = P.mean(0), Q.mean(0)
    U, S, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, qc - R @ pc


def pose_rmsd(lig_ref, prot_ref, lig_cur, prot_cur, box, pocket_idx=None):
    """Receptor-frame ligand pose RMSD with translation retained.

    Args:
      lig_ref, prot_ref: reference (build/crystal) ligand and receptor heavy atoms
      lig_cur, prot_cur: the same atoms in the frame under test
      box: 3x3 box (row vectors)
      pocket_idx: optional subset of prot_* to superpose on (default: all)

    Returns:
      (rmsd, lig_centroid_displacement) both in angstrom.
    """
    lig_ref = np.asarray(lig_ref, float)
    prot_ref = np.asarray(prot_ref, float)
    lig_cur = np.asarray(lig_cur, float)
    prot_cur = np.asarray(prot_cur, float)
    b = as_box(box)

    # put current ligand and protein in a common image
    lig_cur = _ligand_in_pocket_image(lig_cur, prot_cur, b)
    d = prot_cur.mean(0) - prot_ref.mean(0)
    prot_cur = prot_cur - np.round(d @ np.linalg.inv(b)) @ b

    idx = slice(None) if pocket_idx is None else np.asarray(pocket_idx)
    R, t = kabsch(prot_cur[idx], prot_ref[idx])
    # apply the pocket transform to the LIGAND, keeping its offset from the pocket
    lig_moved = lig_cur @ R.T + t
    rmsd = float(np.sqrt(((lig_moved - lig_ref) ** 2).sum(1).mean()))
    disp = float(np.linalg.norm(lig_moved.mean(0) - lig_ref.mean(0)))
    return rmsd, disp


def selftest(verbose=True):
    """The test the review demanded: a 40 A translation must NOT be invisible."""
    rng = np.random.default_rng(0)
    box = np.diag([80.0, 80.0, 90.0])
    prot = rng.normal(0, 8, size=(120, 3)) + np.array([40.0, 40.0, 45.0])
    lig = rng.normal(0, 1.5, size=(12, 3)) + prot.mean(0) + np.array([3.0, 4.0, 0.0])

    r0, d0 = pose_rmsd(lig, prot, lig, prot, box)
    ok_identity = r0 < 1e-6

    lig_far = lig + np.array([40.0, 0.0, 0.0])
    r1, d1 = pose_rmsd(lig, prot, lig_far, prot, box)
    ok_translation = r1 > 30.0 and abs(d1 - 40.0) < 1e-6

    # rotation-only perturbation (ligand turned in place) must be seen too
    th = np.deg2rad(180)
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    c = lig.mean(0)
    lig_rot = (lig - c) @ Rz.T + c
    r2, d2 = pose_rmsd(lig, prot, lig_rot, prot, box)
    ok_rotation = r2 > 1.0 and d2 < 1e-6

    # protein moved together with the ligand: pose is preserved, must score ~0
    shift = np.array([5.0, -3.0, 2.0])
    r3, d3 = pose_rmsd(lig, prot, lig + shift, prot + shift, box)
    ok_comove = r3 < 1e-6

    # ligand wrapped across a box face: molecule-aware unwrapping must recover it
    lig_wrap = lig + np.array([80.0, 0.0, 0.0])
    r4, d4 = pose_rmsd(lig, prot, lig_wrap, prot, box)
    ok_wrap = r4 < 1e-6

    if verbose:
        print("T1 selftest of pose_rmsd (receptor-frame, translation retained)")
        print("  identity frame                        rmsd=%.3g  disp=%.3g   %s"
              % (r0, d0, "PASS" if ok_identity else "FAIL"))
        print("  ligand translated 40 A                rmsd=%.2f  disp=%.2f   %s"
              % (r1, d1, "PASS" if ok_translation else "FAIL"))
        print("  ligand rotated in place (180 deg)     rmsd=%.2f  disp=%.2f   %s"
              % (r2, d2, "PASS" if ok_rotation else "FAIL"))
        print("  ligand+protein moved together         rmsd=%.3g  disp=%.3g   %s"
              % (r3, d3, "PASS" if ok_comove else "FAIL"))
        print("  ligand wrapped by one box vector      rmsd=%.3g  disp=%.3g   %s"
              % (r4, d4, "PASS" if ok_wrap else "FAIL"))
    ok = ok_identity and ok_translation and ok_rotation and ok_comove and ok_wrap
    if verbose:
        print("  => %s" % ("ALL PASS" if ok else "SELFTEST_FAILED"))
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if selftest() else 1)
