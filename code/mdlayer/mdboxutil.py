"""Box helpers: unambiguous box matrices, minimum-image vectors, nm anchors, force-group energy audit."""
import numpy as np


def as_box(box):
    """Coerce to a 3x3 row-vector matrix, loudly."""
    b = np.asarray(box, dtype=float)
    if b.ndim == 1:
        if b.size != 3:
            raise ValueError("1-D box must have 3 entries, got %d" % b.size)
        b = np.diag(b)
    if b.shape != (3, 3):
        raise ValueError("box must be 3x3 (row vectors), got %s" % (b.shape,))
    return b


def min_image(d, box):
    """Minimum image of displacement vector(s) d under box.

    d may be (3,), (N,3) or (N,M,3); box is coerced via as_box.
    """
    b = as_box(box)
    d = np.asarray(d, dtype=float)
    f = d @ np.linalg.inv(b)
    f -= np.round(f)
    return f @ b


def volume_nm3(box):
    """Box volume in nm^3 (box in angstrom)."""
    return float(abs(np.linalg.det(as_box(box)))) / 1000.0


def lengths(box):
    """Box vector lengths (for reporting / NetCDF cell_lengths)."""
    return np.linalg.norm(as_box(box), axis=1)


# --- restraint helpers ---------------------------------------------------------
# addParticle() takes BARE NUMBERS in the Context's default length unit, which is
# nanometres. Every preparation script in this project passed angstrom values (either
# `positions[i].x` or an angstrom float array), so every anchor sat 10x too far away:
# for 281 CA at k=1000 that is a spurious 2.5e6 kJ/mol at t=0, i.e. a ~3.6e4 kJ/mol/nm
# radially outward force on the backbone. It looked like "holding" and behaved like
# "pushing" -- which is what tore the protein apart and produced the apparent ligand
# ejection in every arm. Always convert explicitly through anchors_nm().
def anchors_nm(coords_angstrom):
    """Convert angstrom coordinates to the nanometre anchor values OpenMM expects."""
    return np.asarray(coords_angstrom, dtype=float) / 10.0


def group_energy_audit(sim, unit):
    """Total energy, per-force-group energies, and their sum.

    Returns (total, {group: energy}, sum_of_groups). If the sum does not reproduce the
    total, some force lives in a group that was not queried -- which is exactly how a
    broken restraint hid in build_densify while the check reported restraint == 0.
    """
    total = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit)
    per, acc = {}, 0.0
    for g in range(32):
        e = sim.context.getState(getEnergy=True, groups={g}).getPotentialEnergy().value_in_unit(unit)
        if e != 0.0:
            per[g] = e
            acc += e
    return total, per, acc
