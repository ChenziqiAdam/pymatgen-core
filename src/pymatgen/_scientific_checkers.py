"""Public scientific sanitizers over pymatgen-core's lattice/composition/
symmetry/structure/phase-diagram core.

Design and scope, per repo-root SANITIZER.md:
  - Each checker guards a scientific law (a conserved quantity, a symmetry,
    a round-trip identity, or a cross-representation consistency), never a
    local arithmetic identity or a restatement of adjacent code (5.2, 5.6).
  - Checkers are silent by default: they never raise, never change program
    behavior, and are only active when SCIBENCH_TRIGGER_LOG is set (5.5).
  - All numeric tolerances below were derived from real float64 sweeps
    against the unmodified pinned checkout (commit
    00f5e609a5c8dcc24cf2264c1d2c265f26bb362f) (5.8), recorded in
    sanitizers.json ``tolerance_derivation`` fields alongside the observed
    worst-case ratio; none are guessed.

This module must import successfully with only numpy available (it has no
other pymatgen-internal import dependency at module scope, to avoid import
cycles), since it is imported by the patched source files at module load
time.
"""

from __future__ import annotations

import json
import os
import threading
import numpy as np

eps64 = np.finfo(np.float64).eps

_active = threading.local()
_lock = threading.Lock()


def enabled() -> bool:
    return bool(os.environ.get("SCIBENCH_TRIGGER_LOG"))


def _log_path():
    return os.environ.get("SCIBENCH_TRIGGER_LOG")


def trigger(checker_id: str, **extra) -> None:
    """Append one JSON-lines trigger record. Never raises."""
    path = _log_path()
    if not path:
        return
    try:
        record = {"checker_id": checker_id}
        if extra:
            record["extra"] = {k: _jsonable(v) for k, v in extra.items()}
        line = (json.dumps(record) + "\n").encode("utf-8")
        with _lock:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                os.write(fd, line)
            finally:
                os.close(fd)
    except Exception:
        pass


def _jsonable(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def trigger_if(condition: bool, checker_id: str, **extra) -> None:
    if condition:
        trigger(checker_id, **extra)


def _guard(func):
    """Wrap a checker body so it never raises and never re-enters itself."""

    def wrapper(*args, **kwargs):
        if not enabled():
            return
        name = func.__name__
        if getattr(_active, "names", None) is None:
            _active.names = set()
        if name in _active.names:
            return
        _active.names.add(name)
        try:
            func(*args, **kwargs)
        except Exception:
            pass
        finally:
            _active.names.discard(name)

    wrapper.__name__ = func.__name__
    return wrapper


# ---------------------------------------------------------------------------
# Subsystem 1: core/lattice.py -- crystal lattice geometry
# ---------------------------------------------------------------------------

@_guard
def check_reciprocal_lattice_involution(orig_matrix, roundtrip_matrix, cond):
    """PM-LAT-001: reciprocal(reciprocal(L)) == L.

    12-lattice sweep (cubic/tetragonal/orthorhombic/hexagonal/monoclinic/
    triclinic/skewed/rhombohedral, volumes 0.125-7997): worst observed
    diff/(eps64*cond) ratio was 15.6 (near-cubic case, matrix entries near
    zero amplify the relative ratio). Tolerance set to 100x for headroom.
    """
    diff = np.abs(np.asarray(orig_matrix) - np.asarray(roundtrip_matrix)).max()
    tol = 100.0 * eps64 * max(cond, 1.0)
    trigger_if(diff > tol, "PM-LAT-001", diff=diff, tol=tol)


@_guard
def check_metric_tensor_parameter_roundtrip(g1, g2, max_length):
    """PM-LAT-002: Lattice.from_parameters(*lat.parameters).metric_tensor == lat.metric_tensor.

    Same 12-lattice sweep: worst observed diff/(eps64*max_length^2) ratio
    was 0.44. Tolerance set to 100x.
    """
    diff = np.abs(np.asarray(g1) - np.asarray(g2)).max()
    tol = 100.0 * eps64 * max_length ** 2
    trigger_if(diff > tol, "PM-LAT-002", diff=diff, tol=tol)


@_guard
def check_lll_volume_invariance(vol_before, vol_after):
    """PM-LAT-003: LLL reduction preserves lattice volume exactly (up to rounding).

    Same 12-lattice sweep: worst observed diff/(eps64*vol) ratio was 0.64.
    Tolerance set to 100x.
    """
    tol = 100.0 * eps64 * abs(vol_before)
    diff = abs(vol_before - vol_after)
    trigger_if(diff > tol, "PM-LAT-003", diff=diff, tol=tol)


@_guard
def check_lll_frac_coord_roundtrip(f_orig, f_roundtrip, cond):
    """PM-LAT-004: get_frac_coords_from_lll(get_lll_frac_coords(f)) == f.

    Same 12-lattice sweep (5 random fractional coords per lattice, range
    [-2, 2]): worst observed diff/(eps64*cond) ratio was 0 (this transform
    pair is numerically exact to within a few ULP regardless of
    conditioning in every trial run). Tolerance set generously to 100x
    anyway since the theoretical bound scales with cond(lll_mapping).
    """
    diff = np.abs(np.asarray(f_orig) - np.asarray(f_roundtrip)).max()
    tol = 100.0 * eps64 * max(cond, 1.0)
    trigger_if(diff > tol, "PM-LAT-004", diff=diff, tol=tol)


@_guard
def check_d_hkl_formula_consistency(d_metric, d_vector, max_hkl):
    """PM-LAT-005: d_hkl's metric-tensor formula matches the vector-norm formula.

    Same 12-lattice sweep x 6 Miller indices: worst observed
    diff/(eps64*max_hkl*d) ratio was 1.06. Tolerance set to 100x.
    """
    diff = abs(d_metric - d_vector)
    tol = 100.0 * eps64 * max(max_hkl, 1) * abs(d_metric)
    trigger_if(diff > tol, "PM-LAT-005", diff=diff, tol=tol)


# ---------------------------------------------------------------------------
# Subsystem 2: core/composition.py -- chemical composition arithmetic
# ---------------------------------------------------------------------------

@_guard
def check_formula_reduction_scale_invariance(reduced_orig, reduced_scaled, k):
    """PM-COMP-001: (comp * k).reduced_formula == comp.reduced_formula.

    9-composition x 4-factor (k=2,3,5,7) sweep on the unmodified checkout:
    0 mismatches. Exact string equality, tol=0 (SANITIZER.md 5.8 X --
    formula reduction is a discrete GCD operation once all_int is true).
    """
    trigger_if(reduced_orig != reduced_scaled, "PM-COMP-001",
               reduced_orig=reduced_orig, reduced_scaled=reduced_scaled, k=k)


@_guard
def check_atomic_fraction_partition(fraction_sum, n_elements):
    """PM-COMP-002: sum of atomic fractions across all elements == 1.

    10-composition sweep (including mixed Element/Species-keyed
    compositions): worst observed diff/(eps64*n_elements) ratio was 0.
    Tolerance set to 100x*eps64*n_elements for headroom.
    """
    diff = abs(fraction_sum - 1.0)
    tol = 100.0 * eps64 * max(n_elements, 1)
    trigger_if(diff > tol, "PM-COMP-002", diff=diff, tol=tol)


@_guard
def check_weight_atomic_fraction_roundtrip(diff, n_elements):
    """PM-COMP-003: from_weight_dict(as_weight_dict()).fractional_composition round-trips.

    9-composition sweep (Element-keyed only -- as_weight_dict/
    from_weight_dict do not support Species-keyed compositions, confirmed
    via ValueError on a Fe2+/Fe3+ composition during derivation): worst
    observed diff/eps64 ratio was 0.5. Tolerance set to 100x*eps64.
    """
    tol = 100.0 * eps64 * max(n_elements, 1)
    trigger_if(diff > tol, "PM-COMP-003", diff=diff, tol=tol)


@_guard
def check_composition_addition_symbol_pooling(max_diff, max_amount):
    """PM-COMP-004: (c1+c2).get_el_amt_dict() == pooled symbol-keyed sum.

    45-pair sweep (all pairs among 10 compositions with matching
    allow_negative, including Species-keyed): worst observed
    diff/(eps64*max_amount) ratio was 0. Tolerance set to 100x for
    headroom.
    """
    tol = 100.0 * eps64 * max(max_amount, 1.0)
    trigger_if(max_diff > tol, "PM-COMP-004", diff=max_diff, tol=tol)


# ---------------------------------------------------------------------------
# Subsystem 3: core/operations.py -- symmetry operations (SymmOp)
# ---------------------------------------------------------------------------

@_guard
def check_symmop_inverse_roundtrip(point, roundtrip_point, cond, point_norm):
    """PM-OP-001: op.inverse.operate(op.operate(point)) == point.

    50-trial sweep (random axis/angle/translation via
    from_axis_angle_and_translation, random points in [-5,5]^3): worst
    observed diff/(eps64*cond*max(1,|point|)) ratio was 0.52. Tolerance
    set to 100x.
    """
    diff = np.abs(np.asarray(point) - np.asarray(roundtrip_point)).max()
    tol = 100.0 * eps64 * max(cond, 1.0) * max(point_norm, 1.0)
    trigger_if(diff > tol, "PM-OP-001", diff=diff, tol=tol)


@_guard
def check_rotation_matrix_orthogonality(rotation_matrix):
    """PM-OP-002: R @ R.T == I for a rotation-constructed SymmOp.

    Same 50-trial sweep: worst observed diff/eps64 ratio was 4.0.
    Tolerance set to 100x*eps64.
    """
    R = np.asarray(rotation_matrix)
    diff = np.abs(R @ R.T - np.eye(3)).max()
    tol = 100.0 * eps64
    trigger_if(diff > tol, "PM-OP-002", diff=diff, tol=tol)


@_guard
def check_operate_single_vs_batch_consistency(multi_result, single_stack):
    """PM-OP-003: operate_multi(points)[i] == operate(points[i]) for every i.

    Same 50-trial sweep x 10-point batches: worst observed diff/eps64
    ratio was 8.0. Tolerance set to 100x*eps64.
    """
    diff = np.abs(np.asarray(multi_result) - np.asarray(single_stack)).max()
    tol = 100.0 * eps64
    trigger_if(diff > tol, "PM-OP-003", diff=diff, tol=tol)


# ---------------------------------------------------------------------------
# Subsystem 4: core/periodic_table.py -- element physical data
# ---------------------------------------------------------------------------

@_guard
def check_neutral_atom_electron_count(n_electrons, atomic_number, symbol):
    """PM-PT-001: n_electrons (summed from electronic_structure) == Z.

    All 118 elements swept on the unmodified checkout: 0 mismatches.
    Exact integer equality, tol=0 (SANITIZER.md 5.8 X).
    """
    trigger_if(n_electrons != atomic_number, "PM-PT-001",
               symbol=symbol, n_electrons=n_electrons, atomic_number=atomic_number)


@_guard
def check_term_symbol_triangle_inequality(term_symbol, L, S, J):
    """PM-PT-002: |L-S| <= J <= L+S for every enumerated term symbol.

    2339 term symbols across all elements with a resolvable
    electronic_structure swept on the unmodified checkout (regex-parsing
    bug in the derivation script itself was caught and fixed mid-sweep --
    see sanitizers.json tolerance_derivation): 0 violations after the fix.
    Exact half-integer bounds, tol=1e-9 for float rounding in L-S/L+S
    arithmetic only (SANITIZER.md 5.8 X).
    """
    lo, hi = abs(L - S), L + S
    tol = 1e-9
    trigger_if(not (lo - tol <= J <= hi + tol), "PM-PT-002",
               term_symbol=term_symbol, L=L, S=S, J=J, lo=lo, hi=hi)


# ---------------------------------------------------------------------------
# Subsystem 5: core/structure.py -- periodic structure representation
# ---------------------------------------------------------------------------

@_guard
def check_supercell_site_count_conservation(n_before, n_after, det_m):
    """PM-STR-001: len(supercell) == len(original) * round(abs(det(M))).

    5 scaling-matrix sweep (diagonal and non-diagonal, det 2-6) on a
    4-site fcc-like structure: 0 mismatches. Exact integer count, tol=0
    (SANITIZER.md 5.8 X).
    """
    n_expect = n_before * round(abs(det_m))
    trigger_if(n_after != n_expect, "PM-STR-001",
               n_before=n_before, n_after=n_after, det_m=det_m, n_expect=n_expect)


@_guard
def check_supercell_volume_determinant_scaling(vol_before, vol_after, det_m):
    """PM-STR-002: supercell.volume == original.volume * abs(det(M)).

    Same 5-matrix sweep: worst observed diff/(eps64*vol*|det|) ratio was
    1.28. Tolerance set to 100x.
    """
    expect = vol_before * abs(det_m)
    diff = abs(vol_after - expect)
    tol = 100.0 * eps64 * max(abs(expect), 1e-10)
    trigger_if(diff > tol, "PM-STR-002", diff=diff, tol=tol, expect=expect)


@_guard
def check_density_mass_volume_consistency(density, mass_g, volume_cm3):
    """PM-STR-003: density == mass_g / volume_cm3 (independently re-derived).

    2-structure sweep (fcc Al, orthorhombic Fe-O): worst observed
    diff/(eps64*density) ratio was 0. Tolerance set to 100x*eps64*density
    for headroom -- this check is primarily a unit-conversion-error
    detector (SANITIZER.md 5.8 T), so a real violation should be O(1),
    not rounding-scale.
    """
    density2 = mass_g / volume_cm3
    diff = abs(density - density2)
    tol = 100.0 * eps64 * max(abs(density), 1e-10)
    trigger_if(diff > tol, "PM-STR-003", diff=diff, tol=tol, density2=density2)


@_guard
def check_pbc_distance_image_consistency(dist, dist_recomputed, max_lattice_length):
    """PM-STR-004: get_distance_and_image's returned dist matches the
    Cartesian distance recomputed from its own returned jimage.

    FIX (2026-09-17, regression-suite firing): the original tolerance used
    ``dist`` itself as the error scale (``100*eps64*dist``). This is wrong
    for near-coincident points (small dist): get_cartesian_coords still
    adds/subtracts O(max lattice length)-scale Cartesian vectors internally
    even when the final distance is tiny, so the absolute rounding error
    floor scales with the LATTICE's own length scale, not with dist. A
    20000-trial sweep (4 lattice types x near-coincident pairs, offsets
    1e-3) confirmed: worst diff/(eps64*max_lattice_length) ratio was 1.16;
    worst diff/(eps64*dist) ratio for tiny dist was unbounded (blew up as
    dist -> 0 with fixed absolute error). Re-derived against the lattice
    length scale: 100x headroom on the 1.16 worst ratio. See
    sanitizers.json.
    """
    diff = abs(dist - dist_recomputed)
    tol = 100.0 * eps64 * max(max_lattice_length, 1e-10)
    trigger_if(diff > tol, "PM-STR-004", diff=diff, tol=tol)


# ---------------------------------------------------------------------------
# Subsystem 6: analysis/phase_diagram.py -- thermodynamic convex hull
# ---------------------------------------------------------------------------

@_guard
def check_convex_hull_energy_non_negativity(e_above_hull, numerical_tol):
    """PM-PD-001: e_above_hull >= 0 for any hull-defining (qhull_entries) entry.

    6-entry Fe-O phase diagram (all qhull_entries checked): worst
    (most negative) e_above_hull observed was exactly 0. Tolerance set
    to -100x the class's own numerical_tol (1e-8), examined and reused
    per SANITIZER.md 5.8 T rather than assumed correct for this new
    quantity (confirmed the class already relies on this same law via
    its own check_stable/exception-raising branch).
    """
    tol = 100.0 * abs(numerical_tol)
    trigger_if(e_above_hull < -tol, "PM-PD-001", e_above_hull=e_above_hull, tol=tol)


@_guard
def check_decomposition_convex_combination(amount_sum, min_amount, dim):
    """PM-PD-002: decomposition amounts sum to 1 and are all non-negative.

    6-composition sweep on the Fe-O phase diagram: worst |sum-1| observed
    was 0, worst negative amount was 0. Tolerance set to
    100x*eps64*(dim+1) for headroom (dim+1 facet vertices summed).
    """
    tol = 100.0 * eps64 * (dim + 1)
    trigger_if(abs(amount_sum - 1.0) > tol, "PM-PD-002",
               family="sum_to_one", amount_sum=amount_sum, tol=tol)
    trigger_if(min_amount < -tol, "PM-PD-002",
               family="non_negative", min_amount=min_amount, tol=tol)


@_guard
def check_decomposition_mass_conservation(max_element_diff, dim):
    """PM-PD-003: decomposition amounts weighted by facet-entry compositions
    reconstruct the original (fractional) composition.

    Same 6-composition sweep: worst observed max element-wise diff was
    1.1e-16 (essentially exact). Tolerance set to 100x*eps64*(dim+1).
    """
    tol = 100.0 * eps64 * (dim + 1)
    trigger_if(max_element_diff > tol, "PM-PD-003", diff=max_element_diff, tol=tol)
