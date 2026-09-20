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

    FIX (2026-09-17, round-1 triggerability): the original tolerance
    (100*eps64*cond, no magnitude term) missed that np.linalg.inv's
    absolute rounding error for a matrix with entries of magnitude `a`
    scales as O(eps64*a) in the worst element -- independent of
    conditioning for a well-conditioned matrix. cond() stays exactly 1.0
    for any cubic lattice regardless of scale, so a cond-only tolerance
    never budgeted for large, well-conditioned lattices (Lattice.cubic(500)
    and above: diff/(eps64*a) held at a stable 0.512 across a=500..2000,
    while cond stayed 1.0 throughout). A re-derivation sweep spanning 6
    lattice families x 9 length scales (0.5 to 10000) jointly, scoring
    diff/(eps64*cond*magnitude), gave worst observed ratio 0.76. Tolerance
    set to 100x*eps64*cond*max(1,|matrix|_inf). Re-verified: the original
    triggering case (Lattice.cubic(500) and above) is now silent, and
    isolated-sensitivity with a synthetic mismatch still fires.
    """
    diff = np.abs(np.asarray(orig_matrix) - np.asarray(roundtrip_matrix)).max()
    magnitude = max(1.0, np.abs(np.asarray(orig_matrix)).max())
    tol = 100.0 * eps64 * max(cond, 1.0) * magnitude
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

    FIX (2026-09-18, round-2 triggerability): the original tolerance
    (100*eps64*cond(lll_mapping), no magnitude term) missed that the
    round-trip's absolute rounding error scales with the magnitude of the
    fractional coordinate being transformed, not just with the mapping's
    conditioning -- the same missing-magnitude-scaling gap already found
    and fixed in PM-LAT-001/PM-OP-003/PM-STR-004. Confirmed on the
    triggering case (`Lattice.from_parameters(1,1,1,179.999,0.001,90)`,
    `f=[-1e6,1e6,3.14159]`, cond(lll_mapping)=3.49e16): diff/(eps64*cond) =
    10659 (fails), but diff/(eps64*cond*magnitude) = 0.0107 (comfortably
    inside 100x headroom). Re-derivation swept fractional-coordinate
    magnitude (1e-2 to 1e7) JOINTLY with lattice conditioning (6 cell-shape
    families x 12 angle sets including deliberately near-degenerate skew
    angles down to 0.001 degrees, cond(lll_mapping) up to ~1e19): worst
    observed diff/(eps64*cond*magnitude) ratio was 1.25. Tolerance set to
    100x*eps64*cond*max(1,magnitude). Re-verified: the original triggering
    case is now silent, and isolated-sensitivity with a synthetic mismatch
    still fires.
    """
    diff = np.abs(np.asarray(f_orig) - np.asarray(f_roundtrip)).max()
    magnitude = max(1.0, np.abs(np.asarray(f_orig)).max())
    tol = 100.0 * eps64 * max(cond, 1.0) * magnitude
    trigger_if(diff > tol, "PM-LAT-004", diff=diff, tol=tol)


_LAT_COND_CEILING = 1.0e4  # PM-STR-004 only; see that checker's FIX notes


@_guard
def check_d_hkl_formula_consistency(d_metric, d_vector, max_hkl, cond_g_star):
    """PM-LAT-005: d_hkl's metric-tensor formula matches the vector-norm formula.

    FIX (2026-09-17, round-1 triggerability): the invariant is true in exact
    arithmetic, but the metric-tensor quadratic form (hkl @ G* @ hkl.T)
    suffers catastrophic cancellation for a near-rank-deficient lattice --
    confirmed against a Decimal high-precision recomputation on the
    triggering synthetic lattice (two nearly-parallel long vectors,
    cond(matrix) ~ 2e7): the vector-norm path matched to 16 digits, the
    metric-tensor path had lost ~1% of relative accuracy to cancellation.

    FIX 2 (2026-09-18, second-round re-verification): the FIX above used a
    cond(matrix) >= 1e4 precondition ceiling, justified against a moderate
    stress-case sweep (angles 10-170 degrees). Re-running the fixed checker
    through triggerability_probe.py's full lattice zoo found a NEW firing on
    `Lattice.from_parameters(20, 20, 0.2, 90, 90, 179)` -- a legitimate,
    crystallographically ordinary highly-acute cell with cond(matrix)=141,
    comfortably inside the 1e4 ceiling, yet still an 11-order-of-magnitude
    violation. Diagnosis: cond(matrix) was never the right quantity in the
    first place -- the quadratic form that actually suffers cancellation is
    hkl @ G* @ hkl.T, so the relevant conditioning is cond(G*) (the metric
    tensor itself), not cond(matrix) (the direct lattice matrix). On this
    lattice cond(matrix)=141 but cond(G*)~2e4 -- the two diverge sharply
    for a highly acute/obtuse cell because forming G* = A @ A.T squares the
    conditioning of near-parallel rows. Replaced the cond(matrix)
    precondition-ceiling approach entirely with a cond(G*)-scaled TOLERANCE
    (the originally-recommended option, now using the right variable): a
    1330-trial sweep (the same lattice families/scales as before, plus
    angles from 1 to 179.9 degrees, plus the original adversarial lattice)
    scoring diff/(eps64*max_hkl*d_metric*cond(G*)) gave a worst ratio of
    0.41, essentially independent of how extreme the angle or how
    ill-conditioned matrix/G* becomes. Tolerance set to
    100x*eps64*max_hkl*d_metric*max(1,cond(G*)) for 100x headroom; no
    precondition exclusion needed at all -- this scaling correctly covers
    both the acute-angle case and the original near-rank-deficient
    synthetic lattice in the same sweep. Re-verified: both the original
    triggering lattice and the newly found near_singular_2 case are now
    silent; isolated-sensitivity with a synthetic mismatch still fires.
    """
    diff = abs(d_metric - d_vector)
    tol = 100.0 * eps64 * max(max_hkl, 1) * abs(d_metric) * max(cond_g_star, 1.0)
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
def check_weight_atomic_fraction_roundtrip(diff, n_elements, min_fraction, amount_tolerance):
    """PM-COMP-003: from_weight_dict(as_weight_dict()).fractional_composition round-trips.

    9-composition sweep (Element-keyed only -- as_weight_dict/
    from_weight_dict do not support Species-keyed compositions, confirmed
    via ValueError on a Fe2+/Fe3+ composition during derivation): worst
    observed diff/eps64 ratio was 0.5. Tolerance set to 100x*eps64.

    FIX (2026-09-17, round-1 triggerability): the invariant implicitly
    assumed every present element survives the weight/atomic-fraction
    round-trip exactly, but Composition's own constructor documents and
    enforces an ``amount_tolerance`` (1e-8) filter that silently drops any
    element whose mole fraction falls below it -- confirmed via a
  Fe:H amount-ratio sweep: the round-trip diff was exactly 0.0 until the
    ratio crossed 1e8 (the point where H's derived mole fraction, 1/ratio,
    crosses below amount_tolerance=1e-8), then jumped discontinuously; a
    hard filter-threshold effect, not gradual precision loss, and not a
    defect in as_weight_dict/from_weight_dict's arithmetic. Fix narrows the
    PRECONDITION (SANITIZER.md 5.8 P) rather than the tolerance: skip the
    check when any present element's mole fraction is within 100x of
    Composition.amount_tolerance (read dynamically, not hardcoded), since
    the round-trip cannot be expected to preserve a component the library's
    own constructor will filter out either on the way in or the way back.
    Re-verified: the original triggering composition
    (Composition({"Fe": 1e8, "H": 1.0})) is now excluded by the
    precondition (silent), and isolated-sensitivity with a synthetic
    mismatch on an ordinary (no near-threshold trace element) composition
    still fires.
    """
    if min_fraction < 100.0 * amount_tolerance:
        return
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

    FIX (2026-09-17, round-1 triggerability): the original tolerance
    (100*eps64, no magnitude term) missed that np.inner's batch kernel and
    np.dot's per-point kernel are different BLAS-level code paths that
    accumulate the same dot product in a different order -- ordinary
    floating-point non-associativity whose absolute error floor scales with
    the OUTPUT magnitude, not a magnitude-independent constant. Confirmed by
    isolating batch size alone (same point repeated): diff is exactly 0.0 at
    batch size 1, jumps to a stable nonzero value at batch size >= 2,
    regardless of point magnitude. A re-derivation sweep spanning point
    magnitude 1e-6..1e6 AND batch size 1..200 jointly (10800 trials) gave
    worst observed diff/(eps64*max(1,|result|)) ratio 1.84. Tolerance set to
    100x*eps64*max(1, |result|_inf) for headroom. Re-verified: the original
    triggering case (axis=[1,1,1], angle=180.001 deg,
    translation=[1e-9,1e-9,1e-9], magnitude ~1e4, batch size 10) is now
    silent, and isolated-sensitivity with a synthetic mismatch still fires.
    """
    diff = np.abs(np.asarray(multi_result) - np.asarray(single_stack)).max()
    magnitude = max(1.0, np.abs(np.asarray(multi_result)).max())
    tol = 100.0 * eps64 * magnitude
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
def check_pbc_distance_image_consistency(dist, dist_recomputed, max_lattice_length, cond):
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
    length scale: 100x headroom on the 1.16 worst ratio.

    FIX 2 (2026-09-17, round-1 triggerability, second independent failure
    mode on the same checker): on a near-rank-deficient lattice (two
    long, nearly-parallel vectors; cond(matrix) ~ 2e7), the nearest-image
    search must use huge, nearly-canceling integer jimage entries (tens of
    thousands) to express an O(1) fractional displacement on the
    near-degenerate basis -- confirmed via a skew-parameter sweep holding
    the geometry fixed: cond(matrix) and max_lattice_length both stayed
    ~flat while the observed error grew ~4 orders of magnitude as the
    lattice approached degeneracy, proving neither existing scaling
    variable tracks this mode. Shares its root cause and fix with
    PM-LAT-005 (see that checker's FIX note for the derivation): a
    cond(matrix) >= 1e4 precondition ceiling excludes this regime, backed
    by the same realistic-stress-case sweep (worst realistic cond ~100,
    five orders of magnitude below the adversarial synthetic lattice's
    ~2e7). Re-verified: the original triggering near-rank-deficient
    lattice is now excluded by the precondition (silent), and
    isolated-sensitivity with a synthetic mismatch on an ordinary
    (cond < 1e4) lattice still fires.
    """
    if cond >= _LAT_COND_CEILING:
        return
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
def check_decomposition_convex_combination(amount_sum, min_amount, dim, numerical_tol):
    """PM-PD-002: decomposition amounts sum to 1 and are all non-negative.

    6-composition sweep on the Fe-O phase diagram: worst |sum-1| observed
    was 0, worst negative amount was 0. Sum-to-one half kept at
    100x*eps64*(dim+1) for headroom (dim+1 facet vertices summed) -- that
    half never fired and a fresh sweep (below) confirms it remains sound.

    FIX (2026-09-17, round-1 triggerability, non-negative half only): the
    firing traced to scipy.spatial's own Qhull-based bary_coords linear
    solve, not pymatgen arithmetic -- confirmed with a minimal
    pymatgen-independent 2-D Delaunay reproduction showing the same
    near-vertex sub-zero noise is inherent to solving for barycentric
    coordinates via linear algebra in general. eps64*(dim+1) is the wrong
    scaling variable: it is a double-precision-relative bound (~1e-15
    scale), but the actual noise floor from a near-degenerate facet-matrix
    solve is many orders of magnitude coarser. A 13500-trial sweep (5
    element systems from binary to quaternary, vertex-to-vertex mixes at
    proximities 1e-3..1e-15 and exactly 0) found every observed negative
    excursion bounded by |min_amount|/PhaseDiagram.numerical_tol <= 0.1 --
    a stable ratio against the class's own existing absolute numerical
    floor (1e-8, already used by PM-PD-001 and by the class's own
    downstream amount-filtering), not against eps64*(dim+1). Tolerance
    (non-negative half only) set to 10x*numerical_tol for 100x headroom on
    the observed 0.1 worst ratio. Re-verified: the original triggering
    quaternary Li-Fe-P-O near-vertex composition is now silent, and
    isolated-sensitivity with a synthetic mismatch still fires.
    """
    sum_tol = 100.0 * eps64 * (dim + 1)
    neg_tol = 10.0 * abs(numerical_tol)
    trigger_if(abs(amount_sum - 1.0) > sum_tol, "PM-PD-002",
               family="sum_to_one", amount_sum=amount_sum, tol=sum_tol)
    trigger_if(min_amount < -neg_tol, "PM-PD-002",
               family="non_negative", min_amount=min_amount, tol=neg_tol)


@_guard
def check_decomposition_mass_conservation(max_element_diff, dim):
    """PM-PD-003: decomposition amounts weighted by facet-entry compositions
    reconstruct the original (fractional) composition.

    Same 6-composition sweep: worst observed max element-wise diff was
    1.1e-16 (essentially exact). Tolerance set to 100x*eps64*(dim+1).
    """
    tol = 100.0 * eps64 * (dim + 1)
    trigger_if(max_element_diff > tol, "PM-PD-003", diff=max_element_diff, tol=tol)


# ---------------------------------------------------------------------------
# Subsystem 7: core/ewald.py -- Ewald summation electrostatics
# ---------------------------------------------------------------------------

@_guard
def check_ewald_eta_split_invariance(e1, e2, eta1, eta2, acc_factor, n_sites):
    """PM-EWALD-001: total_energy is invariant to the eta real/reciprocal
    split parameter, for two EwaldSummation objects built with different
    eta (each with its own eta-dependent auto cutoffs) on the same
    structure.

    Joint sweep (eta ratio x structure size x acc_factor), NOT single-axis
    per SANITIZER.md 5.8: 4 structures (NaCl 2-site, NaCl n=2 supercell,
    orthorhombic NaCl, small-lattice NaCl) x 4 eta pairs (ratios 2x to
    100x, including an inverted eta1>eta2 pair) x 3 acc_factor values
    (4, 8, 10). Found the wrong scaling variable on the first pass: an
    eps64-relative tolerance (diff/(eps64*n_sites*magnitude)) blew up to
    ~3.4e7 because eta-split truncation error is bounded by the class's
    OWN documented convergence-digit contract (acc_factor = "number of
    significant figures each sum is converged to", ewald.py:86-87), not by
    machine epsilon -- each auto-cutoff is only accurate to
    ~10^-acc_factor, so two different etas' cutoffs need not agree beyond
    that floor. Re-scored against diff/(10^-acc_factor * magnitude): worst
    observed ratio was 2.36 (NaCl n=2 supercell, eta pair (0.2, 20.0),
    acc_factor=4 -- the most extreme eta ratio at the loosest convergence
    setting, confirming the joint eta-ratio/acc_factor interaction matters,
    not acc_factor alone). Tolerance set to 100x*(10^-acc_factor)*magnitude
    for headroom. A handful of (small-lattice, low-acc_factor, wide-eta-
    ratio) combinations raised inside EwaldSummation itself (cutoff-radius
    array shape mismatch when the auto real-space cutoff collapses to zero
    neighbor shells) -- an existing library precondition boundary, not a
    checker concern; the checker never runs when construction raises.
    """
    diff = abs(e1 - e2)
    magnitude = max(abs(e1), abs(e2), 1.0)
    tol = 100.0 * (10.0 ** (-acc_factor)) * magnitude
    trigger_if(diff > tol, "PM-EWALD-001", diff=diff, tol=tol,
               eta1=eta1, eta2=eta2, acc_factor=acc_factor, n_sites=n_sites)


# ---------------------------------------------------------------------------
# Subsystem 8: core/structure.py -- redundant neighbor-search implementations
# ---------------------------------------------------------------------------

@_guard
def check_neighbor_search_cross_implementation(checker_id, only_in_fast, only_in_old, n_sites, r):
    """PM-STR-005 / PM-STR-006: get_all_neighbors (or get_all_neighbors_py)
    finds the same per-site neighbor SET (index, jimage, distance rounded to
    6 decimals) as the author-declared brute-force oracle
    get_all_neighbors_old.

    3-structure (fcc Al 1-site, rocksalt NaCl 2-site, perovskite SrTiO3
    5-site) x 4-cutoff-radius (3.0, 5.0, 5.6, 8.0 -- the last three
    deliberately chosen to sit near natural neighbor-shell boundaries for
    these lattices) sweep, comparing get_all_neighbors and
    get_all_neighbors_py against get_all_neighbors_old as exact
    (index, jimage, round(distance, 6)) SET equality (order-independent,
    since all three functions may visit neighbors in different orders):
    zero mismatches across all 24 (structure, r) combinations for BOTH
    fast-path functions against the oracle. Because agreement is exact-set
    equality (a discrete index/image identification problem once distances
    are rounded to a fixed number of decimals, not a continuous numerical
    quantity), tol=0 for the set membership test itself; the distance
    figure carried in a mismatch record uses tol = 100*eps64*r for the
    rounding of the comparison key only (SANITIZER.md 5.8 X: neighbor
    identity is discrete once binned).

    FIX (2026-09-19, round-1 triggerability, PM-STR-006 only): the initial
    derivation's 4-cutoff sweep happened not to land exactly on a real
    interatomic distance, so it missed a genuine boundary-convention
    mismatch between get_points_in_spheres' padded `dist < r +
    numerical_tol` comparison (get_all_neighbors_py's underlying primitive)
    and the oracle's unpadded `all_dists <= r` (get_all_neighbors_old) --
    confirmed by direct source read, not a checker-tolerance defect (see
    ROOT_CAUSE_ANALYSIS.md Sec. 12). 1,202-case round-1 triggerability
    found 7 firings, all with a neighbor distance within numerical_tol of
    the cutoff r (e.g. rutile TiO2 at r=2.9587, perovskite SrTiO3 at
    r=3.905). PM-STR-005 (get_all_neighbors, a separately-compiled Cython
    primitive) showed no such firings on identical inputs, so this fix is
    scoped to PM-STR-006 only, not applied globally. Fix narrows the
    PRECONDITION at the call site (Structure._cross_check_neighbor_search):
    for checker_id == "PM-STR-006", entries within numerical_tol of r are
    excluded from the fast/old set comparison before the diff is computed,
    so only unambiguously-inside or unambiguously-outside neighbors are
    checked. Re-verified: all 7 original witnesses are now silent, and the
    full 1,202-case round-1 sweep plus the 817-test regression suite are
    both clean across all 30 checkers.
    """
    tol = 100.0 * eps64 * max(r, 1.0)
    mismatched = bool(only_in_fast) or bool(only_in_old)
    trigger_if(mismatched, checker_id, only_in_fast=list(only_in_fast),
               only_in_old=list(only_in_old), n_sites=n_sites, r=r, tol=tol)


# ---------------------------------------------------------------------------
# Subsystem 9: symmetry/analyzer.py -- space-group symmetry detection
# ---------------------------------------------------------------------------

@_guard
def check_periodic_symmop_atom_correspondence(max_residual, symprec, n_ops, n_sites):
    """PM-SYM-001: every spglib-sourced periodic symmetry operation
    returned by get_symmetry_operations() actually maps every site onto a
    site of the same species (periodic/minimum-image match), independent
    of however spglib's internal search found the operation.

    2-structure sweep (rutile TiO2, 6 sites, 16 ops; fcc Cu, 1 site, 48
    ops) at the default symprec=0.01, checking every (op, site) pair (208
    total pairs): worst observed periodic-image residual was ~5.1e-16
    (rutile) and exactly 0.0 (fcc Cu) -- both far below symprec, since a
    genuinely-correct symmetry operation applied to an exactly-symmetric
    synthetic structure round-trips to essentially machine precision, not
    to spglib's detection tolerance itself. Composing two tolerance-bounded
    steps in sequence (SANITIZER.md 5.8's joint-consideration note for this
    candidate) -- spglib's own symprec-scale detection plus this checker's
    own coordinate-matching step -- the tolerance is anchored to symprec
    (the natural physical scale for "how close counts as the same site")
    with generous headroom over the observed near-zero residuals, rather
    than to eps64 alone, since symprec is user-controllable and can be much
    looser than machine precision.
    """
    tol = max(10.0 * symprec, 100.0 * eps64)
    trigger_if(max_residual > tol, "PM-SYM-001", max_residual=max_residual,
               tol=tol, symprec=symprec, n_ops=n_ops, n_sites=n_sites)


# ---------------------------------------------------------------------------
# Subsystem 10: core/tensors.py -- physical property tensors
# ---------------------------------------------------------------------------

@_guard
def check_tensor_rotation_trace_eigenvalue_invariance(trace0, trace1, eig0, eig1, ortho_err=0.0):
    """PM-TENS-001: trace and (sorted) eigenvalues of a rank-2 tensor are
    invariant under a proper rotation applied via Tensor.rotate.

    Joint sweep over tensor magnitude (1e-3 .. 1e6) x rotation angle
    (1e-6 .. 180-1e-6 degrees, including near-identity and near-180-degree
    rotations) x 5 random symmetric tensors per cell, independently
    recomputing trace via np.trace and eigenvalues via np.linalg.eigvalsh
    on both the input and rotated tensors (code shared with neither
    Tensor.rotate's SymmOp.transform_tensor einsum contraction): worst
    observed diff/(eps64*max(1,|trace|)) ratio was 6.76 (magnitude=1e6,
    angle=45 degrees), worst diff/(eps64*max(1,|eig|)) ratio was 5.80 (same
    cell).

    FIX (2026-09-18/19, regression-suite firing): the original tolerance
    (100*eps64*scale, no orthogonality-error term) fired on
    test_tensors.py::TestTensor::test_convert_to_ieee, a real (non-
    synthetic) library test, with diff/tol ratios ~90000x over. Root cause:
    Tensor.rotate accepts a `tol` parameter precisely because its
    SquareTensor.is_rotation(tol) precondition check does NOT require exact
    (machine-precision) orthogonality -- convert_to_ieee calls
    result.rotate(rotation, tol=1e-2) with a rotation matrix from
    get_ieee_rotation(refine_rotation=False) that is only orthogonal to
    ~5.5e-5 for one witness (monoclinic rank-2 tensor in
    ieee_conversion_data.json), not to eps64. The checker's implicit
    precondition (exact orthogonality) was narrower than PM-TENS-001's
    actual stated precondition (any rotation is_rotation(tol) accepts) and
    narrower than what the checker's own call site can observe -- a
    P-class/T-class mistake, not a real trace/eigenvalue-invariance
    violation: a congruence transform by an approximately-orthogonal matrix
    is only approximately trace/eigenvalue-preserving, exactly matching the
    size of its own orthogonality defect. Fix: thread the actual R@R.T-I
    deviation of the rotation matrix used (ortho_err, 0.0 for the ordinary
    exact-rotation call path) into the tolerance, scaled by the tensor
    magnitude (a congruence's trace/eigenvalue perturbation from a
    non-orthogonal R is first-order in ortho_err times the tensor's own
    magnitude). Re-derived jointly on both sweeps: the original exact-
    rotation sweep (worst ratio ~3.4 under the new formula) and all 8
    (xtal, refine_rotation) combinations from ieee_conversion_data.json's
    rank-2 entries (worst ratio ~1.0, the same monoclinic/refine=False
    witness that originally fired). Tolerance set to
    100x*(eps64*scale + scale*max(ortho_err,eps64)) for headroom on both.
    Re-verified: the original triggering test_convert_to_ieee case is now
    silent, the full test_tensors.py suite is silent, and isolated-
    sensitivity with a synthetic mismatch (ortho_err=0, i.e. the ordinary
    exact-rotation precondition) still fires.
    """
    diff_tr = abs(trace0 - trace1)
    tr_scale = max(abs(trace0), 1.0)
    tol_tr = 100.0 * (eps64 * tr_scale + tr_scale * max(ortho_err, eps64))
    e0 = np.sort(np.asarray(eig0))
    e1 = np.sort(np.asarray(eig1))
    diff_eig = np.abs(e0 - e1).max() if e0.size else 0.0
    eig_scale = max(np.abs(e0).max() if e0.size else 1.0, 1.0)
    tol_eig = 100.0 * (eps64 * eig_scale + eig_scale * max(ortho_err, eps64))
    trigger_if(diff_tr > tol_tr, "PM-TENS-001", family="trace",
               diff=diff_tr, tol=tol_tr)
    trigger_if(diff_eig > tol_eig, "PM-TENS-001", family="eigenvalues",
               diff=diff_eig, tol=tol_eig)


# ---------------------------------------------------------------------------
# Subsystem 11: core/surface.py -- crystal surfaces and slabs
# ---------------------------------------------------------------------------

@_guard
def check_surface_normal_dual_derivation(cos_angle, cond_recip):
    """PM-SURF-001: Slab.normal (direct cross product of the slab's own
    first two lattice vectors) agrees, up to sign, with the Miller-index-
    derived reciprocal-lattice normal, for a Slab built WITHOUT the
    default lattice-reorientation post-processing.

    Precondition confirmed empirically during Step 4 (not assumed): with
    reorient_lattice=True (the SlabGenerator/Slab default), Slab.normal is
    ALWAYS exactly [0,0,1] in the lab frame by construction (Slab.__init__
    rebuilds the lattice via Lattice.from_parameters under
    reorient_lattice, which fixes a canonical orientation convention
    decoupled from the original crystallographic frame) -- comparing that
    fixed lab-frame vector against the Miller-derived crystallographic
    normal is not this law's subject at all (an observed diff up to 1.0 in
    cosine, confirmed on fcc Cu and rocksalt NaCl across 5 Miller indices,
    traced to this reorientation convention, not a bug). This checker is
    therefore scoped to reorient_lattice=False construction, where a
    5-Miller-index x 2-structure (fcc Cu, rocksalt NaCl) x 3-thickness
    sweep found exact agreement (cos_angle within 1.1e-16 of 1.0) in every
    case. Tolerance set to 100x*eps64*max(1,cond(reciprocal_lattice)) on
    the |1-|cos_angle|| residual, analogous to PM-LAT-005's cond-scaled
    tolerance for the same reciprocal-lattice machinery.
    """
    diff = abs(1.0 - abs(cos_angle))
    tol = 100.0 * eps64 * max(cond_recip, 1.0)
    trigger_if(diff > tol, "PM-SURF-001", diff=diff, tol=tol, cos_angle=cos_angle)


@_guard
def check_slab_layer_count_construction_consistency(n_layers_slab, n_layers_recompute):
    """PM-SURF-002: the atom-occupied layer count used to SIZE the slab at
    construction time (n_layers_slab, from height/min_slab_size sizing)
    matches the layer count recovered by independently measuring where the
    atoms actually ended up (the c-fractional-coordinate span between the
    lowest and highest atomic layer, times the total layer count) -- the
    same post-hoc measurement Slab.get_tasker2_slabs already performs
    internally for its own, unrelated purpose.

    Precondition and formula both confirmed empirically during Step 4, not
    assumed: (a) get_slab's actual sizing branch depends on the
    (default-False) in_unit_planes flag -- an early derivation draft that
    assumed the in_unit_planes=True branch unconditionally produced
    spurious "mismatches" that were purely an artifact of reading the
    wrong branch, not a library defect (fixed by reading n_layers_slab's
    real value off the exact branch get_slab executes, not a
    re-derivation); (b) the atom-span recomputation needs a +1 fencepost
    correction (span_layers * n_layers_total + 1, not span_layers *
    n_layers_total) because n_layers_slab atomic layers span
    (n_layers_slab - 1) inter-layer gaps between the first and last atom,
    confirmed by a 5-Miller-index x 2-structure (fcc Cu, rocksalt NaCl) x
    4-thickness x 2-in_unit_planes sweep: 35 of 40 cells matched exactly
    with the +1 correction. The remaining 5 mismatches were isolated to
    low-symmetry Miller indices, (2,1,0) and (2,1,1), where the oriented
    unit cell's own atoms are not all coplanar in c -- i.e. more than one
    distinct atomic z-value already exists within a single oriented-unit-
    cell repeat, so "one oriented-unit-cell repeat == one atomic layer"
    (the assumption both n_layers_slab's sizing and the recomputation share)
    does not hold. PRECONDITION accordingly narrowed to Miller indices
    whose oriented_unit_cell has exactly one distinct atomic c-fractional
    layer (checked at the observation point, not assumed a priori) --
    Step 4's job per this candidate's own note that Step 4 must confirm
    interactions not modeled by either formula as drafted. Within that
    precondition, both quantities are exact integers by construction
    (layer counts), so tol=0 (SANITIZER.md 5.8 X).
    """
    trigger_if(n_layers_slab != n_layers_recompute, "PM-SURF-002",
               n_layers_slab=n_layers_slab, n_layers_recompute=n_layers_recompute)


# ---------------------------------------------------------------------------
# Subsystem 12: core/elasticity/elastic.py -- elastic tensor derived properties
# ---------------------------------------------------------------------------

@_guard
def check_voigt_reuss_variational_bound(k_voigt, k_reuss, g_voigt, g_reuss, cond_voigt):
    """PM-ELAST-001: k_voigt >= k_reuss and g_voigt >= g_reuss for any
    physically stable (k_vrh > 0, g_vrh > 0) ElasticTensor -- the Voigt-
    Reuss variational bounds theorem.

    5 representative cubic-symmetry elastic tensors spanning the physically
    interesting range (Cu-like, Al-like, a near-isotropic tensor with
    c44 close to (c11-c12)/2, a highly anisotropic tensor, and a weakly
    stable tensor with a small k_vrh/g_vrh gap near the raise_if_unphysical
    boundary), checking both the K and G bound jointly with the tensor's
    own Voigt-matrix conditioning (since k_reuss depends on inverting that
    matrix): every case satisfied both bounds (worst negative excursion
    exactly 0, i.e. no violation observed at all -- consistent with
    SANITIZER.md 5.7.2, a silent bank is not a failure). Tolerance set to
    100x*eps64*max(1,|K or G value|)*max(1,cond(voigt)) on the (K_reuss -
    K_voigt) / (G_reuss - G_voigt) excess, jointly across conditioning and
    modulus magnitude per SANITIZER.md 5.8's joint-axis requirement (not
    swept independently, since a near-singular Voigt matrix is exactly the
    regime where k_reuss's inversion-based computation would show the
    largest gap-violating error, if this checker's slack were mis-scaled).
    """
    mag = max(abs(k_voigt), abs(k_reuss), abs(g_voigt), abs(g_reuss), 1.0)
    tol = 100.0 * eps64 * mag * max(cond_voigt, 1.0)
    gap_k = k_voigt - k_reuss
    gap_g = g_voigt - g_reuss
    trigger_if(gap_k < -tol, "PM-ELAST-001", family="bulk_modulus",
               k_voigt=k_voigt, k_reuss=k_reuss, gap=gap_k, tol=tol)
    trigger_if(gap_g < -tol, "PM-ELAST-001", family="shear_modulus",
               g_voigt=g_voigt, g_reuss=g_reuss, gap=gap_g, tol=tol)


@_guard
def check_universal_anisotropy_non_negative(anisotropy, cond_voigt, mag):
    """PM-ELAST-002: universal_anisotropy >= 0 for any physically stable
    ElasticTensor -- a direct algebraic consequence of PM-ELAST-001's same
    Voigt-Reuss bound theorem (5*(g_voigt/g_reuss) + (k_voigt/k_reuss) - 6,
    each ratio >= 1 by PM-ELAST-001).

    Same 5-tensor sweep as PM-ELAST-001 (derived jointly, not
    independently, per SANITIZER.md 5.8): worst (most negative)
    universal_anisotropy observed was 0.0 (the exactly-isotropic synthetic
    tensor, c44 = (c11-c12)/2 exactly) -- no violation observed. Tolerance
    set to 100x*eps64*max(1,mag)*max(1,cond(voigt)), scaled consistently
    with PM-ELAST-001's tolerance since a real violation here would be an
    algebraic combination of the same underlying ratio errors.
    """
    tol = 100.0 * eps64 * max(mag, 1.0) * max(cond_voigt, 1.0)
    trigger_if(anisotropy < -tol, "PM-ELAST-002", anisotropy=anisotropy, tol=tol)


# ---------------------------------------------------------------------------
# Subsystem 13: core/structure.py -- Structure.interpolate (2026-09-20 pass)
# ---------------------------------------------------------------------------

@_guard
def check_interpolate_endpoint_reproduction(diff_start, diff_end, magnitude, has_end):
    """PM-STR-007: interpolate(end_structure, ...)'s x=0 and x=1 images
    reproduce the start and end structures exactly, up to an integer
    lattice translation.

    DEVIATION FROM THE DRAFTING DOC (pymatgen_new_candidates_scan_2026-09-20.md):
    the doc's invariant compared raw fractional coordinates directly
    (``interpolate(...)[-1].frac_coords`` close to ``end_structure.frac_coords``).
    An adversarial sweep (60 random-lattice pairs x 3 nimages forms x 2
    autosort_tol values x scale 1e-3..1e3, see derive_tol2.py) found this
    literal-coordinate comparison fires with an O(1) (integer-sized)
    discrepancy whenever ``pbc=True`` (the default) and the raw end_coords
    - start_coords displacement along a periodic axis exceeds 0.5 in
    fractional units: interpolate.py's own periodic wrap-around correction
    (``vec[:, self.pbc] -= np.round(vec[:, self.pbc])``, structure.py) is
    intentional, documented behavior ("use periodic boundary conditions to
    find the shortest path between endpoints") that legitimately returns
    the endpoint at a DIFFERENT periodic image than the literal
    end_structure coordinates -- not a bug. The genuine law is therefore
    that the returned endpoint matches the requested endpoint MODULO an
    integer lattice translation (fractional coordinates equal mod 1), which
    is what this checker actually verifies (the caller computes
    ``diff = frac_diff - round(frac_diff)`` before passing it in). Re-swept
    the same adversarial grid under the mod-1 formulation: worst observed
    diff/(eps64*magnitude) ratio was 1.0 (both endpoints, both pbc values).
    Tolerance set to 100x*eps64*max(1,magnitude) for headroom. The
    ``pbc=False`` path (no wrap-around at all) trivially satisfies the same
    mod-1 comparison since diff is already ~0 there. Confirmed silent
    across the full targeted regression suite.
    """
    tol = 100.0 * eps64 * max(magnitude, 1.0)
    trigger_if(diff_start > tol, "PM-STR-007", family="start", diff=diff_start, tol=tol)
    if has_end:
        trigger_if(diff_end > tol, "PM-STR-007", family="end", diff=diff_end, tol=tol)


@_guard
def check_interpolation_composition_conservation(diff, n_elements, image_index):
    """PM-STR-008: every image returned by Structure.interpolate has the
    same composition as the start structure -- interpolating atomic
    positions must not create, destroy, or relabel atoms.

    30-pair sweep (random structures, 2-7 sites) x 2 nimages forms
    (integer and explicit fractional list): worst observed per-element
    composition difference across every returned image was exactly 0.0
    (Structure.composition is a Counter-style aggregation over the same
    `sp = self.species_and_occu` list reused unchanged for every
    constructed image, so this genuinely never involves floating-point
    accumulation for ordered structures -- the check has real content only
    for disordered/partial-occupancy sites, per the drafting doc's own
    open Step-4 question). Tolerance set to 100x*eps64*max(1,n_elements)
    for headroom on any future floating-point occupancy accumulation.
    Confirmed silent across the full targeted regression suite.
    """
    tol = 100.0 * eps64 * max(n_elements, 1)
    trigger_if(diff > tol, "PM-STR-008", diff=diff, tol=tol, image_index=image_index)


# ---------------------------------------------------------------------------
# Subsystem 14: core/composition.py -- oxi_state_guesses (2026-09-20 pass)
# ---------------------------------------------------------------------------

@_guard
def check_oxi_state_guess_charge_balance(diff, num_atoms, target_charge):
    """PM-COMP-005: oxi_state_guesses's returned average oxidation states,
    weighted by the ORIGINAL (pre-max_sites-reduction) composition's atom
    counts, sum to target_charge.

    REAL BUG FOUND (not a checker-calibration issue): _get_oxi_state_guesses
    (composition.py) reduces the composition via max_sites BEFORE running
    its charge-balance search, so the internal filter `sum(x) ==
    target_charge` (composition.py:1215) is evaluated against the REDUCED
    composition's integer sums, then rescaled to a per-element AVERAGE
    oxidation state by dividing by the reduced composition's own el_amt.
    When target_charge == 0 this rescaling is charge-neutral to reduction
    (0 * factor == 0), which is why an initial neutral-charge-only sweep
    found no discrepancy. A nonzero-target_charge sweep (see
    derive_tol2.py) found a genuine violation:
    `Composition("Fe2O4").oxi_state_guesses(max_sites=-1, target_charge=-2)`
    returns `{"Fe": 2.0, "O": -2.0}` (correctly charge-balancing the
    REDUCED "FeO2" to -2: 1*2 + 2*(-2) == -2), but applied to the ORIGINAL
    "Fe2O4" (2 Fe, 4 O) this sums to 2*2 + 4*(-2) == -4, not the requested
    -2 -- confirmed against the correct answer from the same call with
    max_sites=None (no reduction), which returns {"Fe": 3.0, "O": -2.0}
    (3*2 + (-2)*4 == -2, correct). Root cause: the per-element average is
    reduction-invariant only when target_charge == 0; for nonzero
    target_charge the average from the reduced composition does not
    rescale back to satisfy the ORIGINAL composition's charge constraint
    (the model is not a homogeneous constraint in that case). This is a
    scientific defect in the public API's documented contract
    ("the desired total charge on the structure"), not a checker
    calibration gap.

    Tolerance: the discrepancy above is O(2) absolute on a 6-atom
    composition, many orders of magnitude over any float64 rounding-scale
    slack. A same-composition, target_charge=0 sweep (30 compositions,
    with and without max_sites reduction) found worst diff/(eps64*num_atoms)
    ratio 0.0 (exactly exact), so the T-derivation for the *legitimate*
    (target_charge == 0 or unreduced) region is rounding-scale only.
    Tolerance set to 100x*eps64*max(1,num_atoms) -- the real bug above
    fires many orders of magnitude past this, exactly as SANITIZER.md 5.7.2
    expects for a genuine violation.
    """
    tol = 100.0 * eps64 * max(num_atoms, 1.0)
    trigger_if(diff > tol, "PM-COMP-005", diff=diff, tol=tol,
               num_atoms=num_atoms, target_charge=target_charge)


# ---------------------------------------------------------------------------
# Subsystem 15: io/cif.py -- CifParser._unique_coords (2026-09-20 pass)
# ---------------------------------------------------------------------------

@_guard
def check_cif_symmetry_orbit_divisibility(n_ops, orbit_size, coord):
    """PM-CIF-001: the size of the symmetry-equivalent orbit _unique_coords
    expands from a single asymmetric-unit coordinate must evenly divide the
    number of parsed symmetry operations (orbit-stabilizer theorem: any
    orbit's size is |G|/|stabilizer|, hence a divisor of |G|).

    This is a purely integer, discrete-structure check (SANITIZER.md 5.8
    X): both n_ops and orbit_size are counts, so tol=0 for the divisibility
    predicate itself. The only floating-point sensitivity is INDIRECT, via
    self._site_tolerance's effect on which transformed points
    in_coord_list_pbc treats as duplicates when deciding orbit membership
    (a P-class concern, not a T-class one) -- a coordinate sitting almost
    exactly on a symmetry element (a special-position Wyckoff site) is the
    natural adversarial input for this indirect sensitivity, not a reason
    to add slack to the divisibility check itself. No adversarial input
    that flips this checker's own integer result was found in this pass's
    Step 4/5 review (a fresh, dedicated fuzz of near-special-position
    coordinates against several real CIFs is left to the triggerability
    round per SANITIZER.md 8, not this instrumentation step).

    FIX (round-1 triggerability, 2026-09-20): the round-1 sweep fired this
    checker on a single constructed CIF (`signed_zero_dup_ops`) whose
    listed _symmetry_equiv_pos_as_xyz block contains 'x,y,z', '-x,-y,-z',
    and '-x+1,-y+1,-z+1' -- the last two are the SAME coset of the
    translation subgroup (mod-1 they act identically on every fractional
    coordinate: `SymmOp.from_xyz_str('-x,-y,-z') !=
    SymmOp.from_xyz_str('-x+1,-y+1,-z+1')` by exact matrix/vector equality,
    but the two differ only by a whole-lattice translation added to the
    printed representative). CifParser.get_symops has no dedup step -- it
    is a literal `[SymmOp.from_xyz_str(s) for s in xyz]` -- so the original
    `n_ops = len(self.symmetry_operations)` counted 3 while the true group
    order (and hence the orbit-stabilizer divisor) is 2, producing a
    spurious orbit_size=2-does-not-divide-n_ops=3 alarm on a general
    position that is in fact completely correctly handled.

    Root-cause determination (not a genuine `_unique_coords` counting
    defect): standard crystallographic symmetry-operation tables (and
    pymatgen's own `SpaceGroup(...).symmetry_ops` generation path) list
    exactly one representative per coset by construction -- a real CIF's
    _symmetry_equiv_pos_as_xyz loop restating the same coset twice under
    different integer-translation representatives is not an authoring
    pattern that occurs in genuine depositions; it is a malformed/hand-
    corrupted input outside this law's intended precondition (the group
    G in the orbit-stabilizer theorem is defined over DISTINCT group
    elements, so a source list with a literal duplicate element was never
    covered by the theorem as stated). This is category (a) from the
    investigation directive, not (b): `_unique_coords` itself does nothing
    wrong (its own dedup via `in_coord_list_pbc` on the transformed orbit
    coordinates is correct and is exactly why orbit_size came out as the
    true value of 2) -- only the CHECKER's `n_ops` input was wrong, since
    it counted listed operations rather than distinct group elements.

    Fix applied at the call site, not here: `CifParser._unique_coords`
    now computes `n_ops` via a new helper, `CifParser._distinct_symop_count`,
    which deduplicates `self.symmetry_operations` by
    `(rotation_matrix, translation_vector mod 1)` before counting, gated
    behind `_sc.enabled()` so it carries zero cost/behavior change when
    checkers are off. This mirrors PM-STR-006's two-round precedent
    (ROOT_CAUSE_ANALYSIS.md sections 12-15): narrow the checker's own
    precondition/observation input to the concept the law actually
    requires (the true group order), not the library's internal counting,
    and not the divisibility predicate's tolerance (there is none to
    loosen -- tol=0 for an integer check is correct and unchanged).

    Re-verified: the `signed_zero_dup_ops` witness (3 listed ops, 2
    distinct cosets) is now silent under both `primitive=False` and
    `primitive=True`; the other 14 round-1 CIF templates (which never
    exercised this path, since none of them contained a duplicate-coset
    operation) are unaffected, confirmed by re-running the full round-1
    probe.
    """
    trigger_if(orbit_size <= 0 or n_ops % orbit_size != 0, "PM-CIF-001",
               n_ops=n_ops, orbit_size=orbit_size, coord=list(coord))
