"""Tests for the fsolve backup of the speciation re-equilibration.

When the internal Newton / LM speciation solve (the free-variable system
that restores chemical equilibrium after a zone's kinetic step) fails,
the Python driver re-solves the same residual with
``scipy.optimize.fsolve`` (``_fsolve_backup_speciation``) and finalizes
the step through ``RtZone.step`` with ``x_free_override``, mirroring the
existing kinetic-step backup. This module covers, without running a full
model:

* the ``subsystem`` tag on the typed error that dispatches speciation vs
  kinetic failures (the two are indistinguishable by failed-vector
  dimension alone on networks whose null space carries as many free
  variables as aqueous species, as here: 5 = 5);
* bit-identical parity of the ``x_free_override`` finalization path
  against the internal solve, and of the explicit ``None`` (default)
  path;
* the backup's acceptance logic: pre-accepted candidates (fsolve never
  called), the residual-bar and physical-validity arbiters (fsolve's
  convergence flag is diagnostic-only and never decides acceptance),
  fsolve rejections (non-finite solution, stalled above the residual
  bar), and mis-dimensioned error payloads;
* the ``SolverBackupWarning`` emitted when the backup gives up (a real
  ``warnings`` warning, so batch and calibration scripts can silence it),
  with the zone, the step index and the specific rejection reason in the
  message.
"""

from __future__ import annotations

from typing import NoReturn

import numpy as np
import pytest
from numpy.typing import NDArray

from .. import reactive_transport_model
from ..core import (
    EquilibriumParameters,
    HydroForcing,
    HydroStep,
    MineralParameters,
    OptimizationError,
    RtForcing,
    RtParameters,
    RtStep,
    RtZone,
    ZoneDimensions,
)
from ..reactive_transport.common_networks import get_carbon_network
from ..reactive_transport_model import (
    SPECIATION_ACCEPT_RESIDUAL_MAX,
    SolverBackupWarning,
    _fsolve_backup_speciation,
)

# One entry per network species, in network order:
# DOC, HCO3-, H+, CO2(aq), CO3--, SOC(s), X-, XDOC.
# The aqueous block solved by the kinetic step is the first five entries;
# the mineral / exchange tail (indices 5-7) is frozen by the fused residual.
C0: NDArray[np.float64] = np.array([1e-3, 1e-3, 1e-7, 1e-4, 1e-10, 1e-2, 1e-3, 1e-3])
DT_DAYS: float = 1.0

# Every field of an ``RtStep``, for whole-step comparisons.
RT_STEP_FIELDS: tuple[str, ...] = (
    "state",
    "mineral_rates",
)


def _forcing(conc_in: NDArray[np.float64], forc_flux: float = 0.0) -> RtForcing:
    """A minimal forcing: no precipitation / inflow, optional surface
    forcing flux (which also opens the water denominator for mass
    coupling)."""
    hydro_step = HydroStep(
        state=10.0,
        forc_flux=forc_flux,
        lat_flux=0.0,
        vert_flux=0.0,
        vap_flux=0.0,
        q_in=0.0,
        lat_flux_ext=0.0,
        vert_flux_ext=0.0,
    )
    hydro_forc = HydroForcing(
        precip=0.0,
        temp=1.5,
        pet=0.0,
        q_in=0.0,
    )
    return RtForcing(
        np.asarray(conc_in, dtype=np.float64), hydro_step, hydro_forc, 0.03
    )


def _make_zone() -> RtZone:
    """A fresh zone with the default speciation cache (a zeros warm
    start), as a zone is in a newly built model."""
    mp = MineralParameters(
        np.array([0.1]), np.array([2.0]), np.array([1.5]), np.array([10.0])
    )
    dim = ZoneDimensions(20.0, 10.0)
    return RtZone(
        get_carbon_network(),
        RtParameters(dim, mp),
        do_reactions=True,
        do_speciation=True,
        name="speciation_backup_zone",
    )


def _fake_optimization_error(best_x: list[float]) -> OptimizationError:
    """A synthetic, fully-populated ``OptimizationError`` standing in for a
    failed speciation solve.

    Constructed with positional arguments only: CPython never forwards
    keyword arguments to the ``__new__`` of an exception class, so the
    constructor is positional-only at call time; the trailing ``subsystem``
    argument defaults to ``""`` (the backup is invoked directly here, so no
    tag is needed).
    """
    zeros: list[float] = [0.0] * len(best_x)
    return OptimizationError(
        100,
        1e6,
        [1e6],
        best_x,
        zeros,
        best_x,
        [],
        [],
        [],
        [],
        [],
        best_x,
        1e6,
        "synthetic speciation failure (test)",
    )


def _x_root_for(
    zone: RtZone, d: RtForcing
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The deterministic kinetic re-solve of a fresh zone and the
    free-variable root that re-equilibrates it — the same calls the
    backup itself makes, so the residual bar below is evaluated against
    exactly what the backup sees."""
    n_free: int = zone.eq.get_stoich_null_space().shape[1]
    c_after_rt: NDArray[np.float64] = np.asarray(zone.solve_rt_step(C0, d, DT_DAYS))
    x_root: NDArray[np.float64] = np.asarray(
        zone.eq.speciation_x_free_solve(c_after_rt, np.zeros(n_free))
    )
    return c_after_rt, x_root


def _near_miss_x(zone: RtZone, d: RtForcing) -> NDArray[np.float64]:
    """A free-variable vector whose residual sits strictly between the
    internal solver's 1e-6 target bar and the acceptance bar: the true
    root perturbed along a fixed direction until the residual norm lands
    in that window. Such an iterate is what the 90 real-world failing
    speciation cases end on (a stalled solve a little above the target
    bar), so this stands in for their failed-solve best iterate."""
    c_after_rt, x_root = _x_root_for(zone, d)
    direction: NDArray[np.float64] = np.ones(len(x_root))
    for scale in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11):
        x_try: NDArray[np.float64] = x_root + scale * direction
        f_max: float = float(
            np.max(np.abs(zone.eq.speciation_residual(c_after_rt, x_try)))
        )
        if 1e-6 < f_max <= SPECIATION_ACCEPT_RESIDUAL_MAX:
            return x_try
    raise AssertionError(
        "could not place a near-miss iterate strictly between the 1e-6 "
        "target bar and the acceptance bar"
    )


# ===================== subsystem tagging ===================== #
def test_speciation_failure_is_tagged_speciation() -> None:
    """A speciation solve whose target totals sit far outside the bounded
    box (all species at 1e300, i.e. ~300 log10 decades beyond what any
    in-box free-variable vector can reproduce) exhausts the Newton / LM
    budget and fails with a typed ``OptimizationError`` tagged
    ``subsystem="speciation"`` — the discriminator the driver uses to
    route the failure to the speciation backup."""
    zone = _make_zone()
    n_free: int = zone.eq.get_stoich_null_space().shape[1]
    chms: NDArray[np.float64] = np.full(len(C0), 1e300)
    with pytest.raises(OptimizationError) as excinfo:
        zone.eq.speciation_x_free_solve(chms, np.zeros(n_free))
    err = excinfo.value
    assert isinstance(err, OptimizationError)
    assert err.subsystem == "speciation"
    assert "Levenberg-Marquardt" in str(err)
    assert len(err.last_x) == n_free


def test_kinetic_failure_is_tagged_kinetic() -> None:
    """A kinetic-solve failure (a NaN in the aqueous warm start) is tagged
    ``subsystem="kinetic"``. On this network the two failure types are
    indistinguishable by failed-vector dimension alone
    (n_free == num_aqueous_species == 5), which is exactly why the tag
    exists: without it, this error would be mis-routed to the aqueous-
    block backup."""
    zone = _make_zone()
    n_aq: int = zone.network.num_aqueous_species
    c_0: NDArray[np.float64] = C0.copy()
    c_0[0] = np.nan
    with pytest.raises(OptimizationError) as excinfo:
        zone.solve_rt_step(c_0, _forcing(np.zeros(len(C0))), DT_DAYS)
    err = excinfo.value
    assert isinstance(err, OptimizationError)
    assert err.subsystem == "kinetic"
    assert len(err.last_x) == n_aq


# ===================== x_free_override finalization ===================== #
def test_override_step_matches_internal_solve_bit_for_bit() -> None:
    """Finalizing a step with the free-variable vector the internal solve
    would have found (``x_free_override``) is bit-identical to running
    the internal solve: same kinetic result, same adopted vector, same
    bookkeeping. Both fresh zones warm-start their speciation solve from
    zeros, so the explicit ``speciation_x_free_solve`` from zeros
    reproduces the reference zone's internal solve exactly."""
    zone_a = _make_zone()
    zone_b = _make_zone()
    n_free: int = zone_a.eq.get_stoich_null_space().shape[1]
    d = _forcing(np.zeros(len(C0)))

    c_after_rt: NDArray[np.float64] = np.asarray(zone_a.solve_rt_step(C0, d, DT_DAYS))
    x_ref: NDArray[np.float64] = np.asarray(
        zone_a.eq.speciation_x_free_solve(c_after_rt, np.zeros(n_free))
    )

    step_a = zone_a.step(C0, d, DT_DAYS, x_free_override=x_ref)
    step_b = zone_b.step(C0, d, DT_DAYS)

    for field in RT_STEP_FIELDS:
        np.testing.assert_array_equal(
            np.asarray(getattr(step_a, field)),
            np.asarray(getattr(step_b, field)),
            err_msg=f"RtStep field '{field}' differs",
        )


def test_explicit_none_override_matches_normal_step() -> None:
    """Passing ``x_free_override=None`` explicitly takes exactly the
    normal path: the step is bit-identical to a call that does not pass
    the keyword at all (the change is strictly additive on the default
    path)."""
    zone_a = _make_zone()
    zone_b = _make_zone()
    d = _forcing(np.zeros(len(C0)))

    step_a = zone_a.step(C0, d, DT_DAYS, x_free_override=None)
    step_b = zone_b.step(C0, d, DT_DAYS)

    for field in RT_STEP_FIELDS:
        np.testing.assert_array_equal(
            np.asarray(getattr(step_a, field)),
            np.asarray(getattr(step_b, field)),
            err_msg=f"RtStep field '{field}' differs",
        )


# ===================== _fsolve_backup_speciation ===================== #
def test_backup_preaccepts_candidate_root_without_fsolve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A candidate (the failed solver's best iterate) that already
    satisfies the acceptance bar (residual at most
    ``SPECIATION_ACCEPT_RESIDUAL_MAX``, physically valid) is accepted
    directly: fsolve is never spent on it (a raising stand-in proves
    this), and the finalized step matches a fresh zone's internal
    solve."""
    zone = _make_zone()
    d = _forcing(np.zeros(len(C0)))
    _c_after_rt, x_root = _x_root_for(zone, d)
    err = _fake_optimization_error(best_x=list(x_root))

    def _fsolve_must_not_run(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("fsolve must not be called for a pre-accepted candidate")

    monkeypatch.setattr(reactive_transport_model, "fsolve", _fsolve_must_not_run)

    rescued = _fsolve_backup_speciation(zone, C0, d, DT_DAYS, err, 0, False)

    assert rescued is not None
    reference = _make_zone().step(C0, d, DT_DAYS)
    for field in RT_STEP_FIELDS:
        np.testing.assert_allclose(
            np.asarray(getattr(rescued, field)),
            np.asarray(getattr(reference, field)),
            rtol=1e-9,
            atol=0.0,
            err_msg=f"RtStep field '{field}' differs",
        )


def test_backup_preaccepts_near_miss_without_fsolve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed solve whose best iterate sits strictly above the internal
    solver's 1e-6 target bar but at or under the acceptance bar
    (``SPECIATION_ACCEPT_RESIDUAL_MAX``, 1e-3), and is physically valid,
    is accepted directly: fsolve is never spent on it (a raising stand-in
    proves this) and the step is finalized from the best iterate through
    the ``x_free_override`` path. This is the relaxed-acceptance behavior
    that turns a stalled-but-close speciation into an accepted model step
    instead of a run failure; under the former 1e-6 bar this exact
    iterate would have been rejected and the run stopped."""
    zone = _make_zone()
    d = _forcing(np.zeros(len(C0)))
    x_near = _near_miss_x(zone, d)

    def _fsolve_must_not_run(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("fsolve must not be called for a pre-accepted candidate")

    monkeypatch.setattr(reactive_transport_model, "fsolve", _fsolve_must_not_run)

    err = _fake_optimization_error(best_x=list(x_near))
    rescued = _fsolve_backup_speciation(zone, C0, d, DT_DAYS, err, 0, False)

    assert rescued is not None
    # The adopted state is the near-miss iterate itself, not a re-solved
    # root: its residual sits strictly above the 1e-6 target bar (and at
    # or under the acceptance bar).
    c_after_rt, _x_root = _x_root_for(zone, d)
    f_max: float = float(
        np.max(np.abs(zone.eq.speciation_residual(c_after_rt, x_near)))
    )
    assert 1e-6 < f_max <= SPECIATION_ACCEPT_RESIDUAL_MAX
    # The step's bookkeeping is the normal post-solve path applied to the
    # adopted vector: the outgoing masses computed from the rescued state
    # match those of a reference step run with the same override.
    reference = _make_zone().step(C0, d, DT_DAYS, x_free_override=x_near)
    resc_lat: NDArray[np.float64]
    resc_vert: NDArray[np.float64]
    resc_lat, resc_vert = zone.outgoing_mass(np.asarray(rescued.state), d)
    ref_lat: NDArray[np.float64]
    ref_vert: NDArray[np.float64]
    ref_lat, ref_vert = zone.outgoing_mass(np.asarray(reference.state), d)
    np.testing.assert_allclose(
        resc_lat,
        ref_lat,
        rtol=1e-9,
        atol=0.0,
        err_msg="RtStep field 'lat_mass' differs",
    )
    np.testing.assert_allclose(
        resc_vert,
        ref_vert,
        rtol=1e-9,
        atol=0.0,
        err_msg="RtStep field 'vert_mass' differs",
    )


def test_backup_rejects_failed_fsolve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An fsolve result that is non-finite, or that stalls above the
    residual bar, is not used: the backup returns ``None`` so the driver
    stops the run with the original failure. The candidate (7.0 per free
    variable) is finite and well-dimensioned but far from a root, so both
    mocks below are actually reached."""
    zone = _make_zone()
    n_free: int = zone.eq.get_stoich_null_space().shape[1]
    d = _forcing(np.zeros(len(C0)))
    err = _fake_optimization_error(best_x=[7.0] * n_free)

    def _nan_solution(
        _func: object, x0: NDArray[np.float64], **_kwargs: object
    ) -> tuple[NDArray[np.float64], dict[str, object], int, str]:
        return np.full(n_free, np.nan), {"fvec": np.zeros(n_free)}, 1, "converged"

    monkeypatch.setattr(reactive_transport_model, "fsolve", _nan_solution)
    assert _fsolve_backup_speciation(zone, C0, d, DT_DAYS, err, 0, False) is None

    def _above_bar(
        _func: object, x0: NDArray[np.float64], **_kwargs: object
    ) -> tuple[NDArray[np.float64], dict[str, object], int, str]:
        stalled: NDArray[np.float64] = (
            SPECIATION_ACCEPT_RESIDUAL_MAX * 10.0 * np.ones(n_free)
        )
        return np.full(n_free, 7.0), {"fvec": stalled}, 1, "stalled above the bar"

    monkeypatch.setattr(reactive_transport_model, "fsolve", _above_bar)
    assert _fsolve_backup_speciation(zone, C0, d, DT_DAYS, err, 0, False) is None


def test_backup_accepts_under_bar_root_regardless_of_ier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The acceptance decision never consults fsolve's convergence flag:
    a result at or under the acceptance bar with a physically valid root
    is accepted whatever ``ier`` fsolve reported, covering the
    real-world case where hybrd reports ``ier=5`` ("xtol too small" at a
    stationary point) for an iterate already at or under the bar. The
    root returned here is the zone's true free-variable root, so the
    finalized step matches a fresh zone's internal solve."""
    zone = _make_zone()
    d = _forcing(np.zeros(len(C0)))
    _c_after_rt, x_root = _x_root_for(zone, d)
    n_free: int = len(x_root)
    err = _fake_optimization_error(best_x=[7.0] * n_free)

    rescued: RtStep | None = None
    for ier in (0, 5):

        def _stalled_at_bar(
            _func: object,
            x0: NDArray[np.float64],
            **_kwargs: object,
        ) -> tuple[NDArray[np.float64], dict[str, object], int, str]:
            return x_root.copy(), {"fvec": np.zeros(n_free)}, ier, "at the bar"

        monkeypatch.setattr(reactive_transport_model, "fsolve", _stalled_at_bar)
        rescued = _fsolve_backup_speciation(zone, C0, d, DT_DAYS, err, 0, False)
        assert rescued is not None

    reference = _make_zone().step(C0, d, DT_DAYS)
    for field in RT_STEP_FIELDS:
        np.testing.assert_allclose(
            np.asarray(getattr(rescued, field)),
            np.asarray(getattr(reference, field)),
            rtol=1e-9,
            atol=0.0,
            err_msg=f"RtStep field '{field}' differs",
        )


def test_backup_rejects_converged_but_invalid_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The physical-validity check is the arbiter: a candidate and an
    fsolve result that meet the residual bar but imply non-finite or
    non-positive concentrations are rejected (the backup returns
    ``None``) no matter how cleanly fsolve converges — the same gate the
    internal solve and the ``x_free_override`` finalization apply."""
    zone = _make_zone()
    d = _forcing(np.zeros(len(C0)))
    _c_after_rt, x_root = _x_root_for(zone, d)
    n_free: int = len(x_root)
    err = _fake_optimization_error(best_x=list(x_root))

    def _conc_always_invalid(
        _eq: EquilibriumParameters, _x: NDArray[np.float64]
    ) -> bool:
        return False

    monkeypatch.setattr(
        reactive_transport_model, "_speciation_conc_valid", _conc_always_invalid
    )

    def _converged(
        _func: object, x0: NDArray[np.float64], **_kwargs: object
    ) -> tuple[NDArray[np.float64], dict[str, object], int, str]:
        return x0.copy(), {"fvec": np.zeros(n_free)}, 1, "converged"

    monkeypatch.setattr(reactive_transport_model, "fsolve", _converged)

    assert _fsolve_backup_speciation(zone, C0, d, DT_DAYS, err, 0, False) is None


def test_backup_rejects_misdimensioned_error_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An error payload whose failed vector does not dimension the
    speciation system (``n_free + 2`` here) cannot be re-solved: the
    backup returns ``None`` before it spends any work on the zone
    (fsolve never called)."""
    zone = _make_zone()
    n_free: int = zone.eq.get_stoich_null_space().shape[1]
    d = _forcing(np.zeros(len(C0)))
    err = _fake_optimization_error(best_x=[7.0] * (n_free + 2))

    def _fsolve_must_not_run(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("fsolve must not be called for a mis-dimensioned payload")

    monkeypatch.setattr(reactive_transport_model, "fsolve", _fsolve_must_not_run)

    assert _fsolve_backup_speciation(zone, C0, d, DT_DAYS, err, 0, False) is None


# ===================== warning emission ===================== #
def test_rejection_emits_solver_backup_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no candidate satisfies the acceptance bar, the backup emits a
    single ``SolverBackupWarning`` naming the zone and the step index — a
    real ``warnings`` warning, not plain ``print`` output — so a run
    stopped by the backup is self-describing on screen and batch and
    calibration scripts can silence it with
    ``warnings.filterwarnings("ignore",
    category=potions.SolverBackupWarning)``."""
    zone = _make_zone()
    n_free: int = zone.eq.get_stoich_null_space().shape[1]
    d = _forcing(np.zeros(len(C0)))
    err = _fake_optimization_error(best_x=[7.0] * n_free)

    def _nan_solution(
        _func: object, x0: NDArray[np.float64], **_kwargs: object
    ) -> tuple[NDArray[np.float64], dict[str, object], int, str]:
        return np.full(n_free, np.nan), {"fvec": np.zeros(n_free)}, 1, "converged"

    monkeypatch.setattr(reactive_transport_model, "fsolve", _nan_solution)

    with pytest.warns(SolverBackupWarning) as record:
        assert _fsolve_backup_speciation(zone, C0, d, DT_DAYS, err, 37, False) is None
    assert len(record) == 1
    message: str = str(record[0].message)
    assert "zone 'speciation_backup_zone'" in message
    assert "step 37" in message


def test_rejection_warning_carries_rejection_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``SolverBackupWarning`` message carries *why* the final
    candidate was rejected, not merely that the backup failed: here
    fsolve reaches a point at or under the residual bar, so the deciding
    check is the physical-validity gate, and the reason in the message
    names that gate (with the residual the candidate reached)."""
    zone = _make_zone()
    d = _forcing(np.zeros(len(C0)))
    _c_after_rt, x_root = _x_root_for(zone, d)
    n_free: int = len(x_root)
    err = _fake_optimization_error(best_x=list(x_root))

    def _conc_always_invalid(
        _eq: EquilibriumParameters, _x: NDArray[np.float64]
    ) -> bool:
        return False

    monkeypatch.setattr(
        reactive_transport_model, "_speciation_conc_valid", _conc_always_invalid
    )

    def _converged(
        _func: object, x0: NDArray[np.float64], **_kwargs: object
    ) -> tuple[NDArray[np.float64], dict[str, object], int, str]:
        return x0.copy(), {"fvec": np.zeros(n_free)}, 1, "converged"

    monkeypatch.setattr(reactive_transport_model, "fsolve", _converged)

    with pytest.warns(SolverBackupWarning) as record:
        assert _fsolve_backup_speciation(zone, C0, d, DT_DAYS, err, 0, False) is None
    assert len(record) == 1
    message: str = str(record[0].message)
    assert "no acceptable solution" in message
    assert "physically invalid" in message
