"""Tests for the typed ``OptimizationError`` and the ``scipy.optimize.fsolve``
kinetic-solve backup path.

Covers:

* a failed Rust Newton / LM solve raising a *typed* ``OptimizationError``
  (``isinstance`` match) with the ``best_x`` / ``best_err`` diagnostics
  populated, which the Python driver relies on;
* ``RtZone.step_with_solution`` finalizing an externally computed
  post-reaction state identically to ``RtZone.step``;
* ``_fsolve_backup_step`` rescuing a zone whose kinetic solve failed and
  refusing to re-solve failures that are not the aqueous kinetic block
  (dimension mismatch) or whose fsolve result does not meet the acceptance
  checks (non-convergence, non-finite vector);
* ``run_zone_steps`` resume semantics: the ``start`` /
  ``lat_mass_init`` / ``vert_mass_init`` parameters reproduce an
  uninterrupted sweep, and invalid resume requests fail with a
  ``ValueError``.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from .. import reactive_transport_model
from ..core import (
    HydroForcing,
    HydroStep,
    MineralParameters,
    OptimizationError,
    RtForcing,
    RtParameters,
    RtZone,
    ZoneDimensions,
    run_zone_steps,
)
from ..reactive_transport.common_networks import get_carbon_network
from ..reactive_transport_model import (
    FSOLVE_BACKUP_RESIDUAL_MAX,
    _fsolve_backup_step,
)

# One entry per network species, in network order:
# DOC, HCO3-, H+, CO2(aq), CO3--, SOC(s), X-, XDOC.
# The aqueous block solved by the kinetic step is the first five entries;
# the mineral / exchange tail (indices 5-7) is frozen by the fused residual.
C0: NDArray[np.float64] = np.array([1e-3, 1e-3, 1e-7, 1e-4, 1e-10, 1e-2, 1e-3, 1e-3])
DT_DAYS: float = 1.0


def _forcing(conc_in: NDArray[np.float64], forc_flux: float = 0.0) -> RtForcing:
    """A minimal forcing: no precipitation / inflow, optional surface forcing
    flux (which also opens the water denominator for mass coupling)."""
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


def _fake_optimization_error(best_x: list[float]) -> OptimizationError:
    """A synthetic, fully-populated ``OptimizationError`` standing in for a
    failed Newton / LM solve.

    Constructed with positional arguments only: CPython never forwards
    keyword arguments to the ``__new__`` of an exception class, so the
    14-argument constructor is positional-only at call time.
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
        "synthetic failure (test)",
    )


@pytest.fixture(scope="module")
def zone() -> RtZone:
    mp = MineralParameters(
        np.array([0.1]), np.array([2.0]), np.array([1.5]), np.array([10.0])
    )
    dim = ZoneDimensions(20.0, 10.0)
    return RtZone(
        get_carbon_network(),
        RtParameters(dim, mp),
        do_reactions=True,
        do_speciation=True,
        name="backup_zone",
    )


# ===================== Typed OptimizationError ===================== #
def test_nan_state_raises_typed_optimization_error(zone: RtZone) -> None:
    """A NaN state makes the Newton / LM chain fail with a *typed*
    ``OptimizationError`` (the historical bug was a bare ``Exception``),
    with the best-iterate diagnostics populated for the Python driver.

    Two trigger classes: a NaN inside the aqueous block (the warm start
    itself is non-finite, so the recorded best iterate is non-finite too)
    and a NaN in the frozen mineral / exchange tail (the warm start stays
    finite and can seed the fsolve backup, but the residual evaluation is
    non-finite, so the solve still fails).
    """
    n_aq: int = zone.network.num_aqueous_species

    c_0: NDArray[np.float64] = C0.copy()
    c_0[0] = np.nan
    with pytest.raises(OptimizationError) as excinfo:
        zone.solve_rt_step(c_0, _forcing(np.zeros(len(C0))), DT_DAYS)
    err = excinfo.value
    # The core of the fix: the raised object matches the class, so
    # `isinstance(e, OptimizationError)` in the driver works.
    assert isinstance(err, OptimizationError)
    assert "NaN" in str(err)
    assert len(err.last_x) == n_aq
    assert len(err.best_x) == n_aq
    # The best-iterate residual is non-finite: this is a non-convergence
    # failure, not a converged-but-wrong solution.
    assert np.isnan(err.best_err)
    assert not np.all(np.isfinite(np.asarray(err.best_x)))

    c_tail: NDArray[np.float64] = C0.copy()
    c_tail[5] = np.nan
    with pytest.raises(OptimizationError) as excinfo:
        zone.solve_rt_step(c_tail, _forcing(np.zeros(len(C0))), DT_DAYS)
    tail_err = excinfo.value
    assert "NaN" in str(tail_err)
    assert len(tail_err.last_x) == n_aq
    assert np.all(np.isfinite(np.asarray(tail_err.best_x)))


def test_run_zone_steps_failure_triple_is_typed(zone: RtZone) -> None:
    """The failure triple returned by ``run_zone_steps`` preserves the typed
    exception object (not a stringified or re-wrapped one)."""
    c_0: NDArray[np.float64] = C0.copy()
    c_0[0] = np.nan
    states: NDArray[np.float64] = c_0[None, :]
    d = _forcing(np.zeros(len(C0)))

    steps, failure = run_zone_steps(
        [zone], states, [d], np.zeros((1, 1)), np.zeros((1, 1)), [], DT_DAYS
    )

    assert steps == []
    assert failure is not None
    i, err, conc_in = failure
    assert i == 0
    assert isinstance(err, OptimizationError)
    assert "NaN" in str(err)
    assert conc_in.shape[0] == len(C0)


# ===================== step_with_solution ===================== #
def test_step_with_solution_matches_step(zone: RtZone) -> None:
    """Feeding a state produced by ``RtZone.step`` back through
    ``step_with_solution`` reproduces the step's bookkeeping exactly
    (the post-solve path is shared)."""
    d = _forcing(np.zeros(len(C0)))
    reference = zone.step(C0, d, DT_DAYS)

    rescued = zone.step_with_solution(C0, reference.state, d, DT_DAYS)

    np.testing.assert_allclose(rescued.state, reference.state, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(rescued.lat_mass, reference.lat_mass)
    np.testing.assert_allclose(rescued.vert_mass, reference.vert_mass)
    np.testing.assert_allclose(rescued.conc_in, reference.conc_in)
    np.testing.assert_allclose(rescued.total_moles, reference.total_moles)
    np.testing.assert_allclose(rescued.mineral_rates, reference.mineral_rates)


# ===================== _fsolve_backup_step ===================== #
def test_fsolve_backup_rescues_converging_zone(zone: RtZone) -> None:
    """With a synthetic failed-solve error whose best iterate is a residual
    root (a stall short of convergence), the fsolve backup re-solves the
    aqueous block, passes its acceptance checks, and the finalized step
    matches the internal solver's to solver tolerance."""
    n_aq: int = zone.network.num_aqueous_species
    d = _forcing(np.zeros(len(C0)))

    reference = zone.step(C0, d, DT_DAYS)
    # The kinetic-solve solution stands in for Newton's best iterate: a
    # finite aqueous block at (near) a residual root, which is what the
    # driver hands to fsolve in a rescuable failure.
    sol_aq: NDArray[np.float64] = np.asarray(zone.solve_rt_step(C0, d, DT_DAYS))[:n_aq]
    err = _fake_optimization_error(best_x=list(sol_aq))

    rescued = _fsolve_backup_step(zone, C0, d, DT_DAYS, err, 0, verbose=False)

    assert rescued is not None
    # fsolve started at a residual root, so it stays there: the finalized
    # step matches the internal solver's bookkeeping to solver tolerance.
    np.testing.assert_allclose(
        np.asarray(rescued.state),
        np.asarray(reference.state),
        rtol=1e-6,
        atol=1e-12,
    )
    # The mass bookkeeping of the rescued step matches the internal
    # solver's: same finalization path from the same kinetic solution.
    np.testing.assert_allclose(
        np.asarray(rescued.lat_mass),
        np.asarray(reference.lat_mass),
        rtol=1e-9,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(rescued.vert_mass),
        np.asarray(reference.vert_mass),
        rtol=1e-9,
        atol=0.0,
    )
    # The acceptance bound is evaluated at the kinetic (pre-finalization)
    # solution: fsolve was warm-started at ``sol_aq``, a residual root, and
    # accepted there. The post-solve speciation re-equilibration then
    # redistributes the secondary aqueous species, so the *final* state is
    # not expected to satisfy the kinetic residual bound (a normal
    # ``zone.step`` final state has the same residual for this network).
    x_full: NDArray[np.float64] = np.concatenate(
        [sol_aq, np.asarray(C0)[n_aq:]]
    )
    f_max: float = float(
        np.max(np.abs(zone.residual_function(C0, x_full, d, DT_DAYS)[:n_aq]))
    )
    assert f_max <= FSOLVE_BACKUP_RESIDUAL_MAX


def test_fsolve_backup_refuses_dimension_mismatch(zone: RtZone) -> None:
    """The backup only re-solves failures of the aqueous kinetic block:
    a mismatched ``last_x`` size (e.g. a speciation solve) or a state that
    does not match the network both return ``None``."""
    n_aq: int = zone.network.num_aqueous_species
    d = _forcing(np.zeros(len(C0)))

    # ``last_x`` the wrong size: not an aqueous kinetic failure.
    wrong_dim_err = _fake_optimization_error(best_x=[0.0] * (n_aq + 3))
    assert _fsolve_backup_step(zone, C0, d, DT_DAYS, wrong_dim_err, 0, False) is None

    # A state that does not match the network size.
    short_state: NDArray[np.float64] = np.zeros(len(C0) - 1)
    good_err = _fake_optimization_error(best_x=list(C0[:n_aq]))
    assert (
        _fsolve_backup_step(zone, short_state, d, DT_DAYS, good_err, 0, False)
        is None
    )


def test_fsolve_backup_rejects_failed_fsolve(
    zone: RtZone, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fsolve that does not converge (``ier != 1``) or returns a
    non-finite vector is not used as a step result: the backup returns
    ``None`` so the driver can stop the run with the original failure."""
    n_aq: int = zone.network.num_aqueous_species
    d = _forcing(np.zeros(len(C0)))
    err = _fake_optimization_error(best_x=list(C0[:n_aq]))

    def _not_converged(
        _func: object, x0: NDArray[np.float64], **_kwargs: object
    ) -> tuple[NDArray[np.float64], dict[str, object], int, str]:
        return np.full(n_aq, 1.0), {"fvec": np.ones(n_aq)}, 0, "no convergence"

    monkeypatch.setattr(reactive_transport_model, "fsolve", _not_converged)
    assert _fsolve_backup_step(zone, C0, d, DT_DAYS, err, 0, False) is None

    def _nan_solution(
        _func: object, x0: NDArray[np.float64], **_kwargs: object
    ) -> tuple[NDArray[np.float64], dict[str, object], int, str]:
        return np.full(n_aq, np.nan), {"fvec": np.zeros(n_aq)}, 1, "converged"

    monkeypatch.setattr(reactive_transport_model, "fsolve", _nan_solution)
    assert _fsolve_backup_step(zone, C0, d, DT_DAYS, err, 0, False) is None


# ===================== run_zone_steps resume ===================== #
def test_run_zone_steps_resume_matches_full_run(zone: RtZone) -> None:
    """Resuming a two-zone sweep from zone 1 with seeded mass buffers
    reproduces the uninterrupted sweep's result for the remaining zone
    (the Gauss-Seidel coupling is identical)."""
    n_spec: int = len(C0)
    d0 = _forcing(np.zeros(n_spec), forc_flux=0.5)
    d1 = _forcing(np.zeros(n_spec), forc_flux=0.5)
    states: NDArray[np.float64] = np.stack([C0, C0])
    # Zone 1 receives all of zone 0's outgoing lateral mass.
    lat_mat: NDArray[np.float64] = np.array([[0.0, 0.0], [1.0, 0.0]])
    vert_mat: NDArray[np.float64] = np.zeros((2, 2))

    full_steps, full_failure = run_zone_steps(
        [zone, zone], states, [d0, d1], lat_mat, vert_mat, [], DT_DAYS
    )
    assert full_failure is None
    assert len(full_steps) == 2
    # Coupling is active: zone 1 sees a nonzero incoming concentration.
    assert np.any(np.asarray(full_steps[1].conc_in) != 0.0)

    lat_init: NDArray[np.float64] = np.zeros((2, n_spec))
    lat_init[0] = np.asarray(full_steps[0].lat_mass)
    vert_init: NDArray[np.float64] = np.zeros((2, n_spec))
    vert_init[0] = np.asarray(full_steps[0].vert_mass)

    resumed, failure = run_zone_steps(
        [zone, zone],
        states,
        [d0, d1],
        lat_mat,
        vert_mat,
        [],
        DT_DAYS,
        start=1,
        lat_mass_init=lat_init,
        vert_mass_init=vert_init,
    )

    assert failure is None
    assert len(resumed) == 1
    # Same incoming concentration, same initial state, deterministic solver:
    # the resumed step matches the uninterrupted one.
    np.testing.assert_allclose(
        np.asarray(resumed[0].conc_in), np.asarray(full_steps[1].conc_in)
    )
    np.testing.assert_allclose(
        np.asarray(resumed[0].state), np.asarray(full_steps[1].state), rtol=1e-12
    )


def test_run_zone_steps_resume_validation(zone: RtZone) -> None:
    """Invalid resume requests fail with informative ``ValueError``s:
    resuming past the last zone, and resuming without the required mass
    buffers."""
    n_spec: int = len(C0)
    d = _forcing(np.zeros(n_spec))
    states: NDArray[np.float64] = np.stack([C0])
    lat_mat: NDArray[np.float64] = np.zeros((1, 1))
    vert_mat: NDArray[np.float64] = np.zeros((1, 1))

    # `start` beyond the number of zones.
    steps, failure = run_zone_steps(
        [zone], states, [d], lat_mat, vert_mat, [], DT_DAYS, start=2
    )
    assert steps == []
    assert failure is not None
    assert isinstance(failure[1], ValueError)
    assert "start" in str(failure[1])

    # Resuming (`start > 0`) without the required mass buffers.
    steps, failure = run_zone_steps(
        [zone], states, [d], lat_mat, vert_mat, [], DT_DAYS, start=1
    )
    assert steps == []
    assert failure is not None
    assert isinstance(failure[1], ValueError)
    assert "lat_mass_init" in str(failure[1])

    # A wrong-shaped buffer is rejected too.
    steps, failure = run_zone_steps(
        [zone],
        states,
        [d],
        lat_mat,
        vert_mat,
        [],
        DT_DAYS,
        start=1,
        lat_mass_init=np.zeros((2, n_spec)),
        vert_mass_init=np.zeros((1, n_spec)),
    )
    assert steps == []
    assert failure is not None
    assert isinstance(failure[1], ValueError)
    assert "shape" in str(failure[1])
