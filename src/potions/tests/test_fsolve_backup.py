"""Tests for the typed ``OptimizationError`` and the ``scipy.optimize.fsolve``
kinetic-solve backup path.

Covers:

* a failed Rust Newton / LM solve raising a *typed* ``OptimizationError``
  (``isinstance`` match) with the ``best_x`` / ``best_err`` diagnostics
  populated, which the Python driver relies on;
* ``RtZone.step_with_solution`` finalizing an externally computed
  post-reaction state identically to ``RtZone.step``;
* ``fsolve_backup_step`` rescuing a zone whose kinetic solve failed and
  refusing to re-solve failures that are not the aqueous kinetic block
  (dimension mismatch) or whose fsolve result does not meet the acceptance
  checks (stalled above the residual bar, non-finite vector; fsolve's
  ``ier`` flag is diagnostic-only and never decides acceptance);
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
    fsolve_backup_step,
)

# One entry per network species, in network order:
# DOC, HCO3-, H+, CO2(aq), CO3--, SOC(s), X-, XDOC.
# The aqueous block solved by the kinetic step is the first five entries;
# the mineral / exchange tail (indices 5-7) is frozen by the fused residual.
C0: NDArray[np.float64] = np.array([1e-3, 1e-3, 1e-7, 1e-4, 1e-10, 1e-2, 1e-3, 1e-3])
DT_DAYS: float = 1.0


def forcing(conc_in: NDArray[np.float64], forc_flux: float = 0.0) -> RtForcing:
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


def forcing_fluxes(
    conc_in: NDArray[np.float64],
    lat_flux: float,
    vert_flux: float,
    lat_flux_ext: float,
    vert_flux_ext: float,
) -> RtForcing:
    """A forcing with explicit internal (`lat_flux` / `vert_flux`) and
    external (`lat_flux_ext` / `vert_flux_ext`) water fluxes, for
    exercising the outgoing-mass bookkeeping formula."""
    hydro_step = HydroStep(
        state=10.0,
        forc_flux=0.0,
        lat_flux=lat_flux,
        vert_flux=vert_flux,
        vap_flux=0.0,
        q_in=0.0,
        lat_flux_ext=lat_flux_ext,
        vert_flux_ext=vert_flux_ext,
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


def fake_optimization_error(best_x: list[float]) -> OptimizationError:
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
        zone.solve_rt_step(c_0, forcing(np.zeros(len(C0))), DT_DAYS)
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
        zone.solve_rt_step(c_tail, forcing(np.zeros(len(C0))), DT_DAYS)
    tail_err = excinfo.value
    assert "NaN" in str(tail_err)
    assert len(tail_err.last_x) == n_aq
    assert np.all(np.isfinite(np.asarray(tail_err.best_x)))


def test_run_zone_steps_failure_payload_is_typed(zone: RtZone) -> None:
    """The failure payload returned by ``run_zone_steps`` preserves the
    typed exception object (not a stringified or re-wrapped one) and
    carries the driver's coupling buffers as they stand at the failure,
    so the Python driver can re-seed its own buffers after a rescue."""
    c_0: NDArray[np.float64] = C0.copy()
    c_0[0] = np.nan
    states: NDArray[np.float64] = c_0[None, :]
    d = forcing(np.zeros(len(C0)))

    steps, failure = run_zone_steps(
        [zone], states, [d], np.zeros((1, 1)), np.zeros((1, 1)), [], DT_DAYS
    )

    assert steps == []
    assert failure is not None
    i, err, conc_in, lat_buf, vert_buf = failure
    assert i == 0
    assert isinstance(err, OptimizationError)
    assert "NaN" in str(err)
    assert np.asarray(conc_in).shape[0] == len(C0)
    # A real step failure reports the coupling buffers at full size
    # (not the empty placeholders of an early validation failure): no
    # zone has been solved yet (start=0), so every entry is zero.
    lat_buf = np.asarray(lat_buf, dtype=np.float64)
    vert_buf = np.asarray(vert_buf, dtype=np.float64)
    assert lat_buf.shape == (1, len(C0))
    assert vert_buf.shape == (1, len(C0))
    assert np.all(lat_buf == 0.0)
    assert np.all(vert_buf == 0.0)


# ===================== step_with_solution ===================== #
def test_step_with_solution_matches_step(zone: RtZone) -> None:
    """Feeding a state produced by ``RtZone.step`` back through
    ``step_with_solution`` reproduces the step's result exactly
    (the post-solve path is shared)."""
    d = forcing(np.zeros(len(C0)))
    reference = zone.step(C0, d, DT_DAYS)

    rescued = zone.step_with_solution(C0, reference.state, d, DT_DAYS)

    np.testing.assert_allclose(rescued.state, reference.state, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(rescued.mineral_rates, reference.mineral_rates)


# ===================== outgoing_mass ===================== #
def test_outgoing_mass_formula(zone: RtZone) -> None:
    """`RtZone.outgoing_mass` implements the Gauss-Seidel coupling
    bookkeeping: the exported moles are the pass-through external flux
    (`q_external * c_in`) plus the state-weighted internal flux
    (`q_internal * state`), with the mineral / exchange entries zeroed
    (they are immobile), split by the external flux fractions of the total
    outgoing water. A zone with no outgoing water at all reports the
    1e-20 sentinel for every species, matching what the driver's coupling
    buffers record for such a zone."""
    n_aq: int = zone.network.num_aqueous_species
    n_spec: int = len(C0)
    c_in: NDArray[np.float64] = np.full(n_spec, 1e-4)

    # 1) Bypass only (no internal water): the full pass-through mass
    #    (q_external = 2 + 1 = 3, i.e. 3e-4 per aqueous species) splits
    #    by the external fractions 2/3 (lateral) and 1/3 (vertical).
    d_bypass = forcing_fluxes(c_in, 0.0, 0.0, 2.0, 1.0)
    lat, vert = zone.outgoing_mass(C0, d_bypass)
    lat = np.asarray(lat)
    vert = np.asarray(vert)
    for sp in range(n_aq):
        assert lat[sp] == pytest.approx(2e-4)
        assert vert[sp] == pytest.approx(1e-4)
    for sp in range(n_aq, n_spec):
        assert lat[sp] == 0.0
        assert vert[sp] == 0.0

    # 2) Mixed internal (0.3 + 0.2 = 0.5) and external (0.9 + 0.4 - 0.5
    #    = 0.8) water: per aqueous species the exported moles are
    #    0.8 * c_in + 0.5 * state, split 0.9/1.3 lateral and 0.4/1.3
    #    vertical of the total outgoing water (1.3).
    q_int: float = 0.5
    q_ext: float = 0.8
    d_mix = forcing_fluxes(c_in, 0.3, 0.2, 0.9, 0.4)
    lat, vert = zone.outgoing_mass(C0, d_mix)
    lat = np.asarray(lat)
    vert = np.asarray(vert)
    total: float = q_int + q_ext
    for sp in range(n_aq):
        tot_out: float = q_ext * c_in[sp] + q_int * C0[sp]
        assert lat[sp] == pytest.approx(tot_out * 0.9 / total)
        assert vert[sp] == pytest.approx(tot_out * 0.4 / total)
    for sp in range(n_aq, n_spec):
        assert lat[sp] == 0.0
        assert vert[sp] == 0.0

    # 3) No outgoing water at all: the 1e-20 sentinel for every species.
    d_dry = forcing_fluxes(c_in, 0.0, 0.0, 0.0, 0.0)
    lat, vert = zone.outgoing_mass(C0, d_dry)
    assert np.all(np.asarray(lat) == 1e-20)
    assert np.all(np.asarray(vert) == 1e-20)


# ===================== fsolve_backup_step ===================== #
def test_fsolve_backup_rescues_converging_zone(zone: RtZone) -> None:
    """With a synthetic failed-solve error whose best iterate is a residual
    root (a stall short of convergence), the fsolve backup re-solves the
    aqueous block, passes its acceptance checks, and the finalized step
    matches the internal solver's to solver tolerance."""
    n_aq: int = zone.network.num_aqueous_species
    d = forcing(np.zeros(len(C0)))

    reference = zone.step(C0, d, DT_DAYS)
    # The kinetic-solve solution stands in for Newton's best iterate: a
    # finite aqueous block at (near) a residual root, which is what the
    # driver hands to fsolve in a rescuable failure.
    sol_aq: NDArray[np.float64] = np.asarray(zone.solve_rt_step(C0, d, DT_DAYS))[:n_aq]
    err = fake_optimization_error(best_x=list(sol_aq))

    rescued = fsolve_backup_step(zone, C0, d, DT_DAYS, err, 0, verbose=False)

    assert rescued is not None
    # fsolve started at a residual root, so it stays there: the finalized
    # step matches the internal solver's bookkeeping to solver tolerance.
    np.testing.assert_allclose(
        np.asarray(rescued.state),
        np.asarray(reference.state),
        rtol=1e-6,
        atol=1e-12,
    )
    # The outgoing mass of the rescued step matches the internal
    # solver's: both final states feed the same bookkeeping formula.
    ref_lat: NDArray[np.float64]
    ref_vert: NDArray[np.float64]
    ref_lat, ref_vert = zone.outgoing_mass(np.asarray(reference.state), d)
    resc_lat: NDArray[np.float64]
    resc_vert: NDArray[np.float64]
    resc_lat, resc_vert = zone.outgoing_mass(np.asarray(rescued.state), d)
    np.testing.assert_allclose(
        resc_lat,
        ref_lat,
        rtol=1e-9,
        atol=0.0,
    )
    np.testing.assert_allclose(
        resc_vert,
        ref_vert,
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
    d = forcing(np.zeros(len(C0)))

    # ``last_x`` the wrong size: not an aqueous kinetic failure.
    wrong_dim_err = fake_optimization_error(best_x=[0.0] * (n_aq + 3))
    assert fsolve_backup_step(zone, C0, d, DT_DAYS, wrong_dim_err, 0, False) is None

    # A state that does not match the network size.
    short_state: NDArray[np.float64] = np.zeros(len(C0) - 1)
    good_err = fake_optimization_error(best_x=list(C0[:n_aq]))
    assert (
        fsolve_backup_step(zone, short_state, d, DT_DAYS, good_err, 0, False)
        is None
    )


def test_fsolve_backup_rejects_failed_fsolve(
    zone: RtZone, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fsolve result that is non-finite, or that stalls above the
    residual bar, is not used as a step result: the backup returns
    ``None`` so the driver can stop the run with the original failure.
    The first stub below is rejected because its residual (1.0) sits
    far above the bar, whatever its ``ier``; the second because its
    vector is non-finite."""
    n_aq: int = zone.network.num_aqueous_species
    d = forcing(np.zeros(len(C0)))
    err = fake_optimization_error(best_x=list(C0[:n_aq]))

    def not_converged(
        _func: object, x0: NDArray[np.float64], **_kwargs: object
    ) -> tuple[NDArray[np.float64], dict[str, object], int, str]:
        return np.full(n_aq, 1.0), {"fvec": np.ones(n_aq)}, 0, "no convergence"

    monkeypatch.setattr(reactive_transport_model, "fsolve", not_converged)
    assert fsolve_backup_step(zone, C0, d, DT_DAYS, err, 0, False) is None

    def nan_solution(
        _func: object, x0: NDArray[np.float64], **_kwargs: object
    ) -> tuple[NDArray[np.float64], dict[str, object], int, str]:
        return np.full(n_aq, np.nan), {"fvec": np.zeros(n_aq)}, 1, "converged"

    monkeypatch.setattr(reactive_transport_model, "fsolve", nan_solution)
    assert fsolve_backup_step(zone, C0, d, DT_DAYS, err, 0, False) is None


def test_fsolve_backup_accepts_under_bar_result_regardless_of_ier(
    zone: RtZone, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An fsolve result that is finite and at or under the residual bar is
    accepted whatever ``ier`` fsolve reported (hybrd flags a stationary
    point "xtol too small" even at or under the bar), mirroring the
    speciation backup's acceptance criterion: the flag is diagnostic-only
    and the residual norm is the arbiter."""
    n_aq: int = zone.network.num_aqueous_species
    d = forcing(np.zeros(len(C0)))
    sol_aq: NDArray[np.float64] = np.asarray(zone.solve_rt_step(C0, d, DT_DAYS))[:n_aq]
    err = fake_optimization_error(best_x=[7.0] * n_aq)

    def stalled_at_bar(
        _func: object, _x0: NDArray[np.float64], **_kwargs: object
    ) -> tuple[NDArray[np.float64], dict[str, object], int, str]:
        return sol_aq.copy(), {"fvec": np.zeros(n_aq)}, 5, "xtol too small"

    monkeypatch.setattr(reactive_transport_model, "fsolve", stalled_at_bar)

    rescued = fsolve_backup_step(zone, C0, d, DT_DAYS, err, 0, False)

    assert rescued is not None
    reference = zone.step(C0, d, DT_DAYS)
    np.testing.assert_allclose(
        np.asarray(rescued.state),
        np.asarray(reference.state),
        rtol=1e-6,
        atol=1e-12,
    )


# ===================== run_zone_steps resume ===================== #
def test_run_zone_steps_resume_matches_full_run(zone: RtZone) -> None:
    """Resuming a two-zone sweep from zone 1 with seeded mass buffers
    reproduces the uninterrupted sweep's result for the remaining zone
    (the Gauss-Seidel coupling is identical)."""
    n_spec: int = len(C0)
    d0 = forcing(np.zeros(n_spec), forc_flux=0.5)
    d1 = forcing(np.zeros(n_spec), forc_flux=0.5)
    states: NDArray[np.float64] = np.stack([C0, C0])
    # Zone 1 receives all of zone 0's outgoing lateral mass.
    lat_mat: NDArray[np.float64] = np.array([[0.0, 0.0], [1.0, 0.0]])
    vert_mat: NDArray[np.float64] = np.zeros((2, 2))

    full_steps, full_failure = run_zone_steps(
        [zone, zone], states, [d0, d1], lat_mat, vert_mat, [], DT_DAYS
    )
    assert full_failure is None
    assert len(full_steps) == 2

    # Zone 1's incoming concentration is the mass the zones solved so far
    # exported (per the coupling matrix) over the water entering zone 1.
    # Recompute it with the same formula the driver uses —
    # `mass_in = lat_mass.T @ lat_mat[i] + vert_mass.T @ vert_mat[i]`, a
    # per-species dot over the *rows* of the mass buffers (the sending
    # zones) — since the `RtStep`s no longer carry incoming concentrations
    # or outgoing masses.
    lat_out0: NDArray[np.float64]
    vert_out0: NDArray[np.float64]
    lat_out0, vert_out0 = zone.outgoing_mass(np.asarray(full_steps[0].state), d0)
    lat_init: NDArray[np.float64] = np.zeros((2, n_spec))
    lat_init[0] = lat_out0
    vert_init: NDArray[np.float64] = np.zeros((2, n_spec))
    vert_init[0] = vert_out0
    denom: float = float(d1.hydro_step.q_in + d1.hydro_step.forc_flux)
    assert denom > 1e-6
    conc_in_expected: NDArray[np.float64] = (
        lat_init.T @ lat_mat[1] + vert_init.T @ vert_mat[1]
    ) / denom
    # Coupling is active: zone 1 sees a nonzero incoming concentration.
    assert np.any(conc_in_expected != 0.0)

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
        np.asarray(resumed[0].state), np.asarray(full_steps[1].state), rtol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(resumed[0].mineral_rates), np.asarray(full_steps[1].mineral_rates)
    )


def test_run_zone_steps_resume_validation(zone: RtZone) -> None:
    """Invalid resume requests fail with informative ``ValueError``s:
    resuming past the last zone, and resuming without the required mass
    buffers."""
    n_spec: int = len(C0)
    d = forcing(np.zeros(n_spec))
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
