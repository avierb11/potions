from __future__ import annotations

import traceback
import warnings
from collections.abc import Callable, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, DatetimeIndex, Series, TimedeltaIndex
from scipy.optimize import fsolve

from potions.objective_functions import DEFAULT_OBJECTIVE_FUNCTIONS
from potions.reactive_transport.kinetic_structures import PARAMETERS_PER_MINERAL
from potions.reactive_transport.parameter_docs import (
    river_zone_rows,
    render_describe,
    rt_zone_rows,
)
from potions.reactive_transport.rt_zone import (
    calculate_moisture_fraction,
    get_hydro_steps,
)
from potions.utils import ModelInputError, RtNumericalError

from .common_types import (
    ChemicalState,
    DerivedSpecies,
    ForcingData,
    HydroModelResults,
    RtModelResults,
    RtZoneConfiguration,
)
from .core import (
    EquilibriumParameters,
    HydrologicZone,
    OptimizationError,
    ReactionNetwork,
    RiverZone,
    RtForcing,
    RtParameters,
    RtStep,
    RtZone,
    run_zone_steps,
)
from .hydro_model import SIM_Q_NAME, HydrologicalModel
from .result_saving import is_save_enabled, save_rt_run

# A successful fsolve backup must drive the step residual at least this far
# down before the backed-up solution is accepted (in absolute terms); a
# backup that stalls above this is rejected and the run fails with the
# original solver error.
FSOLVE_BACKUP_RESIDUAL_MAX: float = 1e-3

# A speciation re-equilibration the internal Newton / LM solve failed to
# converge (that solver's own target bar is 1e-6) is still accepted, and the
# model step proceeds, when the best residual reached is at most this
# (in absolute terms) *and* the state it implies is physically valid; the
# same bar decides an fsolve re-solve when the internal best falls short of
# it. A failure that stalls above this in both attempts, or lands on an
# invalid state, is rejected and the run fails with the original solver
# error. Symmetric with the kinetic backup's
# ``FSOLVE_BACKUP_RESIDUAL_MAX``.
SPECIATION_ACCEPT_RESIDUAL_MAX: float = 1e-3

# Common shape of the two fsolve backup entry points: a failed-zone step
# (zone, start state, forcing, step length, typed solver error, step index,
# verbosity) mapped to the rescued step, or ``None`` when the backup cannot
# converge to an acceptable solution.
FsolveBackup = Callable[
    [
        RtZone,
        NDArray[np.float64],
        RtForcing,
        float,
        OptimizationError,
        int,
        bool,
    ],
    RtStep | None,
]


class SolverBackupWarning(UserWarning):
    """Warning for a solver rescue (fsolve backup) that could not save a step.

    Emitted by the fsolve backup entry points and by
    ``ReactiveTransportModel.step_rt_model`` when a step cannot be accepted
    and the run is about to stop. Each message names the zone, the step
    index, and the specific reason (residual, physical validity, or the
    exception that escaped), so a stopped run is self-describing on screen
    even in non-verbose mode.

    These are real :mod:`warnings`, not plain ``print`` output: batch and
    calibration scripts that run many evaluations can silence them with
    ``warnings.filterwarnings("ignore", category=potions.SolverBackupWarning)``
    (or a blanket ignore) without affecting any other output.

    Attributes:
        None.
    """


def _validate_derived_species(
    derived_species: Sequence[DerivedSpecies],
    network: ReactionNetwork,
) -> dict[str, tuple[str, ...]]:
    """Validate and index the declared derived measurements.

    A derived measurement is a measured quantity that is not a species of
    the reaction network, but is defined as the sum of the river
    concentrations of several network species (for example DIC, the sum of
    the aqueous carbonate species).  Every component must be an aqueous
    (primary or secondary) species of the network, and the derived name
    must not collide with a network species or with another derived
    measurement.

    Args:
        derived_species: The declared derived measurements.
        network: The reaction network the components are drawn from.

    Returns:
        A mapping from each derived name to its component species names.

    Raises:
        ValueError: If a derived name collides with a network species or
            with another derived name, declares no components, or names a
            component that is not an aqueous species of the network.
    """
    species_names: set[str] = set(network.species_names)
    species_types: dict[str, str] = network.species_types
    derived: dict[str, tuple[str, ...]] = {}

    for measurement in derived_species:
        if measurement.name in species_names:
            raise ValueError(
                f"Derived measurement {measurement.name!r} is already a species "
                "of the reaction network; a derived measurement must not share "
                "its name with a network species"
            )
        if measurement.name in derived:
            raise ValueError(
                f"Derived measurement {measurement.name!r} is declared more than once"
            )
        if not measurement.components:
            raise ValueError(
                f"Derived measurement {measurement.name!r} has no component species"
            )
        for component in measurement.components:
            if component not in species_names:
                raise ValueError(
                    f"Derived measurement {measurement.name!r} names component "
                    f"{component!r}, which is not a species of the reaction "
                    f"network. Network species: {network.species_names}"
                )
            if species_types[component] not in ("primary", "secondary"):
                raise ValueError(
                    f"Derived measurement {measurement.name!r} names component "
                    f"{component!r}, which is a {species_types[component]!r} "
                    "species; derived-measurement components must be aqueous "
                    "(primary or secondary) species"
                )
        derived[measurement.name] = tuple(measurement.components)

    return derived


def _fsolve_backup_step(
    zone: RtZone,
    s_i: NDArray[np.float64],
    d_i: RtForcing,
    dt_days: float,
    err: OptimizationError,
    step_index: int,
    verbose: bool,
) -> RtStep | None:
    """Rescue a failed zone step with a ``scipy.optimize.fsolve`` re-solve.

    The internal Rust Newton / LM chain has already failed for this zone.
    This re-solves *only the aqueous block* of the step residual — the same
    system the Rust solver was driving, with the mineral (non-aqueous)
    entries frozen at their initial values — using
    ``scipy.optimize.fsolve``, a damped least-squares solver with a
    different convergence strategy, and finalizes the step through
    ``RtZone.step_with_solution``, the exact same post-solve path as a
    normal step.

    The residual is the first ``n_aq`` entries of
    ``zone.residual_function(c_0, x_full, d_i, dt_days)`` with
    ``x_full = [x_aq, c_0[n_aq:]]``, mirroring the fused residual the Rust
    kinetic solve minimizes.

    Candidates, in order of preference: the iterate with the best residual
    the failed solver reached (``err.best_x``), when finite and dimensioned,
    then the initial aqueous concentrations ``c_0[:n_aq]``.

    Each candidate is evaluated against the backup bar *before* asking
    fsolve for a re-solve: the backup's acceptance criterion is a residual
    norm of at most ``FSOLVE_BACKUP_RESIDUAL_MAX``, so a candidate that
    already satisfies it is accepted directly and fsolve is only invoked
    to *reach* the bar. This matters because the warm-start candidate is
    precisely the failed solver's best iterate — a point the solver itself
    proved finite and close. An fsolve *result* is accepted on the same
    criterion — finite and at or under the bar — without consulting its
    ``ier`` flag: hybrd reports a nonzero ``ier`` ("xtol too small") for
    stationary points that already sit at or under the bar, so the flag is
    diagnostic-only.

    Args:
        zone:
            The ``RtZone`` whose step failed.
        s_i:
            The full state vector the failed solve started from.
        d_i:
            The forcing the zone was about to step with, including the
            computed incoming concentration.
        dt_days:
            Time-step length in days.
        err:
            The typed ``OptimizationError`` raised by the failed solve.
        step_index:
            The model time-step index, for log messages only.
        verbose:
            When true, print each warm-start attempt and the residual it
            achieved.

    Returns:
        The finalized ``RtStep`` when a candidate (pre-accepted or reached
        by fsolve) satisfies a residual norm of at most
        ``FSOLVE_BACKUP_RESIDUAL_MAX``, otherwise ``None``.
    """
    n_aq: int = zone.network.num_aqueous_species
    n_spec: int = zone.network.num_species
    c_0: NDArray[np.float64] = np.asarray(s_i, dtype=np.float64)
    if c_0.shape[0] != n_spec or len(err.last_x) != n_aq:
        # Not the aqueous kinetic block that failed (e.g. a speciation solve,
        # or a state that does not match the network): there is no meaningful
        # fsolve re-solve for it.
        return None

    # The fused residual the Rust Newton / LM solver minimizes: the full
    # step residual with the mineral entries frozen at their initial values,
    # keeping only the aqueous block.
    frozen_tail: NDArray[np.float64] = c_0[n_aq:]

    def _residual(x_aq: NDArray[np.float64]) -> NDArray[np.float64]:
        x_full: NDArray[np.float64] = np.concatenate([x_aq, frozen_tail])
        return zone.residual_function(c_0, x_full, d_i, dt_days)[:n_aq]

    # Candidates, in order of preference. The payload may arrive
    # two-dimensional (a PyO3 vector wrapped as a 1-row array); flatten it.
    candidates: list[NDArray[np.float64]] = []
    best_x: NDArray[np.float64] = np.asarray(err.best_x, dtype=np.float64).reshape(-1)
    if best_x.shape[0] == n_aq and np.all(np.isfinite(best_x)):
        candidates.append(best_x)
    x_init: NDArray[np.float64] = c_0[:n_aq]
    if not any(np.array_equal(x, x_init) for x in candidates):
        candidates.append(x_init)

    x_sol: NDArray[np.float64] | None = None
    for i, x0 in enumerate(candidates):
        # Evaluate the candidate's residual before spending an fsolve on
        # it: if it already sits under the backup bar, it satisfies the
        # backup's acceptance criterion by itself.
        f0: NDArray[np.float64] = _residual(x0)
        f0_max: float = float(np.max(np.abs(f0))) if f0.size else 0.0
        if bool(np.all(np.isfinite(f0))) and f0_max <= FSOLVE_BACKUP_RESIDUAL_MAX:
            x_sol = x0
            if verbose:
                print(
                    f"  fsolve backup (zone '{zone.name}', step {step_index}): "
                    f"candidate {i} pre-accepted with max|f|={f0_max:.3e}"
                )
            break

        x_try, info, ier, _ = fsolve(_residual, x0, full_output=True)
        f_max: float = float(np.max(np.abs(info["fvec"])))
        # The acceptance decision never consults fsolve's convergence flag
        # ``ier`` (hybrd flags a stationary point "xtol too small" even at
        # or under the bar): finiteness and the residual norm are the
        # arbiters, the same criterion as the pre-acceptance above and the
        # one the speciation backup applies.
        if bool(np.all(np.isfinite(x_try))) and f_max <= FSOLVE_BACKUP_RESIDUAL_MAX:
            x_sol = x_try
            if verbose:
                print(
                    f"  fsolve backup (zone '{zone.name}', step {step_index}): "
                    f"accepted with max|f|={f_max:.3e} (ier={ier})"
                )
            break
        if verbose:
            print(
                f"  fsolve backup (zone '{zone.name}', step {step_index}): "
                f"candidate {i} rejected (ier={ier}, max|f|={f_max:.3e})"
            )

    if x_sol is None:
        return None

    x_full_sol: NDArray[np.float64] = np.concatenate([x_sol, frozen_tail])
    return zone.step_with_solution(c_0, x_full_sol, d_i, dt_days, verbose)


def _speciation_conc_valid(
    eq: EquilibriumParameters,
    x_free: NDArray[np.float64],
) -> bool:
    """Check whether a speciation free-variable vector is physically valid.

    A free-variable vector is physically valid when the concentrations it
    implies (``eq.conc_func``, i.e. ``10^(N x_free + b)``) are all finite
    and strictly positive. This is the same physical-validity gate the
    internal Rust speciation solve applies to the vector it accepts, and
    the arbiter the fsolve backup mirrors: a root that satisfies the
    residual bar on a non-physical plateau (an absurd ``x_free`` magnitude
    whose concentrations under/overflow) is rejected.

    Args:
        eq:
            The zone's equilibrium parameters (the concentration function).
        x_free:
            The candidate free-variable vector.

    Returns:
        True when every implied concentration is finite and positive.
    """
    concs: NDArray[np.float64] = eq.conc_func(x_free)
    return bool(np.all(np.isfinite(concs)) and np.all(concs > 0.0))


def _fsolve_backup_speciation(
    zone: RtZone,
    s_i: NDArray[np.float64],
    d_i: RtForcing,
    dt_days: float,
    err: OptimizationError,
    step_index: int,
    verbose: bool,
) -> RtStep | None:
    """Rescue a failed speciation re-equilibration with an ``fsolve`` re-solve.

    The zone's *kinetic* solve succeeded, but its speciation
    re-equilibration (the free-variable solve that restores chemical
    equilibrium after the reaction step) failed: its Newton / LM chain
    stalled above its own 1e-6 target bar, or converged onto a physically
    invalid plateau. A failed solve whose best residual is within
    ``SPECIATION_ACCEPT_RESIDUAL_MAX`` is accepted as is (the model step
    proceeds; see that constant for the rationale), and this re-solve is
    only spent when the internal best falls short of that bar. It
    re-solves the same free-variable system — residual
    ``c_tot - A g(x_free)`` with the analytic Jacobian — using
    ``scipy.optimize.fsolve``, a damped least-squares solver with a
    different convergence strategy, from the failed solver's best iterate,
    and finalizes the step through ``RtZone.step`` with
    ``x_free_override``: the accepted vector is re-gated on the same
    physical-validity check and the zone's warm-start cache is updated,
    exactly as in a normal step.

    The species totals ``c_tot`` the residual depends on are taken from a
    deterministic re-run of the (already succeeded) kinetic solve,
    ``zone.solve_rt_step``.

    Candidates, in order of preference: the iterate with the best residual
    the failed solver reached (``err.best_x``), then the warm start it
    actually used (``err.initial_x``), in each case when finite and
    dimensioned. Each candidate is evaluated against the acceptance bar
    *before* asking fsolve for a re-solve: the acceptance criterion is a
    residual norm of at most ``SPECIATION_ACCEPT_RESIDUAL_MAX`` *and*
    physical validity, so a candidate that already satisfies both is
    accepted directly and fsolve is only invoked to *reach* the bar. The
    same criterion decides an fsolve result: the solver's convergence flag
    ``ier`` is never consulted (hybrd flags a stationary point "xtol too
    small" even at or under the bar). The internal solver's 1e-6 bar
    remains its *target*: a step that meets it never reaches this backup,
    and the 1e-3 acceptance bar only takes over when that target is not
    met.

    Args:
        zone:
            The ``RtZone`` whose speciation re-equilibration failed.
        s_i:
            The full state vector the step started from.
        d_i:
            The forcing the zone stepped with, including the computed
            incoming concentration.
        dt_days:
            Time-step length in days.
        err:
            The typed ``OptimizationError`` raised by the failed speciation
            solve (tagged ``subsystem="speciation"``).
        step_index:
            The model time-step index, for log messages only.
        verbose:
            When true, print each warm-start attempt and the residual it
            achieved.

    Returns:
        The finalized ``RtStep`` when a candidate (pre-accepted or reached
        by fsolve) satisfies the acceptance bar
        (``SPECIATION_ACCEPT_RESIDUAL_MAX``) and is physically valid,
        otherwise ``None``.

    Side effect:
        When no candidate is accepted, a single diagnostic
        ``SolverBackupWarning`` is emitted recording why the final rejection
        happened (convergence status, residual, physical validity), so a run
        stopped by the backup is self-describing even in non-verbose mode.
        It is a real ``warnings`` warning, so batch scripts can filter it
        with ``warnings.filterwarnings``.
    """
    n_free: int = len(err.last_x)
    if zone.eq.get_stoich_null_space().shape[1] != n_free:
        # The error payload does not dimension the speciation system (it is
        # a failure from another subsystem, or a malformed payload): there
        # is no meaningful fsolve re-solve for it.
        return None

    # The post-reaction concentrations the speciation solve was asked to
    # re-equilibrate: a deterministic re-run of the (succeeded) kinetic
    # solve. The residual depends on their conserved totals only.
    c_after_rt: NDArray[np.float64] = zone.solve_rt_step(s_i, d_i, dt_days, verbose)

    def _residual(x_free: NDArray[np.float64]) -> NDArray[np.float64]:
        return zone.eq.speciation_residual(c_after_rt, x_free)

    def _jacobian(x_free: NDArray[np.float64]) -> NDArray[np.float64]:
        return zone.eq.speciation_jacobian(x_free)

    # Candidates, in order of preference. The payload may arrive
    # two-dimensional (a PyO3 vector wrapped as a 1-row array); flatten it.
    candidates: list[NDArray[np.float64]] = []
    for payload in (err.best_x, err.initial_x):
        x: NDArray[np.float64] = np.asarray(payload, dtype=np.float64).reshape(-1)
        if x.shape[0] != n_free or not np.all(np.isfinite(x)):
            continue
        if not any(np.array_equal(x, c) for c in candidates):
            candidates.append(x)
    if not candidates:
        return None

    x_sol: NDArray[np.float64] | None = None
    # Why the last rejected candidate was rejected, reported at the end.
    # ``None`` validity means the physical-validity gate was never the
    # deciding check for that candidate (mirroring the acceptance chain's
    # short-circuit order below).
    last_reject: tuple[int, float, bool | None] | None = None
    for i, x0 in enumerate(candidates):
        # Evaluate the candidate's residual before spending an fsolve on
        # it: if it already sits at or under the acceptance bar and is
        # physically valid, it satisfies the acceptance criterion by
        # itself — no fsolve is spent. This is how a failed solve that
        # only overshot the internal solver's 1e-6 target bar (still
        # within ``SPECIATION_ACCEPT_RESIDUAL_MAX``) is accepted.
        f0: NDArray[np.float64] = _residual(x0)
        f0_max: float = float(np.max(np.abs(f0))) if f0.size else 0.0
        if (
            bool(np.all(np.isfinite(f0)))
            and f0_max <= SPECIATION_ACCEPT_RESIDUAL_MAX
            and _speciation_conc_valid(zone.eq, x0)
        ):
            x_sol = x0
            if verbose:
                print(
                    f"  fsolve backup (zone '{zone.name}', step {step_index}): "
                    f"candidate {i} pre-accepted with max|f|={f0_max:.3e}"
                )
            break

        x_try, info, ier, _ = fsolve(
            _residual, x0, fprime=_jacobian, full_output=True, maxfev=900
        )
        f_max: float = float(np.max(np.abs(info["fvec"])))
        # The acceptance decision never consults fsolve's convergence flag
        # ``ier`` (hybrd flags a stationary point "xtol too small" even at
        # or under the bar): the residual norm and physical validity are
        # the arbiters, the same criterion as the pre-acceptance above.
        x_finite: bool = bool(np.all(np.isfinite(x_try)))
        under_bar: bool = bool(
            np.isfinite(f_max) and f_max <= SPECIATION_ACCEPT_RESIDUAL_MAX
        )
        if x_finite and under_bar and _speciation_conc_valid(zone.eq, x_try):
            x_sol = x_try
            if verbose:
                print(
                    f"  fsolve backup (zone '{zone.name}', step {step_index}): "
                    f"reached the backup bar with max|f|={f_max:.3e} (ier={ier})"
                )
            break
        if not (x_finite and under_bar):
            last_reject = (ier, f_max, None)
        else:
            # fsolve reached a finite point at or under the backup bar:
            # the physical-validity gate is what rejected this candidate.
            last_reject = (ier, f_max, _speciation_conc_valid(zone.eq, x_try))
        if verbose:
            print(
                f"  fsolve backup (zone '{zone.name}', step {step_index}): "
                f"candidate {i} rejected (ier={ier}, max|f|={f_max:.3e})"
            )

    if x_sol is None:
        if last_reject is not None:
            ier_r, f_max_r, valid_r = last_reject
            if valid_r is None:
                reason = (
                    f"no solution at or under the "
                    f"{SPECIATION_ACCEPT_RESIDUAL_MAX:.0e} acceptance bar "
                    f"(max|f|={f_max_r:.3e}, ier={ier_r})"
                )
            else:
                reason = (
                    f"found at max|f|={f_max_r:.3e} but the solution is "
                    f"physically invalid (non-finite or non-positive "
                    f"concentration; ier={ier_r})"
                )
            warnings.warn(
                f"fsolve speciation backup (zone '{zone.name}', "
                f"step {step_index}): no acceptable solution: {reason}",
                SolverBackupWarning,
            )
        return None

    # Finalize through the normal step path with the solved vector supplied:
    # the speciation branch adopts it (re-gating physical validity) instead
    # of re-solving, so the whole step — kinetic solve, bookkeeping,
    # outgoing mass — is identical to a normal step.
    return zone.step(s_i, d_i, dt_days, verbose, x_free_override=x_sol)


class ReactiveTransportModel(HydrologicalModel):

    def __init__(
        self,
        # Hydrology
        zones: dict[str, HydrologicZone] | None = None,
        scales: list[float] | None = None,
        # Reactive transport
        network: ReactionNetwork | None = None,
        rt_zones: dict[str, RtZone] | None = None,
        # River reactions
        river_zone: RiverZone | None = None,
        # Other things
        verbose: bool = False,
        # Derived measurements
        derived_species: Sequence[DerivedSpecies] | None = None,
    ) -> None:
        super().__init__(zones, scales, verbose=verbose)

        # ==== Initialize reactive transport ==== #
        # Make sure that all arguments are present
        if (network is not None) == (rt_zones is not None):
            if verbose:
                print("Using reactive transport model")
        else:
            raise ValueError(
                "When doing reactive transport simulation, you must pass both `network` and `rt_zones` parameters, not just one of them"
            )

        # ==== Derived measurements ==== #
        # A derived measurement is not a species of the network: its
        # simulated river concentration is the sum of the river
        # concentrations of its component species, so it can only be
        # declared on a model that has a reaction network.
        self._derived_species: dict[str, tuple[str, ...]] = {}
        if derived_species:
            if network is None:
                raise ValueError(
                    "`derived_species` requires a reactive transport `network`; "
                    "a hydrology-only model has no species to combine"
                )
            self._derived_species = _validate_derived_species(
                derived_species, network
            )

        if (network is not None) and (rt_zones is not None):
            self._rt_zones: dict[str, RtZone] = rt_zones  # type: ignore
            rt_params: dict[str, RtParameters] = {
                zone_name: zone.parameters for zone_name, zone in rt_zones.items()  # type: ignore
            }

            self._has_rt = True
            self._network: ReactionNetwork = network
            self._rt_params: dict[str, RtParameters] = rt_params
            # Make sure that all of the required zones are in the reactive transport zones
            model_zone_names = set(self.zone_names)
            rt_zone_names = set(rt_params.keys())

            if model_zone_names != rt_zone_names:
                raise ValueError(
                    f"Incorrect structure for `rt_params`. Expected dictionary with keys {
                        list(model_zone_names)
                    }, got keys {list(rt_zone_names)}"
                )

            rt_configuration: dict[str, RtZoneConfiguration] = {
                zone_name: RtZoneConfiguration(
                    do_reactions=zone.do_reactions, do_speciation=zone.do_speciation
                )
                for zone_name, zone in rt_zones.items()  # type: ignore
            }

            # Get the configurations
            cfg: dict[str, RtZoneConfiguration] = {}
            for zone_name in self.zone_names:
                if zone_name in rt_configuration:
                    cfg[zone_name] = rt_configuration[zone_name]
                else:
                    cfg[zone_name] = RtZoneConfiguration(
                        do_reactions=True, do_speciation=True
                    )

            self._zone_configs: dict[str, RtZoneConfiguration] = cfg

        if river_zone is not None:
            self._river_zone: RiverZone = river_zone
        else:
            self._river_zone = None  # type: ignore

    @property
    def reaction_network(self) -> ReactionNetwork | None:
        if self._has_rt:
            return self._network
        else:
            print("This model does not have reactive transport capabilities")
            return None

    @property
    def network(self) -> ReactionNetwork:
        if self._has_rt:
            return self._network
        else:
            raise ValueError("Model does not have reactive-transport capabilities")

    @property
    def derived_species(self) -> dict[str, tuple[str, ...]]:
        """The declared derived measurements, mapping each name to its
        component species.

        The simulated river concentration of each derived measurement is
        the sum of the ``{component}_riv`` columns of its components, and a
        measurement column named after a derived measurement is scored
        against that sum, exactly like a measured network species is scored
        against its own ``{species}_riv`` column.
        """
        return self._derived_species

    @property
    def rt_zones(self) -> dict[str, RtZone]:
        return self._rt_zones

    @property
    def has_river_zone(self) -> bool:
        return self._river_zone is not None

    @property
    def river_zone(self) -> RiverZone:
        if self._river_zone is not None:
            return self._river_zone
        else:
            raise ValueError("No river zone in this model")

    def _rt_parameter_rows(self) -> list[dict]:
        """Build parameter rows for the RT zones and the river zone, if any."""
        rows: list[dict] = []
        if getattr(self, "_has_rt", False):
            for zone_name, zone in self._rt_zones.items():
                rows += rt_zone_rows(zone)
        river = getattr(self, "_river_zone", None)
        if river is not None:
            rows += river_zone_rows(river)
        return rows

    def describe_parameters(self) -> DataFrame:
        """Like the hydro model, but with RT-zone and river-zone rows appended."""
        rows = self._hydro_parameter_rows() + self._rt_parameter_rows()  # type: ignore[attr-defined]
        return DataFrame(rows, columns=["zone", "name", "value", "unit", "description"])

    def describe(self) -> str:
        """Like the hydro model, but also describing RT and river-zone parameters."""
        rows = self._hydro_parameter_rows() + self._rt_parameter_rows()  # type: ignore[attr-defined]
        return render_describe(rows, title="Model parameters")

    @classmethod
    def get_num_rt_parameters(
        cls: type[ReactiveTransportModel], config: dict[str, RtZoneConfiguration]
    ) -> int:
        """
        Get the number of parameters for a zone given the configuration of the reactive transport setup
        """
        num_dim_params: int = 2 * cls.get_num_zones()
        num_min_params: int = 0
        for _zone_name, zone in config.items():
            num_min_params += PARAMETERS_PER_MINERAL * zone.do_reactions

        return num_dim_params + num_min_params

    def rt_to_array(self) -> NDArray:
        """Convert the reactive transport structures into an array"""
        components: list[NDArray] = []

        # The reactive transport zones
        for zone_name in self.zone_names:
            components.append(self.rt_zones[zone_name].to_array())

        # The river zone
        if self.has_river_zone:
            components.append(self.river_zone.to_array())

        return np.concatenate(components)

    @classmethod
    def rt_params_from_array(
        cls: type[ReactiveTransportModel],
        arr: NDArray,
        network: ReactionNetwork,
        configs: dict[str, RtZoneConfiguration],
        verbose: bool = False,
        natural_scales: bool = True,
    ) -> dict[str, RtParameters]:
        # num_zones: int = cls.get_num_zones()
        zone_names: list[str] = cls.get_zone_names()
        # do_reactions: list[bool] = [z.do_reactions for z in configs.values()]
        num_minerals: int = network.num_minerals
        params_per_zone: list[int] = [
            2 + z.do_reactions * num_minerals * PARAMETERS_PER_MINERAL
            for zone_name, z in configs.items()
            if zone_name != "river"
        ]

        if len(arr) != sum(params_per_zone):
            raise ValueError(
                f"Incorrect number of parameters passed to `Model.rt_params_from_array`, expected {sum(params_per_zone)}, received {len(arr)}"
            )

        params_list: list[NDArray] = []
        for np_i in params_per_zone:
            params_list.append(arr[:np_i])
            arr = arr[np_i:]

        zone_params: dict[str, RtParameters] = {}
        for row_i, zone_name_i in zip(params_list, zone_names, strict=True):
            zone_params[zone_name_i] = RtParameters.from_array(row_i)

        return zone_params

    @classmethod
    def rt_zones_from_array(
        cls: type[ReactiveTransportModel],
        arr: NDArray,
        network: ReactionNetwork,
        config: list[RtZoneConfiguration],
    ) -> dict[str, RtZone]:
        # params_per_zone: int = 3 + network.num_mineral_parameters
        params_per_zone: list[int] = [
            3 * z.do_reactions * network.num_minerals * PARAMETERS_PER_MINERAL
            for z in config
        ]

        # params = arr.reshape((num_zones, params_per_zone))
        params_list: list[NDArray] = []
        for np_i in params_per_zone:
            zone_params = arr[:np_i]
            arr = arr[np_i:]
            params_list.append(zone_params)

        rt_zones: dict[str, RtZone] = {
            name_i: RtZone.from_array(
                row,
                network,
                do_reactions=config_i.do_reactions,
                do_speciation=config_i.do_speciation,
                name=name_i,  # type: ignore
            )
            for row, config_i, name_i in zip(
                params_list,
                config,
                cls.get_zone_names(),
                strict=True,  # type: ignore
            )
        }

        return rt_zones

    @property
    def zone_configurations(self) -> dict[str, RtZoneConfiguration]:
        return self._zone_configs

    def construct_rt_forcing_matrix(
        self,
        forc: ForcingData | Iterable[ForcingData],
        hydro_sim_df: DataFrame,
        precip_conc: NDArray,
        model_rt_params: dict[str, RtZone],
    ) -> NDArray:
        """
        Construct the reactive transport forcing data

        precip_conc: The precipitation concentrations at each step. Note that this has shape: (number of surface zones, number of steps)
        """
        sw_vals = calculate_moisture_fraction(model_rt_params, hydro_sim_df)
        hydro_steps = get_hydro_steps(hydro_sim_df)
        hydro_forcing: NDArray = self.construct_hydro_forcing_matrix(forc).T

        num_zones: int = hydro_forcing.shape[0]
        num_steps: int = hydro_forcing.shape[1]

        rt_forcings: NDArray = np.empty((num_zones, num_steps), dtype=object)

        for i in range(num_zones):
            for j in range(num_steps):
                d_ij = RtForcing(
                    conc_in=np.zeros(self.reaction_network.num_species),  # type: ignore
                    hydro_step=hydro_steps[i, j],
                    hydro_forc=hydro_forcing[i, j],
                    s_w=sw_vals[i, j],
                )
                rt_forcings[i, j] = d_ij

        # Now, insert the precipitation chemistry for each zone
        if precip_conc.shape[0] != len(self.surface_zone_ids):  # type: ignore
            raise ModelInputError(
                f"The first dimension of precipitation chemistry must be the same as the the number of surface zones: {
                    self.num_surface_zones
                }, not {precip_conc.shape[0]}"  # type: ignore
            )

        if precip_conc.shape[1] != len(hydro_sim_df):
            raise ModelInputError(
                f"The second dimension of precipitation chemistry must be the same as the the number of steps in the hydro simulation: {
                    len(hydro_sim_df)
                }, not {precip_conc.shape[1]}"
            )

        # Non-finite (NaN/Inf) concentrations would corrupt the reaction
        # solver, so reject them before the simulation starts.
        precip_conc_values = np.asarray(precip_conc, dtype=np.float64)
        if not np.all(np.isfinite(precip_conc_values)):
            bad_positions = [
                tuple(pos) for pos in np.argwhere(~np.isfinite(precip_conc_values))[:5]
            ]
            raise ModelInputError(
                "precip_conc contains non-finite (NaN/Inf) concentration "
                f"values at positions (surface, step, species) "
                f"{bad_positions} (and possibly more). Replace the missing / "
                "corrupted concentration values before running the model."
            )

        for i, zone_data in enumerate(precip_conc):
            for j, species_conc in enumerate(zone_data):
                rt_forcings[i, j].conc_in = species_conc.copy()

        return rt_forcings

    def to_array(self) -> NDArray:
        """Convert this model and turn it into a numpy array"""
        return np.concat([self.hydro_to_array(), self.rt_to_array()])

    def _failed_repro_context(
        self,
        index: int,
        zone: RtZone,
        day_index: int,
        dt_days: float,
        s_i: NDArray,
        d_i: RtForcing,
        err: BaseException,
    ) -> dict:
        """Build a self-describing, fully-reproducible context for a step failure.

        The returned dict carries everything needed to reconstruct the exact
        failing ``zone.step(s_i, d_i, dt_days)`` call in isolation: the zone
        identity, its flags, its own parameter array (``RtParameters``/
        ``RtZone`` are not picklable, so they are stored as their ``to_array()``
        values), the network (which *is* picklable), the full model parameter
        array, the state, the full forcing (``RtForcing`` is picklable), and the
        raw exception (type, string, and traceback).

        Args:
            index: Integer position of the zone in the iteration order.
            zone: The ``RtZone`` that failed to step.
            day_index: Step / time index that was being advanced when the
                failure occurred (so a run-up replay can target it).
            dt_days: Time-step length in days.
            s_i: State (concentrations) of the zone at the start of the step.
            d_i: The full forcing applied to the zone for the step.
            err: The exception that caused the step failure.

        Returns:
            A picklable dictionary describing the failure.
        """
        zone_network = zone.network

        try:
            zone_param_array: list[float] = (
                zone.parameters.to_array().astype(float).tolist()
            )
        except Exception:
            zone_param_array = []

        opt = err if isinstance(err, OptimizationError) else None
        math_info: dict | None = None
        if opt is not None:
            try:
                math_info = {
                    "iterations": opt.iterations,
                    "final_err": opt.final_err,
                    "last_x": opt.last_x,
                    "last_f_x": opt.last_f_x,
                    "jacobian": opt.jacobian,
                    "message": opt.message,
                }
            except Exception:
                math_info = {"message": str(opt)}

        summary = (
            f"reactive-transport step failure: zone '{zone.name}' (index {index}) "
            f"at step {day_index} (dt={dt_days}d) | do_reactions={zone.do_reactions}, "
            f"do_speciation={zone.do_speciation} | "
            f"{type(err).__name__}: {err}"
        )

        return {
            "model_type": type(self).__name__,
            "zone_name": zone.name,
            "zone_index": index,
            "do_reactions": zone.do_reactions,
            "do_speciation": zone.do_speciation,
            "species_names": zone.all_species,
            "step_index": day_index,
            "dt_days": float(dt_days),
            "state_s_i": s_i.astype(float).tolist(),
            "forcing": d_i.copy(),
            "zone_parameter_array": zone_param_array,
            "model_parameter_array": self.to_array().astype(float).tolist(),
            "network": zone_network,
            "exception_type": type(err).__name__,
            "exception": str(err),
            "exception_traceback": traceback.format_exc(),
            "optimization_error": math_info,
            "summary": summary,
        }

    def step_rt_model(
        self,
        model: dict[str, RtZone],
        state: NDArray,
        ds: Iterable[RtForcing],
        dt_days: float,
        verbose: bool = False,
        failed_dir: str | None = None,
        step_index: int = 0,
    ) -> list[RtStep]:
        num_zones: int = len(model)
        zones: list[RtZone] = list(model.values())
        ds_list: list[RtForcing] = list(ds)

        # Run the entire per-zone sweep (Gauss-Seidel coupling of each zone's
        # incoming concentration from the upstream zones already solved this
        # step) in a pure-Rust call, resuming it whenever a zone's Newton /
        # LM solve fails: a failed kinetic solve has its aqueous block
        # re-solved with `scipy.optimize.fsolve` (see
        # `_fsolve_backup_step`), a failed speciation re-equilibration has
        # its free-variable system re-solved the same way (see
        # `_fsolve_backup_speciation`, dispatched on the `subsystem` tag of
        # the typed error), and each rescued step is finalized through the
        # normal post-solve path before the sweep is resumed from the next
        # zone. Each resumed call seeds its
        # mass-coupling buffers with the outgoing mass of the zones solved so
        # far, so every downstream zone sees exactly the incoming
        # concentrations an uninterrupted sweep would have produced. The run
        # stops only if the backup also fails (or the failure is not one of
        # the two backup-able subsystems); a backup that converges but whose
        # finalization fails is wrapped in a typed ``RtNumericalError`` with
        # the repro context attached, so no raw solver exception escapes this
        # method.
        #
        # Each `run_zone_steps` call returns `(steps, failure)` where
        # `failure` is `None` on success or a `(zone_index, exception,
        # conc_in, lat_mass, vert_mass)` 5-tuple for the zone that failed,
        # the last two entries being the driver's coupling buffers as they
        # stand (rows `start` up to the failing zone filled); `steps` holds
        # the successfully solved zones of the current chunk (zone `start`
        # up to but not including the failing one), and `all_steps`
        # accumulates the full zone-ordered result list returned to the
        # caller.
        n_species: int = state.shape[1]
        all_steps: list[RtStep] = []
        lat_mass_solved: NDArray[np.float64] = np.zeros(
            (num_zones, n_species), dtype=np.float64
        )
        vert_mass_solved: NDArray[np.float64] = np.zeros(
            (num_zones, n_species), dtype=np.float64
        )
        start: int = 0

        while start < num_zones:
            steps, failure = run_zone_steps(
                zones,
                np.asarray(state, dtype=np.float64),
                ds_list,
                np.asarray(self.lat_mat, dtype=np.float64),
                np.asarray(self.vert_mat, dtype=np.float64),
                list(self.surface_zone_ids),
                dt_days,
                verbose=verbose,
                start=start,
                lat_mass_init=lat_mass_solved if start > 0 else None,
                vert_mass_init=vert_mass_solved if start > 0 else None,
            )
            all_steps.extend(steps)

            if failure is None:
                break

            i: int = failure[0]
            e: Exception = failure[1]
            # Reconstruct the failing forcing exactly as the Python loop built
            # it: a copy of the zone's forcing with the (computed) incoming
            # concentration that the zone was about to step from.
            d_i: RtForcing = ds_list[i].copy()
            d_i.conc_in = np.asarray(failure[2], dtype=np.float64)  # type: ignore
            s_i: NDArray = np.asarray(state[i], dtype=np.float64).copy()
            zone: RtZone = zones[i]

            # Which fsolve backup applies is decided by the `subsystem` tag
            # on the typed error — the machine-readable discriminator: the
            # dimension of the failed vector is ambiguous, because on
            # networks whose speciation null space carries as many free
            # variables as the network has aqueous species, the two failure
            # types are indistinguishable by dimension alone. A speciation
            # failure re-solves the free-variable system
            # (`_fsolve_backup_speciation`); a kinetic failure re-solves the
            # aqueous block of the step residual (`_fsolve_backup_step`).
            # Failures with no recognized tag propagate as before.
            opt_err: OptimizationError | None = (
                e if isinstance(e, OptimizationError) else None
            )
            is_speciation: bool = (
                opt_err is not None
                and opt_err.subsystem == "speciation"
                and zone.do_speciation
            )
            is_kinetic: bool = (
                opt_err is not None
                and len(opt_err.last_x) == zone.network.num_aqueous_species
            )
            backup_attempted: bool = is_speciation or is_kinetic
            rescued: RtStep | None = None
            if backup_attempted and opt_err is not None:
                backup: FsolveBackup = (
                    _fsolve_backup_speciation
                    if is_speciation
                    else _fsolve_backup_step
                )
                try:
                    rescued = backup(
                        zone, s_i, d_i, dt_days, opt_err, step_index, verbose
                    )
                except Exception as backup_err:
                    # Finalizing the fsolve-rescued step failed: surface it
                    # exactly like a direct failure of this zone, with the
                    # repro context attached, so no raw solver exception
                    # ever escapes `step_rt_model`.
                    backup_context = self._failed_repro_context(
                        i, zone, step_index, dt_days, s_i, d_i, backup_err
                    )
                    backup_err.__rt_repro__ = backup_context  # type: ignore[attr-defined]
                    warnings.warn(
                        f"Failed on zone {i} ('{zone.name}') step {step_index} "
                        f"during fsolve backup finalization: "
                        f"exception: {type(backup_err).__name__}: {backup_err}",
                        SolverBackupWarning,
                    )
                    if isinstance(backup_err, OptimizationError):
                        backup_rt_err: RtNumericalError = RtNumericalError(
                            "other",
                            model_type=type(self),
                            zone=zone,
                            parameters=self.to_array(),
                            state=s_i,
                            rt_forcing=d_i.copy(),
                            math_err=backup_err,
                        )
                        backup_rt_err.__rt_repro__ = (  # type: ignore[attr-defined]
                            backup_context
                        )
                        raise backup_rt_err from backup_err
                    raise

            if rescued is not None:
                # The internal Newton / LM chain stalled but the fsolve backup
                # produced an acceptable step: use it and continue the run. A
                # rescued step is normal operation, not a failure, so it is
                # reported only in verbose runs; the user is notified (with a
                # warning) only when every solver fails and the run stops.
                if verbose:
                    if is_speciation:
                        print(
                            f"Newton/LM speciation solve failed on zone {i} "
                            f"('{zone.name}') step {step_index} ({e}); the fsolve "
                            f"backup solution is used instead and the run continues."
                        )
                    else:
                        print(
                            f"Newton/LM kinetic solve failed on zone {i} "
                            f"('{zone.name}') step {step_index} ({e}); the fsolve "
                            f"backup solution is used instead and the run continues."
                        )
                all_steps.append(rescued)
                # Seed the coupling buffers with the outgoing mass of every
                # zone solved so far so the resumed sweep matches an
                # uninterrupted one: the driver's own buffers (rows
                # `start..i` filled, the rest zero) are copied over
                # wholesale, then row `i` is filled with the rescued zone's
                # outgoing mass, computed from its final state exactly as the
                # driver computes it for a normal step.
                lat_mass_solved[:] = np.asarray(failure[3], dtype=np.float64)
                vert_mass_solved[:] = np.asarray(failure[4], dtype=np.float64)
                resc_lat: NDArray[np.float64]
                resc_vert: NDArray[np.float64]
                resc_lat, resc_vert = zone.outgoing_mass(
                    np.asarray(rescued.state, dtype=np.float64), d_i
                )
                lat_mass_solved[i] = resc_lat
                vert_mass_solved[i] = resc_vert
                start = i + 1
                continue

            if backup_attempted:
                if is_speciation:
                    warnings.warn(
                        f"the fsolve backup also failed to converge for "
                        f"the speciation re-equilibration of zone {i} "
                        f"('{zone.name}') step {step_index}; the run stops.",
                        SolverBackupWarning,
                    )
                else:
                    warnings.warn(
                        f"the fsolve backup also failed to converge for "
                        f"zone {i} ('{zone.name}') step {step_index}; the run stops.",
                        SolverBackupWarning,
                    )

            if isinstance(e, OptimizationError):
                # A solver (Newton / Levenberg-Marquardt) failure: the Rust
                # `OptimizationError` message already carries the iteration,
                # residual, offending vector and Jacobian diagnostics. Wrap and
                # attach the full repro context as well.
                context: dict = self._failed_repro_context(
                    i, zone, step_index, dt_days, s_i, d_i, e
                )
                e.__rt_repro__ = context  # type: ignore[attr-defined]
                warnings.warn(
                    f"Failed on zone {i} ('{zone.name}') step {step_index}: {e}",
                    SolverBackupWarning,
                )
                rt_err: RtNumericalError = RtNumericalError(
                    "other",
                    model_type=type(self),
                    zone=zone,
                    parameters=self.to_array(),
                    state=s_i,
                    rt_forcing=d_i.copy(),
                    math_err=e,
                )
                rt_err.__rt_repro__ = context  # type: ignore[attr-defined]
                raise rt_err from e
            else:
                # Any other (hard) failure: name the zone, the step and the
                # *actual* exception (type + message), and attach the full
                # repro context so the caller can persist it.
                context = self._failed_repro_context(
                    i, zone, step_index, dt_days, s_i, d_i, e
                )
                e.__rt_repro__ = context  # type: ignore[attr-defined]
                warnings.warn(
                    "\n".join(
                        (
                            f"Failed on zone {i} ('{zone.name}') step {step_index}:",
                            f"{s_i=}",
                            f"{d_i=}",
                            f"{dt_days=}",
                            f"exception: {type(e).__name__}: {e}",
                        )
                    ),
                    SolverBackupWarning,
                )
                raise e

        if verbose:
            step: RtStep
            for zone, step in zip(zones, all_steps):
                print(f"Finished zone '{zone.name}', final concentration: {
                        np.array2string(
                            step.state, formatter={"all": lambda x: f"{x:.2e}"}
                        )
                    }")
                print("\n\n")
            print("=" * 25 + "\n")

        return all_steps

    def run_rt_model(
        self,
        # Forcing data
        hydro_res: HydroModelResults,
        precip_conc: NDArray,
        # Model states and parameters
        initial_state: dict[
            str, ChemicalState
        ],  # Initial chemical state for each species
        # Model performance-related things
        meas_river_conc: DataFrame | None = None,
        objective_functions: list[
            tuple[str, Callable[[Series, Series], float]]
        ] = DEFAULT_OBJECTIVE_FUNCTIONS,
        # Optional diagnostic data
        verbose: bool = False,
        return_partial: bool = False,
        failed_dir: str | None = None,
        save_results: bool | None = None,
        results_dir: str | None = None,
    ) -> RtModelResults:
        """Run the reactive transport simulation forwards in time.

        Steps every reactive transport zone forward over the hydrological
        simulation's time steps, coupling the zones through their lateral and
        vertical concentrations, and assembles the per-step states and
        mineral rates into the run's output DataFrame.

        Args:
            hydro_res: The results of the hydrological run whose fluxes and
                storages force the transport simulation.
            precip_conc: The precipitation concentrations, one matrix per
                surface zone, shape `(num_surface_zones, num_steps,
                num_species)`.
            initial_state: The initial chemical state of every zone
                (aqueous, secondary, mineral, and exchange concentrations).
            meas_river_conc: Optional measured river concentrations; when
                provided, the simulated ones are scored against them.
            objective_functions: The metrics to evaluate when measured
                river concentrations are provided.
            verbose: Print progress information as the run proceeds.
            return_partial: On a failed step, return the results computed
                up to that step instead of raising.
            failed_dir: Optional directory for failed-step diagnostic
                files.
            save_results: Whether to save the run's results and parameters
                to disk (see the Saving section below).
            results_dir: Base directory under which the saved run folder is
                written.

        Returns:
            An `RtModelResults` whose `simulation` DataFrame carries, per
            time step, each zone's species concentrations
            (`{species}_{zone}`), the mineral reaction rates of every zone
            in which reactions run (`{mineral}_rate_{zone}`), and the
            simulated river concentrations of every network species and
            declared derived measurement (`{name}_riv`);
            `objective_functions` is a DataFrame of metric values per
            measured species, or None when no measured river concentrations
            were provided.

        Saving:
        Like `run_hydro_model`, the run's results and the model's parameters are
        saved to disk by default. Pass `save_results=False` to disable, or set
        `results_dir` to redirect the output folder. The returned
        `RtModelResults.saved_dir` holds the written folder, or None if saving
        was disabled.
        """
        # ==== Set up the forcing ==== #
        zones_list: list[RtZone] = list(self.rt_zones.values())

        # Measured river-concentration columns must name either a species of
        # the reaction network or a declared derived measurement (for
        # example DIC); reject unknown names before the simulation starts.
        if meas_river_conc is not None:
            scored_names: set[str] = set(self.network.species_names) | set(
                self._derived_species
            )
            unknown_species: list[str] = [
                spec
                for spec in meas_river_conc.columns
                if spec not in scored_names
            ]
            if unknown_species:
                raise ModelInputError(
                    f"meas_river_conc columns {unknown_species} are neither species "
                    f"of the reaction network {self.network.species_names} nor "
                    f"declared derived measurements {list(self._derived_species)}"
                )

        hydro_sim_df: DataFrame = hydro_res.simulation
        forc: ForcingData | list[ForcingData] = hydro_res.forcing

        rt_forcing = self.construct_rt_forcing_matrix(
            hydro_sim_df=hydro_sim_df,
            forc=forc,
            precip_conc=precip_conc,
            model_rt_params=self.rt_zones,
        )

        dates: DatetimeIndex = hydro_sim_df.index  # type: ignore
        num_steps: int = dates.size
        time_deltas: TimedeltaIndex = (
            dates[1:] - dates[:-1]
        )  # Time deltas are calculated in nanoseconds
        time_delta_seconds: NDArray[np.float64] = (
            time_deltas.total_seconds().to_numpy()
        )  # Need the time step in terms of seconds
        time_step_days: list[float] = [1.0] + (time_delta_seconds / 86_400.0).astype(  # type: ignore
            np.float64
        ).tolist()  # Now, need the time delta int terms of days

        # Construct the initial concentrations
        contains_minerals: bool = self.network.num_minerals > 0
        zone_initial_states: list[np.ndarray] = []
        for zone_name in self.zone_names:
            do_reactions: bool = self.rt_zones[zone_name].do_reactions
            do_speciation: bool = self.rt_zones[zone_name].do_speciation
            if zone_name in initial_state and do_reactions and contains_minerals:
                if zone_name not in initial_state and contains_minerals:
                    raise ValueError(
                        f"You must pass mineral concentrations to your simulation for zone {zone_name}"
                    )

            zone_conc: ChemicalState
            if zone_name in initial_state:
                zone_conc = initial_state[zone_name]
            else:
                zone_conc = ChemicalState()

            zone_state: dict[str, float] = dict()

            # Primary aqueous species
            for prim_aq in self.network.primary_aqueous:
                if prim_aq.name in zone_conc.primary:
                    zone_state[prim_aq.name] = zone_conc.primary[prim_aq.name]
                else:
                    zone_state[prim_aq.name] = 1e-3  # default value

            # Secondary species
            for sec in self.network.secondary:
                if sec.name in zone_conc.secondary:
                    zone_state[sec.name] = zone_conc.secondary[sec.name]
                else:
                    zone_state[sec.name] = 1e-3  # default value

            # Mineral species
            for min_val in self.network.mineral:
                if min_val.name in zone_conc.mineral:
                    zone_state[min_val.name] = zone_conc.mineral[min_val.name]
                else:
                    if do_reactions and contains_minerals:
                        raise ValueError(
                            f"Missing initial concentration of mineral {min_val.name} for reactive transport simulation"
                        )
                    else:
                        # Zone is not doing reactions, so it does not need minerals
                        zone_state[min_val.name] = 1e-20

            # Equilibrium adsorption species
            if self.network.has_exchange:
                # Make sure that the exchange site concentration is set
                if "X-" not in zone_conc.exchange and do_speciation:
                    raise ValueError(
                        f"You must specify the initial exchange site concentraitons ('X-') for zone {zone_name} or turn speciation off"
                    )

                if do_speciation:
                    zone_state["X-"] = zone_conc.exchange["X-"]
                else:
                    zone_state["X-"] = 1e-20

                for exch in self.network.exchange_species:
                    if exch.name in zone_conc.exchange:
                        zone_state[exch.name] = zone_conc.exchange[exch.name]
                    else:
                        zone_state[exch.name] = 1e-6  # default value

            zone_init_conc_list: list[float] = []
            for spec_name in self.network.species_order:
                zone_init_conc_list.append(zone_state[spec_name])

            zone_init_state = np.array(zone_init_conc_list, dtype=np.float64)
            if not np.all(np.isfinite(zone_init_state)):
                bad_species = [
                    spec
                    for spec, val in zip(self.network.species_order, zone_init_state)
                    if not np.isfinite(val)
                ][:5]
                raise ModelInputError(
                    f"Initial chemical state for zone '{zone_name}' contains "
                    f"non-finite (NaN/Inf) concentrations for species {
                        bad_species
                    } (and possibly more). Provide finite initial values for "
                    "every species."
                )
            zone_initial_states.append(zone_init_state)

        state: np.ndarray = np.array(zone_initial_states)
        if verbose:
            print("Initial state: ")
            for zone_name, zone_conc in zip(self.zone_names, state):
                print(f"{zone_name}: {zone_conc}")
        # ============================ #

        # ==== Run the model forwards ==== #
        # `steps` accumulates every zone's per-step state and mineral rates;
        # it is the sole source for the output DataFrame and is not kept
        # beyond the return value of this method.
        steps: list[list[RtStep]] = []

        final_completed: int = len(dates)
        try:
            for i in range(len(hydro_sim_df)):
                # print(f"Old state: {state[:, 0]}")
                # ds_for_step: list[RtForcing] = [x.copy() for x in rt_forcing[:, i]]
                ds_for_step: list[RtForcing] = rt_forcing[:, i].tolist()

                dt_days: float = time_step_days[i]
                if dt_days < 1e-6:
                    raise ValueError(f"Time step on step {i} is too small: {dt_days}")

                # print(f"Surface state before: {state[1]}")

                if verbose:
                    print(f"Starting step {i}")
                step: list[RtStep] = self.step_rt_model(
                    self.rt_zones,
                    state,
                    ds_for_step,
                    time_step_days[i],
                    verbose=verbose,
                    failed_dir=failed_dir,
                    step_index=i,
                )
                steps.append(step)
                new_state = np.array([x.state.copy() for x in step]).copy()
                # print(f"Surface state after: {new_state[1]}")

                state[:] = new_state[:]
                # print(f"New state: {state[:, 0]}")

                if verbose:
                    print(f"Finished step {i}")

        except Exception as e:
            if return_partial:
                final_completed = i  # type: ignore
            else:
                # print(f"Failed on step {i}")  # type: ignore
                raise e

        # ================================ #

        # ==== Construct outputs ==== #
        species_names: list[str] = zones_list[0].all_species
        mineral_names: list[str] = zones_list[0].mineral_species
        zone_names: list[str] = self.zone_names
        data_cols: dict[str, NDArray] = {}

        # Get the concentrations of each species
        for zone_id, zone_name in enumerate(zone_names):
            for species_id, species_name in enumerate(species_names):
                conc: NDArray = np.empty(num_steps, dtype=np.float64)
                for i in range(final_completed):
                    single_step: RtStep = steps[i][zone_id]
                    conc[i] = single_step.state[species_id]

                col_name: str = f"{species_name}_{zone_name}"
                data_cols[col_name] = conc

            # Mineral reaction rates only exist for zones in which the
            # mineral reactions run (the zone's own `do_reactions` flag);
            # for the other zones the solver returns zero rates, so no
            # `{mineral}_rate_{zone}` columns are emitted for them.
            if self.rt_zones[zone_name].do_reactions:
                for species_id, species_name in enumerate(mineral_names):
                    conc = np.empty(num_steps, dtype=np.float64)
                    for i in range(final_completed):
                        single_step = steps[i][zone_id]
                        conc[i] = single_step.mineral_rates[species_id]

                    col_name = f"{species_name}_rate_{zone_name}"
                    data_cols[col_name] = conc

        res_df: DataFrame = DataFrame(data=data_cols, index=dates)

        # Quantify the river concentrations
        river_zone_ids = self.get_river_zone_ids()
        river_zone_names: list[str] = [self.zone_names[i] for i in river_zone_ids]
        # Weight each river zone's lateral outflow by its area fraction so the
        # mixing fractions are consistent with the area-weighted SIM_Q_NAME.
        q_components: dict[str, Series] = {
            n: hydro_sim_df[f"q_lat_{n}"] * float(self.zone_scales[river_zone_ids[k]])
            for k, n in enumerate(river_zone_names)
        }
        q_sim: Series = hydro_sim_df[SIM_Q_NAME]
        flow_fractions: dict[str, Series] = {
            name: val / q_sim for name, val in q_components.items()
        }

        for spec_name in self.network.species_names:
            comp_concs: dict[str, Series] = {}
            for n in river_zone_names:
                comp_concs[n] = res_df[f"{spec_name}_{n}"]

            conc_comps: list[Series] = [
                flow_fractions[n] * comp_concs[n] for n in river_zone_names
            ]

            weighted_comps: Series = 0.0  # type: ignore
            for cc in conc_comps:
                weighted_comps += cc

            res_df[f"{spec_name}_riv"] = weighted_comps

        # Add the river concentration of every declared derived measurement
        # (for example DIC, the sum of the aqueous carbonate species), so a
        # measurement column named after a derived measurement can be scored
        # against it just like a measured network species.
        for derived_name, components in self._derived_species.items():
            derived_conc: Series = res_df[f"{components[0]}_riv"]
            for component in components[1:]:
                derived_conc = derived_conc + res_df[f"{component}_riv"]
            res_df[f"{derived_name}_riv"] = derived_conc

        # Calculate the objective functions. The measured columns were
        # validated against the network (and declared derived measurements)
        # before the simulation started, so every column has a `{spec}_riv`
        # series to score against.
        if meas_river_conc is not None:
            test_species: list[str] = list(meas_river_conc.columns)  # type: ignore
            obj_df_cols: list[str] = [x[0] for x in objective_functions]
            obj_df_index: list[str] = test_species
            data = np.full((len(obj_df_index), len(obj_df_cols)), fill_value=np.nan)

            for j, (_obj_name, obj_func) in enumerate(objective_functions):
                for i, spec in enumerate(test_species):
                    sim_col_name: str = f"{spec}_riv"
                    c_meas: Series = meas_river_conc[spec]
                    c_sim: Series = res_df[sim_col_name]
                    data[i, j] = obj_func(c_meas, c_sim)

            obj_df = DataFrame(columns=obj_df_cols, index=obj_df_index, data=data)
        else:
            obj_df = None

        rt_sim_res: RtModelResults = RtModelResults(
            simulation=res_df,
            objective_functions=obj_df,
            zone_names=list(zone_names),
        )

        # Save the run's results and parameters (on by default; see result_saving).
        if is_save_enabled(save_results):
            saved_to = save_rt_run(self, rt_sim_res, results_dir)
            rt_sim_res.saved_dir = str(saved_to)
            print(f"Saving run results to {saved_to}")

        # =========================== #

        return rt_sim_res  # type: ignore
