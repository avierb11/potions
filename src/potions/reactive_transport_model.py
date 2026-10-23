from __future__ import annotations

import traceback
from collections.abc import Callable, Iterable

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
from potions.utils import DO_LOGGING, ModelInputError, RtNumericalError

from .common_types import (
    ChemicalState,
    ForcingData,
    HydroModelResults,
    RtModelResults,
    RtZoneConfiguration,
)
from .core import (
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

    Warm-start candidates, in order of preference: the iterate with the best
    residual the failed solver reached (``err.best_x``), when finite and
    dimensioned, then the initial aqueous concentrations ``c_0[:n_aq]``.

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
        The finalized ``RtStep`` when fsolve converged to a residual norm of
        at most ``FSOLVE_BACKUP_RESIDUAL_MAX``, otherwise ``None``.
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

    # Warm-start candidates, in order of preference.
    candidates: list[NDArray[np.float64]] = []
    best_x: NDArray[np.float64] = np.asarray(err.best_x, dtype=np.float64)
    if best_x.shape[0] == n_aq and np.all(np.isfinite(best_x)):
        candidates.append(best_x)
    x_init: NDArray[np.float64] = c_0[:n_aq]
    if not any(np.array_equal(x, x_init) for x in candidates):
        candidates.append(x_init)

    x_sol: NDArray[np.float64] | None = None
    for x0 in candidates:
        x_try, info, ier, _ = fsolve(_residual, x0, full_output=True)
        f_max: float = float(np.max(np.abs(info["fvec"])))
        if ier == 1 and bool(np.all(np.isfinite(x_try))) and f_max <= FSOLVE_BACKUP_RESIDUAL_MAX:
            x_sol = x_try
            if verbose:
                print(
                    f"  fsolve backup (zone '{zone.name}', step {step_index}): "
                    f"converged with max|f|={f_max:.3e}"
                )
            break
        if verbose:
            print(
                f"  fsolve backup (zone '{zone.name}', step {step_index}): "
                f"warm start {x0} did not converge (ier={ier}, max|f|={f_max:.3e})"
            )

    if x_sol is None:
        return None

    x_full_sol: NDArray[np.float64] = np.concatenate([x_sol, frozen_tail])
    return zone.step_with_solution(c_0, x_full_sol, d_i, dt_days, verbose)


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
        # LM kinetic solve fails: the failed zone's aqueous block is re-solved
        # with `scipy.optimize.fsolve` (see `_fsolve_backup_step`), the
        # rescued step is finalized through the normal post-solve path, and
        # the sweep is resumed from the next zone. Each resumed call seeds its
        # mass-coupling buffers with the outgoing mass of the zones solved so
        # far, so every downstream zone sees exactly the incoming
        # concentrations an uninterrupted sweep would have produced. The run
        # stops only if the backup also fails (or the failure is not a
        # kinetic one).
        #
        # Each `run_zone_steps` call returns `(steps, failure)` where
        # `failure` is `None` on success or a `(zone_index, exception,
        # conc_in)` triple for the zone that failed; `steps` holds the
        # successfully solved zones of the current chunk (zone `start` up to
        # but not including the failing one), and `all_steps` accumulates the
        # full zone-ordered result list returned to the caller.
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

            # Only a *kinetic* solve failure (the failed vector is the
            # aqueous block) can be re-solved with fsolve; speciation and
            # other failures propagate as before.
            backup_attempted: bool = False
            rescued: RtStep | None = None
            if isinstance(e, OptimizationError):
                backup_attempted = (
                    len(e.last_x) == zone.network.num_aqueous_species
                )
                if backup_attempted:
                    rescued = _fsolve_backup_step(
                        zone, s_i, d_i, dt_days, e, step_index, verbose
                    )

            if rescued is not None:
                # The internal Newton / LM chain stalled but the fsolve backup
                # produced an acceptable step: use it and continue the run. A
                # rescued step is normal operation, not a failure, so it is
                # reported only in verbose runs; the user is notified (with a
                # warning) only when every solver fails and the run stops.
                if verbose:
                    print(
                        f"Newton/LM kinetic solve failed on zone {i} "
                        f"('{zone.name}') step {step_index} ({e}); the fsolve "
                        f"backup solution is used instead and the run continues."
                    )
                all_steps.append(rescued)
                # Seed the coupling buffers with the outgoing mass of every
                # zone solved so far (the steps returned by the failed call
                # plus the rescued zone) so the resumed sweep matches an
                # uninterrupted one.
                for j, st in enumerate(steps):
                    lat_mass_solved[start + j] = np.asarray(
                        st.lat_mass, dtype=np.float64
                    )
                    vert_mass_solved[start + j] = np.asarray(
                        st.vert_mass, dtype=np.float64
                    )
                lat_mass_solved[i] = np.asarray(rescued.lat_mass, dtype=np.float64)
                vert_mass_solved[i] = np.asarray(rescued.vert_mass, dtype=np.float64)
                start = i + 1
                continue

            if backup_attempted:
                print(
                    f"WARNING: the fsolve backup also failed to converge for "
                    f"zone {i} ('{zone.name}') step {step_index}; the run stops."
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
                print(f"Failed on zone {i} ('{zone.name}') step {step_index}: {e}")
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
                print(f"Failed on zone {i} ('{zone.name}') step {step_index}:")
                print(f"{s_i=}")
                print(f"{d_i=}")
                print(f"{dt_days=}")
                print(f"exception: {type(e).__name__}: {e}")
                raise e

        if verbose:
            step: RtStep
            for i, (zone, step) in enumerate(zip(zones, all_steps)):
                print(f"Incoming concentration to zone {i}: {step.conc_in}")
                print(f"Lateral mass transfer outwards: {step.lat_mass}")
                print(f"Vertical mass transfer outwards: {step.vert_mass}")
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
        """Run the reactive transport simulation forwards
        `mineral_conc` is a dictionary containing the mineral volume fractions (0,porosity) of the minerals
        of each of the zones

        Saving:
        Like `run_hydro_model`, the run's results and the model's parameters are
        saved to disk by default. Pass `save_results=False` to disable, or set
        `results_dir` (or the `POTIONS_SAVE_RESULTS` / `POTIONS_RESULTS_DIR`
        environment variables) to control the behavior. The returned
        `RtModelResults.saved_dir` holds the written folder, or None if saving
        was disabled.
        """
        # ==== Set up the forcing ==== #
        zones_list: list[RtZone] = list(self.rt_zones.values())

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
        steps: list[list[RtStep]] = []
        old_states: list[NDArray] = []
        new_states: list[NDArray] = []

        final_completed: int = len(dates)
        try:
            for i in range(len(hydro_sim_df)):
                # print(f"Old state: {state[:, 0]}")
                # ds_for_step: list[RtForcing] = [x.copy() for x in rt_forcing[:, i]]
                ds_for_step: list[RtForcing] = rt_forcing[:, i].tolist()

                dt_days: float = time_step_days[i]
                if dt_days < 1e-6:
                    raise ValueError(f"Time step on step {i} is too small: {dt_days}")

                old_states.append(state)

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

                new_states.append(new_state[:])
                state[:] = new_state[:]
                # print(f"New state: {state[:, 0]}")

                if verbose:
                    print(f"Finished step {i}")

        except Exception as e:
            if DO_LOGGING:
                print(f"Failed on step {i} with error: {e}")  # type: ignore
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
                mass_in: NDArray = np.empty(num_steps, dtype=np.float64)
                lat_mass: NDArray = np.empty(num_steps, dtype=np.float64)
                lat_conc: NDArray = np.empty(num_steps, dtype=np.float64)
                vert_mass: NDArray = np.empty(num_steps, dtype=np.float64)
                vert_conc: NDArray = np.empty(num_steps, dtype=np.float64)
                tot_moles: NDArray = np.empty(num_steps, dtype=np.float64)
                for i in range(final_completed):
                    single_step: RtStep = steps[i][zone_id]
                    conc[i] = single_step.state[species_id]
                    mass_in[i] = single_step.mass_in[species_id]
                    lat_mass[i] = single_step.lat_mass[species_id]
                    lat_conc[i] = single_step.lat_conc[species_id]
                    vert_mass[i] = single_step.vert_mass[species_id]
                    vert_conc[i] = single_step.vert_conc[species_id]
                    tot_moles[i] = single_step.total_moles[species_id]

                col_name: str = f"{species_name}_{zone_name}"
                data_cols[col_name] = conc
                data_cols[f"{species_name}_{zone_name}_mass_in"] = mass_in
                data_cols[f"{species_name}_{zone_name}_lat_mass"] = lat_mass
                data_cols[f"{species_name}_{zone_name}_lat_conc"] = lat_conc
                data_cols[f"{species_name}_{zone_name}_vert_mass"] = vert_mass
                data_cols[f"{species_name}_{zone_name}_vert_conc"] = vert_conc
                data_cols[f"{species_name}_{zone_name}_total_moles"] = tot_moles

            # Get the mineral species
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

        # Calculate the objective functions
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
            rt_forcings=rt_forcing,
            steps=steps,
            other={"old_states": old_states, "new_states": new_states},
        )

        # Save the run's results and parameters (on by default; see result_saving).
        if is_save_enabled(save_results):
            saved_to = save_rt_run(self, rt_sim_res, results_dir)
            rt_sim_res.saved_dir = str(saved_to)
            print(f"Saving run results to {saved_to}")

        # =========================== #

        return rt_sim_res  # type: ignore
