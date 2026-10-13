from __future__ import annotations

import os
from typing import Callable, Optional, TYPE_CHECKING
from multiprocessing import Pool
import numpy as np
from numpy import float64 as f64
from numpy.typing import NDArray
from pandas import Series
from scipy.optimize import differential_evolution, minimize, OptimizeResult

from .common_types import (
    CalibrationOptions,
    CalibrationResults,
    EnsembleResults,
    ChemicalState,
    ModelResults,
    ObjectiveSpec,
    RtCalibrationData,
    RtZoneConfiguration,
)
from .objective_functions import DEFAULT_RT_OBJECTIVE_SPEC, evaluate_components
from .utils import setup_logging

if TYPE_CHECKING:
    from .reactive_transport.reaction_network import ReactionNetwork
    from .model import Model

setup_logging(__file__)


def _run_twice(f: Callable[[NDArray], float], x1: NDArray, x2: NDArray) -> tuple[float, float]:
    return (f(x1), f(x2))


def parallel_numerical_gradient(
    f: Callable[[NDArray], float],
    x: NDArray,
    num_threads: Optional[int] = None,
    rel_dx: float = 1e-3,
    dx: float = 1e-8,
) -> NDArray:
    """Calculate the gradient of the function in parallel"""
    dxs_list: list[float] = []
    args: list[tuple[Callable, NDArray, NDArray]] = []

    for i, x_i in enumerate(x):
        dx_i: float
        if abs(x_i) <= 1e-12:
            dx_i = dx
        else:
            dx_i = abs(rel_dx * x_i)

        dxs_list.append(dx_i)
        xs_dn: NDArray = x.copy()
        xs_dn[i] -= dx_i
        xs_up: NDArray = x.copy()
        xs_up[i] += dx_i

        args.append((f, xs_dn, xs_up))

    if num_threads is None:
        num_threads = os.cpu_count()

    with Pool(num_threads) as pool:
        res: list[tuple[float, float]] = pool.starmap(_run_twice, args)

    fx_dn: NDArray = np.array([x[0] for x in res])
    fx_up: NDArray = np.array([x[1] for x in res])
    dxs: NDArray = np.array(dxs_list)

    return (fx_up - fx_dn) / (2.0 * dxs)

_MINERAL_FIELDS: tuple[str, str, str, str] = (
    "sw_threshold",
    "sw_exp",
    "q_10",
    "reactivity",
)

_MINERAL_BOUNDS: dict[str, tuple[float, float]] = {
    "sw_threshold": (0.0, 1.0),
    "sw_exp": (0.5, 2.5),
    "q_10": (1.0, 5.0),
    "reactivity": (-16.0, -6.0),
}


def build_rt_bounds(
    model_type: type["Model"],
    network: ReactionNetwork,
    zone_configs: dict[str, RtZoneConfiguration],
    hydro_bounds: Optional[dict[str, tuple[float, float]]] = None,
) -> dict[str, tuple[float, float]]:
    """Build an ordered :class:`dict` of parameter bounds for a
    differential-evolution calibration over the hydro and reactive-transport
    parameters of ``model_type``.

    The ordering matches :meth:`Model.from_array`: hydro parameters first,
    then per-zone (non-river) dimensions, then per-mineral reaction
    parameters for each reactive non-river zone, finally river parameters
    if a "river" key is present in ``zone_configs``.
    """
    bounds: dict[str, tuple[float, float]] = {}

    if hydro_bounds is None:
        hydro_bounds = model_type.default_parameter_ranges()

    for name, bnd in hydro_bounds.items():
        bounds[name] = bnd

    num_minerals: int = network.num_minerals

    non_river_zones: list[str] = [z for z in zone_configs if z != "river"]
    river_cfg: Optional[RtZoneConfiguration] = zone_configs.get("river")

    # Non-river zone dimensions and mineral params
    for zone_name in non_river_zones:
        cfg: RtZoneConfiguration = zone_configs[zone_name]
        bounds[f"{zone_name}.max_water"] = (100.0, 1500.0)
        bounds[f"{zone_name}.passive_water_storage"] = (0.0, 0.0)

        if cfg.do_reactions:
            for mineral_id in range(num_minerals):
                for field_name in _MINERAL_FIELDS:
                    bounds[f"{zone_name}.{field_name}[{mineral_id}]"] = _MINERAL_BOUNDS[field_name]

    # River zone (parameters go at end of array, per `from_array`)
    if river_cfg is not None:
        bounds["river.bed_depth"] = (0.1, 5.0)
        bounds["river.passive_water_storage"] = (0.0, 0.0)
        if river_cfg.do_reactions:
            for mineral_id in range(num_minerals):
                for field_name in _MINERAL_FIELDS:
                    bounds[f"river.{field_name}[{mineral_id}]"] = _MINERAL_BOUNDS[field_name]

    return bounds


def _build_init_states(data: RtCalibrationData) -> dict[str, ChemicalState]:
    """Build the initial chemical state for every non-river zone."""
    init_states: dict[str, ChemicalState] = {}
    mean_target: float = (
        data.mean_target_conc
        if data.mean_target_conc is not None
        else float(data.conc_df[data.target_species].mean())
    )

    for zone_name in data.zone_configs:
        if zone_name == "river":
            continue
        init_states[zone_name] = ChemicalState(
            primary={
                sp.name: (mean_target if sp.name == data.target_species else data.init_aq_conc)
                for sp in data.network.primary_aqueous
            },
            secondary={sp.name: data.init_aq_conc for sp in data.network.secondary},
        )

    return init_states


def arr_to_results(arr: NDArray, data: RtCalibrationData) -> ModelResults:
    """Run a fresh :class:`~potions.model.Model` built from a single flat
    parameter array against the forcing and concentration data in ``data``,
    returning the combined hydro and reactive-transport results.

    The parameter layout is exactly the one produced by
    :func:`build_rt_bounds`.

    Args:
        arr: The flat parameter array.
        data: The full calibration data object.

    Returns:
        ModelResults: The complete simulation outcome.
    """
    model: Model = data.model_type.from_array(arr, data.network, data.zone_configs)  # type: ignore[arg-type]
    init_states: dict[str, ChemicalState] = _build_init_states(data)

    return model.run_both_models(
        forc=data.forc,
        precip_conc=data.precip_conc,
        init_conc=init_states,
        meas_streamflow=data.q_meas,
        meas_river_conc=data.conc_df,
    )


def make_objective(
    data: RtCalibrationData,
    spec: Optional[ObjectiveSpec] = None,
) -> Callable[[NDArray], float]:
    """Return a callable ``arr -> float`` suitable for
    :func:`scipy.optimize.differential_evolution`.

    Args:
        data: The calibration data object.
        spec: The objective specification (defaults to the nine-term preset).

    Returns:
        Callable[[NDArray], float]: The scalar objective used by the optimizer.
    """
    if spec is None:
        spec = DEFAULT_RT_OBJECTIVE_SPEC

    def _objective(arr: NDArray) -> float:
        try:
            res: ModelResults = arr_to_results(arr, data)

            hydro_sim: Series = res.hydro.simulation["streamflow_sim"]
            if "meas_streamflow_mmd" in res.hydro.simulation.columns:
                q_meas: Series = res.hydro.simulation["meas_streamflow_mmd"]
            else:
                q_meas = data.q_meas

            target_col: str = f"{data.target_species}_riv"
            rt_sim: Optional[Series] = None
            if (
                res.reactive_transport is not None
                and target_col in res.reactive_transport.simulation.columns
            ):
                rt_sim = res.reactive_transport.simulation[target_col]

            doc_meas: Optional[Series] = data.conc_df[data.target_species]
            doc_sim: Optional[Series] = rt_sim

            return float(
                evaluate_components(
                    spec,
                    q_meas=q_meas,
                    q_sim=hydro_sim,
                    doc_meas=doc_meas,
                    doc_sim=doc_sim,
                    target_species=data.target_species,
                )
            )
        except Exception as e:
            print(f"Calibration run failed (params[:5]={arr[:5]}): {e}")
            return float("inf")

    return _objective


def calibrate(
    data: RtCalibrationData,
    spec: Optional[ObjectiveSpec] = None,
    bounds: Optional[dict[str, tuple[float, float]]] = None,
    options: Optional[CalibrationOptions] = None,
) -> CalibrationResults:
    """Calibrate a :class:`~potions.model.Model`'s hydro and RT parameters
    against streamflow and target-solute concentration using differential
    evolution.

    Args:
        data: The full calibration data object.
        spec: The composite objective specification.
        bounds: The parameter bounds; when ``None``, :func:`build_rt_bounds` is used.
        options: Optimizer options; when ``None``, :class:`CalibrationOptions` defaults are used.

    Returns:
        CalibrationResults: The best-fit parameters, results, objective value, and
        raw :class:`scipy.optimize.OptimizeResult`.
    """
    if spec is None:
        spec = DEFAULT_RT_OBJECTIVE_SPEC
    if bounds is None:
        bounds = build_rt_bounds(data.model_type, data.network, data.zone_configs)
    if options is None:
        options = CalibrationOptions()

    obj_fn: Callable[[NDArray], float] = make_objective(data, spec)
    bounds_list: list[tuple[float, float]] = list(bounds.values())

    opt_res: OptimizeResult = differential_evolution(
        func=obj_fn,
        # scipy stubs expect a fixed-width tuple; a homogeneous list of
        # (lo, hi) pairs is accepted at runtime.
        bounds=bounds_list,  # type: ignore[arg-type]
        maxiter=options.maxiter,
        tol=options.tol,
        popsize=options.popsize,
        seed=options.seed,
        polish=options.polish,
        workers=options.workers,
        strategy=options.strategy,
        mutation=options.mutation,
        updating=options.updating,
        atol=options.atol,
    )

    best_arr: NDArray = opt_res.x
    best_res: ModelResults = arr_to_results(best_arr, data)
    objective_value: float = obj_fn(best_arr)

    return CalibrationResults(
        param_names=list(bounds.keys()),
        best_params=np.array(best_arr, dtype=f64),
        best_results=best_res,
        objective=objective_value,
        optimize_result=opt_res,
    )


def polish(
    data: RtCalibrationData,
    start: CalibrationResults,
    spec: Optional[ObjectiveSpec] = None,
    method: str = "L-BFGS-B",
    bounds: Optional[list[tuple[float, float]]] = None,
) -> CalibrationResults:
    """Locally polish a differential-evolution result with
    :func:`scipy.optimize.minimize`.

    Args:
        data: The same calibration-data object used for :func:`calibrate`.
        start: The DE result to refine.
        spec: The objective spec.
        method: One of "L-BFGS-B", "Powell", "Nelder-Mead".
        bounds: Optional per-parameter bounds.

    Returns:
        CalibrationResults: A new result with the polished parameters.
    """
    if spec is None:
        spec = DEFAULT_RT_OBJECTIVE_SPEC

    obj_fn: Callable[[NDArray], float] = make_objective(data, spec)
    x0: NDArray = np.array(start.best_params, dtype=f64)

    kwargs: dict = {"fun": obj_fn, "x0": x0, "method": method}
    if bounds is not None:
        kwargs["bounds"] = bounds
    kwargs["options"] = {"maxiter": 100, "ftol": 1e-8}

    opt_res: OptimizeResult = minimize(**kwargs)

    new_arr: NDArray = opt_res.x
    new_res: ModelResults = arr_to_results(new_arr, data)
    objective_value: float = obj_fn(new_arr)

    return CalibrationResults(
        param_names=list(start.param_names),
        best_params=np.array(new_arr, dtype=f64),
        best_results=new_res,
        objective=objective_value,
        optimize_result=opt_res,
    )


def run_ensemble(
    data: RtCalibrationData,
    base: CalibrationResults,
    top_n: int = 10,
    spec: Optional[ObjectiveSpec] = None,
    seed: int = 0,
) -> EnsembleResults:
    """Produce the ``top_n`` best parameter sets by sampling a local basin
    around the DE best result and ranking by objective value.

    Args:
        data: The same calibration-data object.
        base: The calibration output.
        top_n: Number of best candidates to return (default 10).
        spec: The objective spec; must agree with the spec used for ``base``.
        seed: RNG seed for reproducibility.

    Returns:
        EnsembleResults: One entry per member, sorted by increasing objective value.
    """
    if spec is None:
        spec = DEFAULT_RT_OBJECTIVE_SPEC

    obj_fn: Callable[[NDArray], float] = make_objective(data, spec)
    best_arr: NDArray = np.array(base.best_params, dtype=f64)
    rng = np.random.default_rng(seed=seed)

    pool: list[tuple[NDArray, float]] = [(np.array(best_arr), float(base.objective))]
    n_perturb: int = max(2 * len(best_arr), top_n * 2)

    for _ in range(n_perturb):
        neighbor = best_arr + rng.normal(scale=0.05 * np.abs(best_arr) + 1e-8, size=best_arr.shape)
        pool.append((neighbor, float(obj_fn(neighbor))))

    pool.sort(key=lambda kv: kv[1])
    top = pool[:top_n]

    member_params: list[NDArray] = [kv[0] for kv in top]
    objectives: list[float] = [kv[1] for kv in top]
    member_results: list[ModelResults] = [arr_to_results(p, data) for p in member_params]

    return EnsembleResults(
        member_params=member_params,
        member_results=member_results,
        objectives=objectives,
    )
