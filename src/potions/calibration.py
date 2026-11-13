from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Optional, TYPE_CHECKING
from multiprocessing import Pool, get_context
import numpy as np
from numpy import float64 as f64
from numpy.typing import NDArray
from pandas import Series
from scipy.optimize import differential_evolution, OptimizeResult

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
from .core import SubsurfaceZone, SubsurfaceZoneB
from .line_search import (
    LineSearchConfig,
    LineSearchResult,
    PointKey,
    feasible_t_interval,
    forward_gradient,
    point_key,
    wave_search,
)
from .objective_functions import DEFAULT_RT_OBJECTIVE_SPEC, evaluate_components

if TYPE_CHECKING:
    from .reactive_transport.reaction_network import ReactionNetwork
    from .model import Model


def run_twice(f: Callable[[NDArray], float], x1: NDArray, x2: NDArray) -> tuple[float, float]:
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
        res: list[tuple[float, float]] = pool.starmap(run_twice, args)

    fx_dn: NDArray = np.array([x[0] for x in res])
    fx_up: NDArray = np.array([x[1] for x in res])
    dxs: NDArray = np.array(dxs_list)

    return (fx_up - fx_dn) / (2.0 * dxs)

#: Mineral auxiliary-parameter fields, in the order used by
#: :meth:`~potions.core.MineralParameters.from_array`.
MINERAL_FIELDS: tuple[str, str, str, str] = (
    "sw_threshold",
    "sw_exp",
    "q_10",
    "k_a",
)

#: Default (min, max) bounds for the mineral auxiliary parameters of zones
#: whose reaction rate depends on soil moisture and temperature (surface
#: zones; the river zone reuses these same free bounds).
SURFACE_MINERAL_BOUNDS: dict[str, tuple[float, float]] = {
    "sw_threshold": (0.0, 1.0),
    "sw_exp": (0.5, 2.5),
    "q_10": (1.0, 5.0),
    "k_a": (-15.0, -5.0),
}

#: Degenerate bounds for subsurface zones: the soil-moisture and
#: temperature dependencies are fixed (no dependence), so only ``k_a``
#: varies.
SUBSURFACE_MINERAL_BOUNDS: dict[str, tuple[float, float]] = {
    "sw_threshold": (0.0, 0.0),
    "sw_exp": (0.0, 0.0),
    "q_10": (1.0, 1.0),
    "k_a": (-15.0, -5.0),
}


def mineral_bounds_for_zone(
    model_type: type["Model"],
    zone_name: str,
) -> dict[str, tuple[float, float]]:
    """Return the default mineral auxiliary bounds that apply to ``zone_name``.

    Zones whose class is a subclass of :class:`SubsurfaceZone` or
    :class:`SubsurfaceZoneB` have no soil-moisture or temperature
    dependence and receive the degenerate bounds from
    :data:`SUBSURFACE_MINERAL_BOUNDS`; every other zone class receives the
    free bounds from :data:`SURFACE_MINERAL_BOUNDS`.

    Args:
        model_type: The model class that declares the zone structure.
        zone_name: Name of a *non-river* zone in the model structure
            (the river zone is never classified this way).

    Returns:
        dict[str, tuple[float, float]]: The applicable per-field bounds.

    Raises:
        ValueError: Raised by :meth:`Model.get_zone_type` when the zone
            name is not present in the model structure.
    """
    zone_class: type = model_type.get_zone_type(zone_name)
    if issubclass(zone_class, (SubsurfaceZone, SubsurfaceZoneB)):
        return SUBSURFACE_MINERAL_BOUNDS
    return SURFACE_MINERAL_BOUNDS


def default_rt_parameter_ranges(
    model_type: type["Model"],
    network: ReactionNetwork,
    zone_configs: dict[str, RtZoneConfiguration],
) -> dict[str, tuple[float, float]]:
    """Build an ordered :class:`dict` of default bounds for the
    reactive-transport parameters of ``model_type``.

    For each non-river zone in ``zone_configs`` (in insertion order), the
    bounds are ``max_water`` in ``(100, 1500)`` mm, a fixed
    ``passive_water_storage`` of ``(0, 0)``, and -- when the zone performs
    reactions -- one bound per mineral (counted from
    ``network.num_minerals``) and per auxiliary field in
    :data:`MINERAL_FIELDS`. Subsurface zones (subclasses of
    :class:`SubsurfaceZone` / :class:`SubsurfaceZoneB`) use degenerate
    bounds that fix the soil-moisture and temperature dependencies, so
    only ``k_a`` varies; all other zones use free bounds.

    If ``zone_configs`` contains a "river" entry, its bounds (``bed_depth``
    in ``(0.1, 5)`` mm, a fixed ``passive_water_storage``, and free
    per-mineral bounds when it performs reactions) are appended last,
    matching the parameter order used by :meth:`Model.from_array`.

    Args:
        model_type: The model class that declares the zone structure.
        network: The reaction network (supplies ``num_minerals``).
        zone_configs: One :class:`RtZoneConfiguration` per reactive zone,
            keyed by zone name, with an optional ``"river"`` key.

    Returns:
        dict[str, tuple[float, float]]: Ordered ``(min, max)`` bounds for
        every reactive-transport parameter.
    """
    bounds: dict[str, tuple[float, float]] = {}
    num_minerals: int = network.num_minerals

    for zone_name in (z for z in zone_configs if z != "river"):
        cfg: RtZoneConfiguration = zone_configs[zone_name]
        bounds[f"{zone_name}.max_water"] = (100.0, 1500.0)
        bounds[f"{zone_name}.passive_water_storage"] = (0.0, 0.0)

        if cfg.do_reactions:
            field_bounds: dict[str, tuple[float, float]] = mineral_bounds_for_zone(
                model_type, zone_name
            )
            for mineral_id in range(num_minerals):
                for field_name in MINERAL_FIELDS:
                    bounds[f"{zone_name}.{field_name}[{mineral_id}]"] = field_bounds[field_name]

    # River zone (parameters go at end of array, per `from_array`)
    river_cfg: Optional[RtZoneConfiguration] = zone_configs.get("river")
    if river_cfg is not None:
        bounds["river.bed_depth"] = (0.1, 5.0)
        bounds["river.passive_water_storage"] = (0.0, 0.0)
        if river_cfg.do_reactions:
            for mineral_id in range(num_minerals):
                for field_name in MINERAL_FIELDS:
                    bounds[f"river.{field_name}[{mineral_id}]"] = SURFACE_MINERAL_BOUNDS[
                        field_name
                    ]

    return bounds


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
    then the reactive-transport bounds from
    :func:`default_rt_parameter_ranges` (per-zone dimensions and per-mineral
    reaction parameters for each reactive non-river zone, then river
    parameters if a "river" key is present in ``zone_configs``).
    """
    if hydro_bounds is None:
        hydro_bounds = model_type.default_parameter_ranges()

    bounds: dict[str, tuple[float, float]] = {}
    for name, bnd in hydro_bounds.items():
        bounds[name] = bnd

    for name, bnd in default_rt_parameter_ranges(model_type, network, zone_configs).items():
        bounds[name] = bnd

    return bounds


def build_init_states(data: RtCalibrationData) -> dict[str, ChemicalState]:
    """Build the initial chemical state for every non-river zone."""
    init_states: dict[str, ChemicalState] = {}
    mean_target: float = (
        data.mean_target_conc
        if data.mean_target_conc is not None
        else float(data.conc_df[data.target_species].mean())
    )

    for zone_name, zone_cfg in data.zone_configs.items():
        if zone_name == "river":
            continue
        init_states[zone_name] = ChemicalState(
            mineral=(
                # The RT model requires an initial state for every mineral in
                # any zone that does reactions; give all reactive zones the
                # documented ``data.mineral_conc`` value.
                {sp.name: data.mineral_conc for sp in data.network.mineral}
                if zone_cfg.do_reactions
                else {}
            ),
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
    init_states: dict[str, ChemicalState] = build_init_states(data)

    return model.run_both_models(
        forc=data.forc,
        precip_conc=data.precip_conc,
        init_conc=init_states,
        meas_streamflow=data.q_meas,
        meas_river_conc=data.conc_df,
        # Don't persist individual calibration evaluations.
        save_results=False,
    )


@dataclass(slots=True)
class RtObjective:
    """Picklable scalar objective: the composite fit score at one parameter vector.

    Picklable because a forkserver pool pickles its initargs into the
    worker processes; a closure over local state is unpicklable across
    a spawn-style boundary, so the objective's captured data lives in
    plain fields instead.

    Attributes:
        data: The calibration data object the model runs use.
        spec: The composite objective specification.
    """

    data: RtCalibrationData
    spec: ObjectiveSpec

    def __call__(self, arr: NDArray) -> float:
        """Score one flat parameter array against the calibration data.

        Args:
            arr: The flat parameter array, in :func:`build_rt_bounds` order.

        Returns:
            float: The composite objective (lower is better); ``inf``
            when the model run or scoring fails, so a bad point is
            never better than a good one.
        """
        try:
            res: ModelResults = arr_to_results(arr, self.data)

            hydro_sim: Series = res.hydro.simulation["streamflow_sim"]
            if "meas_streamflow_mmd" in res.hydro.simulation.columns:
                q_meas: Series = res.hydro.simulation["meas_streamflow_mmd"]
            else:
                q_meas = self.data.q_meas

            target_col: str = f"{self.data.target_species}_riv"
            rt_sim: Optional[Series] = None
            if (
                res.reactive_transport is not None
                and target_col in res.reactive_transport.simulation.columns
            ):
                rt_sim = res.reactive_transport.simulation[target_col]

            doc_meas: Optional[Series] = self.data.conc_df[self.data.target_species]
            doc_sim: Optional[Series] = rt_sim

            return float(
                evaluate_components(
                    self.spec,
                    q_meas=q_meas,
                    q_sim=hydro_sim,
                    doc_meas=doc_meas,
                    doc_sim=doc_sim,
                    target_species=self.data.target_species,
                )
            )
        except Exception as e:
            print(f"Calibration run failed (params[:5]={arr[:5]}): {e}")
            return float("inf")

    def __repr__(self) -> str:
        """Return a concise representation naming the target species."""
        return f"RtObjective(target_species={self.data.target_species!r})"


def make_objective(
    data: RtCalibrationData,
    spec: Optional[ObjectiveSpec] = None,
) -> RtObjective:
    """Return a picklable ``arr -> float`` objective suitable for
    :func:`scipy.optimize.differential_evolution` and for
    :func:`map_objectives`' forkserver pool.

    Args:
        data: The calibration data object.
        spec: The objective specification (defaults to the nine-term preset).

    Returns:
        RtObjective: The scalar objective used by the optimizers.
    """
    if spec is None:
        spec = DEFAULT_RT_OBJECTIVE_SPEC
    return RtObjective(data=data, spec=spec)


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


#: The objective one forkserver worker initialized itself with, set by
#: :func:`init_objective` in each worker process.
OBJECTIVE: Optional[RtObjective] = None


def init_objective(objective: RtObjective) -> None:
    """Store ``objective`` in the worker process's module globals.

    Args:
        objective: The objective every batch point is scored against.
    """
    global OBJECTIVE
    OBJECTIVE = objective


def evaluate_objective(arr: NDArray) -> float:
    """Score one parameter array against the worker's initialized objective.

    Args:
        arr: The flat parameter array.

    Returns:
        float: The objective value; NaN when the objective raises, so
        one bad point can never take down the pool.

    Raises:
        RuntimeError: Raised when the worker was never initialized
            with an objective.
    """
    if OBJECTIVE is None:
        raise RuntimeError(
            "evaluate_objective ran in a worker that init_objective "
            "never initialized with an objective."
        )
    try:
        return float(OBJECTIVE(np.asarray(arr, dtype=np.float64)))
    except Exception:
        return float("nan")


def map_objectives(
    objective: RtObjective,
    points: Sequence[NDArray],
    workers: Optional[int] = None,
) -> list[float]:
    """Evaluate a batch of parameter arrays concurrently in a forkserver pool.

    The forkserver context starts workers from a clean server process
    rather than forking a caller that may hold GUI state, and the warm
    workers score the whole batch with the objective installed once as
    the pool's initarg.

    Args:
        objective: The objective the workers score against.
        points: The parameter arrays to score, in result order.
        workers: Worker count; ``None`` uses ``os.cpu_count()``.

    Returns:
        list[float]: One objective value per point, aligned; a value is
        NaN when that point's evaluation raised.
    """
    context = get_context("forkserver")
    with context.Pool(
        processes=workers or os.cpu_count() or 1,
        initializer=init_objective,
        initargs=(objective,),
    ) as pool:
        return pool.map(evaluate_objective, [np.asarray(p, dtype=np.float64) for p in points])


def polish_unchanged(
    start: CalibrationResults,
    x0: NDArray,
    evaluations: int,
    message: str,
) -> CalibrationResults:
    """Return ``start`` unchanged with an :class:`OptimizeResult` carrying ``message``.

    Used when the gradient phase finds no usable descent direction, so
    no line-phase point is run and the start's results are adopted as
    they are.

    Args:
        start: The result being polished.
        x0: The start's parameter vector.
        evaluations: The number of objective evaluations already spent.
        message: Why the polish stopped before searching.

    Returns:
        CalibrationResults: A copy of ``start`` with the explanatory
        ``optimize_result``.
    """
    return CalibrationResults(
        param_names=list(start.param_names),
        best_params=np.array(x0, dtype=f64),
        best_results=start.best_results,
        objective=float(start.objective),
        optimize_result=OptimizeResult(
            x=x0,
            fun=float(start.objective),
            nit=0,
            nfev=evaluations,
            status=0,
            message=message,
        ),
    )


def polish(
    data: RtCalibrationData,
    start: CalibrationResults,
    spec: Optional[ObjectiveSpec] = None,
    method: str = "L-BFGS-B",
    bounds: Optional[list[tuple[float, float]]] = None,
) -> CalibrationResults:
    """Locally polish a differential-evolution result with the parallel
    wave-search line search of :mod:`potions.line_search`.

    The forward-difference gradient of the objective at the start
    point gives the descent direction; the objective is then searched
    along that direction by the wave schedule -- a grid of steps
    across the interval the bounds allow, then recentered waves around
    the best steps found.  Each batch of points runs concurrently in a
    forkserver pool, and the search stops on the engine's improvement,
    bracket-width, and point-budget bars rather than on strict
    tolerances.  When the gradient phase yields no usable direction,
    ``start`` is returned unchanged with the reason in its
    ``optimize_result.message``.

    Args:
        data: The same calibration-data object used for :func:`calibrate`.
        start: The DE result to refine.
        spec: The objective spec.
        method: Kept for call compatibility and ignored: the search is
            always the wave line search.
        bounds: Optional per-parameter ``(lo, hi)`` pairs; when
            ``None``, :func:`build_rt_bounds` values in parameter order.

    Returns:
        CalibrationResults: A new result with the polished parameters,
        the best point's fresh model results, and an
        :class:`OptimizeResult` whose ``message`` names the engine's
        stop reason.

    Raises:
        ValueError:
            Raised when the bounds do not supply exactly one ``(lo,
            hi)`` pair per parameter, or (from the engine) when no
            searched point produced a finite value.
    """
    if spec is None:
        spec = DEFAULT_RT_OBJECTIVE_SPEC
    objective: RtObjective = make_objective(data, spec)
    x0: NDArray = np.array(start.best_params, dtype=f64)
    bound_pairs: list[tuple[float, float]] = (
        list(bounds)
        if bounds is not None
        else list(build_rt_bounds(data.model_type, data.network, data.zone_configs).values())
    )
    if len(bound_pairs) != x0.size:
        raise ValueError(
            f"Expected one (lo, hi) bound per parameter: got {len(bound_pairs)} "
            f"bounds for {x0.size} parameters."
        )

    evaluation_count: int = 0

    def evaluate(points: Sequence[np.ndarray]) -> list[float]:
        nonlocal evaluation_count
        evaluation_count += len(points)
        return map_objectives(objective, points)

    # The start point's finished DE evaluation seeds both phases'
    # memos, so it is never re-run.
    initial_values: dict[PointKey, float] = {point_key(x0): float(start.objective)}

    positions: list[int] = list(range(x0.size))
    offsets: list[np.ndarray] = []
    for position, (lo, hi) in zip(positions, bound_pairs):
        # 1% of the parameter's range, clamped so the perturbed point
        # stays inside the bound (a parameter at its bound gets a zero
        # step, which the gradient phase scores as 0 without a run).
        target: float = min(float(x0[position]) + 0.01 * (hi - lo), float(hi))
        offset: np.ndarray = np.zeros(x0.size, dtype=np.float64)
        offset[position] = target - float(x0[position])
        offsets.append(offset)

    gradient: np.ndarray = forward_gradient(evaluate, x0, offsets, positions, initial_values)
    norm: float = float(np.linalg.norm(gradient))
    if norm == 0.0 or not np.isfinite(norm):
        return polish_unchanged(
            start, x0, evaluation_count, "no descent direction: flat objective"
        )
    direction: np.ndarray = -gradient / norm

    interval: tuple[float, float] | None = feasible_t_interval(
        x0, direction, positions, bound_pairs
    )
    if interval is None:
        return polish_unchanged(start, x0, evaluation_count, "no feasible descent step")
    t_lo, t_hi = interval

    result: LineSearchResult = wave_search(
        evaluate,
        x0,
        direction,
        t_lo,
        t_hi,
        LineSearchConfig(minimize=True),
        initial_values,
    )

    best_arr: NDArray = x0 + result.t_best * direction
    # The engine only reports finite best values; the re-evaluation
    # guards against a future engine that would not.
    objective_value: float = float(result.best_value)
    if not np.isfinite(objective_value):
        objective_value = float(objective(best_arr))
    best_res: ModelResults = arr_to_results(best_arr, data)

    return CalibrationResults(
        param_names=list(start.param_names),
        best_params=np.array(best_arr, dtype=f64),
        best_results=best_res,
        objective=objective_value,
        optimize_result=OptimizeResult(
            x=best_arr,
            fun=objective_value,
            nit=len(result.t_points),
            nfev=result.evaluations,
            status=0,
            message=result.stopped_by,
        ),
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
