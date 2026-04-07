from __future__ import annotations
import datetime
import itertools
import os
import random
import warnings
from copy import deepcopy
from dataclasses import dataclass
from multiprocessing import Pool
from typing import (
    Callable,
    Final,
    Iterable,
    Iterator,
    Literal,
    Optional,
    TypedDict,
    TypeVar,
)

import emcee  # type: ignore
import numpy as np
import scipy.optimize as opt
from numpy import float64 as f64
from numpy.typing import NDArray
from pandas import DataFrame, DatetimeIndex, Series, TimedeltaIndex

from potions.hydro_model import HydrologicalModel
from potions.reactive_transport.kinetic_structures import PARAMETERS_PER_MINERAL

# from potions.reactive_transport.rt_zone import RtStep

from .common_types import (
    ForcingData,
    HydroModelResults,
    RtModelResults,
    ModelResults,
)

from potions.core import (  # type: ignore
    # Hydrology
    LapseRateParameters,
    HydrologicZone,
    OptimizationError,
    RiverZone,
    # Reactive Transport
    RtForcing,
    RtStep,
    RtZone,
    ReactionNetwork,
    RtParameters,
)

from .interfaces import Zone
from .objective_functions import DEFAULT_OBJECTIVE_FUNCTIONS
from .reactive_transport import (
    calculate_moisture_fraction,
    calculate_water_table_depth,
    get_hydro_steps,
)
from .utils import (
    DO_LOGGING,
    PotionsError,
    RtNumericalError,
    log_probability,
    objective_function,
    rt_minerals_to_array,
)
from .utils import setup_logging
from .model_components import Layer

setup_logging(__file__)

# Define a TypeVar for Zones to make Layer, Hillslope, and Model generic
ZoneType = TypeVar("ZoneType", bound=Zone)

# ==== Constants ==== #
OUTPUT_COLUMNS_PER_ZONE: Final[int] = (
    # Number of columns for each zone in the output. Includes 1 state + 6 fluxes (4 normal + 2 external)
    8
)
# =================== #


@dataclass(frozen=True)
class RtZoneConfiguration:
    do_reactions: bool
    do_speciation: bool


class BatchResults(TypedDict):
    simulations: dict[int, DataFrame]
    objective_functions: DataFrame


class BatchParams(TypedDict):
    """A dictionary specifying parameters for a batch model run.

    This is used internally by `Model.run_batch` to pass configuration
    to the worker processes.

    Attributes:
        output_dir: The directory path to save simulation results.
        threshold_function: A callable that takes the results dictionary of a
            single run and returns True if the results should be saved.
        return_results: If True, the full simulation results are returned
            from the worker process.
        save_results: If True, worker processes will save results to disk.
    """

    output_dir: Optional[str]
    threshold_function: Optional[Callable[[HydroModelResults], bool]]
    save_results: bool


def _run_model(
    cls: type[Model],
    i: int,
    params: Series,
    forc: ForcingData | list[ForcingData],
    init_state: Optional[NDArray[f64]],
    meas_streamflow: Optional[Series],
    batch_params: BatchParams,
) -> tuple[int, HydroModelResults]:
    """Worker function to run a single model instance in a parallel batch.

    This function is designed to be called by `multiprocessing.Pool` in the
    `Model.run_batch` method. It instantiates a model from a parameter series,
    runs it, and optionally saves the results.

    Args:
        cls (type[Model]): The model class to instantiate (e.g., `HbvModel`).
        i (int): The index of the run, used for tracking and saving files.
        params (Series): A pandas Series of parameters for this model run.
        forc (ForcingData | list[ForcingData]): The forcing data for the simulation.
        init_state (Optional[NDArray[f64]]): The initial state of the model.
        meas_streamflow (Optional[Series]): Observed streamflow for metric calculation.
        batch_params (BatchParams): A dictionary of parameters controlling the batch execution.

    Returns:
        tuple[int, Optional[HydroModelResults], dict]: A tuple containing:
            - The run index `i`.
            - The full results dictionary (if `return_results` is True), else None.
            - A dictionary of scalar metrics from the run.
    """
    try:
        model = cls.hydro_from_series(params)
        run_res: HydroModelResults = model.run_hydro_model(
            forc=forc,
            init_state=init_state,
            meas_streamflow=meas_streamflow,
            verbose=False,
        )

        if batch_params["save_results"]:
            save_model = True
            if batch_params["threshold_function"] is not None:
                save_model = batch_params["threshold_function"](run_res)

            if batch_params["output_dir"] is None:
                raise ValueError("Output directory must be specified")

            if save_model:
                run_res["simulation"].to_csv(  # type: ignore
                    os.path.join(batch_params["output_dir"], f"{i}.csv")
                )

        return i, run_res
    except Exception as e:
        print(f"Failed on run {i} with error: {e}")
        return i, None  # type: ignore


class Model(HydrologicalModel):
    """An abstract base class for defining and running hydrologic models.

    The `Model` class serves as the engine for simulations. It takes a defined
    `structure` (a list of lists of `HydrologicZone` objects), builds a
    connectivity graph, and provides methods to run simulations, calibrate
    parameters, and manage model state.

    To create a new model, subclass `Model` and define the `structure` class
    attribute.

    Example:
        >>> class MySimpleModel(Model):
        ...     structure = [[SnowZone()], [SurfaceZone()]]
        >>> model = MySimpleModel()
    """

    structure: list[list[HydrologicZone]] = []

    def __init__(
        self,
        # Hydrology arguments
        zones: Optional[dict[str, HydrologicZone]] = None,
        scales: Optional[list[float]] = None,
        lapse_rates: Optional[list[LapseRateParameters]] = None,
        # Reactive transport arguments
        network: Optional[ReactionNetwork] = None,
        rt_params: Optional[dict[str, RtParameters]] = None,
        rt_configuration: Optional[dict[str, RtZoneConfiguration]] = None,
        river_zone: Optional[RiverZone] = None,
        verbose: bool = False,
    ) -> None:
        """Initializes the model engine.

        Args:
            zones (Optional[dict[str, HydrologicZone]]): A dictionary to
                override the default zones in the structure with custom-
                parameterized ones. Keys are zone names. Defaults to None.
            scales (Optional[list[float]]): A list of fractional areas for each
                surface zone, defining their contribution to the total catchment.
                Must sum to 1. Defaults to equal scaling.
            lapse_rates (Optional[list[LapseRateParameters]]): A list of lapse
                rate parameters, one for each surface zone. Defaults to None.
            verbose (bool): If True, prints additional information during
                initialization. Defaults to False.
        """

        super().__init__(zones, scales, verbose=verbose)

        # ==== Initialize reactive transport ==== #
        # Make sure that all arguments are present
        if (network is not None) == (rt_params is not None):
            pass
        else:
            raise ValueError(
                "When doing reactive transport simulation, you must pass both `network` and `rt_zones` parameters, not just one of them"
            )

        if (network is not None) and (rt_params is not None):
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

            if rt_configuration is None:
                rt_configuration = dict()

            # Get the configurations
            cfg: dict[str, RtZoneConfiguration] = {}
            for zone_name in self.zone_names:
                if zone_name in rt_configuration:
                    cfg[zone_name] = rt_configuration[zone_name]
                else:
                    cfg[zone_name] = RtZoneConfiguration(
                        do_reactions=True, do_speciation=True
                    )

            self._rt_zones: dict[str, RtZone] = {
                key: RtZone(
                    network=network,
                    params=val,
                    name=key,
                    do_reactions=cfg[key].do_reactions,
                    do_speciation=cfg[key].do_speciation,
                )
                for key, val in rt_params.items()
            }
            self._zone_configs: dict[str, RtZoneConfiguration] = cfg

        # Set the configuration for each reactive transport zone

    @property
    def reaction_network(self) -> Optional[ReactionNetwork]:
        if self._has_rt:
            return self._network
        else:
            print("This model does not have reactive transport capabilities")
            return None

    @property
    def rt_zones(self) -> dict[str, RtZone]:
        return self._rt_zones

    def __len__(self) -> int:
        """Returns the number of layers in the model."""
        return len(self.layers)

    def __iter__(self) -> Iterator[Layer]:
        """Returns an iterator over the layers in the model."""
        return iter(self.layers)

    def get_river_zone_ids(self) -> list[int]:
        """Gets the indices of zones that discharge laterally to the river.

        By convention, these are the last zones in each layer.

        Returns:
            list[int]: A list of indices for river-contributing zones.
        """
        river_zones: list[int] = []
        current_zone_idx: int = 0
        for layer in self.layers:
            # The last zone in each layer of each hillslope is considered to flow into the river
            # This assumes a rectangular domain for simplicity in this method
            if len(layer) > 0:
                river_zones.append(current_zone_idx + len(layer) - 1)
            current_zone_idx += len(layer)
        return river_zones

    # Newer definitions
    @classmethod
    def get_num_zone_parameters(cls) -> int:
        """Gets the total number of tunable parameters across all zones.

        Returns:
            int: The count of zone-specific parameters.
        """
        num_zone_params: int = 0
        for layer in cls.structure:
            for zone in layer:
                num_zone_params += len(zone.param_list())

        return num_zone_params

    @classmethod
    def get_num_size_parameters(cls) -> int:
        """Gets the number of tunable size (area) parameters.

        This is equal to one less than the number of surface zones, as the
        last one is determined by the constraint that they sum to 1.

        Returns:
            int: The number of size parameters.
        """
        return len(cls.structure[0]) - 1

    @classmethod
    def hydro_from_array(
        cls, arr: NDArray, latent: bool = False, natural_scales: bool = True
    ) -> Model:
        """Creates a new model instance from a 1D NumPy array of parameters.

        Args:
            arr (NDArray): A flat array of parameter values.
            latent (bool): If True, treats size parameters as being in a latent
                space, requiring transformation. Defaults to False.

        Returns:
            Model: A new, parameterized model instance.
        """

        num_zone_params: int = cls.get_num_zone_parameters()
        num_size_params: int = cls.get_num_size_parameters()

        zone_params: NDArray = arr[:num_zone_params]
        size_params: list[float] = arr[
            num_zone_params : num_zone_params + num_size_params
        ].tolist()

        if latent:
            fractions: list[float] = []
            remainder: float = 1.0
            for size in size_params:
                fraction: float = size / remainder
                fractions.append(fraction)
                remainder -= size

            size_params = fractions

        size_params.append(1 - sum(size_params))
        # print(f"Size params: {size_params}")

        lapse_rate_params: NDArray = arr[num_zone_params + num_size_params :]

        new_zones: dict[str, HydrologicZone] = dict()
        for layer in cls.structure:
            for zone in layer:
                ps, zone_params = (
                    zone_params[: zone.num_parameters()],  # type: ignore
                    zone_params[zone.num_parameters() :],  # type: ignore
                )
                new_zones[zone.name] = zone.from_array(ps, natural_scales=natural_scales)  # type: ignore

        new_lapse_rates: list[LapseRateParameters] = [
            LapseRateParameters(
                temp_factor=temp_factor,
                precip_factor=precip_factor,
            )
            for precip_factor, temp_factor in zip(
                lapse_rate_params[::2], lapse_rate_params[1::2], strict=True
            )
        ]

        return cls(
            zones=new_zones,
            scales=size_params,
            lapse_rates=new_lapse_rates,
        )

    def hydro_parameter_names(self) -> list[str]:
        """Gets an ordered list of all parameter names in the model.

        Returns:
            list[str]: A list of parameter names (e.g., "soil.fc").
        """
        param_names: list[str] = []
        for layer in self.structure:
            for zone in layer:
                for param in zone.parameter_names():  # type: ignore
                    param_names += [f"{zone.name}.{param}"]  # type: ignore
        for i, _ in enumerate(self.structure[0][:-1]):
            param_names.append(f"proportion.{i + 1}")

        # for i, _ in enumerate(self.lapse_rates):
        #     param_names += [
        #         f"lapse_rate.{i + 1}.precip_factor",
        #         f"lapse_rate.{i + 1}.temp_factor",
        #     ]

        return param_names

    @classmethod
    def from_dict(cls, params: dict) -> Model:
        """Creates a new model instance from a dictionary of parameters.

        Args:
            params (dict): A dictionary mapping parameter names (e.g., "soil.fc")
                to their values.

        Returns:
            Model: A new, parameterized model instance.
        """
        params_list: list[tuple[str, float]] = [
            (key, val) for key, val in params.items()
        ]

        zone_params_list: list[tuple[str, float]] = [
            x for x in params_list if x[0].split(".")[0] in cls.get_zone_names()
        ]
        proportion_params_list: list[tuple[str, float]] = [
            x for x in params_list if x[0].split(".")[0] == "proportion"
        ]
        lapse_rate_params_list: list[tuple[str, float]] = [
            x for x in params_list if x[0].split(".")[0] == "lapse_rate"
        ]

        decomposed_vals: list[tuple[str, str, float]] = [
            (key.split(".")[0], key.split(".")[1], val) for key, val in zone_params_list
        ]

        # Check the zone names
        zone_names: list[str] = list(set(map(lambda x: x[0], decomposed_vals)))

        for zone_name in zone_names:
            if zone_name not in cls.get_zone_names():
                raise ValueError(f"Unknown zone name: {zone_name}")

        # Now, group the parameters into their zones
        zone_params: dict[str, dict[str, float]] = {}
        zone_param_dict: dict[str, float]
        for key, group in itertools.groupby(decomposed_vals, lambda x: x[0]):
            zps: list[tuple[str, str, float]] = list(group)
            zone_param_dict = {x[1]: x[2] for x in zps}

            zone_params[key] = zone_param_dict

        # Now, construct the zones
        zones: dict[str, HydrologicZone] = {}
        for zone_name in zone_names:
            zone_param_dict = zone_params[zone_name]
            zone_type: type[HydrologicZone] = cls.get_zone_type(zone_name)

            new_zone = zone_type.from_dict(zone_param_dict)  # type: ignore
            zones[zone_name] = new_zone

        # Get the sizes (if applicable)
        scales_ordered: list[tuple[int, float]] = [
            (int(key.split(".")[1]), val) for key, val in proportion_params_list
        ]
        scales_ordered.sort(key=lambda x: x[0])
        scales: list[float] | None = [float(val) for _, val in scales_ordered]
        if len(scales) == 0:  # type: ignore
            scales = None

        if scales is not None:
            scales.append(1 - sum(scales))

        # Get the lapse rate parameters (if applicable)
        lapse_rates: list[LapseRateParameters] = []
        lapse_rates_grouped: list[tuple[int, str, float]] = [
            (int(key.split(".")[1]), key.split(".")[2], val)
            for key, val in lapse_rate_params_list
        ]
        lapse_rates_grouped.sort(key=lambda x: x[0])

        lapse_rate_groups: dict[int, list[tuple[int, str, float]]] = {
            k: list(v)
            for k, v in itertools.groupby(lapse_rates_grouped, lambda x: x[0])
        }

        for _k, v in lapse_rate_groups.items():
            param_dict = {}
            for _, key, val in v:
                param_dict[key] = val
            lapse_rates.append(LapseRateParameters.from_dict(param_dict))  # type: ignore

        return cls(zones=zones, scales=scales, lapse_rates=lapse_rates)

    @classmethod
    def get_zone_type(cls, name: str) -> type:
        """Gets the class type of a zone by its name.

        Args:
            name (str): The name of the zone.

        Returns:
            type: The class of the specified zone (e.g., `SurfaceZone`).
        """
        for layer in cls.structure:
            for zone in layer:
                if zone.name == name:  # type: ignore
                    return type(zone)
        raise ValueError(f"Unknown zone name: {name}")

    @classmethod
    def run_batch(
        cls,
        params: DataFrame,
        forc: ForcingData | list[ForcingData],
        init_state: Optional[NDArray[f64]] = None,
        meas_streamflow: Optional[Series] = None,
        num_threads: int = -1,
        write_time_series_results: bool = False,
        threshold_function: Optional[Callable[[HydroModelResults], bool]] = None,
        output_dir: str = "batch_results",
    ) -> BatchResults:
        """Runs a batch of model simulations in parallel.

        Args:
            params (DataFrame): A DataFrame where each row is a parameter set
                and columns are parameter names.
            forc (ForcingData | list[ForcingData]): The forcing data for the simulations.
            init_state (Optional[NDArray[f64]]): The initial state for all runs.
                Defaults to None.
            meas_streamflow (Optional[Series]): Observed streamflow for all runs.
                Defaults to None.
            num_threads (int): The number of parallel processes to use. If -1,
                uses all available CPU cores. Defaults to -1.
            write_time_series_results (bool): If True, saves the full time
                series output of selected runs to CSV files. Defaults to False.
            threshold_function (Optional[Callable]): A function that returns True
                if a run's results should be saved. Defaults to None (save all).
            output_dir (str): Directory to save results. Defaults to "batch_results".
            return_results (bool): If True, returns the full results objects.
                Defaults to True.

        Returns:
            tuple[DataFrame, list[tuple[Model, HydroModelResults]]]: A tuple containing:
                - A DataFrame of scalar metrics for all runs.
                - A list of tuples, each with a `Model` instance and its results.
        """
        results: list[tuple[int, HydroModelResults]] = []
        batch_params: Optional[BatchParams] = None

        if write_time_series_results:
            if os.path.exists(output_dir):
                cur_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                new_output_dir: str = f"{output_dir}.{cur_time}"
                print(
                    f"Specified directory at {output_dir} exists, saving to {
                        new_output_dir
                    }"
                )
                output_dir = new_output_dir

            if threshold_function is None and write_time_series_results:
                print("No threshold function specified, saving all model results")
            os.makedirs(output_dir, exist_ok=True)

        bps: BatchParams = {
            "output_dir": output_dir,
            "threshold_function": threshold_function,
            "save_results": write_time_series_results,
        }

        batch_params = bps

        args: list[
            tuple[
                type[Model],
                int,
                Series,
                ForcingData | list[ForcingData],
                Optional[NDArray[f64]],
                Optional[Series],
                BatchParams,
            ],
        ] = [
            (
                cls,
                i,
                row,
                forc,
                init_state,
                meas_streamflow,
                batch_params,
            )  # type: ignore
            for i, row in params.iterrows()
        ]

        if num_threads < 1:
            num_threads = os.cpu_count()  # type: ignore
        if num_threads is None:
            num_threads = 1
            warnings.warn(
                "Failed to get number of threads, defaulting to 1", stacklevel=1
            )

        with Pool(num_threads) as pool:
            results = pool.starmap(_run_model, args)  # type: ignore

        indices = [x[0] for x in results]
        obj_series: list[Series] = [
            x[1]["objective_functions"].to_dict() for x in results  # type: ignore
        ]
        sim_results: dict[int, DataFrame] = {x[0]: x[1].simulation for x in results}

        # Now construct the dataframe of the resulting values
        res_df: DataFrame = DataFrame(obj_series, index=indices).sort_index()

        results.sort(key=lambda x: x[0])

        return {"objective_functions": res_df, "simulations": sim_results}

    @classmethod
    def default_parameter_ranges(
        cls, include_lapse_rates: bool = False
    ) -> dict[str, tuple[float, float]]:
        """Gets the default parameter ranges for calibration.

        Args:
            include_lapse_rates (bool): If True, includes default ranges for
                lapse rate parameters. Defaults to False.

        Returns:
            dict[str, tuple[float, float]]: A dictionary mapping parameter
                names to their (min, max) bounds.
        """
        param_ranges: dict[str, tuple[float, float]] = {}
        # Add the hydrologic zone parameters
        for layer in cls.structure:
            for zone in layer:
                zone_range = zone.default_parameter_range()  # type: ignore

                for param_name in zone.parameter_names():
                    param_ranges[
                        f"{zone.name}.{
                        param_name}"
                    ] = zone_range[param_name]

        # Add the size parameters
        for i, _ in enumerate(cls.structure[0][:-1]):
            param_ranges[f"proportion.{i + 1}"] = (0.1, 0.9)

        # Add the lapse rate parameters
        if include_lapse_rates:
            for i, _ in enumerate(cls.structure[0]):
                params: dict[str, tuple[float, float]] = (
                    LapseRateParameters.default_parameter_range()  # type: ignore
                )
                for param_name, param_range in params.items():
                    param_ranges[
                        f"lapse_rate.{
                        i + 1}.{param_name}"
                    ] = param_range

        return param_ranges

    @classmethod
    def simple_calibration(
        cls,
        forc: ForcingData | Iterable[ForcingData],
        meas_streamflow: Series,
        metric: (
            Literal["nse", "kge", "combined"] | Callable[[HydroModelResults], float]
        ),
        use_lapse_rates: bool = False,
        num_threads: int = -1,
        polish: bool = False,
        maxiter=10,
        print_values: bool = False,
        param_ranges: Optional[dict[str, tuple[float, float]]] = None,
    ) -> tuple[dict[str, float], HydroModelResults, opt.OptimizeResult]:
        """Performs a simple calibration using differential evolution.

        Args:
            forc (ForcingData | Iterable[ForcingData]): The forcing data.
            meas_streamflow (Series): The observed streamflow for optimization.
            metric (Literal["nse", "kge", "combined"]): The objective function to optimize
                ("nse", "kge", "metric").
            use_lapse_rates (bool): Whether to include lapse rates in the
                calibration. Defaults to False.
            num_threads (int): Number of parallel workers for the optimizer.
                Defaults to -1 (all cores).
            polish (bool): If True, refines the result with a local optimizer.
                Defaults to False.
            maxiter (int): Maximum number of generations for the optimizer.
                Defaults to 10.
            print_values (bool): If True, prints metric values during
                optimization. Defaults to False.

        Returns:
            tuple[dict, dict, OptimizeResult]: A tuple containing:
                - The dictionary of the best-fit parameters.
                - The results dictionary from the best-fit model run.
                - The full `OptimizeResult` object from SciPy.
        """

        bounds: dict[str, tuple[float, float]]
        if param_ranges is None:
            bounds = cls.default_parameter_ranges(include_lapse_rates=use_lapse_rates)
        else:
            bounds = param_ranges

        bounds_list: list[tuple[float, float]] = list(bounds.values())

        args = (cls, forc, meas_streamflow, metric, print_values)

        opt_res = opt.differential_evolution(
            func=objective_function,
            bounds=bounds_list,  # type: ignore
            maxiter=maxiter,
            tol=0.01,
            rng=0,
            polish=polish,
            workers=num_threads,
            args=args,
            updating="deferred",
        )

        # Now, run the model and return the optimum results
        model = cls.hydro_from_array(opt_res.x)
        best_results = model.run_hydro_model(
            forc=forc,  # type: ignore
            init_state=cls.default_hydro_init_state(),
            meas_streamflow=meas_streamflow,
            verbose=False,
        )

        opt_params: dict[str, float] = {
            key: val for key, val in zip(bounds.keys(), opt_res.x, strict=True)
        }

        return opt_params, best_results, opt_res

    def gradient(
        self,
        obj_func: Callable[[HydroModelResults], float] | str,
        forc: ForcingData | list[ForcingData],
        meas_streamflow: Series,
        rel_step: float = 0.01,
        mean_elev: Optional[list[float]] = None,
        include_lapse_rates: bool = False,
    ) -> np.ndarray:
        """
        Compute the gradient of some objective function that takes the `HydroModelResults` as an argument
        and returns a float representing the gradient
        """
        # Get the objective function
        if isinstance(obj_func, str):

            def operational_obj_func(res: HydroModelResults) -> float:
                return res.objective_functions[obj_func]

        else:
            operational_obj_func = obj_func  # type: ignore

        params = self.hydro_to_array()

        grad_vals: list[float] = []
        bounds: dict = self.default_parameter_ranges(
            include_lapse_rates=include_lapse_rates
        )
        for i, (x_i, (min_val, max_val)) in enumerate(
            zip(params, bounds.values(), strict=True)
        ):
            # Compute the gradient at x_i using centered finite differences
            state_left = params.copy()
            state_right = params.copy()

            dx: float = abs(state_left[i] * rel_step)

            # Make sure that the steps are nonzero
            if abs(dx) < 1e-6:
                dx = abs(0.5 * (max_val - min_val))

            x_i_left = max(min_val, x_i - dx)
            x_i_right = min(max_val, x_i + dx)

            state_left[i] = x_i_left
            state_right[i] = x_i_right

            left_model = self.hydro_from_array(state_left)
            right_model = self.hydro_from_array(state_right)

            left_res = left_model.run_hydro_model(
                forc=forc, elevations=mean_elev, meas_streamflow=meas_streamflow
            )
            right_res = right_model.run_hydro_model(
                forc=forc, elevations=mean_elev, meas_streamflow=meas_streamflow
            )

            grad_vals.append(
                (operational_obj_func(right_res) - operational_obj_func(left_res))
                / (x_i_right - x_i_left)
            )

        return np.array(grad_vals)

    @classmethod
    def mcmc(
        cls: type[Model],
        forc: ForcingData | list[ForcingData],
        meas_streamflow: Series,
        include_lapse_rates: bool = False,
        elevations: Optional[list[float]] = None,
        bounds: dict[str, tuple[float, float]] | None = None,
        num_threads: int = -1,
        num_walkers: Optional[int] = None,
        num_samples: int = 1_000,
        metric: Callable[[HydroModelResults], float] | Literal["kge", "nse"] = "kge",
        initial_state: Optional[NDArray[f64]] = None,
        random_seed: int = 0,
    ) -> tuple[DataFrame, DataFrame, emcee.EnsembleSampler, emcee.State]:
        """Runs a Markov Chain Monte Carlo simulation."""
        if num_threads < 1:
            num_threads = os.cpu_count()  # type: ignore

        if bounds is None:
            bounds = cls.default_parameter_ranges(
                include_lapse_rates=include_lapse_rates
            )

        if initial_state is None:
            initial_state = np.array(
                [0.5 * (x[0] + x[1]) for x in bounds.values()]
            )  # Have the initial guess be in the middle of the parameter ranges
        param_ranges = np.array([x[1] - x[0] for x in bounds.values()])
        ndim = initial_state.size

        if num_walkers is None:
            num_walkers = 2 * ndim

        # ==== Set random state ==== #
        random.seed(random_seed)
        np.random.seed(random_seed)
        # ========================== #

        initial_walker_states = [
            initial_state + 0.25 * param_ranges * np.random.randn(ndim)
            for _ in range(num_walkers)
        ]

        args = (
            cls,
            forc,
            meas_streamflow,
            bounds,
            metric,
            elevations,
        )

        if num_threads != 1:
            with Pool(num_threads) as pool:
                sampler = emcee.EnsembleSampler(
                    num_walkers, ndim, log_prob_fn=log_probability, pool=pool, args=args
                )

                mc_result = sampler.run_mcmc(
                    initial_walker_states, num_samples, progress=True
                )
        else:
            sampler = emcee.EnsembleSampler(
                num_walkers, ndim, log_prob_fn=log_probability, args=args
            )

            mc_result = sampler.run_mcmc(
                initial_walker_states, num_samples, progress=True
            )

        # Now, examine the outputs
        blob_arr: np.ndarray = sampler.get_blobs()  # type: ignore
        blob_arr = blob_arr.reshape(
            (blob_arr.shape[0] * blob_arr.shape[1], blob_arr.shape[2])
        )

        # Get the examples
        # base_model = cls()
        column_names: list[str] = ["kge", "nse", "log_kge", "log_nse", "bias"]
        # base_model.run(
        #     forc=forc, meas_streamflow=meas_streamflow
        # )["objective_functions"].index.tolist()

        obj_func_df = DataFrame(blob_arr, columns=column_names)

        sample_arr: np.ndarray = sampler.get_chain()  # type: ignore
        sample_arr = sample_arr.reshape(
            (sample_arr.shape[0] * sample_arr.shape[1], sample_arr.shape[2])
        )
        sample_df = DataFrame(sample_arr, columns=list(bounds.keys()))

        return sample_df, obj_func_df, sampler, mc_result  # type: ignore

    def _validate_model_structure(self) -> None:
        """Validates the model's structure and configuration.

        Checks for:
        - Correct types in the `structure` attribute.
        - Non-empty structure.
        - Valid layer connectivity (each layer must have 1 zone or the same
          number of zones as the layer above it).
        - `scales` that sum to 1.
        """
        # Check if all types are correct in the structure
        for i, layer in enumerate(self.structure):
            if not isinstance(layer, (list, tuple, np.ndarray, Series)):
                raise TypeError(
                    f"Layer {i} in structure has an incorrect type: {
                        type(layer)
                    }, ensure that all layers are of type list, tuple, or other ordered iterable"
                )
            for j, zone in enumerate(layer):
                if not isinstance(zone, HydrologicZone):
                    raise TypeError(
                        f"Zone {j} in layer {i} has an incorrect type: {
                            type(zone)
                        }, ensure that all zones are of type HydrologicZone"
                    )

        # Check if model is empty
        if self.structure == []:
            raise ValueError(
                "Model structure is empty, make sure you define a model structure class property describing your model"
            )

        # Check if model geometry is correct
        layer_lengths: list[int] = [len(layer) for layer in self.structure]
        prev_zones: int = len(self.structure[0])

        for layer_size in layer_lengths:
            if layer_size not in (1, prev_zones):
                raise ValueError(
                    f"Invalid model structure: The next layer can only have either 1 zone or the same number as the zone above, but your model has the structure {
                        layer_lengths
                    }"
                )
            prev_zones = layer_size

        # Check if all model scales add up to 1
        if abs(sum(self.scales) - 1) > 1e-3:
            raise ValueError(
                f"Model scales do not add up to 1: {
                    self.scales
                }. Ensure that the scales equal 1 to maintain water balance"
            )

    def to_dict(self) -> dict[str, float]:
        """Converts the model's parameters to a dictionary.

        Returns:
            dict[str, float]: A dictionary mapping parameter names to values.
        """
        return {
            key: val
            for key, val in zip(
                self.hydro_parameter_names(), self.hydro_to_array(), strict=True
            )
        }

    def to_series(self) -> Series:
        """Converts the model's parameters to a pandas Series.

        Returns:
            Series: A Series with parameter names as the index.
        """
        return Series(self.hydro_to_array(), index=self.hydro_parameter_names())

    @property
    def surface_zone_ids(self) -> list[int]:
        """
        Get a list of the numerical indices of the surface zones in the model.

        Note that these must be the top zones in the model. It's just the indices of the first later of zones
        """
        return [i for i, _ in enumerate(self.structure[0])]

    @classmethod
    def hydro_from_series(cls, series: Series) -> Model:
        """Creates a new model instance from a pandas Series of parameters.

        Args:
            series (Series): A Series of parameters with names as the index.

        Returns:
            Model: A new, parameterized model instance.
        """
        return cls.from_dict(series.to_dict())

    @property
    def num_hydro_parameters(self) -> int:
        """The total number of optimizable parameters in the model."""
        return len(self.hydro_to_array())

    @classmethod
    def get_num_hydro_parameters(cls: type[Model]) -> int:
        """The total number of optimizable parameters in the model."""
        num_params: int = 0
        for layer in cls.structure:
            for zone in layer:
                num_params += zone.num_parameters()  # type: ignore

        return num_params + cls.get_num_size_parameters()

    @classmethod
    def get_num_rt_parameters(
        cls: type[Model], config: dict[str, RtZoneConfiguration]
    ) -> int:
        """
        Get the number of parameters for a zone given the configuration of the reactive transport setup
        """
        num_dim_params: int = 3 * cls.get_num_zones()
        num_min_params: int = 0
        for _zone_name, zone in config.items():
            num_min_params += PARAMETERS_PER_MINERAL * zone.do_reactions

        return num_dim_params + num_min_params

    def rt_to_array(self) -> NDArray:
        """Convert the reactive transport structures into an array"""
        components: list[NDArray] = []

        for zone_name in self.zone_names:
            components.append(self.rt_zones[zone_name].to_array())

        return np.concatenate(components)

    @classmethod
    def rt_params_from_array(
        cls: type[Model],
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
            3 + z.do_reactions * num_minerals * PARAMETERS_PER_MINERAL
            for z in configs.values()
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
            zone_params[zone_name_i] = RtParameters.from_array(
                row_i, natural_scales=natural_scales
            )

        return zone_params

    @classmethod
    def rt_zones_from_array(
        cls: type[Model],
        arr: NDArray,
        network: ReactionNetwork,
        config: list[RtZoneConfiguration],
    ) -> dict[str, RtZone]:
        # params_per_zone: int = 3 + network.num_mineral_parameters
        params_per_zone: list[int] = [
            3 * z.do_reactions * network.num_minerals * PARAMETERS_PER_MINERAL
            for z in config
        ]

        # if arr.size % params_per_zone != 0:
        #     raise ValueError(
        #         f"Wrong size vector passed, expected multiple of {
        #             params_per_zone
        #         }, got {arr.size}"
        #     )

        # num_zones: int = len(config)

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

    def update_rt_params(self, rt_params: dict[str, RtParameters]) -> Model:
        """
        Create a new model by just updating the reactive transport zones
        """
        new_hydro_zones: dict[str, HydrologicZone] = {
            name: type(zone).from_array(np.array(zone.param_list()))  # type: ignore
            for name, zone in self.__zones.items()
        }
        scales = deepcopy(self.scales)
        # lapse_rates = deepcopy(self.lapse_rates)
        network = deepcopy(self.network)

        rt_configurations: dict[str, RtZoneConfiguration] = self.zone_configurations

        return type(self)(
            zones=new_hydro_zones,
            scales=scales,
            # lapse_rates=None,
            network=network,
            rt_params=rt_params,
            rt_configuration=rt_configurations,
        )

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
        zw_vals = calculate_water_table_depth(model_rt_params, hydro_sim_df)
        hydro_steps = get_hydro_steps(hydro_sim_df)
        hydro_forcing: NDArray = self.construct_hydro_forcing_matrix(forc).T

        num_zones: int = hydro_forcing.shape[0]
        num_steps: int = hydro_forcing.shape[1]

        rt_forcings: NDArray = np.empty((num_zones, num_steps), dtype=object)

        for i in range(num_zones):
            for j in range(num_steps):
                d_ij = RtForcing(
                    conc_in=np.zeros(self.network.num_species),
                    hydro_step=hydro_steps[i, j],
                    hydro_forc=hydro_forcing[i, j],
                    s_w=sw_vals[i, j],
                    z_w=zw_vals[i, j],
                )
                rt_forcings[i, j] = d_ij

        # Now, insert the precipitation chemistry for each zone
        if precip_conc.shape[0] != len(self.surface_zone_ids):  # type: ignore
            raise ValueError(
                f"The first dimension of precipitation chemistry must be the same as the the number of surface zones: {
                    self.num_surface_zones
                }, not {precip_conc.shape[0]}"  # type: ignore
            )

        if precip_conc.shape[1] != len(hydro_sim_df):
            raise ValueError(
                f"The second dimension of precipitation chemistry must be the same as the the number of surface zones: {
                    len(hydro_sim_df)
                }, not {precip_conc.shape[1]}"
            )

        for i, zone_data in enumerate(precip_conc):
            for j, species_conc in enumerate(zone_data):
                rt_forcings[i, j].conc_in = species_conc.copy()

        return rt_forcings

    def step_rt_model(
        self,
        model: dict[str, RtZone],
        state: NDArray,
        ds: Iterable[RtForcing],
        dt_days: float,
        verbose: bool = False,
        failed_dir: Optional[str] = None,
    ) -> list[RtStep]:
        num_zones: int = len(model)
        zones: list[RtZone] = list(model.values())

        num_species: int = zones[0].num_species

        lat_mass: NDArray = np.zeros(
            (num_zones, num_species), dtype=np.float64
        )  # Lateral mass transfered by a zone
        vert_mass: NDArray = lat_mass.copy()  # Mass transferred vertically by a zone

        steps: list[RtStep] = []

        i: int
        zone: RtZone
        s_i: NDArray
        d_i: RtForcing
        for i, (zone, s_i, d_i) in enumerate(  # type: ignore
            zip(model.values(), state.copy(), ds, strict=True)
        ):
            d_i = d_i.copy()

            if verbose:
                print(f"Starting solution for zone on step '{zone.name}'")
            tot_water_in: float = d_i.hydro_step.q_in + d_i.hydro_step.forc_flux
            if i not in self.surface_zone_ids:
                lat_dep_row = self.lat_mat[i]
                vert_dep_row = self.vert_mat[i]

                lat_mass_in = lat_mass.T @ lat_dep_row
                vert_mass_in = vert_mass.T @ vert_dep_row

                mass_in = lat_mass_in + vert_mass_in

                if tot_water_in > 1e-6:
                    conc_in = mass_in / tot_water_in
                else:
                    conc_in = np.zeros_like(mass_in)
                d_i.conc_in = conc_in  # type: ignore

                if verbose:
                    print(f"Mass entering zone '{zone.name}': \n{mass_in=}")
                    print(f"Lateral component entering: {lat_mass_in}")
                    print(f"Vertical component entering: {vert_mass_in}")

            else:  # For surface zones, the incoming precipitation provides the mass flux
                mass_in = d_i.conc_in * tot_water_in

            if verbose:
                print(f"Incoming concentration to zone {i}: {d_i.conc_in}")

            try:
                step: RtStep = zone.step(s_i, d_i, dt_days, verbose=verbose)
            except OptimizationError as e:
                raise RtNumericalError(
                    "other",
                    model_type=type(self),
                    zone=zone,
                    parameters=self.to_array(),
                    state=s_i,
                    rt_forcing=d_i.copy(),
                    math_err=e,
                ) from e
            except Exception as e:
                print(f"Failed on zone {i} with values:")
                print(f"{s_i=}")
                print(f"{d_i=}")
                print(f"{dt_days=}")
                raise e
            lat_mass[i] = step.lat_mass
            vert_mass[i] = step.vert_mass
            steps.append(step)
            if verbose:
                print(f"Lateral mass transfer outwards: {step.lat_mass}")
                print(f"Vertical mass transfer outwards: {step.vert_mass}")
                print(
                    f"Finished zone '{zone.name}', final concentration: {
                        np.array2string(
                            step.state, formatter={'all': lambda x: f'{x:.2e}'}
                        )
                    }"
                )

                print("\n\n")
        if verbose:
            print("=" * 25 + "\n")

        return steps

    def run_rt_model(
        self,
        hydro_sim_df: DataFrame,
        forc: ForcingData | Iterable[ForcingData],
        precip_conc: NDArray,
        mineral_conc: Iterable | dict[str, Iterable | dict[str, float]],
        exchange_conc: Optional[dict[str, float]] = None,
        init_conc: Optional[NDArray] = None,
        meas_river_conc: Optional[DataFrame] = None,
        verbose: bool = False,
        return_partial: bool = False,
        failed_dir: Optional[str] = None,
        objective_functions: list[
            tuple[str, Callable[[Series, Series], float]]
        ] = DEFAULT_OBJECTIVE_FUNCTIONS,
    ) -> RtModelResults:
        """Run the reactive transport simulation forwards
        `mineral_conc` is a dictionary containing the mineral volume fractions (0,porosity) of the minerals
        of each of the zones
        """
        # ==== Set up the forcing ==== #
        zones_list: list[RtZone] = list(self.rt_zones.values())

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
        if init_conc is None:
            init_conc_rows = [
                self.network.get_default_aqueous_initial_state()
                for _zone in self.zone_names
            ]
            init_conc = np.vstack(init_conc_rows)

        aqueous_init_state: NDArray = init_conc
        mineral_init_state: NDArray | None = rt_minerals_to_array(
            mineral_conc=mineral_conc,
            mineral_order=self.network.mineral_species_names,
            zone_order=self.zone_names,
        )

        if mineral_init_state is not None:
            state: NDArray = np.hstack([aqueous_init_state, mineral_init_state])
        else:
            state = aqueous_init_state

        if exchange_conc is not None:
            # Construct the exchange concentrations
            site_conc: np.ndarray = np.array(
                [[exchange_conc[n] for n in self.zone_names]]
            ).T
            num_eq_species: int = self.network.num_exchange_species - 1
            exch_eq_conc = np.full((self.num_zones, num_eq_species), 1e-20, dtype=float)
            state = np.hstack([state, site_conc, exch_eq_conc])

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
                print(f"Failed on step {i}")  # type: ignore
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
        q_components: dict[str, Series] = {
            n: hydro_sim_df[f"q_lat_{n}"] for n in river_zone_names
        }
        q_sim: Series = hydro_sim_df["sim_streamflow_mmd"]
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
        # =========================== #

        return rt_sim_res  # type: ignore

    def run_both_models(
        self,
        forc: ForcingData | list[ForcingData],
        precip_conc: NDArray,
        mineral_conc: Iterable | dict[str, Iterable | dict[str, float]],
        exchange_conc: Optional[dict[str, float]] = None,
        init_conc: Optional[NDArray | Series] = None,
        init_hydro_state: Optional[NDArray[f64]] = None,
        meas_streamflow: Optional[Series] = None,
        average_elevation: Optional[float] = None,
        elevations: Optional[list[float]] = None,
        check_water_balance: bool = False,
        meas_river_conc: Optional[DataFrame] = None,
        objective_functions: Optional[
            list[tuple[str, Callable[[Series, Series], float]]]
        ] = None,
        verbose: bool = False,
    ) -> ModelResults:
        if verbose:
            print("Starting hydrologic simulation")

        hydro_res = self.run_hydro_model(
            forc=forc,
            init_state=init_hydro_state,
            meas_streamflow=meas_streamflow,
            average_elevation=average_elevation,
            elevations=elevations,
            check_water_balance=check_water_balance,
            objective_functions=objective_functions,
        )

        if verbose:
            print("Finished hydrologic model simulation")
            print("Starting reactive transport simulation")

        rt_res: RtModelResults = self.run_rt_model(
            hydro_sim_df=hydro_res.simulation,
            forc=forc,
            mineral_conc=mineral_conc,
            precip_conc=precip_conc,
            init_conc=init_conc,  # type: ignore
            meas_river_conc=meas_river_conc,
            verbose=verbose,
            return_partial=False,
            failed_dir=None,
            exchange_conc=exchange_conc,
        )

        return ModelResults(hydro=hydro_res, reactive_transport=rt_res)

    @classmethod
    def from_array(
        cls: type[Model],
        arr: NDArray,
        network: ReactionNetwork,
        config: dict[str, RtZoneConfiguration],
        verbose: bool = False,
        natural_scales: bool = True,
    ) -> Model:
        """
        Creates a new model instance from a single array containing both
        hydrologic and reactive transport parameters.

        Args:
            arr (NDArray): A flat array of all parameter values, with
                hydrologic parameters first, followed by reactive transport
                parameters.
            network (ReactionNetwork): The reaction network for the RT model.
            config (dict[str, RtZoneConfiguration]): A dictionary of RT configurations,
                one for each zone in order.

        Returns:
            Model: A new, fully parameterized model instance.
        """
        # Determine the number of reactive transport (RT) parameters.
        # Each zone has 3 base RT params + one per mineral in the network.
        num_zones: int = cls.get_num_zones()
        num_hydro_params: int = cls.get_num_hydro_parameters()
        num_dim_params: int = 3 * num_zones
        num_min_params: int = 0
        for _zone_name, zone in config.items():
            num_min_params += PARAMETERS_PER_MINERAL * zone.do_reactions
        num_expected: int = num_hydro_params + num_dim_params + num_min_params

        if arr.size != num_expected:
            raise PotionsError(
                f"""Incorrect number of parameters for creating RT model. Expected {num_expected}, received {arr.size},
                with {num_hydro_params} hydrological parameters, {num_dim_params} zone parameters, and
                {num_min_params} mineral parameters"""
            )
        if verbose:
            print(f"Number of zones in the model: {num_zones}")
            print(f"Number of hydrological parameters: {num_hydro_params}")
            print(
                f"Total number of reactive transport parameters: {
                  num_dim_params}"
            )

        # Split the main array into hydrologic and RT parts.
        # Slicing from the end is safer than calculating the hydro param count.
        hydro_params_arr = arr[0:num_hydro_params]
        rt_params_arr = arr[num_hydro_params:]

        if verbose:
            print(f"Hydrological parameters: {hydro_params_arr}")
            print(f"Reactive transport parameters: {rt_params_arr}")

        # Create the base hydrologic model and the RT parameters separately.
        hydro_mod: Model = cls.hydro_from_array(
            hydro_params_arr, natural_scales=natural_scales
        )
        rt_params: dict[str, RtParameters] = cls.rt_params_from_array(
            rt_params_arr,
            network,
            config,
            verbose=verbose,
            natural_scales=natural_scales,
        )

        if verbose:
            print("Reactive transport parameters for each zone:")
            for key, val in rt_params.items():
                print(f"{key}: {val}")

        # The model constructor expects a dictionary for RT configuration,
        # so we convert the input list.

        # Construct the final model using components from the hydrologic model
        # and the newly created RT components.
        return cls(
            zones=hydro_mod.hydro_zones,
            scales=hydro_mod.scales,
            network=network,
            rt_params=rt_params,
            rt_configuration=config,
        )

    def to_array(self) -> NDArray:
        """Convert this model and turn it into a numpy array"""
        return np.concat([self.hydro_to_array(), self.rt_to_array()])

    @property
    def network(self) -> ReactionNetwork:
        if self._has_rt:
            return self._network
        else:
            raise ValueError("Model does not have reactive-transport capabilities")
