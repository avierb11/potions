from __future__ import annotations
import datetime
import os
import warnings
from multiprocessing import Pool
from typing import (
    Callable,
    Final,
    Iterable,
    Optional,
)

from numpy import float64 as f64
from numpy.typing import NDArray
from pandas import DataFrame, Series

from potions.hydro_model import HydrologicalModel
from potions.reactive_transport.kinetic_structures import PARAMETERS_PER_MINERAL
from .reactive_transport_model import ReactiveTransportModel

# from potions.reactive_transport.rt_zone import RtStep

from .common_types import (
    BatchParams,
    ForcingData,
    HydroModelResults,
    RtModelResults,
    ModelResults,
    RtZoneConfiguration,
    BatchResults,
)

from potions.core import (  # type: ignore
    # Hydrology
    HydrologicZone,
    RiverZone,
    # Reactive Transport
    ReactionNetwork,
    RtParameters,
    RtZone,
)


from .utils import (
    PotionsError,
)
from .utils import setup_logging

setup_logging(__file__)

# Define a TypeVar for Zones to make Layer, Hillslope, and Model generic

# ==== Constants ==== #
OUTPUT_COLUMNS_PER_ZONE: Final[int] = (
    # Number of columns for each zone in the output. Includes 1 state + 6 fluxes (4 normal + 2 external)
    8
)
# =================== #


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


class Model(ReactiveTransportModel):
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

    def __init__(
        self,
        # Hydrology arguments
        zones: Optional[dict[str, HydrologicZone]] = None,
        scales: Optional[list[float]] = None,
        # Reactive transport arguments
        network: Optional[ReactionNetwork] = None,
        rt_zones: Optional[dict[str, RtZone]] = None,
        # River zone
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

        super().__init__(
            zones=zones,
            scales=scales,
            network=network,
            rt_zones=rt_zones,
            river_zone=river_zone,
            verbose=verbose,
        )

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
            hydro_res=hydro_res,
            precip_conc=precip_conc,
            init_conc=init_conc,  # type: ignore
            meas_river_conc=meas_river_conc,
            verbose=verbose,
            return_partial=False,
            failed_dir=None,
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
        num_river_params: int = 0
        for zone_name, zone in config.items():
            if zone_name == "river":
                num_river_params += 3 + PARAMETERS_PER_MINERAL * zone.do_reactions
            else:
                num_min_params += PARAMETERS_PER_MINERAL * zone.do_reactions
        num_expected: int = (
            num_hydro_params + num_dim_params + num_min_params + num_river_params
        )

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
        hydro_params_arr: NDArray = arr[0:num_hydro_params]
        rt_zone_params_arr: NDArray = arr[num_hydro_params:-num_river_params]
        river_params_arr: NDArray = arr[-num_river_params:]

        if verbose:
            print(f"Hydrological parameters: {hydro_params_arr}")
            print(f"Reactive transport parameters: {rt_zone_params_arr}")

        # Create the base hydrologic model and the RT parameters separately.
        hydro_mod: HydrologicalModel = cls.hydro_from_array(
            hydro_params_arr, natural_scales=natural_scales
        )
        rt_params: dict[str, RtParameters] = cls.rt_params_from_array(
            rt_zone_params_arr,
            network,
            config,
            verbose=verbose,
            natural_scales=natural_scales,
        )

        if verbose:
            print("Reactive transport parameters for each zone:")
            for key, val in rt_params.items():
                print(f"{key}: {val}")

        rt_zones: dict[str, RtZone] = {}
        for zone_name in rt_params.keys():
            cfg: RtZoneConfiguration = config[zone_name]
            rt_zone = RtZone(
                network,
                params=rt_params[zone_name],
                do_reactions=cfg.do_reactions,
                do_speciation=cfg.do_speciation,
                name=zone_name,
            )
            rt_zones[zone_name] = rt_zone

        river_zone: Optional[RiverZone] = None
        if river_params_arr.size > 0:
            river_config = config["river"]
            river_zone = RiverZone.from_array(
                river_params_arr,
                network=network,
                do_reactions=river_config.do_reactions,
                do_speciation=river_config.do_speciation,
            )

        # The model constructor expects a dictionary for RT configuration,
        # so we convert the input list.

        # Construct the final model using components from the hydrologic model
        # and the newly created RT components.
        return cls(
            zones=hydro_mod.hydro_zones,
            scales=hydro_mod.scales,
            network=network,
            rt_zones=rt_zones,
            river_zone=river_zone,
        )
