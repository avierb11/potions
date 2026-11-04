from __future__ import annotations
import os
import warnings
from multiprocessing import Pool
from typing import (
    Callable,
    Final,
    Optional,
    Sequence,
)
import numpy as np
from numpy import float64 as f64
from numpy.typing import NDArray
from pandas import DataFrame, Series

from potions.hydro_model import HydrologicalModel
from potions.reactive_transport.kinetic_structures import PARAMETERS_PER_MINERAL
from .reactive_transport_model import ReactiveTransportModel

# from potions.reactive_transport.rt_zone import RtStep

from .common_types import (
    BatchParams,
    ChemicalState,
    DerivedSpecies,
    ForcingData,
    HydroModelResults,
    RtModelResults,
    ModelResults,
    RtZoneConfiguration,
    BatchResults,
)

from .core import (
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
from .result_saving import (
    DEFAULT_RESULTS_DIR,
    is_save_enabled,
    make_run_dir,
    save_model_run,
    write_batch_info,
)

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
            save_results=False,  # saving is controlled at the batch level
        )

        if batch_params["save_results"]:
            save_model = True
            if batch_params["threshold_function"] is not None:
                save_model = batch_params["threshold_function"](run_res)

            if batch_params["output_dir"] is None:
                raise ValueError("Output directory must be specified")

            if save_model:
                run_res.simulation.to_csv(
                    os.path.join(batch_params["output_dir"], f"run_{i}.csv")
                )
                model.describe_parameters().to_csv(
                    os.path.join(batch_params["output_dir"], f"run_{i}_parameters.csv"),
                    index=False,
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
        # Derived measurements
        derived_species: Optional[Sequence[DerivedSpecies]] = None,
    ) -> None:
        """Initializes the model engine.

        Args:
            zones (Optional[dict[str, HydrologicZone]]): A dictionary to
                override the default zones in the structure with custom-
                parameterized ones. Keys are zone names. Defaults to None.
            scales (Optional[list[float]]): A list of fractional areas `for each
                surface zone, defining their contribution to the total catchment.
                All values must be strictly positive and sum to 1. These area
                fractions govern forcing distribution, the area-weighted
                lateral transfer between zones, and the area-weighted outlet
                discharge. Defaults to equal scaling.
            lapse_rates (Optional[list[LapseRateParameters]]): A list of lapse
                rate parameters, one for each surface zone. Defaults to None.
            verbose (bool): If True, prints additional information during
                initialization. Defaults to False.
            derived_species (Optional[Sequence[DerivedSpecies]]): Measured
                quantities that are not species of the reaction network but
                are defined as the sum of several network species (for
                example DIC, the sum of the aqueous carbonate species). The
                model scores a measurement column named after a derived
                measurement against the sum of its components' river
                concentrations. Defaults to None.
        """

        super().__init__(
            zones=zones,
            scales=scales,
            network=network,
            rt_zones=rt_zones,
            river_zone=river_zone,
            verbose=verbose,
            derived_species=derived_species,
        )

    @classmethod
    def run_batch(
        cls,
        params: DataFrame,
        forc: ForcingData | list[ForcingData],
        init_state: Optional[NDArray[f64]] = None,
        meas_streamflow: Optional[Series] = None,
        num_threads: int = -1,
        write_time_series_results: Optional[bool] = None,
        threshold_function: Optional[Callable[[HydroModelResults], bool]] = None,
        output_dir: str = DEFAULT_RESULTS_DIR,
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
            write_time_series_results (Optional[bool]): If True, saves the full
                time series output of selected runs to CSV files (under
                ``<output_dir>/run_<timestamp>/``). If None, saving is on by
                default. Defaults to None.
            threshold_function (Optional[Callable]): A function that returns True
                if a run's results should be saved. Defaults to None (save all).
            output_dir (str): Base directory under which the `run_<timestamp>`
                batch folder is created. Defaults to `"results"`.
        Returns:
            BatchResults: A dictionary containing:
                - `objective_functions`: A DataFrame of scalar metrics for all runs.
                - `simulations`: A dictionary mapping run index to the simulation DataFrame.
        """
        do_save: bool = is_save_enabled(write_time_series_results)
        batch_out_dir: str = str(output_dir)

        if do_save:
            batch_out_dir = str(make_run_dir(output_dir))
            if threshold_function is None:
                print("No threshold function specified, saving all model results")

        bps: BatchParams = {
            "output_dir": batch_out_dir if do_save else None,
            "threshold_function": threshold_function,
            "save_results": do_save,
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
        ]  # ty:ignore[invalid-assignment]

        if num_threads < 1:
            num_threads = os.cpu_count()  # type: ignore
        if num_threads is None:
            num_threads = 1
            warnings.warn(
                "Failed to get number of threads, defaulting to 1", stacklevel=1
            )

        with Pool(num_threads) as pool:
            results = pool.starmap(_run_model, args)  # type: ignore

        ok_runs: list[tuple[int, HydroModelResults]] = [
            (idx, res) for idx, res in results if res is not None
        ]
        indices = [x[0] for x in ok_runs]
        obj_dict_list: list[dict[str, float]] = [
            x[1].objective_functions.to_dict() for x in ok_runs
        ]
        sim_results: dict[int, DataFrame] = {x[0]: x[1].simulation for x in ok_runs}

        # Now construct the dataframe of the resulting values
        res_df: DataFrame = DataFrame(obj_dict_list, index=indices).sort_index()

        if do_save:
            write_batch_info(
                batch_out_dir,
                num_runs=len(indices),
                num_threads=num_threads,
                has_threshold=threshold_function is not None,
            )
            print(f"Saving batch results to {batch_out_dir}")

        return {"objective_functions": res_df, "simulations": sim_results}

    def run_both_models(
        self,
        # Hydrology
        forc: ForcingData | list[ForcingData],
        # Reactive transport
        precip_conc: NDArray,
        init_conc: dict[str, ChemicalState],
        # Optional data
        init_hydro_state: Optional[NDArray[f64]] = None,
        meas_streamflow: Optional[Series] = None,
        meas_river_conc: Optional[DataFrame] = None,
        objective_functions: Optional[
            list[tuple[str, Callable[[Series, Series], float]]]
        ] = None,
        verbose: bool = False,
        save_results: Optional[bool] = None,
        results_dir: Optional[str] = None,
    ) -> ModelResults:
        """Runs the hydrologic and reactive transport models in sequence.

        By default, once both models have run, the combined results and the
        model parameters are saved to a fresh ``results/run_<timestamp>``
        directory (see ``potions.result_saving`` for details on how to
        disable or redirect saving).

        Args:
            forc (ForcingData | list[ForcingData]): The forcing data.
            precip_conc (NDArray): Precipitation concentrations for the RT model.
            init_conc (dict[str, ChemicalState]): Initial chemical state for the
                RT model.
            init_hydro_state (Optional[NDArray[f64]]): Initial state of the
                hydrologic model. Defaults to None.
            meas_streamflow (Optional[Series]): Observed streamflow, used to
                calculate objective functions. Missing days may be NaN and are
                excluded from every metric. Defaults to None.
            meas_river_conc (Optional[DataFrame]): Observed river concentrations,
                used to calculate RT objective functions. Defaults to None.
            objective_functions (Optional[list[tuple[str, ...]]]): Custom objective
                functions. Defaults to None (use the defaults).
            verbose (bool): Whether to print progress messages. Defaults to False.
            save_results (Optional[bool]): Whether to save the run's results and
                parameters to disk. If None, saving is on by default.
                Defaults to None.
            results_dir (Optional[str]): Base directory under which the
                ``run_<timestamp>`` folder is created. If None, uses the default
                ``"results"`` directory. Defaults to None.
        """
        if verbose:
            print("Starting hydrologic simulation")

        hydro_res = self.run_hydro_model(
            forc=forc,
            init_state=init_hydro_state,
            meas_streamflow=meas_streamflow,
            objective_functions=objective_functions,
            # Saving happens once, for both models, at the end of this method.
            save_results=False,
        )

        if verbose:
            print("Finished hydrologic model simulation")
            print("Starting reactive transport simulation")

        rt_res: RtModelResults = self.run_rt_model(
            hydro_res=hydro_res,
            precip_conc=precip_conc,
            initial_state=init_conc,
            meas_river_conc=meas_river_conc,
            verbose=verbose,
            return_partial=False,
            failed_dir=None,
            # Saving happens once, for both models, at the end of this method.
            save_results=False,
        )

        if verbose:
            print("Finished reactive transport simulation")

        res = ModelResults(hydro=hydro_res, reactive_transport=rt_res)

        if is_save_enabled(save_results):
            saved_to = save_model_run(self, hydro_res, rt_res, results_dir)
            res.hydro.saved_dir = str(saved_to)
            res.reactive_transport.saved_dir = str(saved_to)
            print(f"Saving run results to {saved_to}")

        return res

    @classmethod
    def from_array(
        cls: type[Model],
        arr: NDArray,
        network: ReactionNetwork,
        config: dict[str, RtZoneConfiguration],
        verbose: bool = False,
        natural_scales: bool = True,
        derived_species: Optional[Sequence[DerivedSpecies]] = None,
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
            verbose (bool): If True, prints additional information about the
                model construction. Defaults to False.
            natural_scales (bool): If True, the dimensional parameters are
                scaled to their natural ranges. Defaults to True.
            derived_species (Optional[Sequence[DerivedSpecies]]): Measured
                quantities that are not species of the reaction network but
                are defined as the sum of several network species (for
                example DIC, the sum of the aqueous carbonate species). The
                model scores a measurement column named after a derived
                measurement against the sum of its components' river
                concentrations. Defaults to None.

        Returns:
            Model: A new, fully parameterized model instance.
        """
        # Determine the number of reactive transport (RT) parameters.
        # Each zone has 2 base RT params (max_water, passive_water_storage)
        # plus one set of mineral params per mineral in the network.
        num_zones: int = cls.get_num_zones()
        num_hydro_params: int = cls.get_num_hydro_parameters()
        num_dim_params: int = 2 * num_zones
        num_minerals: int = network.num_minerals
        num_min_params: int = 0
        num_river_params: int = 0
        for zone_name, zone in config.items():
            if zone_name == "river":
                num_river_params += (
                    2 + num_minerals * PARAMETERS_PER_MINERAL * zone.do_reactions
                )
            else:
                num_min_params += (
                    num_minerals * PARAMETERS_PER_MINERAL * zone.do_reactions
                )
        num_expected: int = (
            num_hydro_params + num_dim_params + num_min_params + num_river_params
        )
        num_rt_zone_params: int = num_dim_params + num_min_params

        if arr.size != num_expected:
            raise PotionsError(
                f"""Incorrect number of parameters for creating RT model. Expected {num_expected}, received {arr.size},
                with {num_hydro_params} hydrological parameters, {num_dim_params} zone parameters,
                {num_min_params} mineral parameters, and {num_river_params} river parameters"""
            )
        if verbose:
            print(f"Number of zones in the model: {num_zones}")
            print(f"Number of hydrological parameters: {num_hydro_params}")
            print(f"Number of mineral parameters: {num_hydro_params}")
            print(f"Total number of dimension parameters: {num_dim_params}")

        # Split the main array into hydrologic and RT parts.
        # Slicing from the end is safer than calculating the hydro param count.
        hydro_params_arr: NDArray = arr[0:num_hydro_params]
        rt_zone_params_arr: NDArray = arr[
            num_hydro_params : num_hydro_params + num_rt_zone_params
        ]
        river_params_arr: np.ndarray = np.array([])
        if num_river_params > 0:
            river_params_arr = arr[-num_river_params:]

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
            derived_species=derived_species,
        )
