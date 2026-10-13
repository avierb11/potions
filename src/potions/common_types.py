from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    TYPE_CHECKING,
    TypeVar,
    TypedDict,
)

import numpy as np
from numpy import float64 as f64
from pandas import DataFrame, Series

from .core import RtStep

if TYPE_CHECKING:
    from scipy.optimize import OptimizeResult

    from .model import Model
    from .reactive_transport.reaction_network import ReactionNetwork


# ==== Types ==== #
M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)
Vector = np.ndarray[tuple[N], np.dtype[f64]]
Matrix = np.ndarray[tuple[M, N], np.dtype[f64]]


@dataclass(frozen=True)
class ForcingData:
    """Represents the time series of meteorological forcing data for a single location.

    Attributes:
        precip: Time series of precipitation (e.g., mm/day).
        temp: Time series of temperature (e.g., °C).
        pet: Time series of potential evapotranspiration (e.g., mm/day).
    """

    precip: Series
    temp: Series
    pet: Series


@dataclass
class LapseRateParameters:
    temp_factor: float  # The slope of the temperature line
    precip_factor: float  # The slope of the precipitation line

    def scale_temperature(
        self, gauge_elevation: float, elev: float, temp_series: Series
    ) -> Series:
        """Scale the temperature time series according to the elevation"""
        return temp_series + self.temp_factor * (elev - gauge_elevation)

    def scale_precipitation(
        self, gauge_elevation: float, elev: float, precip_series: Series
    ) -> Series:
        """Scale the precipitation according to elevation zones"""
        return precip_series + self.precip_factor * (elev - gauge_elevation)

    def scale_forcing_data(
        self, gauge_elevation: float, elev: float, forcing_data: ForcingData
    ) -> ForcingData:
        """Scale the forcing data according to elevation zones"""
        return ForcingData(
            precip=self.scale_precipitation(gauge_elevation, elev, forcing_data.precip),
            temp=self.scale_temperature(gauge_elevation, elev, forcing_data.temp),
            pet=forcing_data.pet,
        )

    @classmethod
    def default_parameter_range(cls) -> dict[str, tuple[float, float]]:
        return {"temp_factor": (-1, 0), "precip_factor": (0, 10)}

    @classmethod
    def from_dict(cls, params: dict) -> "LapseRateParameters":
        try:
            return cls(**params)
        except TypeError:
            raise TypeError(
                f"Invalid parameters for lapse rate: {params}, expected {list(cls.default_parameter_range().keys())}"
            ) from None


@dataclass(frozen=True)
class ChemicalState:
    """Represents the chemical state of a zone, partitioned by species type."""

    mineral: dict[str, float] = field(default_factory=dict)  # Mineral concentrations
    exchange: dict[str, float] = field(
        default_factory=dict
    )  # Exchange species concentrations
    primary: dict[str, float] = field(
        default_factory=dict
    )  # Primary aqueous species concentrations
    secondary: dict[str, float] = field(
        default_factory=dict
    )  # Secondary species concentrations


@dataclass(frozen=True)
class HydroModelStep:
    """Holds the results of a single time step for the entire model.

    This is an immutable data structure that contains the new states and the
    calculated fluxes for all zones in the model for a single time step.

    Attributes (list order corresponds to the flattened model zones):
        state: A list of the updated states for each zone.
        forc_flux: A list of the forcing fluxes for each zone.
        vap_flux: A list of the vaporization fluxes for each zone.
        lat_flux: A list of the lateral fluxes for each zone.
        vert_flux: A list of the vertical fluxes for each zone.
    """

    state: np.ndarray
    forc_flux: np.ndarray
    vap_flux: np.ndarray
    lat_flux: np.ndarray
    vert_flux: np.ndarray
    q_in: np.ndarray
    lat_flux_ext: np.ndarray
    vert_flux_ext: np.ndarray


@dataclass
class HydroSimulation:
    forcing: list[ForcingData]
    storage: DataFrame
    forc_flux: DataFrame
    lat_flux: DataFrame
    vert_flux: DataFrame
    vap_flux: DataFrame
    q_in: DataFrame
    lat_flux_ext: DataFrame
    vert_flux_ext: DataFrame


@dataclass
class HydroModelResults:
    """A type containing the results of a hydrologic model run.

    Attributes:
        simulation (DataFrame): A DataFrame with time series of states and
            fluxes for all zones, plus simulated and measured streamflow.
        objective_functions (Series): A Series with the values of each of the objective functions as keys
        saved_dir (Optional[str]): The directory this run was saved to
            (`"results/run_<timestamp>"` by default); None if saving was disabled.
    """

    forcing: ForcingData | list[ForcingData]
    simulation: DataFrame
    objective_functions: Series
    saved_dir: Optional[str] = None


@dataclass
class RtModelResults:
    simulation: DataFrame
    objective_functions: Optional[DataFrame]
    rt_forcings: np.ndarray
    steps: list[list[RtStep]]
    other: dict[str, Any]
    saved_dir: Optional[str] = None


@dataclass
class ModelResults:
    hydro: HydroModelResults
    reactive_transport: RtModelResults


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


# ==== Reactive-transport calibration ==== #
@dataclass(frozen=True)
class ObjectiveComponent:
    """One term in a composite calibration objective.

    Attributes:
        metric: The metric name (e.g., ``"nse"``, ``"log_nse"``, ``"kge"``).
            The name is looked up in :mod:`potions.objective_functions`.
        target: The series the metric is evaluated on. One of
            ``"q"`` (streamflow), ``"doc"`` (target solute concentration),
            or ``"doc_load"`` (target solute load, conc * q).
        use_flow_weight: If True, apply a flow-weighted variant of the
            metric, weighting each time step by the measured
            streamflow.
        power: The exponent applied to ``1 - metric_value`` when
            computing the composite objective.
    """

    metric: str
    target: str
    use_flow_weight: bool = False
    power: float = 1.0


@dataclass(frozen=True)
class ObjectiveSpec:
    """Composable specification of the composite calibration objective.

    The composite objective is the sum of ``(1 - value) ** power`` over
    every :class:`ObjectiveComponent`, so a perfect fit yields zero.
    """

    components: tuple[ObjectiveComponent, ...]


@dataclass(frozen=True)
class RtCalibrationData:
    """All of the data required to run and calibrate a reactive-transport
    model.

    Attributes:
        model_type: The :class:`~potions.model.Model` subclass to calibrate.
        network: The reaction network describing species, minerals, and
            exchange processes.
        zone_configs: A mapping of zone name to its
            :class:`RtZoneConfiguration`.
        forc: The precipitation, temperature, and PET forcing data.
        precip_conc: A
            ``(num_surface_zones, n_timesteps, num_species)`` array of
            solute concentrations in the precipitation.
        q_meas: The observed streamflow time series.
        conc_df: A DataFrame with the observed river concentrations; every
            column of :attr:`target_species` is used for the solute-term of
            the objective.
        target_species: The name of the species in the reaction network
            that is matched against :attr:`conc_df`.
        mean_target_conc: The value to initialise every zone's aqueous
            target-species concentration with. If ``None``, the mean of
            ``conc_df[target_species]`` is used.
        init_aq_conc: The initial aqueous concentration (mol/L) given to
            every other primary and secondary species in each zone.
        mineral_conc: The concentration (mol/L) assigned to every mineral
            in every reactive zone. The mineral state is frozen over the
            simulation, so this value only has to be a positive, physically
            sensible constant.
    """

    model_type: type["Model"]
    network: "ReactionNetwork"
    zone_configs: dict[str, RtZoneConfiguration]
    forc: ForcingData
    precip_conc: np.ndarray
    q_meas: Series
    conc_df: DataFrame
    target_species: str = "DOC"
    mean_target_conc: Optional[float] = None
    init_aq_conc: float = 1e-3
    mineral_conc: float = 1.0


@dataclass
class CalibrationOptions:
    """Options for the differential-evolution driver used by
    :func:`~potions.calibration.calibrate`.

    Attribute names map 1-to-1 to :func:`scipy.optimize.differential_evolution`
    arguments.
    """

    maxiter: int = 10
    tol: float = 0.01
    popsize: int = 15
    seed: int = 0
    polish: bool = False
    workers: int = -1
    strategy: Literal["best1bin", "best1exp", "best2bin", "best2exp", "rand1bin", "rand1exp", "rand2bin"] = "best1bin"
    mutation: tuple[float, float] = (0.5, 1.0)
    updating: Literal["immediate", "deferred"] = "deferred"
    atol: float = 0.0
    # callback: Optional[Callable[..., bool]] = None


@dataclass
class CalibrationResults:
    """The outcome of a single calibration run.

    Attributes:
        param_names: The ordered list of calibrated parameter names.
        best_params: The best parameter values, aligned with
            :attr:`param_names`.
        best_results: The full :class:`ModelResults` of the best run.
        objective: The composite objective value (lower is better).
        optimize_result: The raw :class:`scipy.optimize.OptimizeResult`.
    """

    param_names: list[str]
    best_params: np.ndarray
    best_results: ModelResults
    objective: float
    optimize_result: "OptimizeResult"


@dataclass
class EnsembleResults:
    """The outcome of an ensemble of top-ranked parameter sets.

    Attributes:
        member_params: A list of the parameter arrays, one per member.
        member_results: A list of the full :class:`ModelResults`, one per
            member.
        objectives: The composite objective value of each member.
    """

    member_params: list[np.ndarray]
    member_results: list[ModelResults]
    objectives: list[float]
