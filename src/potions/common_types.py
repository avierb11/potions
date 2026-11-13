from dataclasses import dataclass, field
from typing import (
    Callable,
    Final,
    Literal,
    Optional,
    TYPE_CHECKING,
    TypeVar,
    TypedDict,
)

import numpy as np
from numpy import float64 as f64
from pandas import DataFrame, Series

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


@dataclass(frozen=True)
class DerivedSpecies:
    """A measured river quantity defined as a sum of network species.

    Some measurements do not name a single species of the reaction
    network, but are derived from several of them.  Dissolved inorganic
    carbon (DIC), for example, is the sum of the aqueous carbonate
    species (``CO2(aq)``, ``HCO3-``, and ``CO3--``).  Declaring such a
    quantity on the model lets the reactive transport model score a
    measurement column named :attr:`name` against the simulated river
    concentration of the sum of :attr:`components`, exactly as it scores
    a measured network species against its own ``{species}_riv`` column.

    Attributes:
        name: The name of the derived measurement (for example ``"DIC"``).
            Must not collide with the name of a network species.
        components: The names of the network species whose river
            concentrations are summed, in concentration units (for
            example ``("CO2(aq)", "HCO3-", "CO3--")``).  Every component
            must be an aqueous (primary or secondary) species of the
            network.
    """

    name: str
    components: tuple[str, ...]


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


# The per-zone quantities a hydrology frame carries, paired with the
# column prefixes the flat frame uses for them. The layout's single
# source of truth is `HydrologicalModel.get_hydro_column_names`; these
# pairs must track it.
HYDRO_QUANTITY_PREFIXES: Final[tuple[tuple[str, str], ...]] = (
    ("storage", "s_"),
    ("precip_flux", "q_forc_"),
    ("evaporation", "q_vap_"),
    ("lateral", "q_lat_"),
    ("vertical", "q_vert_"),
    ("inflow", "q_in_"),
    ("external_lateral", "q_lat_ext_"),
    ("external_vertical", "q_vert_ext_"),
)
# Must track `hydro_model.SIM_Q_NAME` and the measured-streamflow column
# name used by `run_hydro_model`'s output assembly.
STREAMFLOW_SIM_COLUMN: Final[str] = "streamflow_sim"
MEAS_STREAMFLOW_COLUMN: Final[str] = "meas_streamflow_mmd"
PROP_Q_PREFIX: Final[str] = "prop_q_"


@dataclass(frozen=True)
class HydroZoneResults:
    """One zone's per-quantity views of a hydrology run.

    Each series is a relabeled snapshot of the corresponding
    `{prefix}_{zone}` column of the parent `HydroModelResults.simulation`
    frame, taken when the view is created. This project pins pandas
    >= 3, in which Copy-on-Write is unconditional: a write made through
    this object never reaches the stored frame, and a later change to
    the stored frame is not reflected in an already-taken view. To
    mutate the results, write through `HydroModelResults.to_dataframe()`
    or its `simulation` field and re-access: every accessor access
    returns a fresh snapshot of the stored frame.

    Attributes:
        zone: The name of the zone these series belong to.
        storage: The zone's storage over time (`s_{zone}`).
        precip_flux: The zone's forced precipitation flux
            (`q_forc_{zone}`).
        evaporation: The zone's evaporation flux (`q_vap_{zone}`).
        lateral: The zone's lateral (donor) flux (`q_lat_{zone}`).
        vertical: The zone's vertical flux (`q_vert_{zone}`).
        inflow: The zone's inflow (`q_in_{zone}`).
        external_lateral: The zone's external lateral flux
            (`q_lat_ext_{zone}`).
        external_vertical: The zone's external vertical flux
            (`q_vert_ext_{zone}`).
    """

    zone: str
    storage: Series
    precip_flux: Series
    evaporation: Series
    lateral: Series
    vertical: Series
    inflow: Series
    external_lateral: Series
    external_vertical: Series


@dataclass
class HydroModelResults:
    """A type containing the results of a hydrologic model run.

    The flat `simulation` frame is the sole storage: every named
    accessor is a cheap view over it (a renamed sub-frame or column —
    a fresh snapshot object on each access), and `to_dataframe()` is
    the only accessor that returns the stored frame by identity.

    The per-zone views partition the frame's `{prefix}_{zone}` columns,
    treating zone names as opaque labels. If one zone name were a
    suffix of another zone name after an underscore (e.g. zones `b`
    and `a_b`), the partitioning would be ambiguous; model structures
    should avoid such name pairs.

    Attributes:
        simulation (DataFrame): A DataFrame with time series of states and
            fluxes for all zones, plus simulated and measured streamflow.
        objective_functions (Series): A Series with the values of each of the objective functions as keys
        saved_dir (Optional[str]): The directory this run was saved to
            (`"results/run_<timestamp>"` by default); None if saving was disabled.
        zone_names (list[str]): The model's zone names in layer-major
            (structure) order, recording the frame's zone layout for the
            per-zone accessors. Empty when the results were built by
            hand, in which case the layout is parsed from the frame's
            `s_` storage columns.
    """

    forcing: ForcingData | list[ForcingData]
    simulation: DataFrame
    objective_functions: Series
    saved_dir: Optional[str] = None
    zone_names: list[str] = field(default_factory=list)

    def _effective_zone_names(self) -> list[str]:
        """The zone names to partition the frame by.

        Uses the recorded `zone_names` when present; otherwise parses
        them from the frame's `s_` storage columns. The `s_` prefix is
        used by no other column, so a standard hydrology frame is
        self-describing.

        Returns:
            The zone names in frame order.

        Raises:
            ValueError:
                Raised when no zone names are recorded and the frame
                carries no `s_` storage columns to parse.
        """
        if self.zone_names:
            return self.zone_names
        parsed: list[str] = [
            c[len("s_"):]
            for c in self.simulation.columns
            if c.startswith("s_")
        ]
        if not parsed:
            raise ValueError(
                "The hydrology simulation frame carries no `s_` storage "
                "columns and no zone names are recorded, so the zone "
                "names cannot be determined. Record `zone_names` on the "
                "results or use a frame with the standard per-zone "
                "columns."
            )
        return parsed

    def _quantity_frame(self, quantity: str) -> DataFrame:
        """The frame of one per-zone quantity, relabeled to zone names.

        Args:
            quantity: A quantity name of `HYDRO_QUANTITY_PREFIXES`,
                e.g. `"lateral"`.

        Returns:
            A view of the stored frame with one column per zone: the
            `{prefix}_{zone}` columns, relabeled to the zone names.

        Raises:
            ValueError:
                Raised when a recorded zone name has no corresponding
                `{prefix}_{zone}` column in the frame.
        """
        prefix = dict(HYDRO_QUANTITY_PREFIXES)[quantity]
        zone_names = self._effective_zone_names()
        columns = [f"{prefix}{zone}" for zone in zone_names]
        missing = [c for c in columns if c not in self.simulation.columns]
        if missing:
            raise ValueError(
                f"Column(s) {missing} not found in the hydrology "
                f"simulation frame for quantity '{quantity}'."
            )
        return self.simulation[columns].rename(
            columns={f"{prefix}{zone}": zone for zone in zone_names}
        )

    @property
    def storage(self) -> DataFrame:
        """Zone storage: the `s_{zone}` columns, relabeled to zone names."""
        return self._quantity_frame("storage")

    @property
    def precip_flux(self) -> DataFrame:
        """Forced precipitation per zone: the `q_forc_{zone}` columns."""
        return self._quantity_frame("precip_flux")

    @property
    def evaporation(self) -> DataFrame:
        """Evaporation per zone: the `q_vap_{zone}` columns."""
        return self._quantity_frame("evaporation")

    @property
    def lateral(self) -> DataFrame:
        """Lateral (donor) flux per zone: the `q_lat_{zone}` columns."""
        return self._quantity_frame("lateral")

    @property
    def vertical(self) -> DataFrame:
        """Vertical flux per zone: the `q_vert_{zone}` columns."""
        return self._quantity_frame("vertical")

    @property
    def inflow(self) -> DataFrame:
        """Inflow per zone: the `q_in_{zone}` columns."""
        return self._quantity_frame("inflow")

    @property
    def external_lateral(self) -> DataFrame:
        """External lateral flux per zone: the `q_lat_ext_{zone}` columns."""
        return self._quantity_frame("external_lateral")

    @property
    def external_vertical(self) -> DataFrame:
        """External vertical flux per zone: the `q_vert_ext_{zone}` columns."""
        return self._quantity_frame("external_vertical")

    @property
    def simulated_streamflow(self) -> Series:
        """The simulated streamflow: the `streamflow_sim` column."""
        return self.simulation[STREAMFLOW_SIM_COLUMN]

    @property
    def measured_streamflow(self) -> Series | None:
        """The measured streamflow: the `meas_streamflow_mmd` column.

        Returns:
            The measured series, or None when the frame carries no
            measured-streamflow column (the run was not scored against
            a measured streamflow).
        """
        if MEAS_STREAMFLOW_COLUMN not in self.simulation.columns:
            return None
        return self.simulation[MEAS_STREAMFLOW_COLUMN]

    @property
    def river_zone_fractions(self) -> DataFrame:
        """The area fraction of each river zone: the `prop_q_{zone}`
        columns, relabeled to the zone names.

        The `prop_q_` prefix is unique to these columns, so no zone
        names need be recorded to recover them.
        """
        columns = [
            c
            for c in self.simulation.columns
            if c.startswith(PROP_Q_PREFIX)
        ]
        return self.simulation[columns].rename(
            columns={c: c[len(PROP_Q_PREFIX):] for c in columns}
        )

    def to_dataframe(self) -> DataFrame:
        """The stored flat simulation frame, returned by identity.

        This is the only accessor that returns the stored object
        itself, so it is also the way to mutate the results: a write
        through it (or through the `simulation` field) is what later
        accessor accesses read, because each of those returns a fresh
        snapshot of the stored frame. A write made through a named
        accessor does not reach this frame: this project pins pandas
        >= 3, in which Copy-on-Write makes such writes no-ops (and
        pandas may flag them with a `ChainedAssignmentError`).
        """
        return self.simulation

    def zone(self, name: str) -> HydroZoneResults:
        """One zone's per-quantity series, gathered into a view.

        Args:
            name: The name of the zone to gather.

        Returns:
            A `HydroZoneResults` whose series are renamed views of the
            zone's `{prefix}_{name}` columns.

        Raises:
            ValueError:
                Raised when `name` is not one of the results' zones, or
                when the frame is missing one of the zone's columns.
        """
        zone_names = self._effective_zone_names()
        if name not in zone_names:
            raise ValueError(
                f"Unknown zone '{name}': not a zone of these results. "
                f"Known zones: {zone_names}"
            )
        missing = [
            f"{prefix}{name}"
            for _quantity, prefix in HYDRO_QUANTITY_PREFIXES
            if f"{prefix}{name}" not in self.simulation.columns
        ]
        if missing:
            raise ValueError(
                f"Column(s) {missing} not found in the hydrology "
                f"simulation frame for zone '{name}'."
            )
        # HydroZoneResults' fields follow HYDRO_QUANTITY_PREFIXES' order.
        series: list[Series] = [
            self.simulation[f"{prefix}{name}"].rename(quantity)
            for quantity, prefix in HYDRO_QUANTITY_PREFIXES
        ]
        return HydroZoneResults(name, *series)

    @property
    def zones(self) -> dict[str, HydroZoneResults]:
        """One `HydroZoneResults` view per zone, keyed by zone name."""
        return {zone: self.zone(zone) for zone in self._effective_zone_names()}


@dataclass(frozen=True)
class RtZoneResults:
    """One zone's views of a reactive-transport run.

    ``species`` is a relabeled snapshot of the zone's
    ``{species}_{zone}`` columns — one column per network species,
    including mineral and exchange species. ``mineral_rates`` is a
    relabeled snapshot of the zone's ``{mineral}_rate_{zone}``
    columns, or None when the zone performs no mineral reactions. This
    project pins pandas >= 3, in which Copy-on-Write is unconditional:
    a write made through either frame never reaches the parent
    `RtModelResults.simulation` frame, and a later change to that
    frame is not reflected in an already-taken view. To mutate the
    results, write through `RtModelResults.to_dataframe()` or its
    ``simulation`` field and re-access: every accessor access returns
    a fresh snapshot of the stored frame.
    """

    zone: str
    species: DataFrame
    mineral_rates: DataFrame | None


@dataclass
class RtModelResults:
    """A type containing the results of a reactive transport model run.

    The flat `simulation` frame is the sole storage: every named
    accessor is a cheap view over it (a renamed sub-frame — a fresh
    snapshot object on each access), and `to_dataframe()` is the only
    accessor that returns the stored frame by identity.

    Unlike a hydrology frame, an RT frame is not self-describing: no
    column pattern alone recovers the zone names, so the per-zone and
    per-species accessors require `zone_names` to be recorded. An RT
    model declares an RT zone for every model zone, so these equal the
    hydrology model's zone names in layer-major order. Zone names are
    treated as opaque labels: if one zone name were a suffix of another
    zone name after an underscore (e.g. zones `b` and `a_b`), the
    per-zone partitioning would be ambiguous; model structures should
    avoid such name pairs.

    Attributes:
        simulation (DataFrame): A DataFrame with the time series of each
            zone's species concentrations, the mineral reaction rates of
            every zone in which reactions run (``{mineral}_rate_{zone}``),
            and the simulated river concentrations of every network species
            and declared derived measurement (``{name}_riv``).
        objective_functions (Optional[DataFrame]): One row per measured
            species and one column per objective function; None when the
            run was not scored against measured river concentrations.
        saved_dir (Optional[str]): The directory this run was saved to
            (`"results/run_<timestamp>"` by default); None if saving was
            disabled.
        zone_names (list[str]): The model's zone names in layer-major
            (structure) order — for an RT model these equal the
            hydrology model's zone names. Records the frame's zone
            layout for the per-zone and per-species accessors; empty
            when the results were built by hand or unpickled from a
            file written before this field existed, in which case those
            accessors raise `ValueError`.
    """

    simulation: DataFrame
    objective_functions: Optional[DataFrame]
    saved_dir: Optional[str] = None
    zone_names: list[str] = field(default_factory=list)

    def _effective_zone_names(self) -> list[str]:
        """The recorded zone names, or a diagnostic error.

        An RT frame cannot be parsed for zone names the way a
        hydrology frame can, so a results object that records none
        cannot be partitioned by zone.

        Returns:
            The recorded zone names in frame order.

        Raises:
            ValueError:
                Raised when no zone names are recorded.
        """
        if not self.zone_names:
            raise ValueError(
                "These reactive-transport results do not record zone "
                "names, so the simulation frame cannot be partitioned "
                "by zone. Results of a model run record them "
                "automatically; set `zone_names` when building "
                "RtModelResults by hand."
            )
        return self.zone_names

    def _zone_columns(self, zone: str) -> list[str]:
        """The zone's species columns: `_{zone}`-suffixed, excluding
        `*_rate_{zone}` columns, in frame order.
        """
        suffix = f"_{zone}"
        rate_suffix = f"_rate_{zone}"
        return [
            c
            for c in self.simulation.columns
            if c.endswith(suffix) and not c.endswith(rate_suffix)
        ]

    def _zone_rate_columns(self, zone: str) -> list[str]:
        """The zone's mineral-rate columns: `*_rate_{zone}`, in frame
        order.
        """
        rate_suffix = f"_rate_{zone}"
        return [c for c in self.simulation.columns if c.endswith(rate_suffix)]

    @property
    def river(self) -> DataFrame:
        """The simulated river concentrations: all `*_riv` columns, in
        frame order (one per network species and declared derived
        measurement).

        No zone names are needed: the `_riv` suffix identifies these
        columns unambiguously.
        """
        columns = [c for c in self.simulation.columns if c.endswith("_riv")]
        return self.simulation[columns]

    @property
    def mineral_rates(self) -> DataFrame:
        """All mineral reaction rates: the `*_rate_*` columns, in frame
        order (one per mineral reacting in each reactive zone).
        """
        columns = [c for c in self.simulation.columns if "_rate_" in c]
        return self.simulation[columns]

    def species(self, name: str) -> DataFrame:
        """One species' concentrations over time, one column per zone.

        Args:
            name: The species name; the frame's `{name}_{zone}` columns
                are gathered and relabeled to the zone names.

        Returns:
            A view of the stored frame with one column per recorded
            zone.

        Raises:
            ValueError:
                Raised when no zone names are recorded, or when a
                recorded zone has no `{name}_{zone}` column in the
                frame.
        """
        zone_names = self._effective_zone_names()
        columns = [f"{name}_{zone}" for zone in zone_names]
        missing = [c for c in columns if c not in self.simulation.columns]
        if missing:
            raise ValueError(
                f"Column(s) {missing} not found in the reactive-transport "
                f"simulation frame for species '{name}'."
            )
        return self.simulation[columns].rename(
            columns={f"{name}_{zone}": zone for zone in zone_names}
        )

    def zone(self, name: str) -> RtZoneResults:
        """One zone's species concentrations and mineral rates.

        Args:
            name: The name of the zone to gather.

        Returns:
            An `RtZoneResults` whose `species` frame holds the zone's
            `{species}_{name}` columns relabeled to species names, and
            whose `mineral_rates` frame holds the zone's
            `{mineral}_rate_{name}` columns relabeled to mineral names
            (None when the zone performs no mineral reactions).

        Raises:
            ValueError:
                Raised when no zone names are recorded, when `name` is
                not one of the results' zones, or when the frame
                carries no species columns of the zone.
        """
        zone_names = self._effective_zone_names()
        if name not in zone_names:
            raise ValueError(
                f"Unknown zone '{name}': not a zone of these results. "
                f"Known zones: {zone_names}"
            )
        species_columns = self._zone_columns(name)
        if not species_columns:
            raise ValueError(
                f"No species columns found in the reactive-transport "
                f"simulation frame for zone '{name}'."
            )
        species_frame = self.simulation[species_columns].rename(
            columns={c: c[: -len(f"_{name}")] for c in species_columns}
        )
        rate_columns = self._zone_rate_columns(name)
        if rate_columns:
            mineral_rates: DataFrame | None = (
                self.simulation[rate_columns].rename(
                    columns={
                        c: c[: -len(f"_rate_{name}")] for c in rate_columns
                    }
                )
            )
        else:
            mineral_rates = None
        return RtZoneResults(name, species_frame, mineral_rates)

    @property
    def zones(self) -> dict[str, RtZoneResults]:
        """One `RtZoneResults` view per recorded zone, keyed by zone name."""
        return {zone: self.zone(zone) for zone in self._effective_zone_names()}

    def to_dataframe(self) -> DataFrame:
        """The stored flat simulation frame, returned by identity.

        This is the only accessor that returns the stored object
        itself, so it is also the way to mutate the results: a write
        through it (or through the `simulation` field) is what later
        accessor accesses read, because each of those returns a fresh
        snapshot of the stored frame. A write made through a named
        accessor does not reach this frame: this project pins pandas
        >= 3, in which Copy-on-Write makes such writes no-ops (and
        pandas may flag them with a `ChainedAssignmentError`).
        """
        return self.simulation

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore state from a pickle, tolerating pre-`zone_names` files.

        A results pickle written before `zone_names` existed carries no
        such key; it is defaulted to the empty list (the "not recorded"
        state) so those files still unpickle. Pickles written after the
        field was added carry the key and are restored unchanged.
        """
        state = dict(state)
        state.setdefault("zone_names", [])
        self.__dict__.update(state)


@dataclass
class ModelResults:
    hydro: HydroModelResults
    reactive_transport: RtModelResults

    @property
    def rt(self) -> RtModelResults:
        """Alias for :attr:`reactive_transport`, matching the attribute
        name used by ``gui.data.GuiRunResults``.

        Returns the reactive-transport results, which may be
        ``None``-substituted by callers that hold a hydro-only run in
        this field's slot.
        """
        return self.reactive_transport


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
