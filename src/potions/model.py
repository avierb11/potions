from __future__ import annotations
import datetime
from typing import Final, Iterator, Optional, overload, Any, Generic, TypeVar
from functools import reduce
import operator
from dataclasses import dataclass
from warnings import deprecated
import numpy as np
from numpy import float64 as f64
from numpy.typing import NDArray, ArrayLike
from pandas import DataFrame, Index, Series

from .reactive_transport import ReactiveTransportZone
from .interfaces import Zone, StateType, ForcingType
from .hydro import HydroForcing, HydrologicZone  # Still needed for run_hydro_model


# Define a TypeVar for Zones to make Layer, Hillslope, and Model generic
ZoneType = TypeVar("ZoneType", bound=Zone)


@dataclass(frozen=True)
class ForcingData:
    """Represents the time series of meteorological forcing data for a single location.

    Attributes:
        precip: Time series of precipitation (e.g., mm/day).
        temp: Time series of temperature (e.g., °C).
        pet: Time series of potential evapotranspiration (e.g., mm/day).
    """

    precip: Series[f64]
    temp: Series[f64]
    pet: Series[f64]


class Layer:
    """A horizontal collection of computational zones.

    A Layer represents a set of zones that are at the same vertical level
    within a hillslope. It can be initialized with a variable number of Zone
    objects or a list of them.
    """

    @overload
    def __init__(self, *zones: HydrologicZone) -> None:
        """Initializes a Layer with a variable number of Zone objects."""
        ...

    @overload
    def __init__(self, zones: list[HydrologicZone]) -> None:
        """Initializes a Layer with a list of Zone objects."""
        ...

    def __init__(self, *args: Any) -> None:  # type: ignore
        if len(args) == 1 and isinstance(args[0], list):
            self.__zones: list[HydrologicZone] = args[0]
        else:
            self.__zones = list(args)

    @property
    def zones(self) -> list[HydrologicZone]:
        """The list of zones contained within this layer."""
        return self.__zones

    def __iter__(self) -> Iterator[HydrologicZone]:
        return iter(self.zones)

    def __len__(self) -> int:
        return len(self.zones)

    def __getitem__(self, ind: int) -> Optional[HydrologicZone]:
        if 0 <= ind < len(self.zones):
            return self.__zones[ind]
        else:
            return None


@dataclass(frozen=True)
class Hillslope:
    """A vertical stack of Layers, representing a single landscape unit.

    A Hillslope is a fundamental structural component of a model, composed of
    one or more layers.

    Attributes:
        layers: A list of Layer objects, ordered from top to bottom.
    """

    layers: list[Layer]

    def __iter__(self) -> Iterator[Layer]:
        return iter(self.layers)

    def __len__(self) -> int:
        return reduce(operator.add, map(len, self.layers), 0)

    def __getitem__(self, ind: int) -> Optional[Layer]:
        if 0 <= ind < len(self.layers):
            return self.layers[ind]
        else:
            return None

    def flatten(self) -> list[HydrologicZone]:
        """Flattens the hillslope structure into a single list of zones.

        The zones are ordered from top layer to bottom layer, and within each
        layer, from left to right. This sequential list is used by the model
        engine for processing.

        Returns:
            A list of all zones in the hillslope.
        """
        return reduce(operator.add, map(lambda x: x.zones, self.layers), [])


@dataclass(frozen=True)
class ModelStep(Generic[StateType]):
    """Holds the results of a single time step for the entire model.

    This is an immutable data structure that contains the new states and the
    calculated fluxes for all zones in the model over a single time step (`dt`).

    Attributes:
        state: A list of the updated states for each zone.
        forc_flux: A list of the forcing fluxes for each zone.
        vap_flux: A list of the vaporization fluxes for each zone.
        lat_flux: A list of the lateral fluxes for each zone.
        vert_flux: A list of the vertical fluxes for each zone.
    """

    state: list[StateType]
    forc_flux: list[StateType]
    vap_flux: list[StateType]
    lat_flux: list[StateType]
    vert_flux: list[StateType]


class Model(Generic[ZoneType]):
    """The main model engine that orchestrates the simulation.

    This is a generic class that can manage and run simulations for any type of
    computational zone (`ZoneType`) that adheres to the `Zone` interface. It
    constructs a model from a collection of hillslopes, calculates the
    connectivity between all zones, and provides a `step` method to advance
    the simulation in time.
    """

    def __init__(
        self,
        hillslopes: list[Hillslope],
        scales: list[list[float]],
        verbose: bool = False,
    ) -> None:
        """Initializes the model engine.

        Args:
            hillslopes: A list of `Hillslope` objects that make up the model structure.
            scales: A nested list defining the relative area of each hillslope
                and each forcing source within that hillslope.
        """
        self.verbose: bool = verbose
        self.__hillslopes: list[Hillslope] = hillslopes
        self.__scales: list[list[float]] = scales
        # self.__flat_model: list[AnnotatedZone] = self.flatten(
        #     scales
        # )  # Model in linear order to be evaluated
        self.__flat_model: list[ZoneType] = self.flatten()
        self.__size: int = reduce(operator.add, map(len, self.hillslopes), 0)
        self.__lat_matrix: NDArray[f64] = self.get_lat_mat()
        self.__vert_matrix: NDArray[f64] = self.get_vert_mat()
        flat_scales: list[float] = [item for sublist in scales for item in sublist]
        self.flat_scales: list[float] = flat_scales
        if verbose:
            print(f"Flattened scales: {flat_scales}")
        self.__forcing_mat: NDArray[f64] = self.get_forc_mat(flat_scales)
        self.__forcing_rel_mat: NDArray[f64] = self.get_forc_mat(
            flat_scales, relative=True
        )

    @property
    def lat_mat(self) -> NDArray[f64]:
        """The matrix describing lateral connectivity between zones."""
        return self.__lat_matrix

    @property
    def vert_mat(self) -> NDArray[f64]:
        """The matrix describing vertical connectivity between zones."""
        return self.__vert_matrix

    @property
    def precip_mat(self) -> NDArray[f64]:
        """The matrix distributing precipitation forcing to each zone."""
        return self.__forcing_mat

    @property
    def pet_mat(self) -> NDArray[f64]:
        """The matrix distributing PET forcing to each zone."""
        return self.__forcing_mat

    @property
    def temp_mat(self) -> NDArray[f64]:
        """The matrix distributing temperature forcing to each zone."""
        return self.__forcing_rel_mat

    @property
    def hillslopes(self) -> list[Hillslope]:
        """The list of `Hillslope` objects that define the model structure."""
        return self.__hillslopes

    @property
    def scales(self) -> list[list[float]]:
        """The nested list of relative areas for hillslopes and forcing sources."""
        return self.__scales

    def __len__(self) -> int:
        return self.__size

    def __iter__(self) -> Iterator[Hillslope]:
        return iter(self.hillslopes)

    def flatten_old(self, scales: list[list[float]]) -> list[AnnotatedZone]:
        """(Deprecated) Flattens the model structure."""
        positions: list[ZonePosition] = []
        cur_zone: int = 0

        hillslope_id: int
        hillslope: Hillslope
        for hillslope_id, hillslope in enumerate(self.hillslopes):
            layer_id: int
            layer: Layer
            for layer_id, layer in enumerate(hillslope.layers):
                zone_id: int
                # _zone: HydrologicZone
                for zone_id, _zone in enumerate(layer.zones):
                    positions.append(
                        ZonePosition(cur_zone, zone_id, layer_id, hillslope_id)
                    )
                    cur_zone += 1

        # raise NotImplementedError()
        return NotImplemented

    def flatten(self) -> list[ZoneType]:
        """Flattens the model structure into a single list of zones."""
        return reduce(operator.add, map(lambda x: x.flatten(), self.hillslopes), [])

    @property
    def flat_model(self) -> list[ZoneType]:
        """A 1D list of all zones in the model, in the order of evaluation."""
        return self.__flat_model

    def step(
        self, state: list[StateType], ds: list[ForcingType], dt: float
    ) -> ModelStep[StateType]:
        """Step the model given the current state, forcing, and the time step."""
        new_states: list[StateType] = []
        forc_fluxes: list[StateType] = []
        vap_fluxes: list[StateType] = []
        lat_fluxes: list[StateType] = []
        vert_fluxes: list[StateType] = []

        # Determine the 'zero' value for fluxes based on the state type
        zero_flux: Any = 0.0
        if state and not isinstance(state[0], float):
            # If state is a list of arrays, create a zero-array template
            zero_flux = np.zeros_like(state[0])

        i: int
        zone: ZoneType
        s_i: StateType
        d_i: ForcingType
        for i, (zone, (s_i, d_i)) in enumerate(zip(self.flat_model, zip(state, ds))):
            # Calculate incoming flux from previously stepped zones.
            # This works for both floats and numpy arrays.
            q_in_lat = (
                sum(self.lat_mat[i, j] * flux_j for j, flux_j in enumerate(lat_fluxes))
                if lat_fluxes
                else zero_flux
            )
            q_in_vert = (
                sum(
                    self.vert_mat[i, j] * flux_j for j, flux_j in enumerate(vert_fluxes)
                )
                if vert_fluxes
                else zero_flux
            )
            q_in = q_in_lat + q_in_vert

            s_i = float(s_i)  # type: ignore
            if not isinstance(s_i, float):
                raise RuntimeError(f"State must be float, not {type(s_i)}")
            step_res: ModelStep = zone.step(s_i, d_i, dt, q_in)  # type: ignore

            new_states.append(step_res.state)  # type: ignore
            forc_fluxes.append(step_res.forc_flux)  # type: ignore
            vap_fluxes.append(step_res.vap_flux)  # type: ignore
            lat_fluxes.append(step_res.lat_flux)  # type: ignore
            vert_fluxes.append(step_res.vert_flux)  # type: ignore

        return ModelStep(
            state=new_states,
            forc_flux=forc_fluxes,
            vap_flux=vap_fluxes,
            lat_flux=lat_fluxes,
            vert_flux=vert_fluxes,
        )

    def get_hillslope_vert_mat(self, hs: Hillslope) -> NDArray[f64]:
        """Calculates the vertical connectivity matrix for a single hillslope.

        Args:
            hs: The hillslope for which to calculate the matrix.

        Returns:
            A square matrix where `mat[i, j] = 1` if zone `j` flows vertically
            into zone `i`.
        """
        n: Final[int] = len(hs)
        mat: NDArray[f64] = np.zeros((n, n), dtype=float)
        cz: int = 0  # The current zone

        if len(hs) == 0:
            raise ValueError("Hillslope must have at least one layer")

        rect_domain: NDArray = np.zeros((len(hs.layers), len(hs[0])), dtype=int)  # type: ignore
        for i, ly in enumerate(hs):
            if len(ly) == 1:
                rect_domain[i, :] = cz
                cz += 1
            elif len(ly) == rect_domain.shape[1]:
                for j, _ in enumerate(ly):
                    rect_domain[i, j] = cz
                    cz += 1
            else:
                raise ValueError("Invalid model structure encountered")

        cz = 0
        layer: Layer
        for i, layer in enumerate(hs):
            match hs[i + 1]:
                case Layer():
                    for j, _ in enumerate(layer):
                        mat[cz, rect_domain[i + 1, j]] = 1.0
                        cz += 1
                case None:
                    cz += len(layer)
                    continue
                case _:
                    raise ValueError("Invalid model structure")

        return mat.T

    def get_vert_mat(self) -> NDArray[f64]:
        """Assembles the block-diagonal vertical connectivity matrix for the entire model.

        Returns:
            The full vertical connectivity matrix for the model.
        """
        hs_blocks: list[NDArray[f64]] = [self.get_hillslope_vert_mat(hs) for hs in self]
        model_dim = len(self)
        mat: NDArray[f64] = np.zeros((model_dim, model_dim))

        cur: int = 0
        for b in hs_blocks:
            mat[cur : cur + b.shape[0], cur : cur + b.shape[0]] = b
            cur += b.shape[0]

        return mat

    def get_hillslope_lat_mat(self, hs: Hillslope) -> NDArray[f64]:
        """Calculates the lateral connectivity matrix for a single hillslope.

        Args:
            hs: The hillslope for which to calculate the matrix.

        Returns:
            A square matrix where `mat[i, j] = 1` if zone `j` flows laterally
            into zone `i`.
        """
        n: Final[int] = len(hs)
        mat: NDArray[f64] = np.zeros((n, n), dtype=float)
        cz: int = 0  # The current zone

        if len(hs) == 0:
            raise ValueError("Hillslope must have at least one layer")

        rect_domain: NDArray = np.zeros((len(hs.layers), len(hs[0])), dtype=int)  # type: ignore
        for i, ly in enumerate(hs):
            if len(ly) == 1:
                rect_domain[i, :] = cz
                cz += 1
            elif len(ly) == rect_domain.shape[1]:
                for j, _ in enumerate(ly):
                    rect_domain[i, j] = cz
                    cz += 1
            else:
                raise ValueError("Invalid model structure encountered")

        cz = 0
        layer: Layer
        for i, layer in enumerate(hs):
            # For each zone in the layer, check if it has a neighbor to the right
            for j, zone in enumerate(layer):
                if layer[j + 1] is not None:  # Check for neighbor
                    mat[cz, rect_domain[i, j + 1]] = 1.0
                cz += 1

        return mat.T

    def get_lat_mat(self) -> NDArray[f64]:
        """Assembles the block-diagonal lateral connectivity matrix for the entire model.

        Returns:
            The full lateral connectivity matrix for the model.
        """
        hs_blocks: list[NDArray[f64]] = [self.get_hillslope_lat_mat(hs) for hs in self]
        model_dim = len(self)
        mat: NDArray[f64] = np.zeros((model_dim, model_dim))

        cur: int = 0
        for b in hs_blocks:
            mat[cur : cur + b.shape[0], cur : cur + b.shape[0]] = b
            cur += b.shape[0]

        return mat

    def get_size_mat(self) -> NDArray[f64]:
        """Calculates a matrix mapping surface zones to their contributing areas.

        This matrix is used to determine how forcing data applied to a surface
        zone (or forcing source) is distributed to all zones beneath it.

        Returns:
            A matrix mapping forcing sources to zones.
        """
        vert: NDArray[f64] = self.get_vert_mat()
        index_rows: list[int] = [
            i for i, row in enumerate(vert) if row.sum() < 1e-12
        ]  # The zone indices of the surface zones. They have no vertical zones above
        composite_rows: list[int] = list(
            set((i for i in range(vert.shape[0]))).difference(index_rows)
        )

        mat: NDArray[f64] = np.zeros((vert.shape[0], len(index_rows)), dtype=float)
        for i, x_i in enumerate(index_rows):
            mat[x_i, i] = 1

        for c_i in composite_rows:
            mat[c_i] = sum([mat[i] for i, row_i in enumerate(vert[c_i]) if row_i != 0])

        return mat

    def get_forc_mat(self, sizes: ArrayLike, relative: bool = False) -> NDArray[f64]:
        """Calculates the final forcing distribution matrix.

        This matrix combines the connectivity information with the relative
        area of each forcing source to create a final matrix that can be used
        to distribute forcing data (e.g., precipitation) to every zone.

        Args:
            sizes: An array-like object with the relative area of each forcing source.
            relative: If True, normalizes the matrix rows to sum to 1. This is
                used for intensive variables like temperature.
        """
        conn_mat: NDArray[f64] = self.get_size_mat()
        s_arr: NDArray[f64] = np.array(sizes)
        if self.verbose:
            print(f"Connection matrix: {conn_mat}")
            print(f"Sizes array: {s_arr}")
        mat: NDArray[f64] = conn_mat @ np.diag(s_arr)

        if relative:
            row_sums = mat.sum(axis=1)
            # Initialize relative matrix with zeros
            relative_mat = np.zeros_like(mat, dtype=f64)
            # Create a mask for rows where sum is not zero to avoid division by zero
            non_zero_sum_rows = row_sums != 0
            # Perform division only for rows with non-zero sum
            relative_mat[non_zero_sum_rows] = (
                mat[non_zero_sum_rows] / row_sums[non_zero_sum_rows, np.newaxis]
            )
            return relative_mat
        else:
            return mat

    @property
    def column_names(self) -> list[str]:
        """A list of column names for the final output DataFrame."""
        names: list[str] = []
        for i, zone in enumerate(self.flat_model):
            names += zone.columns(i)

        return names


@dataclass(frozen=True)
class ZonePosition:
    """Represents the unique position of a zone within the model's structure.

    Attributes:
        model_id: The global index of the zone in the flattened model.
        zone_id: The index of the zone within its layer (laterally).
        layer_id: The index of the layer within its hillslope (vertically).
        hillslope_id: The index of the hillslope within the model.
    """

    model_id: int
    zone_id: int
    layer_id: int
    hillslope_id: int


@dataclass(frozen=True)
class AnnotatedZone:
    """A wrapper class that holds a zone and its associated metadata. (Deprecated)

    Attributes:
        zone: The hydrologic zone object.
        size: The proportion of the total catchment area this zone represents.
        pos: The `ZonePosition` of this zone in the model.
        incoming_fluxes: A list of model_ids for zones that flow into this one.
    """

    zone: HydrologicZone
    size: float
    pos: ZonePosition
    incoming_fluxes: list[int]


# @dataclass
# class HydroModelResults:
#     """A container for the results of a hydrologic model run.

#     Attributes:
#         state: A DataFrame containing the time series of states for all zones.
#         fluxes: A DataFrame containing the time series of fluxes for all zones.
#     """

#     state: DataFrame
#     fluxes: DataFrame


@deprecated("Uses the older forcing data format.")
def run_hydro_model_older(
    model: Model[HydrologicZone],  # type: ignore
    init_state: NDArray[np.float64],
    forc: list[list[HydroForcing]],
    dates: Series[datetime.date],
    dt: float,
) -> DataFrame:
    """Runs a complete hydrologic simulation using a legacy forcing format.

    This function iterates through time, calling the model's `step` method at
    each interval and collecting the results into a pandas DataFrame.

    Args:
        model: The configured `Model[HydrologicZone]` instance to run.
        init_state: An array of initial storage values for each zone.
        forc: A nested list of `HydroForcing` objects. The outer list represents
            time steps, and the inner list corresponds to each zone.
        dates: A pandas Series of dates for the output DataFrame index.
        dt: The time step duration in days.

    Returns:
        A pandas DataFrame containing the time series of states and fluxes for
        all zones in the model.
    """
    num_steps: Final[int] = len(forc)  # Number of steps
    num_zones: Final[int] = len(model)
    storages: NDArray[np.float64] = np.full(
        (num_steps, num_zones), fill_value=np.nan, dtype=float
    )
    fluxes: NDArray[np.float64] = np.full(
        (num_steps, num_zones, 4), fill_value=np.nan, dtype=float
    )

    state: NDArray[np.float64] = init_state

    for i, forc_i in enumerate(forc):
        try:
            # Convert state array to list for the generic step method
            step_res: ModelStep = model.step(list(state), forc_i, dt)
            storages[i] = np.array(step_res.state)
            fluxes[i, :, 0] = np.array(step_res.forc_flux)
            fluxes[i, :, 1] = np.array(step_res.vap_flux)
            fluxes[i, :, 2] = np.array(step_res.lat_flux)
            fluxes[i, :, 3] = np.array(step_res.vert_flux)
            # Convert state back to array for the next iteration
            state = np.array(step_res.state)
        except ValueError as e:
            print(e)
            print(f"Failed on step {i}, returning early")
            break

    full_array: NDArray[f64] = np.full(
        (num_steps, 5 * num_zones), fill_value=np.nan, dtype=float
    )
    for i in range(num_zones):
        full_array[:, 5 * i] = storages[:, i]
        full_array[:, 5 * i + 1] = fluxes[:, i, 0]  # Forcing
        full_array[:, 5 * i + 2] = fluxes[:, i, 1]  # Vaporization
        full_array[:, 5 * i + 3] = fluxes[:, i, 2]  # Lateral
        full_array[:, 5 * i + 4] = fluxes[:, i, 3]  # Vertical

    col_names: list[str] = model.column_names

    out_df: DataFrame = DataFrame(data=full_array, index=dates, columns=col_names)
    return out_df


def run_hydro_model(
    model: Model[HydrologicZone],  # type: ignore
    init_state: NDArray[f64],
    forc: list[ForcingData],
    dates: Series[datetime.date] | Index[datetime.date],
    dt: float,
) -> DataFrame:
    """Runs a complete hydrologic simulation.

    This function prepares the forcing data by distributing it from sources to
    individual zones based on the model's connectivity matrices. It then
    iterates through time, calling the model's `step` method at each interval
    and collecting the results into a pandas DataFrame.

    Args:
        model: The configured `Model[HydrologicZone]` instance to run.
        init_state: An array of initial storage values for each zone.
        forc: A list of `ForcingData` objects, one for each forcing source
            defined in the model's scale parameters.
        dates: A pandas Series of dates for the output DataFrame index.
        dt: The time step duration in days.

    Returns:
        A pandas DataFrame containing the time series of states and fluxes for
        all zones in the model.

    Raises:
        ValueError: If the number of provided `ForcingData` objects does not
            match what the model expects, or if their time series lengths are
            inconsistent.
    """
    num_steps: Final[int] = len(dates)  # Number of steps
    num_zones: Final[int] = len(model)
    num_forcing_sources_expected: Final[int] = model.precip_mat.shape[1]

    all_precip_sources_matrix: NDArray[f64]
    all_temp_sources_matrix: NDArray[f64]
    all_pet_sources_matrix: NDArray[f64]

    # Validate and prepare source forcing matrices
    if forc:
        if len(forc) != num_forcing_sources_expected:
            raise ValueError(
                f"Model expects {num_forcing_sources_expected} ForcingData objects, "
                f"but {len(forc)} were provided in the 'forc' list."
            )
        fd: ForcingData
        for i, fd in enumerate(forc):
            if not (
                len(fd.precip) == num_steps
                and len(fd.temp) == num_steps
                and len(fd.pet) == num_steps
            ):
                raise ValueError(
                    f"ForcingData at index {i} has series lengths inconsistent with num_steps ({num_steps}). "
                    f"P length: {len(fd.precip)}, T length: {len(fd.temp)}, PET length: {len(fd.pet)}"
                )
        all_precip_sources_matrix: NDArray[f64] = np.vstack(
            [fd.precip.to_numpy() for fd in forc]
        ).T  # type: ignore
        all_temp_sources_matrix: NDArray[f64] = np.vstack(
            [fd.temp.to_numpy() for fd in forc]
        ).T  # type: ignore
        all_pet_sources_matrix: NDArray[f64] = np.vstack(
            [fd.pet.to_numpy() for fd in forc]
        ).T  # type: ignore
    else:  # forc is empty
        if num_forcing_sources_expected > 0:
            raise ValueError(
                f"Model expects {num_forcing_sources_expected} forcing inputs, but 'forc' list is empty."
            )
        # If no forcing sources are expected, create empty (0-column) matrices
        all_precip_sources_matrix = np.zeros((num_steps, 0), dtype=f64)
        all_temp_sources_matrix = np.zeros((num_steps, 0), dtype=f64)
        all_pet_sources_matrix = np.zeros((num_steps, 0), dtype=f64)

    # Distribute source forcings to each zone for all time steps
    # Resulting shape for each: (num_steps, num_zones)
    zone_precip_series = all_precip_sources_matrix @ model.precip_mat.T
    zone_temp_series = all_temp_sources_matrix @ model.temp_mat.T
    zone_pet_series = all_pet_sources_matrix @ model.pet_mat.T

    storages: NDArray[f64] = np.full(
        (num_steps, num_zones), fill_value=np.nan, dtype=float
    )
    fluxes: NDArray[f64] = np.full(
        (num_steps, num_zones, 4), fill_value=np.nan, dtype=float
    )

    state: NDArray[f64] = init_state

    for t_idx in range(num_steps):
        # Forcings for the current time step, one HydroForcing object per zone
        ds_for_step: list[HydroForcing] = [
            HydroForcing(
                precip=zone_precip_series[t_idx, j],
                temp=zone_temp_series[t_idx, j],
                pet=zone_pet_series[t_idx, j],
            )
            for j in range(num_zones)
        ]

        try:
            # Convert state array to list for the generic step method
            step_res: ModelStep = model.step(list(state), ds_for_step, dt)
            storages[t_idx] = np.array(step_res.state)
            fluxes[t_idx, :, 0] = np.array(step_res.forc_flux)
            fluxes[t_idx, :, 1] = np.array(step_res.vap_flux)
            fluxes[t_idx, :, 2] = np.array(step_res.lat_flux)
            fluxes[t_idx, :, 3] = np.array(step_res.vert_flux)
            # Convert state back to array for the next iteration
            state = np.array(step_res.state)
        except ValueError as e:
            # print(f"Failed on step {t_idx}, returning early")
            # break
            raise e

    full_array: NDArray[f64] = np.full(
        (num_steps, 5 * num_zones), fill_value=np.nan, dtype=float
    )
    for i in range(num_zones):
        full_array[:, 5 * i] = storages[:, i]
        full_array[:, 5 * i + 1] = fluxes[:, i, 0]  # Forcing
        full_array[:, 5 * i + 2] = fluxes[:, i, 1]  # Vaporization
        full_array[:, 5 * i + 3] = fluxes[:, i, 2]  # Lateral
        full_array[:, 5 * i + 4] = fluxes[:, i, 3]  # Vertical

    col_names: list[str] = model.column_names

    out_df: DataFrame = DataFrame(data=full_array, index=dates, columns=col_names)
    return out_df


def run_reactive_transport_model(
    model: Model[ReactiveTransportZone],  # type: ignore
    init_state: NDArray[f64],
    hydro_results: HydroModelResults,
    rt_forcing: list[
        ForcingData
    ],  # Assuming ForcingData contains solute concentrations
    dates: Series[datetime.date],
    dt: float,
) -> DataFrame:
    """Runs a complete reactive transport simulation.

    This function simulates the movement and reaction of chemical species through
    the domain defined by the model. It uses the results from a prior
    hydrologic model run to define the flow paths and water volumes.

    Args:
        model: The configured `Model[ReactiveTransportZone]` instance to run.
        init_state: A 2D array of initial chemical concentrations for each zone
            and each species, with shape (num_zones, num_species).
        hydro_results: A `HydroModelResults` object containing the time series
            of states (e.g., storage) and fluxes from the hydrologic model run.
        rt_forcing: A list of `ForcingData` objects, one for each forcing
            source, containing the time series of solute concentrations in
            precipitation or other inputs.
        dates: A pandas Series of dates for the output DataFrame index.
        dt: The time step duration in days.

    Returns:
        A pandas DataFrame containing the time series of concentrations for
        all species in all zones.
    """
    # 1. Get dimensions and validate inputs
    # num_steps = len(dates)
    # num_zones = len(model)
    # num_species = init_state.shape[1]

    # 2. Prepare chemical forcing data
    # Similar to run_hydro_model, distribute the chemical forcing (e.g., solute
    # concentrations in rain) from the sources to each individual zone for all
    # time steps. This will create a `zone_concentration_series` array with
    # shape (num_steps, num_zones, num_species).

    # 3. Initialize storage for results
    # concentrations = np.full((num_steps, num_zones, num_species), np.nan)

    # 4. Set initial state
    # state = init_state  # Shape: (num_zones, num_species)

    # 5. Loop over each time step
    # for t_idx in range(num_steps):
    #     # a. Extract hydrologic data for the current step from hydro_results
    #     #    - Get water volume/storage for each zone.
    #     #    - Get water fluxes (lateral, vertical, forcing) for each zone.

    #     # b. Construct the forcing object (`ds`) for each zone for the current step
    #     #    This will be a list of `RtForcing` objects, one for each zone.
    #     #    Each `RtForcing` object needs to be populated with the hydrologic
    #     #    data from (a) and the chemical forcing data from step 2.
    #     ds_for_step: list[RtForcing] = []

    #     # c. Step the reactive transport model
    #     #    The `state` here is a list of 1D arrays, where each array holds
    #     #    the concentrations for one zone.
    #     # step_res: ModelStep = model.step(list(state), ds_for_step, dt)

    #     # d. Store the new concentrations from step_res
    #     # concentrations[t_idx] = np.array(step_res.state)

    #     # e. Update the state for the next iteration
    #     # state = np.array(step_res.state)

    # 6. Format and return results
    #    - Create a pandas DataFrame from the `concentrations` array.
    #    - Name the columns appropriately to identify the zone and species
    #      (e.g., "z0_Cl", "z0_Na", "z1_Cl", etc.).
    #    - Set the DataFrame index to `dates`.
    #    - Return the DataFrame.

    raise NotImplementedError("Outline complete. Implementation pending.")
