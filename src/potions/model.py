from __future__ import annotations
from abc import ABC
import datetime
import itertools
from multiprocessing import Pool
import os
import threading
from typing import (
    Callable,
    Final,
    Iterable,
    Iterator,
    Literal,
    Optional,
    TypedDict,
    overload,
    Any,
    Generic,
    TypeVar,
)
from functools import reduce
import operator
from dataclasses import dataclass
import warnings
import networkx as nx
import numpy as np
from numpy import float64 as f64
from numpy.typing import NDArray, ArrayLike
from pandas import DataFrame, Index, Series, Timestamp
import scipy.optimize as opt
import xarray as xr

from .common_types import ForcingData, LapseRateParameters, HydroModelResults
from .objective_functions import kge, nse
from .reactive_transport import ReactiveTransportZone
from .utils import objective_function
from .interfaces import Zone, StateType
from .hydro import (
    GroundZone,
    GroundZoneB,
    GroundZoneLinear,
    GroundZoneLinearB,
    HydroForcing,
    HydroStep,
    HydrologicZone,
    SnowZone,
    SoilZone,
)  # Still needed for run_hydro_model


# Define a TypeVar for Zones to make Layer, Hillslope, and Model generic
ZoneType = TypeVar("ZoneType", bound=Zone)
NETCDF_LOCK = threading.Lock()


class BatchParams(TypedDict):
    output_dir: Optional[str]
    threshold_function: Optional[Callable[[HydroModelResults], bool]]
    return_results: bool
    save_results: bool


def run_model(
    cls: type[Model],
    i: int,
    params: Series,
    forc: ForcingData | list[ForcingData],
    init_state: Optional[NDArray[f64]],
    meas_streamflow: Optional[Series],
    batch_params: BatchParams,
) -> tuple[int, Optional[HydroModelResults], dict]:
    model = cls.from_series(params)
    run_res = model.run(
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
            run_res.simulation.to_csv(
                os.path.join(batch_params["output_dir"], f"{i}.csv")
            )

    if batch_params["return_results"]:
        return i, run_res, run_res.to_results_dict()
    else:
        return i, None, run_res.to_results_dict()


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


class Model(ABC):
    """The main model engine that runs the simulation.

    This is a generic class that can manage and run simulations for any type of
    computational zone (`ZoneType`) that adheres to the `Zone` interface. It
    constructs a model from a collection of hillslopes, calculates the
    connectivity between all zones, and provides a `step` method to advance
    the simulation in time.
    """

    structure: list[list[HydrologicZone]] = []

    def __init__(
        self,
        zones: Optional[dict[str, HydrologicZone]] = None,
        scales: Optional[list[float]] = None,
        lapse_rates: Optional[list[LapseRateParameters]] = None,
        verbose: bool = False,
    ) -> None:
        """Initializes the model engine.

        Args:
            layers: A list of `Layer` objects that make up the model structure.
            scales: A nested list defining the relative area of each hillslope
                and each forcing source within that hillslope.
            lapse_rates: A list of `LapseRateParameters` objects for each of the zones
        """
        # Check for empty values
        if scales is None:
            # Get the default
            num_zones: int = len(self.structure[0])
            scales = [1 / num_zones for _ in range(num_zones)]
        if lapse_rates is None:
            lapse_rates = []
        if zones is None:
            zones = {}

        # Construct the zone dictionary with the zones
        self.__zones: dict[str, HydrologicZone] = {}
        for layer in self.structure:
            for zone in layer:
                self.__zones[zone.name] = zone.default()

        for zone_name, zone in zones.items():
            if zone_name not in self.get_zone_names():
                raise ValueError(
                    f"Unknown zone name: {zone_name}. The zone name must be one of {self.get_zone_names()}"
                )
            else:
                self.__zones[zone_name] = zone

        # Construct the linear model
        self.lapse_rates: list[LapseRateParameters] = lapse_rates
        layers: list[Layer] = []
        for layer in self.structure:
            layer_vals: list[HydrologicZone] = []
            for zone in layer:
                layer_vals.append(self.__zones[zone.name])

            layers.append(Layer(layer_vals))

        self.verbose: bool = verbose
        self.__layers: list[Layer] = layers
        self.__scales: list[float] = scales
        self.__flat_model: list[HydrologicZone] = self.flatten()
        self.__size: int = reduce(operator.add, map(len, self.layers), 0)

        # Calculate the connectivity matrices
        self.__zone_graph: nx.DiGraph = self.construct_hydrologic_graph()
        lat, vert, _ = self.get_connection_matrices_with_river_row(self.__zone_graph)
        self.__lat_matrix: NDArray[f64] = lat
        self.__vert_matrix: NDArray[f64] = vert
        self.__forcing_mat: NDArray[f64] = self.get_forc_mat(scales)
        self.__forcing_rel_mat: NDArray[f64] = self.get_forc_mat(scales, relative=True)

    def __getitem__(self, zone_name: str) -> HydrologicZone:
        """
        Access a hydrologic zone with a given name.
        """
        if zone_name in self.get_zone_names():
            return self.__zones[zone_name]
        else:
            raise ValueError(
                f"Unknown zone name: {zone_name}, must be one of {self.get_zone_names()}"
            )

    @classmethod
    def get_zone_names(cls) -> list[str]:
        """
        The names of each of the hydrologic zones in the model
        """

        zone_names: list[str] = []
        for layer in cls.structure:
            for zone in layer:
                zone_names.append(zone.name)
        return zone_names

    @property
    def num_surface_zones(self) -> int:
        return len(self.structure[0])

    @property
    def graph(self) -> nx.DiGraph:
        return self.__zone_graph

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
    def layers(self) -> list[Layer]:
        """The list of `Layer` objects that define the model structure."""
        return self.__layers

    @property
    def scales(self) -> list[float]:
        """The nested list of relative areas for hillslopes and forcing sources."""
        return self.__scales

    def __len__(self) -> int:
        return len(self.layers)

    def __iter__(self) -> Iterator[Layer]:
        return iter(self.layers)

    def flatten(self) -> list[HydrologicZone]:
        """Flattens the model structure into a single list of zones."""
        return reduce(operator.add, map(lambda x: x.zones, self.layers), [])

    @property
    def flat_model(self) -> list[HydrologicZone]:
        """A 1D list of all zones in the model, in the order of evaluation."""
        return self.__flat_model

    def step(
        self, state: NDArray[f64], ds: list[HydroForcing], dt: float
    ) -> ModelStep[float]:
        """Step the model given the current state, forcing, and the time step."""
        new_states: list[float] = []
        forc_fluxes: list[float] = []
        vap_fluxes: list[float] = []
        lat_fluxes: list[float] = []
        vert_fluxes: list[float] = []

        # Determine the 'zero' value for fluxes based on the state type
        zero_flux: Any = 0.0
        if state and not isinstance(state[0], float):
            # If state is a list of arrays, create a zero-array template
            zero_flux = np.zeros_like(state[0])

        i: int
        zone: HydrologicZone
        s_i: float
        d_i: HydroForcing
        for i, (zone, (s_i, d_i)) in enumerate(zip(self.flat_model, zip(state, ds))):
            # Calculate incoming flux using faster NumPy dot products
            num_fluxes = len(lat_fluxes)
            if num_fluxes > 0:
                q_in_lat = np.dot(self.lat_mat[i, :num_fluxes], np.array(lat_fluxes))
                q_in_vert = np.dot(self.vert_mat[i, :num_fluxes], np.array(vert_fluxes))
            else:
                q_in_lat, q_in_vert = zero_flux, zero_flux

            q_in = q_in_lat + q_in_vert

            s_i = float(s_i)  # type: ignore
            d_i_zone = HydroForcing(
                precip=d_i.precip, temp=d_i.temp, pet=d_i.pet, q_in=q_in
            )
            step_res: HydroStep = zone.step(s_i, d_i_zone, dt)

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

    def get_river_zone_ids(self) -> list[int]:
        """
        Get a list of the indices of the zones (in the flattened model) that laterally flow into the river.
        These zones are the final zones in a layer.
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

    def num_zones(self) -> int:
        return len(self.flat_model)

    def get_size_mat(self) -> NDArray[f64]:
        """Calculates a matrix mapping surface zones to their contributing areas.

        This matrix is used to determine how forcing data applied to a surface
        zone (or forcing source) is distributed to all zones beneath it.

        Returns:
            A matrix mapping forcing sources to zones.
        """
        vert: NDArray[f64] = self.__vert_matrix
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

    @property
    def zone_labels(self) -> list[str]:
        """A list of zone labels for the final output DataFrame."""
        return [f"{zone.name}_{i}" for i, zone in enumerate(self.flat_model)]

    # Newer definitions
    def construct_hydrologic_graph(self: Model):
        G: nx.DiGraph = nx.DiGraph()

        # Node identifiers will be tuples: (layer_idx, zone_idx)

        # 1. Add all zones as nodes and create lateral connections
        zone_counter: int = 0
        for l_idx, layer in enumerate(self.layers):
            for z_idx, zone in enumerate(layer.zones):
                node_id = (l_idx, z_idx)
                G.add_node(
                    node_id, obj=zone, name=self.zone_labels[zone_counter]
                )  # Store the actual Zone object if needed

                # Add lateral connections (within the same layer, towards the river)
                # Assuming zones are ordered from upstream to downstream within a layer
                if z_idx < len(layer.zones) - 1:
                    next_zone_node_id = (l_idx, z_idx + 1)
                    G.add_edge(
                        node_id,
                        next_zone_node_id,
                        type="lateral",
                    )

                zone_counter += 1

        # 2. Add vertical connections between layers
        for l_idx in range(len(self.layers) - 1):
            current_layer = self.layers[l_idx]
            next_layer = self.layers[l_idx + 1]

            num_zones_current = len(current_layer.zones)
            num_zones_next = len(next_layer.zones)

            # Rule: Vertical flux flows into a single zone
            if num_zones_current == num_zones_next:
                # Case 1: Same number of zones, each flows to the one directly below
                for z_idx in range(num_zones_current):
                    from_node = (l_idx, z_idx)
                    to_node = (l_idx + 1, z_idx)
                    G.add_edge(from_node, to_node, type="vertical")
            elif num_zones_next == 1:
                # Case 2: All zones flow into a single aggregated zone in the layer below
                for z_idx in range(num_zones_current):
                    from_node = (l_idx, z_idx)
                    to_node = (l_idx + 1, 0)  # The single zone in the next layer
                    G.add_edge(from_node, to_node, type="vertical")
            else:
                # Handle invalid layer configurations according to your rules
                print(
                    f"Warning: Layer {l_idx} has {num_zones_current} zones, "
                    f"but Layer {l_idx + 1} has {num_zones_next} zones. "
                    "No vertical connections added for this layer pair as it violates rules."
                )
        return G

    def get_connection_matrices_with_river_row(
        self: Model, G: nx.DiGraph
    ) -> tuple[NDArray, NDArray, list[str]]:
        """
        Generates lateral and vertical connection matrices, each augmented
        with an additional row representing outflow to the river.

        Args:
            G (nx.DiGraph): The full hydrologic graph.
            model (Model): The model object, used to identify river connection zones.

        Returns:
            tuple: (lateral_matrix_augmented, vertical_matrix_augmented, all_nodes)
                - lateral_matrix_augmented (np.array): (n+1) x n matrix for lateral flows + river outflow.
                - vertical_matrix_augmented (np.array): (n+1) x n matrix for vertical flows + river outflow.
                - all_nodes (list): Ordered list of nodes corresponding to matrix columns (and first n rows).
        """
        # Ensure consistent node ordering across all matrices
        all_nodes = sorted(list(G.nodes()))
        num_zones = len(all_nodes)  # This is 'n'

        # Create separate graphs for lateral and vertical connections (n x n part)
        G_lateral: nx.DiGraph = nx.DiGraph()
        G_vertical: nx.DiGraph = nx.DiGraph()

        # Add all nodes to ensure the adjacency matrices have the same dimensions and node mapping
        G_lateral.add_nodes_from(all_nodes)
        G_vertical.add_nodes_from(all_nodes)

        for u, v, data in G.edges(data=True):
            if data["type"] == "lateral":
                G_lateral.add_edge(u, v)
            elif data["type"] == "vertical":
                G_vertical.add_edge(u, v)

        # Convert to NumPy adjacency matrices (n x n)
        lateral_matrix_nn = nx.to_numpy_array(G_lateral, nodelist=all_nodes).T
        vertical_matrix_nn = nx.to_numpy_array(G_vertical, nodelist=all_nodes).T

        # --- Construct the River Outflow Indicator Row ---
        river_nodes = self.get_river_zone_ids()
        river_outflow_indicator_row = np.zeros(num_zones, dtype=int)
        river_outflow_indicator_row[river_nodes] = 1

        # --- Append the River Outflow Indicator Row to the matrices ---
        lateral_matrix_augmented = np.vstack(
            [lateral_matrix_nn, river_outflow_indicator_row]
        )

        return lateral_matrix_augmented, vertical_matrix_nn, all_nodes

    def to_array(self) -> NDArray:
        """
        Convert the model parameters into an array
        """
        param_list: list[float] = []
        # Get the zone parameters
        for zone_name in self.get_zone_names():
            param_list += self[zone_name].param_list()

        # Get the size parameters
        if len(self.scales) > 1:
            param_list += self.scales[:-1]

        # Get the lapse rate parameters
        for lp in self.lapse_rates:
            param_list += [lp.precip_factor, lp.temp_factor]

        return np.array(param_list)

    @classmethod
    def get_num_zone_parameters(cls) -> int:
        """
        Return the number of parameters in the zones only, excluding the meteorology and sizes
        """
        num_zone_params: int = 0
        for layer in cls.structure:
            for zone in layer:
                num_zone_params += len(zone.param_list())

        return num_zone_params

    @classmethod
    def get_num_size_parameters(cls) -> int:
        """
        Return the number of parameters in the sizes only. This will be
        equal to 1 less than the number of surface zones.
        """
        return len(cls.structure[0]) - 1

    @classmethod
    def from_array(cls, arr: NDArray, latent: bool = False) -> Model:
        """
        Create a new object of this type from a numpy array
        `latent` specifies that the size parameters may be encoded in a latent
        space and need to be
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
                    zone_params[: zone.num_parameters()],
                    zone_params[zone.num_parameters() :],
                )
                new_zones[zone.name] = zone.from_array(ps)

        new_lapse_rates: list[LapseRateParameters] = [
            LapseRateParameters(
                temp_factor=temp_factor,
                precip_factor=precip_factor,
            )
            for precip_factor, temp_factor in zip(
                lapse_rate_params[::2], lapse_rate_params[1::2]
            )
        ]

        return cls(
            zones=new_zones,
            scales=size_params,
            lapse_rates=new_lapse_rates,
        )

    def parameter_names(self) -> list[str]:
        """
        Get all of the parameter names for the model
        """
        param_names: list[str] = []
        for layer in self.structure:
            for zone in layer:
                for param in zone.parameter_names():
                    param_names += [f"{zone.name}.{param}"]
        for i, _ in enumerate(self.structure[0][:-1]):
            param_names.append(f"proportion.{i + 1}")

        for i, _ in enumerate(self.lapse_rates):
            param_names += [
                f"lapse_rate.{i + 1}.precip_factor",
                f"lapse_rate.{i + 1}.temp_factor",
            ]

        return param_names

    @classmethod
    def from_dict(cls, params: dict) -> Model:
        """
        Create a new object of this type from a dictionary of parameters
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
        for key, group in itertools.groupby(decomposed_vals, lambda x: x[0]):
            zps: list[tuple[str, str, float]] = list(group)
            zone_param_dict: dict[str, float] = {x[1]: x[2] for x in zps}

            zone_params[key] = zone_param_dict

        # Now, construct the zones
        zones: dict[str, HydrologicZone] = {}
        for zone_name in zone_names:
            zone_param_dict: dict[str, float] = zone_params[zone_name]
            zone_type = cls.get_zone_type(zone_name)

            new_zone = zone_type.from_dict(zone_param_dict)
            zones[zone_name] = new_zone

        # Get the sizes (if applicable)
        scales_ordered: list[tuple[int, float]] = [
            (int(key.split(".")[1]), val) for key, val in proportion_params_list
        ]
        scales_ordered.sort(key=lambda x: x[0])
        scales: list[float] | None = [float(val) for _, val in scales_ordered]
        if len(scales) == 0:
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

        for k, v in lapse_rate_groups.items():
            param_dict = {}
            for _, key, val in v:
                param_dict[key] = val
            lapse_rates.append(LapseRateParameters.from_dict(param_dict))

        return cls(zones=zones, scales=scales, lapse_rates=lapse_rates)

    @classmethod
    def get_zone_type(cls, name: str) -> type:
        """
        Return the type of the zone with the given name
        """
        for layer in cls.structure:
            for zone in layer:
                if zone.name == name:
                    return type(zone)
        raise ValueError(f"Unknown zone name: {name}")

    @classmethod
    def default_init_state(cls) -> NDArray:
        """
        Get the default initial state for the model based on the default
        initial state for each zone in the model
        """
        return np.array(
            [zone.default_init_state() for layer in cls.structure for zone in layer]
        )

    def run(
        self,
        forc: ForcingData | list[ForcingData],
        init_state: Optional[NDArray[f64]] = None,
        meas_streamflow: Optional[Series] = None,
        average_elevation: Optional[float] = None,
        elevations: Optional[list[float]] = None,
        bounds: Optional[dict[str, tuple[float, float]]] = None,
        verbose: bool = False,
    ) -> HydroModelResults:
        """
        Run the hydrologic simulation forward and calculate the objective functions
        """
        # Check the model structure
        self._validate_model_structure()

        # Check if need initial state
        if init_state is None:
            init_state = self.default_init_state()

        # Construct the forcing data
        forcing_data: list[ForcingData]
        if isinstance(forc, ForcingData):
            forcing_data = [forc] * self.num_surface_zones
        else:
            forcing_data = list(forc)  # type: ignore
            if len(forcing_data) != 1:
                if len(forcing_data) != self.num_surface_zones:
                    raise ValueError(
                        f"The number of forcing data series must be either 1 or {self.num_surface_zones}, not {len(forcing_data)}"
                    )

        # Scale the forcing data based on the lapse rates
        if len(self.lapse_rates) > 0:
            if average_elevation is None or elevations is None:
                raise ValueError(
                    "If using lapse rates, you must pass an average elevation and mean elevations for each band"
                )

            for i, (fd, lp, elev) in enumerate(
                zip(forcing_data, self.lapse_rates, elevations)
            ):
                forcing_data[i] = lp.scale_forcing_data(
                    gauge_elevation=average_elevation, elev=elev, forcing_data=fd
                )

        dates = forcing_data[0].precip.index

        # Run the model forwards
        model_res: DataFrame = run_hydro_model(
            model=self,
            init_state=init_state,
            forc=forcing_data,
            dates=dates,
        )

        streamflow_cols: list[str] = [
            col
            for col in model_res.columns
            if "lat" in col and int(col[-1]) in self.get_river_zone_ids()
        ]

        sim_streamflow = model_res[streamflow_cols].sum(axis=1)
        sim_streamflow.name = "streamflow_mmd"
        model_res["streamflow_mmd"] = sim_streamflow

        # Calculate objective functions
        kge_val: Optional[float] = None
        nse_val: Optional[float] = None
        bias_val: Optional[float] = None
        r_squared_val: Optional[float] = None
        spearman_rho_val: Optional[float] = None

        if meas_streamflow is not None:
            kge_val = kge(sim_streamflow, meas_streamflow)
            nse_val = nse(sim_streamflow, meas_streamflow)
            bias_val = (
                sim_streamflow - meas_streamflow
            ).mean() / meas_streamflow.mean()
            r_squared_val = meas_streamflow.corr(sim_streamflow) ** 2
            spearman_rho_val = meas_streamflow.corr(sim_streamflow, method="spearman")

        return HydroModelResults(
            simulation=model_res,
            kge=kge_val,
            nse=nse_val,
            bias=bias_val,
            r_squared=r_squared_val,
            spearman_rho=spearman_rho_val,
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
        return_results: bool = True,
    ) -> tuple[DataFrame, list[tuple[Model, HydroModelResults]]]:
        """
        Run a series of model runs and return the results
        """
        results: list[tuple[int, Model, HydroModelResults]] = []
        batch_params: Optional[BatchParams] = None

        if write_time_series_results:
            if os.path.exists(output_dir):
                cur_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                new_output_dir: str = f"{output_dir}.{cur_time}"
                print(
                    f"Specified directory at {output_dir} exists, saving to {new_output_dir}"
                )
                output_dir = new_output_dir

            if threshold_function is None and write_time_series_results:
                print("No threshold function specified, saving all model results")
            os.makedirs(output_dir, exist_ok=True)

        bps: BatchParams = {
            "output_dir": output_dir,
            "threshold_function": threshold_function,
            "return_results": return_results,
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
            (cls, i, row, forc, init_state, meas_streamflow, batch_params)
            for i, row in params.iterrows()  # type: ignore
        ]

        if num_threads < 1:
            num_threads = os.cpu_count()  # type: ignore
        if num_threads is None:
            num_threads = 1
            warnings.warn("Failed to get number of threads, defaulting to 1")

        with Pool(num_threads) as pool:
            results = pool.starmap(run_model, args)  # type: ignore

        # Now construct the dataframe of the resulting values
        res_df: DataFrame = DataFrame(
            [x for _, _, x in results], index=[x[0] for x in results]
        ).sort_index()

        results.sort(key=lambda x: x[0])

        return res_df, [results[i][1:] for i in range(len(results))]

    @classmethod
    def default_parameter_ranges(
        cls, include_lapse_rates: bool = False
    ) -> dict[str, tuple[float, float]]:
        """
        Get the default parameter ranges for each of the zones in the model
        """
        param_ranges: dict[str, tuple[float, float]] = {}
        # Add the hydrologic zone parameters
        for layer in cls.structure:
            for zone in layer:
                zone_range = zone.default_parameter_range()
                for param_name, param_range in zone_range.items():
                    param_ranges[f"{zone.name}.{param_name}"] = param_range

        # Add the size parameters
        for i, _ in enumerate(cls.structure[0][:-1]):
            param_ranges[f"proportion.{i + 1}"] = (0.1, 0.9)

        # Add the lapse rate parameters
        if include_lapse_rates:
            for i, _ in enumerate(cls.structure[0]):
                params: dict[str, tuple[float, float]] = (
                    LapseRateParameters.default_parameter_range()
                )
                for param_name, param_range in params.items():
                    param_ranges[f"lapse_rate.{i + 1}.{param_name}"] = param_range

        return param_ranges

    @classmethod
    def simple_calibration(
        cls,
        forc: ForcingData | Iterable[ForcingData],
        meas_streamflow: Series,
        metric: Literal["nse", "kge"],
        use_lapse_rates: bool = False,
        num_threads: int = -1,
        polish: bool = False,
        maxiter=10,
        print_values: bool = False,
    ) -> tuple[dict[str, float], HydroModelResults, opt.OptimizeResult]:
        bounds: dict[str, tuple[float, float]] = cls.default_parameter_ranges(
            include_lapse_rates=use_lapse_rates
        )

        bounds_list: list[tuple[float, float]] = list(bounds.values())

        args = (cls, forc, meas_streamflow, metric, print_values)

        opt_res = opt.differential_evolution(
            func=objective_function,
            bounds=bounds_list,  # type: ignore
            maxiter=maxiter,
            tol=0.1,
            rng=0,
            polish=polish,
            workers=num_threads,
            args=args,
            updating="deferred",
        )

        # Now, run the model and return the optimum results
        model = cls.from_array(opt_res.x)
        best_results = model.run(
            forc=forc,  # type: ignore
            init_state=cls.default_init_state(),
            meas_streamflow=meas_streamflow,
            verbose=False,
        )

        opt_params: dict[str, float] = {
            key: val for key, val in zip(bounds.keys(), opt_res.x)
        }

        return opt_params, best_results, opt_res

    def _validate_model_structure(self) -> None:
        """
        Validate whether the model structure is valid or not. The model
        checks are:
        - Ensure all types are correct in the structure
        - Ensure structure is not empty
        - Ensure all zones are properly connected
        - Scales all add up to 1
        """
        # Check if all types are correct in the structure
        for i, layer in enumerate(self.structure):
            if not isinstance(layer, (list, tuple, np.ndarray, Series)):
                raise TypeError(
                    f"Layer {i} in structure has an incorrect type: {type(layer)}, ensure that all layers are of type list, tuple, or other ordered iterable"
                )
            for j, zone in enumerate(layer):
                if not isinstance(zone, HydrologicZone):
                    raise TypeError(
                        f"Zone {j} in layer {i} has an incorrect type: {type(zone)}, ensure that all zones are of type HydrologicZone"
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
                    f"Invalid model structure: The next layer can only have either 1 zone or the same number as the zone above, but your model has the structure {layer_lengths}"
                )
            prev_zones = layer_size

        # Check if all model scales add up to 1
        if abs(sum(self.scales) - 1) > 1e-3:
            raise ValueError(
                f"Model scales do not add up to 1: {self.scales}. Ensure that the scales equal 1 to maintain water balance"
            )

    def to_dict(self) -> dict[str, float]:
        """
        Convert this object to a dictionary
        """
        return {key: val for key, val in zip(self.parameter_names(), self.to_array())}

    def to_series(self) -> Series:
        """
        Convert this object to a pandas Series
        """
        return Series(self.to_array(), index=self.parameter_names())

    @classmethod
    def from_series(cls, series: Series) -> Model:
        """
        Create a new object of this type from a pandas Series
        """
        return cls.from_dict(series.to_dict())

    @property
    def num_parameters(self) -> int:
        """
        Return the number of parameters in the model
        """
        return len(self.to_array())


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


def run_hydro_model(
    model: Model[HydrologicZone],  # type: ignore
    init_state: NDArray[f64],
    forc: list[ForcingData],
    dates: Series[Timestamp] | Index[Timestamp],
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
    num_zones: Final[int] = model.num_zones()
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
        all_precip_sources_matrix = np.vstack([fd.precip.to_numpy() for fd in forc]).T  # type: ignore
        all_temp_sources_matrix = np.vstack([fd.temp.to_numpy() for fd in forc]).T  # type: ignore
        all_pet_sources_matrix = np.vstack([fd.pet.to_numpy() for fd in forc]).T  # type: ignore
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

    delta_ts: list[float] = [
        (dates[i] - dates[i - 1]).days for i in range(1, len(dates))
    ]
    delta_ts.append(delta_ts[-1])

    for t_idx in range(num_steps):
        # Forcings for the current time step, one HydroForcing object per zone
        dt: float = delta_ts[t_idx]
        ds_for_step: list[HydroForcing] = [
            HydroForcing(
                precip=zone_precip_series[t_idx, j],
                temp=zone_temp_series[t_idx, j],
                pet=zone_pet_series[t_idx, j],
                q_in=0.0,
            )
            for j in range(num_zones)
        ]

        try:
            # Convert state array to list for the generic step method
            step_res: ModelStep = model.step(list(state), ds_for_step, dt)  # type: ignore
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
    hydro_results,
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


# ==== Defined hydrologic models ==== #
class HbvModel(Model):
    structure = [
        [SnowZone(name="snow")],
        [SoilZone(name="soil")],
        [GroundZoneLinear(name="shallow")],
        [GroundZoneLinearB(name="deep")],
    ]


class HbvLateralModel(Model):
    structure = [
        [SnowZone(name="snow_hs"), SnowZone(name="snow_ls")],
        [SoilZone(name="soil_hs"), SoilZone(name="soil_ls")],
        [GroundZoneLinear(name="shallow_hs"), GroundZoneLinear(name="shallow_ls")],
        [GroundZoneLinearB(name="deep_hs"), GroundZoneLinearB(name="deep_ls")],
    ]


class HbvNonlinearModel(Model):
    structure = [
        [SnowZone(name="snow")],
        [SoilZone(name="soil")],
        [GroundZone(name="shallow")],
        [GroundZoneB(name="deep")],
    ]


class ThreeLayerModel(Model):
    structure = [
        [SnowZone(name="snow")],
        [SoilZone(name="soil")],
        [GroundZoneB(name="ground")],
    ]


# =================================== #
