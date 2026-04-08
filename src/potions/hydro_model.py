from __future__ import annotations

import itertools
from multiprocessing import Pool
import operator
from functools import reduce
import os
import random
from typing import Callable, Final, Iterable, Iterator, Literal, Optional

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike, NDArray
from pandas import DataFrame, Index, Series, Timestamp
import scipy.optimize as opt
import emcee  # type: ignore

from .common_types import ForcingData, HydroModelResults, HydroModelStep
from .core import HydroForcing, HydrologicZone, HydroStep, ScalarRootFindingError
from .model_components import Layer
from .objective_functions import DEFAULT_OBJECTIVE_FUNCTIONS
from .utils import HydrologyNumericalError, log_probability, objective_function

# ==== Constants and types ==== #
f64 = np.float64
OUTPUT_COLUMNS_PER_ZONE: Final[int] = (
    # Number of columns for each zone in the output. Includes 1 state + 6 fluxes (4 normal + 2 external)
    8
)
# ============================= #


class HydrologicalModel:

    structure: list[list[HydrologicZone]]

    def __init__(
        self,
        # Hydrology arguments
        zones: Optional[dict[str, HydrologicZone]] = None,
        scales: Optional[list[float]] = None,
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
        # Check for empty values
        if scales is None or scales == []:
            # Get the default
            num_zones: int = len(self.structure[0])
            scales = [1 / num_zones for _ in range(num_zones)]
        if zones is None:
            zones = {}

        # Construct the zone dictionary with the zones
        self.__zones: dict[str, HydrologicZone] = {}
        for layer in self.structure:
            for zone in layer:
                if zone.name in zones:  # type: ignore
                    self.__zones[zone.name] = zones[zone.name]  # type: ignore
                else:
                    self.__zones[zone.name] = zone.default()  # type: ignore

        # Construct the lapse rates
        layers: list[Layer] = []
        for layer in self.structure:
            layer_vals: list[HydrologicZone] = []
            for zone in layer:
                layer_vals.append(self.__zones[zone.name])  # type: ignore

            layers.append(Layer(layer_vals))

        self.verbose: bool = verbose
        self.__layers: list[Layer] = layers
        self.__scales: list[float] = scales
        self.__flat_model: list[HydrologicZone] = self.flatten()
        self.__size: int = reduce(operator.add, map(len, self.layers), 0)

        # Calculate the connectivity matrices
        self.__zone_graph: nx.DiGraph = self.construct_hydrologic_graph()
        lat, vert, _ = self.get_connection_matrices_with_river_row(self.__zone_graph)
        self.__lat_matrix: NDArray = lat
        self.__vert_matrix: NDArray = vert
        self.__forcing_mat: NDArray = self.get_forc_mat(scales)
        self.__forcing_rel_mat: NDArray = self.get_forc_mat(scales, relative=True)

    def __len__(self) -> int:
        """Returns the number of layers in the model."""
        return len(self.layers)

    def __iter__(self) -> Iterator[Layer]:
        """Returns an iterator over the layers in the model."""
        return iter(self.layers)

    @property
    def hydro_zones(self) -> dict[str, HydrologicZone]:
        raise NotImplementedError()

    @classmethod
    def get_zone_names(cls: type[HydrologicalModel]) -> list[str]:
        """Get the names of all hydrologic zones in the model structure.

        Returns:
            list[str]: A list of all zone names.
        """
        zone_names: list[str] = []
        for layer in cls.structure:
            for zone in layer:
                zone_names.append(zone.name)  # type: ignore
        return zone_names

    def __getitem__(self, zone_name: str) -> HydrologicZone:
        """Access a hydrologic zone within the model by its name.

        Args:
            zone_name (str): The name of the zone to retrieve.

        Returns:
            HydrologicZone: The zone object.

        Raises:
            ValueError: If no zone with the given name exists in the model.
        """
        raise NotImplementedError()

    @property
    def num_surface_zones(self) -> int:
        """sThe number of zones in the top layer of the model."""
        return len(self.structure[0])

    @property
    def graph(self) -> nx.DiGraph:
        """The NetworkX graph representing the model's hydrologic connectivity."""
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
        """The list of relative areas for each surface zone."""
        return self.__scales

    @property
    def surface_zone_ids(self) -> list[int]:
        """
        Get a list of the numerical indices of the surface zones in the model.

        Note that these must be the top zones in the model. It's just the indices of the first later of zones
        """
        return [i for i, _ in enumerate(self.structure[0])]

    def flatten(self) -> list[HydrologicZone]:
        """Flattens the nested model structure into a single list of zones.

        Returns:
            list[HydrologicZone]: A 1D list of all zones.
        """
        return reduce(operator.add, map(lambda x: x.zones, self.layers), [])

    @property
    def flat_model(self) -> list[HydrologicZone]:
        """A 1D list of all zones in the model, in the order of evaluation."""
        return self.__flat_model

    @property
    def num_zones(self) -> int:
        """Returns the total number of zones in the model."""
        return len(self.flat_model)

    @classmethod
    def get_num_zones(cls: type[HydrologicalModel]) -> int:
        """Returns the total number of zones in the model."""
        num_zones: int = 0
        for layer in cls.structure:  # type: ignore
            num_zones += len(layer)

        return num_zones

    def get_size_mat(self) -> NDArray[f64]:
        """Calculates a matrix mapping surface zones to their contributing areas.

        This matrix is used to determine how forcing data applied to a surface
        zone (or forcing source) is distributed to all zones beneath it.

        Returns:
            NDArray[f64]: A matrix where `mat[i, j]` is 1 if surface zone `j`
                contributes to zone `i`, and 0 otherwise.
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
            relative (bool): If True, normalizes the matrix rows to sum to 1.
                This is used for intensive variables like temperature where an
                area-weighted average is desired. If False, the matrix contains
                the fractional contribution of each source. Defaults to False.
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

    @classmethod
    def get_hydro_column_names(cls) -> list[str]:
        """A class method to get column names for the final output DataFrame."""
        zone_names: list[str] = cls.get_zone_names()
        col_names: list[str] = []
        for _i, zone_name in enumerate(zone_names):
            col_names += [
                f"s_{zone_name}",
                f"q_forc_{zone_name}",
                f"q_vap_{zone_name}",
                f"q_lat_{zone_name}",
                f"q_vert_{zone_name}",
                f"q_in_{zone_name}",
                f"q_lat_ext_{zone_name}",
                f"q_vert_ext_{zone_name}",
            ]
        return col_names

    @property
    def zone_labels(self) -> list[str]:
        """A list of zone labels for the final output DataFrame."""
        return [f"{zone.name}_{i}" for i, zone in enumerate(self.flat_model)]  # type: ignore

    @property
    def zone_names(self) -> list[str]:
        """A list of the names of the zones in the order that they are evaluated"""
        names: list[str] = []

        for layer in self.structure:
            for zone in layer:
                names.append(zone.name)  # type: ignore

        return names

    @property
    def zone_indices(self) -> dict[str, int]:
        zones: dict[str, int] = {}
        counter: int = 0
        for layer in self.structure:
            for zone in layer:
                zones[zone.name] = counter  # type: ignore
                counter += 1

        return zones

    def construct_hydrologic_graph(self) -> nx.DiGraph:
        """Constructs a directed graph representing hydrologic connectivity.

        This method builds a `networkx.DiGraph` where nodes are zones and
        edges represent the flow of water (laterally or vertically). The graph
        is built based on the model's `structure`.

        Returns:
            nx.DiGraph: The connectivity graph of the model.
        """
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
                    # The single zone in the next layer
                    to_node = (l_idx + 1, 0)
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
        self, G: nx.DiGraph
    ) -> tuple[NDArray, NDArray, list[str]]:
        """Generates lateral and vertical connection matrices from the graph.

        The lateral matrix is augmented with an additional row representing
        outflow to the river.

        Args:
            G (nx.DiGraph): The full hydrologic graph.

        Returns:
            tuple[NDArray, NDArray, list[str]]: A tuple containing:
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

    def hydro_to_array(self) -> NDArray:
        """Serializes all model parameters into a single 1D NumPy array.

        Returns:
            NDArray: A flat array of all model parameters.
        """
        param_list: list[float] = []
        # Get the zone parameters
        for zone_name in self.get_zone_names():
            param_list += self[zone_name].param_list()

        # Get the size parameters
        if len(self.scales) > 1:
            param_list += self.scales[:-1]

        # Get the lapse rate parameters
        return np.array(param_list)

    def construct_hydro_forcing_matrix(
        self,
        forc: ForcingData | Iterable[ForcingData],
        elevations: Optional[list[float]] = None,
        average_elevation: Optional[float] = None,
    ) -> NDArray[np.object_]:
        """
        Take the input of the forcing data and turn it into the Python formats that are useful for potions
        """
        # Construct the forcing data
        forcing_data: list[ForcingData]
        if isinstance(forc, ForcingData):
            forcing_data = [forc] * self.num_surface_zones
        else:
            forcing_data = list(forc)  # type: ignore
            if len(forcing_data) != 1:
                if len(forcing_data) != self.num_surface_zones:
                    raise ValueError(
                        f"The number of forcing data series must be either 1 or {
                            self.num_surface_zones
                        }, not {len(forcing_data)}"
                    )

        # Scale the forcing data based on the lapse rates
        # if len(self.lapse_rates) > 0:
        #     if average_elevation is None or elevations is None:
        #         raise ValueError(
        #             "If using lapse rates, you must pass an average elevation and mean elevations for each band"
        #         )

        #     fd: ForcingData
        #     for i, (fd, lp, elev) in enumerate(
        #         zip(forcing_data, self.lapse_rates, elevations, strict=True)
        #     ):
        #         forcing_data[i] = lp.scale_forcing_data(  # type: ignore
        #             gauge_elevation=average_elevation, elev=elev, forcing_data=fd
        #         )

        # Now, turn these into the values of
        num_steps: Final[int] = len(forcing_data[0].precip)  # Number of steps

        num_forcing_sources_expected: Final[int] = self.precip_mat.shape[1]

        all_precip_sources_matrix: NDArray[f64]
        all_temp_sources_matrix: NDArray[f64]
        all_pet_sources_matrix: NDArray[f64]

        # Validate and prepare source forcing matrices
        if forcing_data:
            if len(forcing_data) != num_forcing_sources_expected:
                raise ValueError(
                    f"Model expects {
                        num_forcing_sources_expected
                    } ForcingData objects, "
                    f"but {len(forcing_data)
                           } were provided in the 'forc' list."
                )
            for i, fd in enumerate(forcing_data):
                if not (
                    len(fd.precip) == num_steps
                    and len(fd.temp) == num_steps
                    and len(fd.pet) == num_steps
                ):
                    raise ValueError(
                        f"ForcingData at index {
                            i
                        } has series lengths inconsistent with num_steps ({
                            num_steps
                        }). "
                        f"P length: {len(fd.precip)}, T length: {
                            len(fd.temp)
                        }, PET length: {len(fd.pet)}"
                    )
            all_precip_sources_matrix = np.vstack(
                [fd.precip.to_numpy() for fd in forcing_data]
            ).T  # type: ignore
            all_temp_sources_matrix = np.vstack(
                [fd.temp.to_numpy() for fd in forcing_data]
            ).T  # type: ignore
            all_pet_sources_matrix = np.vstack(
                [fd.pet.to_numpy() for fd in forcing_data]
            ).T  # type: ignore
        else:  # forc is empty
            if num_forcing_sources_expected > 0:
                raise ValueError(
                    f"Model expects {
                        num_forcing_sources_expected
                    } forcing inputs, but 'forcing_data' list is empty."
                )
            # If no forcing sources are expected, create empty (0-column) matrices
            all_precip_sources_matrix = np.zeros((num_steps, 0), dtype=f64)
            all_temp_sources_matrix = np.zeros((num_steps, 0), dtype=f64)
            all_pet_sources_matrix = np.zeros((num_steps, 0), dtype=f64)

        # Distribute source forcings to each zone for all time steps
        # Resulting shape for each: (num_steps, num_zones)
        zone_precip_series = all_precip_sources_matrix @ self.precip_mat.T
        zone_temp_series = all_temp_sources_matrix @ self.temp_mat.T
        zone_pet_series = all_pet_sources_matrix @ self.pet_mat.T

        # Now, create a matrix of the hydrologic forcing values
        hydro_forcing: NDArray = np.empty(zone_precip_series.shape, dtype=object)
        for i, (ppt_row_i, temp_row_i, pet_row_i) in enumerate(
            zip(zone_precip_series, zone_temp_series, zone_pet_series, strict=True)
        ):
            for j, (ppt_ij, temp_ij, pet_ij) in enumerate(
                zip(ppt_row_i, temp_row_i, pet_row_i, strict=True)
            ):
                hydro_forcing[i, j] = HydroForcing(
                    precip=ppt_ij, temp=temp_ij, pet=pet_ij, q_in=0.0
                )

        return hydro_forcing

    @classmethod
    def default_hydro_init_state(cls) -> NDArray:
        """Gets the default initial state for the model.

        Returns:
            NDArray: An array of default initial storage values for all zones.
        """
        return np.array(
            [
                zone.default_init_state() for layer in cls.structure for zone in layer
            ]  # type: ignore
        )

    def step_hydro_model(
        self,
        state: NDArray[f64],
        ds: Iterable[HydroForcing],
        dt: float,
        check_water_balance: bool = False,
    ) -> HydroModelStep:
        """Advances all zones in the model by a single time step.

        This method orchestrates the computation for one step by:
        1. Calculating incoming fluxes to each zone from its neighbors.
        2. Calling the `step` method of each individual zone.
        3. Collecting the results.

        Args:
            state (NDArray[f64]): The current state (storage) of all zones.
            ds (list[HydroForcing]): A list of forcing data objects, one for
                each zone for the current time step.
            dt (float): The time step duration in days.

        Returns:
            ModelStep[float]: An object containing the new states and all
                calculated fluxes for every zone.
        """
        num_zones: int = self.num_zones

        new_states: np.ndarray = np.zeros((num_zones,), dtype=np.float64)
        forc_fluxes: np.ndarray = new_states.copy()
        vap_fluxes: np.ndarray = new_states.copy()
        lat_fluxes: np.ndarray = new_states.copy()
        vert_fluxes: np.ndarray = new_states.copy()
        q_ins: np.ndarray = new_states.copy()
        lat_ext_fluxes: np.ndarray = new_states.copy()
        vert_ext_fluxes: np.ndarray = new_states.copy()

        i: int
        zone: HydrologicZone
        s_i: float
        d_i: HydroForcing
        for i, (zone, (s_i, d_i)) in enumerate(
            zip(self.flat_model, zip(state, ds, strict=True), strict=True)
        ):
            num_fluxes = i + 1
            if num_fluxes > 0:
                q_in_lat = np.dot(self.lat_mat[i, :num_fluxes], lat_ext_fluxes[: i + 1])
                q_in_vert = np.dot(
                    self.vert_mat[i, :num_fluxes], vert_ext_fluxes[: i + 1]
                )
            else:
                q_in_lat, q_in_vert = 0.0, 0.0

            q_in = q_in_lat + q_in_vert

            d_i_zone = HydroForcing(
                precip=d_i.precip, temp=d_i.temp, pet=d_i.pet, q_in=q_in
            )

            try:
                step_res: HydroStep = zone.step(max(0.0, s_i), d_i_zone, dt)
            except ScalarRootFindingError as e:
                raise HydrologyNumericalError(
                    model_type=type(self),
                    parameters=self.hydro_to_array(),
                    zone=zone,
                    state=s_i,
                    hydro_forcing=d_i.copy(),
                ) from e
            except Exception as e:
                msg = f"""Failed on zone {i} for zone {zone.name}, storages are {state=}, with forcing data {d_i_zone}
                Zone: {zone}, error={e}
                """
                raise ValueError(msg) from e

            # ==== Check the water balance ==== #
            if check_water_balance:
                s_new = step_res.state
                mass_balance = (
                    d_i_zone.q_in
                    + step_res.forc_flux
                    - step_res.vap_flux
                    - step_res.lat_flux
                    - step_res.vert_flux
                )

                mass_err = (s_i - s_new) + dt * mass_balance

                if abs(mass_err) > 1e-3:
                    print(
                        f"Mass balance error on for zone {
                          i} with name {zone.name}"
                    )  # type: ignore
                    print(f"Storage: {s_i}")
                    print(f"Zone: {zone}")
                    print(f"Forcing data: {d_i_zone}")
                    print(f"Step: {step_res}")
                    raise ValueError("Mass balance failed")
            # ================================= #

            new_states[i] = step_res.state
            forc_fluxes[i] = step_res.forc_flux
            vap_fluxes[i] = step_res.vap_flux
            lat_fluxes[i] = step_res.lat_flux
            vert_fluxes[i] = step_res.vert_flux
            q_ins[i] = d_i_zone.q_in
            lat_ext_fluxes[i] = step_res.lat_flux_ext
            vert_ext_fluxes[i] = step_res.vert_flux_ext

        return HydroModelStep(
            state=new_states,
            forc_flux=forc_fluxes,
            vap_flux=vap_fluxes,
            lat_flux=lat_fluxes,
            vert_flux=vert_fluxes,
            q_in=q_ins,
            lat_flux_ext=lat_ext_fluxes,
            vert_flux_ext=vert_ext_fluxes,
        )

    def run_hydro_model(
        self,
        forc: ForcingData | list[ForcingData],
        init_state: Optional[NDArray[f64]] = None,
        meas_streamflow: Optional[Series] = None,
        average_elevation: Optional[float] = None,
        elevations: Optional[list[float]] = None,
        check_water_balance: bool = False,
        objective_functions: Optional[
            list[tuple[str, Callable[[Series, Series], float]]]
        ] = None,
        verbose: bool = False,
    ) -> HydroModelResults:
        """Runs a forward simulation of the hydrologic model.

        Args:
            forc (ForcingData | list[ForcingData]): A single `ForcingData` object
                or a list of them (one for each surface zone).
            init_state (Optional[NDArray[f64]]): The initial storage for each
                zone. If None, uses `default_init_state()`. Defaults to None.
            meas_streamflow (Optional[Series]): A time series of observed
                streamflow for calculating performance metrics. Defaults to None.
            average_elevation (Optional[float]): The average elevation of the
                meteorological gauge (required for lapse rates). Defaults to None.
            elevations (Optional[list[float]]): A list of mean elevations for
                each surface zone (required for lapse rates). Defaults to None.
            verbose (bool): Not currently used.

        Returns:
            HydroModelResults: A `HydroModelResults` dictionary.
        """
        # Check the model structure to make sure that all of the zones are correct
        self._validate_model_structure()

        # Check if need initial state
        if init_state is None:
            init_state = self.default_hydro_init_state()

        # ==== Run the model forwards ==== #
        dates: Series[Timestamp] | Index[Timestamp]
        if isinstance(forc, ForcingData):
            dates = forc.precip.index
        elif isinstance(forc, (list, tuple)):
            dates = forc[0].precip.index
        else:
            raise TypeError("Forcing data is the wrong type")

        num_steps: Final[int] = len(dates)  # Number of steps
        num_zones: Final[int] = self.get_num_zones()

        hydro_forcing = self.construct_hydro_forcing_matrix(forc)

        storages: NDArray[f64] = np.full(
            (num_steps, num_zones), fill_value=np.nan, dtype=float
        )
        fluxes: NDArray[f64] = np.full(
            (num_steps, num_zones, OUTPUT_COLUMNS_PER_ZONE - 1),
            fill_value=np.nan,
            dtype=float,
        )

        state: NDArray[f64] = init_state

        delta_ts: list[float] = [
            (dates[i] - dates[i - 1]).days for i in range(1, len(dates))
        ]
        delta_ts.append(delta_ts[-1])

        if any(map(lambda x: x <= 1e-5, delta_ts)):
            raise ValueError(
                "Invalid step size in hydrologic simulation, make sure all simulation points are >= 1e-5 days (~1 second)"
            )

        for t_idx in range(num_steps):
            # Forcings for the current time step, one HydroForcing object per zone
            dt: float = delta_ts[t_idx]
            ds_for_step: list[HydroForcing] = hydro_forcing[t_idx]

            try:
                # Convert state array to list for the generic step method
                step_res: HydroModelStep = self.step_hydro_model(
                    list(state),  # type: ignore
                    ds_for_step,
                    dt,
                    check_water_balance=check_water_balance,  # type: ignore
                )
                storages[t_idx] = np.array(step_res.state)
                fluxes[t_idx, :, 0] = np.array(step_res.forc_flux)
                fluxes[t_idx, :, 1] = np.array(step_res.vap_flux)
                fluxes[t_idx, :, 2] = np.array(step_res.lat_flux)
                fluxes[t_idx, :, 3] = np.array(step_res.vert_flux)
                fluxes[t_idx, :, 4] = np.array(step_res.q_in)
                fluxes[t_idx, :, 5] = np.array(step_res.lat_flux_ext)
                fluxes[t_idx, :, 6] = np.array(step_res.vert_flux_ext)
                # Convert state back to array for the next iteration
                state = np.array(step_res.state)
            except ValueError as e:
                # print(f"Failed on step {t_idx}, returning early")
                # break
                raise e

        full_array: NDArray[f64] = np.full(
            (num_steps, OUTPUT_COLUMNS_PER_ZONE * num_zones),
            fill_value=np.nan,
            dtype=float,
        )
        for i in range(num_zones):
            full_array[:, OUTPUT_COLUMNS_PER_ZONE * i + 0] = storages[:, i]
            full_array[:, OUTPUT_COLUMNS_PER_ZONE * i + 1] = fluxes[:, i, 0]  # Forcing
            full_array[:, OUTPUT_COLUMNS_PER_ZONE * i + 2] = fluxes[
                :, i, 1
            ]  # Vaporization
            full_array[:, OUTPUT_COLUMNS_PER_ZONE * i + 3] = fluxes[:, i, 2]  # Lateral
            full_array[:, OUTPUT_COLUMNS_PER_ZONE * i + 4] = fluxes[:, i, 3]  # Vertical
            full_array[:, OUTPUT_COLUMNS_PER_ZONE * i + 5] = fluxes[:, i, 4]  # q_in
            full_array[:, OUTPUT_COLUMNS_PER_ZONE * i + 6] = fluxes[
                :, i, 5
            ]  # Lateral (external)
            full_array[:, OUTPUT_COLUMNS_PER_ZONE * i + 7] = fluxes[
                :, i, 6
            ]  # Vertical (external)

        col_names: list[str] = self.get_hydro_column_names()

        model_res: DataFrame = DataFrame(
            data=full_array, index=dates, columns=col_names
        )
        # ================================ #

        river_zone_ids: list[int] = self.get_river_zone_ids()

        def is_streamflow_col(x: str) -> bool:
            if not x.startswith("q_lat_"):
                return False

            zone_name: str = x.replace("q_lat_", "")
            return self.zone_indices[zone_name] in river_zone_ids

        streamflow_cols: list[str] = [
            col
            for col in model_res.columns
            if "lat" in col and ("ext" not in col) and is_streamflow_col(col)
        ]

        sim_streamflow = model_res[streamflow_cols].sum(axis=1)
        sim_streamflow.name = "sim_streamflow_mmd"
        model_res["sim_streamflow_mmd"] = sim_streamflow
        if meas_streamflow is not None:
            model_res["meas_streamflow_mmd"] = meas_streamflow

        # Calculate objective functions
        obj_vals: dict[str, float] = {}
        if meas_streamflow is not None:
            if objective_functions is None:
                # Use the default
                objective_functions = DEFAULT_OBJECTIVE_FUNCTIONS

            for name, obj_func in objective_functions:
                obj_vals[name] = obj_func(meas_streamflow, sim_streamflow)

        obj_val_ser: Series = Series(data=obj_vals)

        # Calculate some other metrics
        streamflow_ids = self.get_river_zone_ids()
        zone_names: list[str] = self.get_zone_names()

        props: dict[str, float] = {}
        for col_id in streamflow_ids:
            zone_name: str = zone_names[col_id]
            col_name: str = f"q_lat_{zone_name}"
            prop_name: str = f"prop_q_{zone_name}"
            model_res[prop_name] = model_res[col_name] / model_res["sim_streamflow_mmd"]
            props[prop_name] = model_res[prop_name].mean()

        # Create the object and save
        res = HydroModelResults(
            simulation=model_res, objective_functions=obj_val_ser, forcing=forc
        )
        return res

    @classmethod
    def hydro_from_array(
        cls, arr: NDArray, latent: bool = False, natural_scales: bool = True
    ) -> HydrologicalModel:
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

        # lapse_rate_params: NDArray = arr[num_zone_params + num_size_params :]

        new_zones: dict[str, HydrologicZone] = dict()
        for layer in cls.structure:
            for zone in layer:
                ps, zone_params = (
                    zone_params[: zone.num_parameters()],  # type: ignore
                    zone_params[zone.num_parameters() :],  # type: ignore
                )
                new_zones[zone.name] = zone.from_array(ps, natural_scales=natural_scales)  # type: ignore

        # new_lapse_rates: list[LapseRateParameters] = [
        #     LapseRateParameters(
        #         temp_factor=temp_factor,
        #         precip_factor=precip_factor,
        #     )
        #     for precip_factor, temp_factor in zip(
        #         lapse_rate_params[::2], lapse_rate_params[1::2], strict=True
        #     )
        # ]

        return cls(
            zones=new_zones,
            scales=size_params,
            # lapse_rates=new_lapse_rates,
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
    def hydro_from_dict(cls, params: dict) -> HydrologicalModel:
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
        # lapse_rate_params_list: list[tuple[str, float]] = [
        #     x for x in params_list if x[0].split(".")[0] == "lapse_rate"
        # ]

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
        # lapse_rates: list[LapseRateParameters] = []
        # lapse_rates_grouped: list[tuple[int, str, float]] = [
        #     (int(key.split(".")[1]), key.split(".")[2], val)
        #     for key, val in lapse_rate_params_list
        # ]
        # lapse_rates_grouped.sort(key=lambda x: x[0])

        # lapse_rate_groups: dict[int, list[tuple[int, str, float]]] = {
        #     k: list(v)
        #     for k, v in itertools.groupby(lapse_rates_grouped, lambda x: x[0])
        # }

        # for _k, v in lapse_rate_groups.items():
        #     param_dict = {}
        #     for _, key, val in v:
        #         param_dict[key] = val
        #     lapse_rates.append(LapseRateParameters.from_dict(param_dict))  # type: ignore

        return cls(zones=zones, scales=scales)

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
    def default_parameter_ranges(
        cls,
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
        # if include_lapse_rates:
        #     for i, _ in enumerate(cls.structure[0]):
        #         params: dict[str, tuple[float, float]] = (
        #             LapseRateParameters.default_parameter_range()  # type: ignore
        #         )
        #         for param_name, param_range in params.items():
        #             param_ranges[
        #                 f"lapse_rate.{
        #                 i + 1}.{param_name}"
        #             ] = param_range

        return param_ranges

    @classmethod
    def simple_calibration(
        cls: type[HydrologicalModel],
        forc: ForcingData | Iterable[ForcingData],
        meas_streamflow: Series,
        metric: (
            Literal["nse", "kge", "combined"] | Callable[[HydroModelResults], float]
        ),
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
            bounds = cls.default_parameter_ranges()
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
        bounds: dict = self.default_parameter_ranges()
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
        cls: type[HydrologicalModel],
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
            bounds = cls.default_parameter_ranges()

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

    @classmethod
    def hydro_from_series(cls, series: Series) -> HydrologicalModel:
        """Creates a new model instance from a pandas Series of parameters.

        Args:
            series (Series): A Series of parameters with names as the index.

        Returns:
            Model: A new, parameterized model instance.
        """
        return cls.hydro_from_dict(series.to_dict())

    @property
    def num_hydro_parameters(self) -> int:
        """The total number of optimizable parameters in the model."""
        return len(self.hydro_to_array())

    @classmethod
    def get_num_hydro_parameters(cls: type[HydrologicalModel]) -> int:
        """The total number of optimizable parameters in the model."""
        num_params: int = 0
        for layer in cls.structure:
            for zone in layer:
                num_params += zone.num_parameters()  # type: ignore

        return num_params + cls.get_num_size_parameters()
