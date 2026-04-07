from __future__ import annotations
from typing import Callable, Final, Iterable, Optional
from pandas import DataFrame, Index, Series, Timestamp
import operator
from functools import reduce
import numpy as np
from numpy.typing import ArrayLike, NDArray
import networkx as nx

from .common_types import HydroModelStep, ForcingData, HydroModelResults
from .core import HydroStep, HydrologicZone, HydroForcing, ScalarRootFindingError
from .model_components import Layer
from .objective_functions import DEFAULT_OBJECTIVE_FUNCTIONS
from .utils import HydrologyNumericalError

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
        res = HydroModelResults(simulation=model_res, objective_functions=obj_val_ser)
        return res
