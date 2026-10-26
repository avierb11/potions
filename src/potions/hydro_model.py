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
from .model_components import Layer, is_bottom_zone
from .objective_functions import DEFAULT_OBJECTIVE_FUNCTIONS
from .reactive_transport.parameter_docs import (
    _fmt,
    render_describe,
    zone_parameter_rows,
)
from .result_saving import is_save_enabled, save_hydro_run
from .utils import (
    HydrologyNumericalError,
    ModelInputError,
    ModelStructureError,
    log_probability,
    objective_function,
)

# ==== Constants and types ==== #
f64 = np.float64
OUTPUT_COLUMNS_PER_ZONE: Final[int] = (
    # Number of columns for each zone in the output. Includes 1 state + 6 fluxes (4 normal + 2 external)
    8
)
SIM_Q_NAME: Final[str] = "streamflow_sim"
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
                All values must be strictly positive and sum to 1. These area
                fractions govern forcing distribution, the area-weighted
                lateral transfer between zones (donor/receiver ratio), and the
                area-weighted outlet discharge. Defaults to equal scaling.
            lapse_rates (Optional[list[LapseRateParameters]]): A list of lapse
                rate parameters, one for each surface zone. Defaults to None.
            verbose (bool): If True, prints additional information during
                initialization. Defaults to False.
        """
        if not self.structure:
            raise ModelStructureError(
                "Model structure is empty, make sure you define a model "
                "structure class property describing your model"
            )
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

        # Per-zone area fractions, propagated down each column from the
        # surface scales, and the area-weighted connectivity matrices. Area
        # factors live on the *boundaries* (edges) of the connectivity graph:
        # an edge d -> r carries the donor/receiver area ratio (s_d / s_r), and
        # the river outflow row carries the receiver scale (s_r) so the outlet
        # is a catchment-mean depth.
        self.__zone_scales: NDArray = self.get_zone_scales(scales)
        self.__lat_matrix_scaled: NDArray = self.get_scaled_lat_mat(
            lat, self.__zone_scales
        )
        self.__vert_matrix_scaled: NDArray = self.get_scaled_vert_mat(
            vert, self.__zone_scales
        )

        # Validate the structure (bottom-zone placement rules, parameter
        # finiteness, scale validity) at construction time, so that a
        # misconfigured model fails immediately with a clear message rather
        # than silently producing wrong results when it is eventually run.
        self._validate_model_structure()

    def __len__(self) -> int:
        """Returns the number of layers in the model."""
        return len(self.layers)

    def __iter__(self) -> Iterator[Layer]:
        """Returns an iterator over the layers in the model."""
        return iter(self.layers)

    @property
    def hydro_zones(self) -> dict[str, HydrologicZone]:
        zones: dict[str, HydrologicZone] = {}
        for zone_name in self.zone_names:
            zones[zone_name] = self.__zones[zone_name]

        return zones

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
        if zone_name not in self.zone_names:
            raise ValueError(f"Model does not contain zone {zone_name}")
        else:
            return self.__zones[zone_name]

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
    def lat_mat_scaled(self) -> NDArray[f64]:
        """The area-weighted matrix describing lateral connectivity.

        Lateral edges carry the donor/receiver area ratio (`s_d / s_r`) and
        the river-outflow row carries the receiver scale (`s_r`). Used at
        run time so that inter-zone transfer and outlet discharge are
        area-consistent.
        """
        return self.__lat_matrix_scaled

    @property
    def zone_scales(self) -> NDArray[f64]:
        """The area fraction of every zone, in the order of evaluation."""
        return self.__zone_scales

    @property
    def vert_mat(self) -> NDArray[f64]:
        """The matrix describing vertical connectivity between zones."""
        return self.__vert_matrix

    @property
    def vert_mat_scaled(self) -> NDArray[f64]:
        """The area-weighted matrix describing vertical connectivity.

        Vertical edges join zones of the same column (ratio 1) except where a
        layer aggregates N zones into one, in which case the edge carries the
        donor/receiver area ratio so mass is conserved.
        """
        return self.__vert_matrix_scaled

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

    def get_zone_scales(self, scales: ArrayLike) -> NDArray[f64]:
        """Propagates the surface-zone area fractions to every zone.

        Each surface zone defines a fractional area of the catchment, and
        every zone beneath it (within the same column) belongs to the same
        area fraction. This mirrors the propagation used by `get_forc_mat`:
        the per-zone scale is `size_mat @ scales`.

        Args:
            scales: An array-like object with the relative area of each
                surface zone (must sum to 1).

        Returns:
            NDArray[f64]: A length-`num_zones` vector of the area fraction
                for every zone, in the order of evaluation.
        """
        conn_mat: NDArray[f64] = self.get_size_mat()
        s_arr: NDArray[f64] = np.array(scales, dtype=f64)
        return conn_mat @ s_arr

    @staticmethod
    def get_scaled_lat_mat(
        lat_mat: NDArray[f64], zone_scales: NDArray[f64]
    ) -> NDArray[f64]:
        """Builds an area-weighted lateral connectivity matrix.

        The plain lateral matrix (`lat_mat`) encodes 0/1 adjacency, with an
        extra final row encoding outflow to the river. This method converts
        that adjacency into an area-consistent transfer operator:

        - A lateral edge from donor zone `d` to receiver zone `r` is weighted
          by the donor/receiver area ratio `s_d / s_r`. This conserves mass:
          all water leaving the donor's area is concentrated onto the
          receiver's (typically smaller) area.
        - The river-outflow row (the last row) is weighted by the receiver
          scale `s_r`, so that summing river-zone contributions yields a
          catchment-mean depth (mm/d).

        Args:
            lat_mat: The `(n + 1) x n` adjacency matrix from
                `get_connection_matrices_with_river_row` (last row = river).
            zone_scales: The length-`n` vector of per-zone area fractions.

        Returns:
            NDArray[f64]: The area-weighted lateral matrix, same shape as
                `lat_mat`.

        Raises:
            ModelStructureError: If any zone has a non-positive area, since
                the donor/receiver ratio would be undefined.
        """
        n: int = zone_scales.shape[0]
        HydrologicalModel._check_positive_scales(zone_scales)

        scaled: NDArray[f64] = np.array(lat_mat, dtype=f64, copy=True)

        # Weight each lateral edge d -> r by s_d / s_r. `lat_mat[r, d] == 1`
        # marks an edge; scale each such entry by the donor scale of column d
        # and divide by the receiver scale of row r.
        receiver_scales = zone_scales[:, np.newaxis]  # shape (n, 1)
        donor_scales = zone_scales[np.newaxis, :]  # shape (1, n)
        scaled[:n, :] = scaled[:n, :] * donor_scales / receiver_scales

        # Weight the river-outflow row by the receiver (river-zone) scale.
        scaled[n, :] = scaled[n, :] * zone_scales

        return scaled

    @staticmethod
    def get_scaled_vert_mat(
        vert_mat: NDArray[f64], zone_scales: NDArray[f64]
    ) -> NDArray[f64]:
        """Builds an area-weighted vertical connectivity matrix.

        A vertical edge normally joins two zones of the *same* column, so the
        donor/receiver ratio is 1 and the matrix is unchanged. When a layer
        aggregates (N zones above, 1 below), however, the receiving zone
        represents the combined area of all donors, so the edge must carry the
        ratio `s_d / s_r` for mass to be conserved.

        Args:
            vert_mat: The `n x n` vertical adjacency matrix.
            zone_scales: The length-`n` vector of per-zone area fractions.

        Returns:
            NDArray[f64]: The area-weighted vertical matrix, same shape as
                `vert_mat`.

        Raises:
            ModelStructureError: If any zone has a non-positive area, since
                the donor/receiver ratio would be undefined.
        """
        HydrologicalModel._check_positive_scales(zone_scales)

        scaled: NDArray[f64] = np.array(vert_mat, dtype=f64, copy=True)
        receiver_scales = zone_scales[:, np.newaxis]  # shape (n, 1)
        donor_scales = zone_scales[np.newaxis, :]  # shape (1, n)
        scaled = scaled * donor_scales / receiver_scales
        return scaled

    @staticmethod
    def _check_positive_scales(zone_scales: NDArray[f64]) -> None:
        """Raise if any per-zone area fraction is not strictly positive.

        The area-weighted transfer uses a donor/receiver ratio, which is
        undefined when a zone occupies zero area.
        """
        if np.any(zone_scales <= 0):
            bad = [int(i) for i in np.where(zone_scales <= 0)[0]]
            raise ModelStructureError(
                "All zones must have a strictly positive area fraction, but "
                f"zones at indices {bad} have scale <= 0. The area-weighted "
                "transfer uses a donor/receiver area ratio that requires "
                "every zone to occupy a non-zero area."
            )

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
                raise ModelStructureError(
                    f"Invalid model structure: layer {l_idx} has "
                    f"{num_zones_current} zones, but layer {l_idx + 1} has "
                    f"{num_zones_next} zones. Each layer must contain either "
                    "1 zone or the same number of zones as the layer above "
                    "it, so that vertical flux is unambiguous."
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

    def structure_diagram(self) -> str:
        """Return a boxed ASCII diagram of the model structure.

        Renders each hydrologic zone (with its parameter values) as a box,
        stacked vertically by layer, with a wavy river channel on the right.

        Returns:
            str: The multi-line ASCII representation.
        """
        num_cols = max(len(layer) for layer in self.structure)

        zone_lines: list[list[list[str]]] = []
        for l, layer in enumerate(self.structure):
            layer_zones = self.layers[l].zones
            layer_lines: list[list[str]] = []
            for c, zone in enumerate(layer_zones):
                names = zone.parameter_names()  # type: ignore[attr-defined]
                values = zone.param_list()  # type: ignore[attr-defined]
                lines: list[str] = [self.structure[l][c].name]  # type: ignore[attr-defined]
                for k, v in zip(names, values):
                    lines.append(f"{k} = {str(round(float(v), 4))}")
                layer_lines.append(lines)
            zone_lines.append(layer_lines)

        col_widths = [0] * num_cols
        for layer_lines in zone_lines:
            for c, lines in enumerate(layer_lines):
                for line in lines:
                    col_widths[c] = max(col_widths[c], len(line))

        RIVER_LABEL = "River"
        river_width = len(RIVER_LABEL)

        rows: list[str] = []

        def make_border() -> str:
            b = "+"
            for c in range(num_cols):
                b += "-" * (col_widths[c] + 2) + "+"
            b += "~" * (river_width + 2) + "+"
            return b

        for i, layer_lines in enumerate(zone_lines):
            if not layer_lines:
                continue
            rows.append(make_border())
            band_height = max(len(lines) for lines in layer_lines)
            for row_idx in range(band_height):
                content = "|"
                for c in range(num_cols):
                    if c < len(layer_lines) and row_idx < len(layer_lines[c]):
                        text = layer_lines[c][row_idx]
                    else:
                        text = ""
                    content += f" {text.ljust(col_widths[c])} |"
                if i == 0 and row_idx == 0:
                    river_text = RIVER_LABEL
                else:
                    river_text = ""
                content += f" {river_text.ljust(river_width)} |"
                rows.append(content)

        rows.append(make_border())
        return "\n".join(rows)

    def _hydro_parameter_rows(self) -> list[dict]:
        """Build parameter description rows for every hydro zone, in layer order."""
        rows: list[dict] = []
        for zone in self.flat_model:
            names = zone.parameter_names()  # type: ignore[attr-defined]
            values = zone.param_list()  # type: ignore[attr-defined]
            rows += zone_parameter_rows(zone, "hydro", names=names, values=values)
        for i in range(len(self.structure[0]) - 1):
            rows.append(
                {
                    "zone": "model",
                    "name": f"proportion.{i + 1}",
                    "value": _fmt(self.__scales[i]),
                    "unit": "",
                    "description": "Fractional area contribution of the surface zone.",
                }
            )
        return rows

    def describe_parameters(self) -> DataFrame:
        """Return a table describing every tunable parameter in the model.

        The base (pure-hydrology) model reports one row per hydro-zone
        parameter plus the catchment area proportions. Subclasses that add
        reactive-transport or river zones override this and append their own
        rows.

        Returns:
            DataFrame: Columns are ``[zone, name, value, unit, description]``.
        """
        rows = self._hydro_parameter_rows()
        return DataFrame(rows, columns=["zone", "name", "value", "unit", "description"])

    def describe(self) -> str:
        """Return a human-readable, grouped description of the model parameters.

        For the base model this covers the hydro zones and the catchment area
        proportions. Reactive-transport subclasses override this to also cover
        their RT zones and (if present) the river zone.

        Returns:
            str: A multi-line, sectioned description of the parameters.
        """
        return render_describe(self._hydro_parameter_rows(), title="Model parameters")

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
        r"""Validates the model's structure and configuration.

        Checks for:
        - Correct types in the `structure` attribute.
        - Non-empty structure.
        - Bottom-zone placement rules:
          * a registered bottom zone may not appear above the last layer
            (nothing may be placed below a bottom zone);
          * the last layer must consist of bottom zones only, so that no
            water is lost through the base of the model domain.
        - Valid layer geometry (each layer must have 1 zone or the same
          number of zones as the layer above it).
        - Zone parameters that are finite.
        - `scales` that are finite, strictly positive, and sum to 1.

        Raises:
            TypeError: if a `structure` entry is not a zone.
            ModelStructureError: if any structural rule is violated.
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
                    raise TypeError(f"Zone {j} in layer {i} has an incorrect type: {
                            type(zone)
                        }, ensure that all zones are of type HydrologicZone")

        # Check if model is empty
        if self.structure == []:
            raise ModelStructureError(
                "Model structure is empty, make sure you define a model structure class property describing your model"
            )

        # ==== Bottom-zone rules ==== #
        # These checks run against the *instantiated* zones (self.layers), so
        # that a zone overridden via `zones={...}` is judged by the type that
        # actually runs, not just the type declared in `structure`.
        layer_zones: list[list[HydrologicZone]] = [layer.zones for layer in self.layers]
        num_layers: int = len(layer_zones)
        last_layer: int = num_layers - 1

        for l_idx, layer in enumerate(layer_zones):
            for z_idx, zone in enumerate(layer):
                if is_bottom_zone(zone) and l_idx != last_layer:
                    raise ModelStructureError(
                        f"Bottom zone '{zone.name}' ({type(zone).__name__}) "
                        f"in layer {l_idx} (position {z_idx}) is not "
                        f"allowed: layer {l_idx + 1} sits below it. A "
                        "bottom zone transmits no vertical flux to a zone "
                        "beneath it, so nothing must be placed below it. "
                        "Move the bottom zone to the last layer, or replace "
                        "it with a non-bottom zone."
                    )
                if l_idx == last_layer and not is_bottom_zone(zone):
                    raise ModelStructureError(
                        f"Zone '{zone.name}' ({type(zone).__name__}) in the "
                        f"last layer (layer {l_idx}, position {z_idx}) is not "
                        "a bottom zone: as the lowest zone of the model, it "
                        "would discharge vertical flux out of the domain and "
                        "silently lose water. Use a bottom zone (e.g. "
                        "GroundZoneB / SubsurfaceZoneB), or mark your custom "
                        "zone as one with register_bottom_zone()."
                    )
        # ============================= #

        # Check if model geometry is correct
        layer_lengths: list[int] = [len(layer) for layer in self.structure]
        prev_zones: int = len(self.structure[0])

        for layer_size in layer_lengths:
            if layer_size not in (1, prev_zones):
                raise ModelStructureError(
                    f"Invalid model structure: The next layer can only have either 1 zone or the same number as the zone above, but your model has the structure {
                        layer_lengths
                    }"
                )
            prev_zones = layer_size

        # Check that all scales are valid and add up to 1
        if any(not np.isfinite(s) or s <= 0 for s in self.scales):
            raise ModelStructureError(
                f"Model scales must be finite and strictly positive, got: {
                    self.scales
                }. Every zone must occupy a non-zero area so that the "
                "area-weighted lateral transfer (donor/receiver ratio) is "
                "well defined."
            )
        if abs(sum(self.scales) - 1) > 1e-3:
            raise ModelStructureError(f"Model scales do not add up to 1: {
                    self.scales
                }. Ensure that the scales equal 1 to maintain water balance")

        # Check that every zone parameter is finite (corrupted parameters
        # would otherwise propagate as NaN/Inf through the simulation).
        # This runs against the instantiated zones (self.layers) so that a
        # zone overridden via `zones={...}` is checked with its actual values.
        for l_idx, layer in enumerate(layer_zones):
            for z_idx, zone in enumerate(layer):
                for pname, pval in zip(
                    zone.parameter_names(), zone.param_list(), strict=True
                ):
                    if not np.isfinite(pval):
                        raise ModelStructureError(
                            f"Zone '{zone.name}' ({type(zone).__name__}) in "
                            f"layer {l_idx} (position {z_idx}) has a "
                            f"non-finite parameter: {pname} = {pval!r}. "
                            "All zone parameters must be finite numbers."
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
                    f"but {len(forcing_data)} were provided in the 'forc' list."
                )
            for i, fd in enumerate(forcing_data):
                if not (
                    len(fd.precip) == num_steps
                    and len(fd.temp) == num_steps
                    and len(fd.pet) == num_steps
                ):
                    raise ValueError(f"ForcingData at index {
                            i
                        } has series lengths inconsistent with num_steps ({
                            num_steps
                        }). " f"P length: {len(fd.precip)}, T length: {
                            len(fd.temp)
                        }, PET length: {len(fd.pet)}")
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
                raise ValueError(f"Model expects {
                        num_forcing_sources_expected
                    } forcing inputs, but 'forcing_data' list is empty.")
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
            [zone.default_init_state() for layer in cls.structure for zone in layer]  # type: ignore
        )

    @staticmethod
    def _series_to_floats(series: Series, label: str) -> NDArray[np.float64]:
        """Convert a forcing time series to a float64 array, raising a
        descriptive error if it contains non-numeric entries."""
        try:
            return np.asarray(series.to_numpy(), dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise ModelInputError(
                f"The '{label}' series must contain only numeric values "
                f"(found non-numeric entries): {e}"
            ) from e

    def _validate_hydro_inputs(
        self,
        forc: ForcingData | Iterable[ForcingData],
        meas_streamflow: Optional[Series] = None,
        init_state: Optional[NDArray[f64]] = None,
    ) -> None:
        """Validate the run-time inputs before any model steps are taken.

        All checks are hard errors (`ModelInputError`) so that corrupted or
        unphysical input data is rejected before the simulation starts,
        rather than producing NaN-filled results part-way through a run.

        Checks:
        - 1 or one-per-surface-zone `ForcingData` objects, each with
          equal-length precip/temp/pet series;
        - no non-finite (NaN/Inf) values in precip, temp, or pet;
        - precipitation and PET are non-negative at every step;
        - measured streamflow (if given) has the same length as the forcing
          series and contains no infinite or negative values (NaN entries
          mark missing measurements and are excluded from the objective
          functions);
        - `init_state` (if given) has one value per zone, all finite and
          non-negative.

        Raises:
            ModelInputError: if any check fails, with a message that
                identifies the offending series/value.
        """
        forcing_series: list[ForcingData]
        if isinstance(forc, ForcingData):
            forcing_series = [forc]
        else:
            forcing_series = list(forc)

        num_surface_zones: int = len(self.surface_zone_ids)
        if len(forcing_series) not in (1, num_surface_zones):
            raise ModelInputError(
                f"Expected 1 or {num_surface_zones} ForcingData objects "
                f"(one per surface zone), got {len(forcing_series)}."
            )

        if not forcing_series:
            raise ModelInputError(
                "No forcing data was provided; a ForcingData object with "
                "precip, temp, and pet series is required."
            )

        num_steps: int = len(forcing_series[0].precip)
        if num_steps == 0:
            raise ModelInputError(
                "The forcing series are empty: there is no data to simulate."
            )

        for i, fd in enumerate(forcing_series):
            for field in ("precip", "temp", "pet"):
                series = Series.__getattribute__(fd, field)
                if len(series) != num_steps:
                    raise ModelInputError(
                        f"ForcingData #{i}: the '{field}' series has length "
                        f"{len(series)}, but the precipitation series has "
                        f"length {num_steps}. precip, temp, and pet must all "
                        "have the same length."
                    )
                values = self._series_to_floats(series, f"'{field}' (ForcingData #{i})")
                if not np.all(np.isfinite(values)):
                    bad_idx = np.where(~np.isfinite(values))[0][:5]
                    raise ModelInputError(
                        f"ForcingData #{i}: the '{field}' series contains "
                        f"non-finite (NaN/Inf) values at index"
                        f"{'s' if len(bad_idx) > 1 else ''} {
                            list(map(int, bad_idx))
                        } (first values: {list(values[bad_idx])}). "
                        "Replace the missing/corrupted values before "
                        "running the model."
                    )

            precip_vals = self._series_to_floats(
                fd.precip, "precip (ForcingData "
                f"#{i})"
            )
            if np.any(precip_vals < 0):
                raise ModelInputError(
                    f"ForcingData #{i}: precipitation contains negative "
                    f"values (min = {float(np.min(precip_vals)):.6g}). "
                    "Precipitation must be >= 0 at every step."
                )

            pet_vals = self._series_to_floats(fd.pet, "pet (ForcingData "
                                                     f"#{i})")
            if np.any(pet_vals < 0):
                raise ModelInputError(
                    f"ForcingData #{i}: PET contains negative values "
                    f"(min = {float(np.min(pet_vals)):.6g}). PET must be "
                    ">= 0 at every step."
                )

        if meas_streamflow is not None:
            if len(meas_streamflow) != num_steps:
                raise ModelInputError(
                    f"meas_streamflow has length {len(meas_streamflow)}, but "
                    f"the forcing series have length {num_steps}. They must "
                    "have the same length."
                )
            stream_vals = self._series_to_floats(meas_streamflow, "meas_streamflow")
            inf_idx = np.where(np.isinf(stream_vals))[0][:5]
            if inf_idx.size:
                raise ModelInputError(
                    "meas_streamflow contains infinite values at "
                    f"{'indices' if inf_idx.size > 1 else 'index'} "
                    f"{list(map(int, inf_idx))}."
                )
            measured_vals = stream_vals[~np.isnan(stream_vals)]
            if measured_vals.size and np.any(measured_vals < 0):
                raise ModelInputError(
                    "meas_streamflow contains negative values "
                    f"(min = {float(np.min(measured_vals)):.6g}). Measured "
                    "streamflow must be >= 0."
                )

        if init_state is not None:
            state = np.asarray(init_state, dtype=np.float64)
            if state.ndim != 1:
                raise ModelInputError(
                    f"init_state must be a 1-dimensional array (got ndim = "
                    f"{state.ndim})."
                )
            if len(state) != self.num_zones:
                raise ModelInputError(
                    f"init_state has length {len(state)}, but the model has "
                    f"{self.num_zones} zones. One initial storage value per "
                    f"zone is required, in the order: {self.zone_names}."
                )
            if not np.all(np.isfinite(state)):
                nans = [
                    int(i)
                    for i in np.where(~np.isfinite(state))[0][:5]
                ]
                bad_zones = [self.zone_names[i] for i in nans]
                raise ModelInputError(
                    f"init_state contains non-finite (NaN/Inf) values at "
                    f"zone index(es) {nans} (zones {bad_zones})."
                )
            if np.any(state < 0):
                raise ModelInputError(
                    "init_state contains negative zone storage values "
                    f"(min = {float(np.min(state)):.6g}). Zone storages must "
                    "be >= 0."
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
                q_in_lat = np.dot(
                    self.lat_mat_scaled[i, :num_fluxes], lat_ext_fluxes[: i + 1]
                )
                q_in_vert = np.dot(
                    self.vert_mat_scaled[i, :num_fluxes], vert_ext_fluxes[: i + 1]
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
                    raise HydrologyNumericalError(
                        type(self),
                        zone,
                        self.hydro_to_array(),
                        s_i,
                        d_i_zone.copy(),
                        "Water balance failed in zone "
                        f"'{zone.name}' ({type(zone).__name__}):\n"
                        f"  initial storage (s_i): {s_i:+.6f}\n"
                        f"  final storage (s_new): {s_new:+.6f}\n"
                        f"  q_in:        {d_i_zone.q_in:+.6f}\n"
                        f"  forc_flux:   {step_res.forc_flux:+.6f}\n"
                        f"  vap_flux:    {step_res.vap_flux:+.6f}\n"
                        f"  lat_flux:    {step_res.lat_flux:+.6f}\n"
                        f"  vert_flux:   {step_res.vert_flux:+.6f}\n"
                        f"  residual (mass_err): {mass_err:+.6f} "
                        f"(tolerance: 1e-3)\n"
                        f"  dt: {dt!r}\n"
                        f"  forcing: {d_i_zone}\n"
                        f"  step:  {step_res}",
                    )
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
        save_results: Optional[bool] = None,
        results_dir: Optional[str] = None,
    ) -> HydroModelResults:
        """Runs a forward simulation of the hydrologic model.

        Args:
            forc (ForcingData | list[ForcingData]): A single `ForcingData` object
                or a list of them (one for each surface zone).
            init_state (Optional[NDArray[f64]]): The initial storage for each
                zone. If None, uses `default_init_state()`. Defaults to None.
            meas_streamflow (Optional[Series]): A time series of observed
                streamflow for calculating performance metrics. Missing days
                may be NaN and are excluded from every metric; infinite and
                negative values are rejected. Defaults to None.
            average_elevation (Optional[float]): The average elevation of the
                meteorological gauge (required for lapse rates). Defaults to None.
            elevations (Optional[list[float]]): A list of mean elevations for
                each surface zone (required for lapse rates). Defaults to None.
            verbose (bool): Not currently used.
            save_results (Optional[bool]): Whether to save this run's results
                (and the model's parameters) to disk. If None, falls back to
                the `POTIONS_SAVE_RESULTS` environment variable, and if that is
                unset, saving is on by default. Defaults to None.
            results_dir (Optional[str]): The base directory under which the
                `run_<timestamp>` folder is created. If None, falls back to the
                `POTIONS_RESULTS_DIR` environment variable, then to the default
                `"results"`. Only used when saving is enabled. Defaults to None.

        Returns:
            HydroModelResults: A `HydroModelResults` dictionary. The
                `saved_dir` attribute of the returned object holds the path
                the run was saved to (or None if saving was disabled).
        """
        # Check the model structure to make sure that all of the zones are correct
        self._validate_model_structure()

        # Check the inputs (forcing data, measured streamflow, initial state)
        # so that missing/corrupted/unphysical data is rejected before any
        # simulation steps, with a message that pinpoints the problem.
        self._validate_hydro_inputs(
            forc=forc, meas_streamflow=meas_streamflow, init_state=init_state
        )

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

        # Weight each river zone's lateral outflow by its area fraction, so
        # the outlet is a catchment-mean depth (mm/d) rather than an
        # unweighted sum of intensive zone fluxes.
        zone_names: list[str] = self.get_zone_names()
        river_zone_names: list[str] = [zone_names[i] for i in river_zone_ids]
        q_components: dict[str, Series] = {
            name: model_res[f"q_lat_{name}"] * float(self.zone_scales[river_zone_ids[k]])
            for k, name in enumerate(river_zone_names)
        }
        sim_streamflow: Series = sum(q_components.values())  # type: ignore
        sim_streamflow.name = SIM_Q_NAME
        model_res[SIM_Q_NAME] = sim_streamflow
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

        # Report the area-weighted share of streamflow from each river zone.
        props: dict[str, float] = {}
        for name in river_zone_names:
            prop_name: str = f"prop_q_{name}"
            model_res[prop_name] = q_components[name] / model_res[SIM_Q_NAME]
            props[prop_name] = model_res[prop_name].mean()

        # Create the object and save
        res = HydroModelResults(
            simulation=model_res, objective_functions=obj_val_ser, forcing=forc
        )

        # Save the run's results and parameters (on by default; see result_saving).
        if is_save_enabled(save_results):
            saved_to = save_hydro_run(self, res, results_dir)
            res.saved_dir = str(saved_to)
            print(f"Saving run results to {saved_to}")

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
                new_zones[zone.name] = zone.from_array(
                    ps, natural_scales=natural_scales
                )  # type: ignore

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
            zone_type: type[HydrologicZone] = cls.get_zone_type(zone_name)  # type: ignore

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
        #     lapse_rates.append(LapseRateParameters.from_dict(param_dict))

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
                    param_ranges[f"{zone.name}.{param_name}"] = zone_range[param_name]

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
        best_results = cls.hydro_from_array(opt_res.x).run_hydro_model(
            forc=forc,  # type: ignore
            init_state=cls.default_hydro_init_state(),
            meas_streamflow=meas_streamflow,
            verbose=False,
            save_results=False,  # don't persist an intermediate optimizer result
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
                forc=forc, elevations=mean_elev, meas_streamflow=meas_streamflow,
                save_results=False,  # don't persist finite-difference evaluations
            )
            right_res = right_model.run_hydro_model(
                forc=forc, elevations=mean_elev, meas_streamflow=meas_streamflow,
                save_results=False,
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
