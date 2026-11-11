"""
Non-Qt tests for the potions.gui package.

Covers the pure-Python layer of the GUI:

* `potions.gui.data` — the input loader and the parameter-array
  conversions (``arr_to_model`` / ``arr_to_results`` /
  ``arr_to_hydro_results`` and the ``GuiRunResults`` results
  wrapper; the real model run is the ``slow``-marked end-to-end
  test).
* `potions.gui.common` — the parameter-label and bound-precision
  helpers and the module constants the panel and the figures use.
* `potions.gui.state` — the ``gui_state.json`` round trip, its
  error paths, the saved-run discovery, the restore of full and
  legacy (pre-GUI) saved runs, and the abbreviated run-history file
  (the record type, appending, and the objective extraction).
* `potions.gui.gradients` — the finite-difference gradient
  calculator: the free-parameter perturbation, the objective
  vector, the difference matrix, and the timestamped CSV export.
* `potions.gui.model_pool` — the worker pool the jobs use to run a
  batch of parameter points: the per-worker initializer and the
  per-point results / error-string reporting.
* `potions.gui.plots` — the four figure builders and the objective
  table, asserted on the figures' artists (titles, axes, series
  counts, legend entries, colors, linestyles, clipping).

The Qt-layer tests live in `test_gui_qt.py`.
"""

import hashlib
import json
import math
import multiprocessing
import os
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterable

import matplotlib
import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
matplotlib.use("Agg")

import potions as pt
from potions.hydro_model import HydrologicalModel
from potions.reactive_transport.common_networks import (
    get_chloride_network,
    get_soc_calcite_network,
)
from potions.result_saving import (
    DEFAULT_RESULTS_DIR,
    save_hydro_run,
    save_model_run,
)

from ..gui import common as gui_common
from ..gui import data as gui_data
from ..gui import gradients as gui_gradients
from ..gui import main as gui_main
from ..gui import model_pool
from ..gui import model_specs
from ..gui import network_editor as gui_network_editor
from ..gui import network_specs
from ..gui import plots as gui_plots
from ..gui import state as gui_state
from ..gui.common import (
    HYDRO_PARAMS,
    LINESTYLE_CYCLE,
    MEAS_Q_COLOR,
    METRIC_LABELS,
    METRIC_NAMES,
    PARAM_LABELS,
    RIVER_SPECIES,
    RIVER_SPECIES_MEAS_COLORS,
    RIVER_SPECIES_SIM_COLORS,
    SIM_Q_COLOR,
    SNOW_ZONES,
    SPECIATION_ZONES,
    WATER_YEARS,
    ZONE_COLORS,
    ZONE_LABELS,
    ZONE_ORDER,
    ZONE_STACK_ORDER,
    bound_decimals,
    param_label,
    param_symbol,
    water_year_window,
)
from ..gui.state import (
    GUI_STATE_VERSION,
    MODE_FULL,
    MODE_HYDRO,
    RUN_HISTORY_FILENAME,
    GuiState,
    RunHistoryEntry,
    append_run_history,
    find_latest_run_dir,
    list_run_dirs,
    load_saved_run,
    objectives_from_results,
    read_gui_state,
    saved_run_model_key,
    write_gui_state,
)
from ..reactive_transport.common_networks import (
    get_carbon_network,
    get_chloride_network,
    get_soc_calcite_network,
)

# ------------------------------------------------------------------ #
# Fixtures and helpers
# ------------------------------------------------------------------ #


@pytest.fixture(scope="session")
def input_data() -> gui_data.GuiInputData:
    """The package's own East River input data (the app's default)."""
    return gui_data.load_input_data(gui_data.DEFAULT_CSV_PATH)


@pytest.fixture(scope="session")
def model_inputs() -> dict[str, gui_data.GuiInputData]:
    """The East River inputs assembled for every curated model spec."""
    return {
        key: gui_data.load_input_data(
            gui_data.DEFAULT_CSV_PATH, model_key=key
        )
        for key in model_specs.MODEL_SPECS
    }


def _synthetic_hydro_sim(
    index: pd.Index,
    zones: list[str] = ZONE_ORDER,
    snow_zones: list[str] = SNOW_ZONES,
) -> pd.DataFrame:
    """A synthetic hydrology simulation frame with the figure columns."""
    data: dict[str, np.ndarray] = {
        "streamflow_sim": np.arange(len(index), dtype=float)
    }
    for i, zone in enumerate(zones):
        share: float = (i + 1) * 0.1
        data[f"s_{zone}"] = np.arange(len(index), dtype=float)
        data[f"q_lat_{zone}"] = share * data["streamflow_sim"]
        # Only the river zones partition the streamflow (the snow
        # zones discharge to the ground zones, not to the river).
        if zone not in snow_zones:
            data[f"prop_q_{zone}"] = np.full(len(index), share)
    return pd.DataFrame(data, index=index)


def _synthetic_rt_sim(
    index: pd.Index,
    minerals: list[str],
    zones: list[str] = ZONE_ORDER,
    snow_zones: list[str] = SNOW_ZONES,
) -> pd.DataFrame:
    """
    A synthetic reactive-transport frame with every plotted column.

    The frame mirrors the model's real schema: every river species is
    carried as a ``{species}_riv`` column, but only the network species
    carry a ``{species}_{zone}`` column.  DIC is a derived species, so
    it appears only as ``DIC_riv``.
    """
    river_species: list[str] = ["DOC", "DIC", "Ca++"]
    network_species: list[str] = ["DOC", "Ca++"]
    non_snow_zones: list[str] = [
        zone for zone in zones if zone not in snow_zones
    ]
    data: dict[str, np.ndarray] = {}
    for species_name in river_species:
        data[f"{species_name}_riv"] = np.arange(len(index), dtype=float)
    for species_name in network_species:
        for zone in zones:
            data[f"{species_name}_{zone}"] = np.arange(len(index), dtype=float)
    for mineral in minerals:
        for zone in non_snow_zones:
            data[f"{mineral}_conc_{zone}"] = np.arange(len(index), dtype=float)
            data[f"{mineral}_rate_{zone}"] = np.arange(len(index), dtype=float)
    return pd.DataFrame(data, index=index)


def _synthetic_results(
    input_data: gui_data.GuiInputData,
) -> tuple[pt.HydroModelResults, pt.RtModelResults]:
    """Synthetic hydrology and reactive-transport results objects."""
    index: pd.Index = input_data.forc.precip.index
    zones: list[str] = list(input_data.spec.zones)
    snow_zones: list[str] = list(input_data.spec.snow_zones)
    hydro: pt.HydroModelResults = pt.HydroModelResults(
        forcing=input_data.forc,
        simulation=_synthetic_hydro_sim(index, zones, snow_zones),
        objective_functions=pd.Series(
            {name: 0.9 * (i + 1) for i, name in enumerate(METRIC_NAMES)},
            dtype=float,
        ),
        zone_names=zones,
    )
    rt: pt.RtModelResults = pt.RtModelResults(
        simulation=_synthetic_rt_sim(
            index,
            list(input_data.network.mineral_names),
            zones,
            snow_zones,
        ),
        objective_functions=pd.DataFrame(
            {
                species: [0.5 + 0.1 * i for i in range(len(METRIC_NAMES))]
                for species in ["DOC", "DIC", "Ca++"]
            },
            index=METRIC_NAMES,
        ).T,
        saved_dir=None,
        zone_names=zones,
    )
    return hydro, rt


def _write_saved_run(
    base_dir: Path,
    input_data: gui_data.GuiInputData,
    state: GuiState | None = None,
    hydro_only: bool = False,
    with_state: bool = True,
) -> Path:
    """
    Write a synthetic saved run into `base_dir / "results"`.

    In full mode the run is saved via the real `save_model_run` (the
    ``hydro`` and ``rt`` bundle pair); in hydrology-only mode via the
    real `save_hydro_run` (the flat hydrology bundle, whose 19-row
    parameters.csv is layer-named and maps only the model-level
    proportion row back onto a bound).  A gui_state.json is written
    when `state` is given and `with_state` is on.

    Returns:
        The created run directory.
    """
    hydro, rt = _synthetic_results(input_data)
    if hydro_only:
        model_class: type[pt.Model] = input_data.spec.model_class
        num_hydro: int = model_class.get_num_hydro_parameters()
        hydro_model: HydrologicalModel = model_class.hydro_from_array(
            _midpoint_vector(input_data)[:num_hydro],
            natural_scales=True,
        )
        run_dir = save_hydro_run(
            hydro_model, hydro, results_dir=base_dir / "results"
        )
    else:
        model, _init_arr = gui_data.arr_to_model(
            _midpoint_vector(input_data), input_data
        )
        run_dir = save_model_run(
            model, hydro, rt, results_dir=base_dir / "results"
        )
    if state is not None and with_state:
        write_gui_state(run_dir, state)
    return run_dir


def _midpoint_vector(input_data: gui_data.GuiInputData) -> np.ndarray:
    """The parameter vector at every bound's midpoint."""
    return np.array(
        [
            0.5 * (lo + hi)
            for lo, hi in input_data.bounds.values()
        ]
    )


# The re-recorded reactive objectives of the default midpoint run
# (task 11, re-record approved at the step-3 STOP).  The preflight
# fixture recorded the pre-change run, whose DOC series sat at the
# ~1e-40 numerical floor (bias exactly -1.0, log KGE NaN).  T5's
# generalized exchange initialization is mathematically identical
# to the old closed form but rounds the solved 1:1 XDOC initial
# concentration one ULP differently, and that last-bit difference
# moves the nine-year DOC integration to its non-degenerate
# attractor (~3.5e-3 M mean), carrying the carbonate metrics with
# it (localization and proof: ledger, task 11).  Pinned here at
# full precision; the run is deterministic, so test_midpoint_run
# compares against these values at rel=1e-9.
_RECORDED_RT_OBJECTIVES: dict[str, dict[str, float]] = {
    "DOC": {
        "kge": -39.46019316723981,
        "log_kge": -22.956510700231803,
        "nse": -4490.633365881205,
        "log_nse": -4327.648745610105,
        "bias": 37.58628058112267,
        "spearman_rho": 0.10428413876788405,
        "r_squared": 0.0022336319882841804
    },
    "DIC": {
        "kge": -4.855306341392926,
        "log_kge": -4.160813100690826,
        "nse": -297.7006248035903,
        "log_nse": -281.20760980171184,
        "bias": 2.505904979193234,
        "spearman_rho": 0.030765726114249017,
        "r_squared": 0.00153195304613059
    },
    "Ca++": {
        "kge": -4.347626716551775,
        "log_kge": -3.8330726417149155,
        "nse": -149.55663151179246,
        "log_nse": -144.9943038130825,
        "bias": 2.1113714014208607,
        "spearman_rho": 0.0680251942572137,
        "r_squared": 0.0014945029868255186
    },
}


def _load_baseline() -> Any:
    """
    The re-recorded default-run baseline (task 11).

    The fixture was re-recorded by this task with the current
    tree's midpoint-run objectives (the pre-change values remain in
    the preflight commit's copy of the file).  The fixture is
    untyped JSON, so the parsed value comes back as `Any`; each use
    below reads a known key and compares it with a float, so the
    looseness stays at this one boundary.
    """
    path: Path = Path(__file__).parent / "baseline_midpoint.json"
    return json.loads(path.read_text())


def _full_state(
    input_data: gui_data.GuiInputData,
    save_history: bool = False,
    mode: str = MODE_FULL,
    objective_terms: list[str] | None = None,
    model_key: str = model_specs.DEFAULT_MODEL_KEY,
    network_fingerprint: str | None = None,
) -> GuiState:
    """A complete gui_state matching the loaded input data."""
    bounds: dict[str, tuple[float, float]] = dict(input_data.bounds)
    values: dict[str, float] = {
        key: 0.5 * (lo + hi) for key, (lo, hi) in bounds.items()
    }
    if "proportion.1" in values:
        values["proportion.1"] = 0.7
    return GuiState(
        csv_path=str(gui_data.DEFAULT_CSV_PATH),
        param_values=values,
        param_bounds=bounds,
        selection=gui_data.default_plot_selection(),
        window_geometry={"x": 10, "y": 20, "width": 1100, "height": 800},
        splitter_sizes=[300, 600],
        save_history=save_history,
        mode=mode,
        objective_terms=objective_terms,
        model_key=model_key,
        network_fingerprint=network_fingerprint,
    )



# ------------------------------------------------------------------ #
# potions.gui.data
# ------------------------------------------------------------------ #


class TestLoadInputData:
    """The default East River input data loads with its known shapes."""

    def test_indices_and_frames(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        index: pd.Index = input_data.forc.precip.index
        assert len(index) == 3287
        assert index.name == "date"
        assert index[0] == pd.Timestamp(2012, 10, 1)
        assert index[-1] == pd.Timestamp(2021, 9, 30)

        q: pd.Series = input_data.q_meas
        assert q.iloc[:730].isna().all()
        assert q.iloc[730:].notna().all()
        assert q.index[q.notna()][0] == pd.Timestamp(2014, 10, 1)

        conc: pd.DataFrame = input_data.conc_df
        assert list(conc.columns) == ["DOC", "Ca++", "DIC"]
        assert conc.shape == (3287, 3)
        assert conc.isna().sum().sum() == 6342

        precip: np.ndarray = input_data.precip_conc
        assert precip.shape == (2, 3287, 11)

    def test_network_and_zones(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        assert input_data.network.num_species == 11
        assert input_data.network.num_minerals == 3
        assert list(input_data.network.species_names) == [
            "DOC",
            "Ca++",
            "HCO3-",
            "H+",
            "CO2(aq)",
            "CO3--",
            "SOC(s)",
            "SOC-lz(s)",
            "Calcite(s)",
            "X-",
            "XDOC",
        ]
        assert list(input_data.network.mineral_names) == [
            "SOC(s)",
            "SOC-lz(s)",
            "Calcite(s)",
        ]
        assert list(input_data.zone_configs) == ZONE_ORDER

    def test_bounds(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        bounds: dict[str, tuple[float, float]] = input_data.bounds
        assert len(bounds) == 83
        pinned = sum(1 for lo, hi in bounds.values() if lo == hi)
        assert pinned == 28
        assert 83 - pinned == 55

        assert list(bounds)[0] == "snow_hs.tt"
        assert list(bounds)[-4:] == [
            "initial_state.surface_hs_x",
            "initial_state.surface_rp_x",
            "initial_state.ground_hs_x",
            "initial_state.ground_rp_x",
        ]
        assert bounds["proportion.1"] == (0.25, 0.95)
        assert bounds["snow_hs.tt"] == (-1.0, 1.0)
        assert bounds["surface_hs.max_water"] == (150.0, 1500.0)
        assert bounds["ground_hs.max_water"] == (100.0, 5000.0)
        assert bounds["ground_hs.alpha"] == (1.0, 1.0)
        assert bounds["ground_rp.k_a[1]"] == (-15.0, -5.0)

    def test_carries_spec(self) -> None:
        data = gui_data.load_input_data(
            gui_data.DEFAULT_CSV_PATH, model_key="hbv"
        )
        assert data.spec.key == "hbv"
        assert list(data.zone_configs) == data.spec.zones

    def test_rejects_unknown_key(self) -> None:
        with pytest.raises(ValueError, match="nope"):
            gui_data.load_input_data(
                gui_data.DEFAULT_CSV_PATH, model_key="nope"
            )


class TestModelBounds:
    """The model-generic bounds of every curated spec."""

    @pytest.mark.parametrize(
        ("key", "total", "pinned"),
        [
            ("lateral_three_layer", 83, 28),
            ("three_layer", 41, 13),
            ("hbv", 59, 23),
            ("hbv_lateral", 119, 50),
        ],
    )
    def test_counts(
        self,
        model_inputs: dict[str, gui_data.GuiInputData],
        key: str,
        total: int,
        pinned: int,
    ) -> None:
        bounds = model_inputs[key].bounds
        assert len(bounds) == total
        assert sum(1 for lo, hi in bounds.values() if lo == hi) == pinned

    def test_lateral_unchanged(
        self, model_inputs: dict[str, gui_data.GuiInputData]
    ) -> None:
        bounds = model_inputs["lateral_three_layer"].bounds
        assert bounds["surface_hs.max_water"] == (150.0, 1500.0)
        assert bounds["ground_hs.max_water"] == (100.0, 5000.0)
        assert bounds["ground_hs.alpha"] == (1.0, 1.0)
        assert bounds["proportion.1"] == (0.25, 0.95)

    def test_three_layer_has_no_proportion(
        self, model_inputs: dict[str, gui_data.GuiInputData]
    ) -> None:
        bounds = model_inputs["three_layer"].bounds
        assert "proportion.1" not in bounds
        assert bounds["ground.alpha"] == (0.5, 3.0)

    def test_hbv_zone_classes(
        self, model_inputs: dict[str, gui_data.GuiInputData]
    ) -> None:
        bounds = model_inputs["hbv"].bounds
        assert bounds["shallow.max_water"] == (150.0, 1500.0)
        assert bounds["deep.max_water"] == (100.0, 5000.0)
        assert bounds["shallow.perc"] == (0.0, 5.0)
        assert bounds["shallow.alpha"] == (0.5, 3.0)
        assert "proportion.1" not in bounds
        assert list(bounds)[-3:] == [
            "initial_state.surface_x",
            "initial_state.shallow_x",
            "initial_state.deep_x",
        ]

    def test_hbv_lateral_pins_all_subsurface_alphas(
        self, model_inputs: dict[str, gui_data.GuiInputData]
    ) -> None:
        bounds = model_inputs["hbv_lateral"].bounds
        assert "proportion.1" in bounds
        for zone in ("shallow_hs", "shallow_rp", "deep_hs", "deep_rp"):
            assert bounds[f"{zone}.alpha"] == (1.0, 1.0)

    def test_initial_state_by_zone_class(
        self, model_inputs: dict[str, gui_data.GuiInputData]
    ) -> None:
        bounds = model_inputs["hbv"].bounds
        assert bounds["initial_state.surface_x"] == (-5.0, -1.0)
        assert bounds["initial_state.shallow_x"] == (-5.0, -1.0)
        assert bounds["initial_state.deep_x"] == (-7.0, -3.0)

    def test_precip_conc_has_one_row_per_surface_zone(
        self, model_inputs: dict[str, gui_data.GuiInputData]
    ) -> None:
        for data in model_inputs.values():
            assert data.precip_conc.shape[0] == len(
                data.spec.model_class.structure[0]
            )


class TestArrToModel:
    """The array-to-model conversion and its description table."""

    def test_midpoint_values(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        model, init_arr = gui_data.arr_to_model(
            _midpoint_vector(input_data), input_data
        )
        assert np.allclose(init_arr, [-3.0, -3.0, -5.0, -5.0])

    def test_describe_parameters(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        model, _init_arr = gui_data.arr_to_model(
            _midpoint_vector(input_data), input_data
        )
        table: pd.DataFrame = model.describe_parameters()
        assert table.shape == (79, 5)
        assert list(table.columns) == [
            "zone",
            "name",
            "value",
            "unit",
            "description",
        ]

    def test_non_midpoint_values(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        x: np.ndarray = _midpoint_vector(input_data).copy()
        x[0] += 0.25  # snow_hs.tt (its midpoint is 0.0)
        x[2] -= 0.5  # snow_rp.tt
        model, _init_arr = gui_data.arr_to_model(x, input_data)
        table: pd.DataFrame = model.describe_parameters()
        # The hydrology parameters are described per hydrology *layer*
        # (zone "snow", "surface", "ground_bottom"), one row pair per
        # region, in zone order; values are stored as strings.
        tt_rows: list[float] = [
            float(value)
            for value in table.loc[table["name"] == "tt", "value"]
        ]
        assert tt_rows == pytest.approx([0.25, -0.5])
        proportion: float = float(
            table.loc[table["zone"] == "model", "value"].iloc[0]
        )
        assert proportion == pytest.approx(0.6)




# ------------------------------------------------------------------ #
# potions.gui.common helpers
# ------------------------------------------------------------------ #

_MINERAL_NAMES: list[str] = ["SOC(s)", "SOC-lz(s)", "Calcite(s)"]


def test_lateral_stack_order_is_deepest_first() -> None:
    """The partitioning stack lists the deepest layer first."""
    assert ZONE_STACK_ORDER == [
        "ground_hs",
        "ground_rp",
        "surface_hs",
        "surface_rp",
        "snow_hs",
        "snow_rp",
    ]


class TestParamLabel:
    """Human-readable labels for every parameter-key form."""

    def test_proportion(self) -> None:
        assert (
            param_label("proportion.1", _MINERAL_NAMES)
            == "Distal proportion of catchment"
        )

    def test_hydro_params(self) -> None:
        assert param_label("snow_hs.tt", _MINERAL_NAMES) == PARAM_LABELS["tt"]
        assert param_label("ground_rp.k", _MINERAL_NAMES) == PARAM_LABELS["k"]

    def test_rt_dimension_params(self) -> None:
        assert (
            param_label("surface_hs.max_water", _MINERAL_NAMES)
            == PARAM_LABELS["max_water"]
        )

    def test_mineral_params(self) -> None:
        label: str = param_label("ground_rp.k_a[1]", _MINERAL_NAMES)
        assert label == f"SOC-lz(s) {PARAM_LABELS['k_a']}"

    def test_initial_state(self) -> None:
        assert (
            param_label("initial_state.ground_rp_x", _MINERAL_NAMES)
            == "Ground (RP) total sorption sites"
        )

    def test_initial_state_label_uses_given_zone_labels(self) -> None:
        labels: dict[str, str] = {"deep_hs": "Deep (HS)"}
        assert (
            param_label(
                "initial_state.deep_hs_x",
                _MINERAL_NAMES,
                zone_labels=labels,
            )
            == "Deep (HS) total sorption sites"
        )

    def test_percolation_label(self) -> None:
        assert (
            param_label("shallow.perc", _MINERAL_NAMES)
            == "percolation rate [mm/d]"
        )

    def test_unknown_key(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            param_label("bogus.key", _MINERAL_NAMES)
        assert str(excinfo.value) == "Unrecognized parameter key: 'bogus.key'"

    def test_unknown_mineral_field(self) -> None:
        # A mineral key naming an unknown field fails at the label
        # lookup (KeyError), unlike the other unrecognized forms.
        with pytest.raises(KeyError):
            param_label("ground_rp.bogus[0]", _MINERAL_NAMES)


class TestBoundDecimals:
    """The spin box's display precision follows the range's span."""

    @pytest.mark.parametrize(
        ("lo", "hi", "decimals"),
        [
            (0.0, 1.0, 4),
            (0.25, 0.95, 5),
            (-1.0, 1.0, 4),
            (150.0, 1500.0, 1),
            (-5.0, -1.0, 4),
            (5000.0, 5000.0, 4),
            (-15.0, -5.0, 3),
            (0.5, 2.5, 4),
            (0.05, 1.0, 5),
            (50.0, 1000.0, 2),
        ],
    )
    def test_bound_decimals(
        self, lo: float, hi: float, decimals: int
    ) -> None:
        assert bound_decimals(lo, hi) == decimals


class TestParamSymbol:
    """The short symbol shown in a cell for each parameter-key form."""

    def test_proportion(self) -> None:
        assert param_symbol("proportion.1") == "proportion"

    def test_hydro(self) -> None:
        assert param_symbol("snow_hs.tt") == "tt"

    def test_reactive_storage(self) -> None:
        assert param_symbol("surface_hs.max_water") == "max_water"

    def test_ground(self) -> None:
        assert param_symbol("ground_rp.k") == "k"

    def test_mineral(self) -> None:
        assert param_symbol("ground_rp.k_a[1]") == "k_a"

    def test_initial_state(self) -> None:
        assert param_symbol("initial_state.ground_rp_x") == "x"

    def test_unknown_field(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            param_symbol("ground_rp.bogus[0]")
        assert (
            str(excinfo.value)
            == "Unrecognized parameter key: 'ground_rp.bogus[0]'"
        )

    def test_no_dot(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            param_symbol("bogus")
        assert str(excinfo.value) == "Unrecognized parameter key: 'bogus'"


class TestSpeciesPalette:
    """The session's species color maps follow the run's measured species."""

    def test_legacy_colors_unchanged(self) -> None:
        """The default network's measured species keep the published colors."""
        try:
            gui_common.set_species_colors(["DOC", "Ca++", "DIC"])
            assert gui_common.RIVER_SPECIES_SIM_COLORS == {
                "DOC": "sandybrown",
                "DIC": "turquoise",
                "Ca++": "lightcoral",
            }
            assert gui_common.RIVER_SPECIES_MEAS_COLORS == {
                "DOC": "peru",
                "DIC": "teal",
                "Ca++": "crimson",
            }
        finally:
            gui_common.set_species_colors(list(RIVER_SPECIES))

    def test_new_species_take_palette_hues(self) -> None:
        """Novel species take the next palette hue in measured order."""
        try:
            gui_common.set_species_colors(["DOC", "Na+"])
            assert gui_common.RIVER_SPECIES_SIM_COLORS["DOC"] == "sandybrown"
            sim: str = gui_common.RIVER_SPECIES_SIM_COLORS["Na+"]
            meas: str = gui_common.RIVER_SPECIES_MEAS_COLORS["Na+"]
            assert sim == gui_common.SPECIES_SIM_PALETTE[1]
            assert meas == gui_common.SPECIES_MEAS_PALETTE[1]
            assert sim == "turquoise"
            assert meas == "teal"
        finally:
            gui_common.set_species_colors(list(RIVER_SPECIES))

    def test_palette_wraps_at_eight(self) -> None:
        """Beyond eight species the palette wraps around."""
        names: list[str] = [f"novel_{i}" for i in range(9)]
        try:
            gui_common.set_species_colors(names)
            sim: dict[str, str] = dict(gui_common.RIVER_SPECIES_SIM_COLORS)
            meas: dict[str, str] = dict(gui_common.RIVER_SPECIES_MEAS_COLORS)
            assert sim[names[0]] == gui_common.SPECIES_SIM_PALETTE[0]
            assert sim[names[8]] == gui_common.SPECIES_SIM_PALETTE[0]
            assert sim[names[3]] == gui_common.SPECIES_SIM_PALETTE[3]
            assert sim[names[7]] == gui_common.SPECIES_SIM_PALETTE[7]
            assert sim[names[0]] == "sandybrown"
            assert meas[names[0]] == gui_common.SPECIES_MEAS_PALETTE[0]
            assert meas[names[8]] == gui_common.SPECIES_MEAS_PALETTE[0]
        finally:
            gui_common.set_species_colors(list(RIVER_SPECIES))


class TestWaterYearWindow:
    """The picker's water years and the plotting window they span."""

    def test_water_years_are_the_site_record(self) -> None:
        # The site record runs 2014-05-02 to 2025-09-18: the water
        # years the picker offers are 2015 through 2025 (the last
        # one partial, ending with the record).
        assert WATER_YEARS == list(range(2015, 2026))

    def test_full_range_is_the_site_record(self) -> None:
        start, end = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        # The start of water year 2015: the first day of the site
        # record the loader keeps (the simulation's two-year
        # spin-up ends before it).
        assert start == pd.Timestamp(2014, 10, 1)
        assert end == pd.Timestamp(2025, 9, 30)

    def test_single_year_window(self) -> None:
        start, end = water_year_window(2018, 2018)
        assert start == pd.Timestamp(2017, 10, 1)
        assert end == pd.Timestamp(2018, 9, 30)

    def test_partial_range(self) -> None:
        start, end = water_year_window(2017, 2020)
        assert start == pd.Timestamp(2016, 10, 1)
        assert end == pd.Timestamp(2020, 9, 30)

    def test_year_before_record_rejected(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            water_year_window(2014, 2015)
        assert "2014" in str(excinfo.value)

    def test_year_after_record_rejected(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            water_year_window(2015, 2026)
        assert "2026" in str(excinfo.value)

    def test_reversed_range_rejected(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            water_year_window(2020, 2017)
        assert "2020" in str(excinfo.value)
        assert "2017" in str(excinfo.value)


class TestDefaultPlotSelection:
    """The panel's starting selection matches the app's defaults."""

    def test_defaults(self, input_data: gui_data.GuiInputData) -> None:
        selection = gui_data.default_plot_selection()
        # The default network's measured species, in species order.
        assert selection.species == ["DOC", "Ca++", "DIC"]
        assert selection.zones == [
            "surface_hs",
            "surface_rp",
            "ground_hs",
            "ground_rp",
        ]
        assert selection.mineral == "SOC(s)"
        assert selection.streamflow is True
        assert selection.window_start == gui_data.DEFAULT_WINDOW_START
        assert selection.window_end == gui_data.DEFAULT_WINDOW_END

    def test_excludes_snow_zones_for_every_spec(
        self, model_inputs: dict[str, gui_data.GuiInputData]
    ) -> None:
        for data in model_inputs.values():
            selection = gui_data.default_plot_selection(data.spec)
            assert selection.zones == [
                zone
                for zone in data.spec.zones
                if zone not in data.spec.snow_zones
            ]

    def test_measured_species_follow_the_network(self) -> None:
        assert gui_data.network_measured_species(
            get_soc_calcite_network()
        ) == ["DOC", "Ca++", "DIC"]
        assert (
            gui_data.network_measured_species(get_carbon_network())
            == ["DOC", "DIC"]
        )
        assert gui_data.network_measured_species(get_chloride_network()) == []

    def test_mineral_follows_the_network(self) -> None:
        assert gui_data.default_plot_mineral() == "SOC(s)"
        assert gui_data.default_plot_mineral(get_carbon_network()) == "SOC(s)"
        assert gui_data.default_plot_mineral(get_chloride_network()) == ""

    def test_chloride_selection_has_no_series(self) -> None:
        """The empty mineral is the signal that hides the rate figure."""
        selection = gui_data.default_plot_selection(network=get_chloride_network())
        assert selection.species == []
        assert selection.mineral == ""



# ------------------------------------------------------------------ #
# potions.gui.state: the gui_state.json round trip
# ------------------------------------------------------------------ #


def _as_dict(raw: dict[str, object], field: str) -> dict[object, object]:
    """
    Narrow a raw gui_state field that must be a mapping.

    `to_dict` returns a `dict[str, object]`, so the nested mappings
    need an explicit isinstance before they can be indexed.
    """
    value: object = raw[field]
    assert isinstance(value, dict), f"Expected the '{field}' field to be a mapping."
    return value


class TestGuiStateRoundTrip:
    """``GuiState`` survives ``to_dict``/``from_dict`` exactly."""

    def test_full_state(self, input_data: gui_data.GuiInputData) -> None:
        state = _full_state(input_data)
        restored = GuiState.from_dict(state.to_dict())
        assert restored == state
        assert GUI_STATE_VERSION == 7
        assert state.to_dict()["version"] == 7

    def test_to_dict_shape(self, input_data: gui_data.GuiInputData) -> None:
        raw: dict[str, object] = _full_state(input_data).to_dict()
        assert raw["version"] == 7
        assert raw["csv_path"] == str(gui_data.DEFAULT_CSV_PATH)
        assert raw["model_key"] == "lateral_three_layer"
        param_values = _as_dict(raw, "param_values")
        assert param_values["proportion.1"] == 0.7
        param_bounds = _as_dict(raw, "param_bounds")
        assert param_bounds["proportion.1"] == [0.25, 0.95]
        selection = _as_dict(raw, "selection")
        # The default network's measured species, in species order.
        assert selection["species"] == ["DOC", "Ca++", "DIC"]
        assert selection["mineral"] == "SOC(s)"
        assert selection["streamflow"] is True
        assert selection["window_start"] == gui_data.DEFAULT_WINDOW_START
        assert selection["window_end"] == gui_data.DEFAULT_WINDOW_END
        # The main vertical splitter's two sections: the figure and
        # the lower (table-and-parameters) row.
        assert raw["splitter_sizes"] == [300, 600]

    def test_legacy_three_size_splitter(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """A pre-version-1 stacked state maps onto the current layout."""
        raw: dict[str, object] = _full_state(input_data).to_dict()
        raw["splitter_sizes"] = [300, 200, 400]
        restored = GuiState.from_dict(raw)
        # The figure keeps its height; the table and the parameters
        # collapse into the lower row at their former combined height.
        assert restored.splitter_sizes == [300, 600]

    def test_legacy_four_size_splitter(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """A version-1 two-splitter state maps onto the current layout."""
        raw: dict[str, object] = _full_state(input_data).to_dict()
        raw["version"] = 1
        raw["splitter_sizes"] = [300, 600, 200, 400]
        restored = GuiState.from_dict(raw)
        # The version-1 inner horizontal pair (table, params) is
        # dropped; the main vertical sections keep their sizes.
        assert restored.splitter_sizes == [300, 600]
        # Version 1 predates the field: a v1-era file restores
        # with no objective selection.
        assert restored.objective_terms is None

    def test_optional_fields_none(self, input_data: gui_data.GuiInputData) -> None:
        state = _full_state(input_data)
        bare = GuiState(
            csv_path=None,
            param_values=state.param_values,
            param_bounds=state.param_bounds,
            selection=state.selection,
            window_geometry=None,
            splitter_sizes=None,
        )
        restored = GuiState.from_dict(bare.to_dict())
        assert restored == bare
        assert restored.csv_path is None
        assert restored.window_geometry is None
        assert restored.splitter_sizes is None
        assert restored.objective_terms is None

    def test_version_key_omitted_means_current(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        raw: dict[str, object] = _full_state(input_data).to_dict()
        del raw["version"]
        assert GuiState.from_dict(raw) == _full_state(input_data)

    def test_save_history_defaults_off(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        state = _full_state(input_data)
        assert state.save_history is False
        raw: dict[str, object] = state.to_dict()
        assert raw["save_history"] is False
        assert GuiState.from_dict(raw) == state

    def test_save_history_on_round_trips(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        state = _full_state(input_data, save_history=True)
        assert state.to_dict()["save_history"] is True
        assert GuiState.from_dict(state.to_dict()) == state

    def test_save_history_absent_restores_off(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """A state saved before the toggle existed restores it off."""
        raw: dict[str, object] = _full_state(input_data).to_dict()
        del raw["save_history"]
        assert GuiState.from_dict(raw).save_history is False

    def test_mode_defaults_to_full(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """A state built without a mode is full and records "full"."""
        state = _full_state(input_data)
        assert state.mode == MODE_FULL
        assert state.to_dict()["mode"] == MODE_FULL
        assert GuiState.from_dict(state.to_dict()) == state

    def test_mode_hydro_round_trips(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        state = _full_state(input_data, mode=MODE_HYDRO)
        assert GuiState.from_dict(state.to_dict()) == state

    def test_mode_absent_restores_full(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """A state saved before the mode existed restores it as full."""
        raw: dict[str, object] = _full_state(input_data).to_dict()
        del raw["mode"]
        assert GuiState.from_dict(raw).mode == MODE_FULL

    def test_model_key_round_trips(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        state = _full_state(input_data, model_key="hbv")
        assert state.to_dict()["model_key"] == "hbv"
        assert GuiState.from_dict(state.to_dict()).model_key == "hbv"

    def test_model_key_defaults_to_lateral(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """A state built without a model key is the lateral model."""
        state = _full_state(input_data)
        assert state.model_key == "lateral_three_layer"
        raw: dict[str, object] = state.to_dict()
        del raw["model_key"]
        assert (
            GuiState.from_dict(raw).model_key == "lateral_three_layer"
        )

    def test_state_v4_defaults_to_lateral(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """A version-4 state predates the model key: it restores lateral."""
        raw: dict[str, object] = _full_state(input_data).to_dict()
        raw["version"] = 4
        del raw["model_key"]
        assert GuiState.from_dict(raw).model_key == "lateral_three_layer"

    def test_unknown_model_key_raises(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        raw: dict[str, object] = _full_state(input_data).to_dict()
        raw["model_key"] = "nope"
        with pytest.raises(
            ValueError, match="Unknown model key"
        ):
            GuiState.from_dict(raw)

    def test_state_v3_objective_terms_roundtrip(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        state = _full_state(input_data, objective_terms=["Streamflow: KGE"])
        raw: dict[str, object] = state.to_dict()
        assert raw["objective_terms"] == ["Streamflow: KGE"]
        assert GuiState.from_dict(raw) == state

        bare = _full_state(input_data)
        raw = bare.to_dict()
        assert raw["objective_terms"] is None
        assert GuiState.from_dict(raw) == bare


class TestSavedRunModelKey:
    """The model key a saved run was produced with."""

    def test_prefers_gui_state(
        self, tmp_path: Path, input_data: gui_data.GuiInputData
    ) -> None:
        (tmp_path / "run_info.json").write_text(
            json.dumps(
                {"model_class": "potions.common_models.ThreeLayerModel"}
            )
        )
        write_gui_state(tmp_path, _full_state(input_data, model_key="hbv"))
        assert saved_run_model_key(tmp_path) == "hbv"

    def test_falls_back_to_run_info(self, tmp_path: Path) -> None:
        (tmp_path / "run_info.json").write_text(
            json.dumps(
                {"model_class": "potions.common_models.ThreeLayerModel"}
            )
        )
        assert saved_run_model_key(tmp_path) == "three_layer"

    def test_falls_back_to_nested_run_info(self, tmp_path: Path) -> None:
        (tmp_path / "hydro").mkdir()
        (tmp_path / "hydro" / "run_info.json").write_text(
            json.dumps({"model_class": "potions.common_models.HbvModel"})
        )
        assert saved_run_model_key(tmp_path) == "hbv"

    def test_defaults_to_lateral(self, tmp_path: Path) -> None:
        assert saved_run_model_key(tmp_path) == "lateral_three_layer"

    def test_ignores_unknown_model_class(self, tmp_path: Path) -> None:
        (tmp_path / "run_info.json").write_text(
            json.dumps({"model_class": "some.UnknownModel"})
        )
        assert saved_run_model_key(tmp_path) == "lateral_three_layer"


class TestFromDictErrors:
    """Each malformed gui_state field produces its documented message."""

    def _base(self, input_data: gui_data.GuiInputData) -> dict[str, object]:
        return _full_state(input_data).to_dict()

    @pytest.mark.parametrize(
        ("patch", "message"),
        [
            (
                {"version": 99},
                "Unsupported gui_state.json version 99 (expected [1, 2, 3, 4, 5, 6, 7]).",
            ),
            (
                {"csv_path": 12},
                "gui_state.json 'csv_path' must be a string or null, got int.",
            ),
            (
                {"param_values": {}},
                "gui_state.json is missing the 'param_values' mapping.",
            ),
            (
                {"param_bounds": {}},
                "gui_state.json is missing the 'param_bounds' mapping.",
            ),
            (
                {"selection": "DOC"},
                "gui_state.json is missing the 'selection' mapping.",
            ),
            (
                {"window_geometry": "10x20"},
                "gui_state.json 'window_geometry' must be a mapping of integers.",
            ),
            (
                {"splitter_sizes": "300 200 400"},
                "gui_state.json 'splitter_sizes' must be a list of integers.",
            ),
            (
                {"splitter_sizes": [1, 2, 3, 4, 5]},
                "gui_state.json 'splitter_sizes' must hold two, three, or four sizes, got 5.",
            ),
            (
                {"save_history": "yes"},
                "gui_state.json 'save_history' must be a boolean, got str.",
            ),
            (
                {"mode": "fuller"},
                "gui_state.json 'mode' must be 'full' or 'hydro', got 'fuller'.",
            ),
            (
                {"mode": 1},
                "gui_state.json 'mode' must be 'full' or 'hydro', got 1.",
            ),
        ],
    )
    def test_documented_messages(
        self,
        input_data: gui_data.GuiInputData,
        patch: dict[str, object],
        message: str,
    ) -> None:
        raw: dict[str, object] = self._base(input_data)
        raw.update(patch)
        with pytest.raises(ValueError) as excinfo:
            GuiState.from_dict(raw)
        assert str(excinfo.value) == message

    def test_non_numeric_param_value(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        raw: dict[str, object] = self._base(input_data)
        values: dict[object, object] = _as_dict(raw, "param_values")
        values["proportion.1"] = "high"
        raw["param_values"] = values
        with pytest.raises(ValueError) as excinfo:
            GuiState.from_dict(raw)
        assert (
            str(excinfo.value)
            == "gui_state.json param_values entry 'proportion.1' is not a number."
        )

    def test_non_numeric_bound(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        raw: dict[str, object] = self._base(input_data)
        bounds: dict[object, object] = _as_dict(raw, "param_bounds")
        bounds["proportion.1"] = [0.25, "1"]
        raw["param_bounds"] = bounds
        with pytest.raises(ValueError) as excinfo:
            GuiState.from_dict(raw)
        assert (
            str(excinfo.value)
            == "gui_state.json param_bounds entry 'proportion.1' "
            "is not a two-number sequence."
        )

    def test_selection_field_not_strings(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        raw: dict[str, object] = self._base(input_data)
        selection: dict[object, object] = _as_dict(raw, "selection")
        selection["zones"] = ["surface_hs", 1]
        raw["selection"] = selection
        with pytest.raises(ValueError) as excinfo:
            GuiState.from_dict(raw)
        assert (
            str(excinfo.value)
            == "gui_state.json selection field 'zones' is not a list of strings."
        )

    def test_streamflow_field_not_boolean(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        raw: dict[str, object] = self._base(input_data)
        selection: dict[object, object] = _as_dict(raw, "selection")
        selection["streamflow"] = "yes"
        raw["selection"] = selection
        with pytest.raises(ValueError) as excinfo:
            GuiState.from_dict(raw)
        assert (
            str(excinfo.value)
            == "gui_state.json selection field 'streamflow' is not a boolean."
        )

    def test_window_date_field_not_a_date(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        raw: dict[str, object] = self._base(input_data)
        selection: dict[object, object] = _as_dict(raw, "selection")
        selection["window_start"] = 2019
        raw["selection"] = selection
        with pytest.raises(ValueError) as excinfo:
            GuiState.from_dict(raw)
        assert (
            str(excinfo.value)
            == "gui_state.json selection field 'window_start' is not a date."
        )

    def test_window_date_field_malformed(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        raw: dict[str, object] = self._base(input_data)
        selection: dict[object, object] = _as_dict(raw, "selection")
        selection["window_end"] = "not-a-date"
        raw["selection"] = selection
        with pytest.raises(ValueError) as excinfo:
            GuiState.from_dict(raw)
        assert (
            str(excinfo.value)
            == "gui_state.json selection field 'window_end' is not a date."
        )

    def test_window_start_after_end(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        raw: dict[str, object] = self._base(input_data)
        selection: dict[object, object] = _as_dict(raw, "selection")
        selection["window_start"] = "2020-01-01"
        selection["window_end"] = "2019-01-01"
        raw["selection"] = selection
        with pytest.raises(ValueError) as excinfo:
            GuiState.from_dict(raw)
        assert (
            str(excinfo.value)
            == (
                "gui_state.json selection window starts on "
                "'2020-01-01', after its end '2019-01-01'."
            )
        )

    def test_version_five_water_years_migrate(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """A version-5 state's water years become a date window."""
        raw: dict[str, object] = self._base(input_data)
        raw["version"] = 5
        selection: dict[object, object] = _as_dict(raw, "selection")
        selection.pop("window_start")
        selection.pop("window_end")
        selection["water_years_from"] = 2017
        selection["water_years_to"] = 2020
        raw["selection"] = selection
        restored = GuiState.from_dict(raw)
        start, end = water_year_window(2017, 2020)
        assert restored.selection.window_start == start.strftime("%Y-%m-%d")
        assert restored.selection.window_end == end.strftime("%Y-%m-%d")

    def test_legacy_selection_migrates(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """A version-3 shape migrates onto the current one."""
        raw: dict[str, object] = self._base(input_data)
        selection: dict[object, object] = _as_dict(raw, "selection")
        # The version-3 shape: the zone figure's species, the
        # time-series figure's streamflow and river-species list, and
        # a list of minerals (no single mineral, no streamflow, no
        # water years).
        selection.pop("mineral")
        selection.pop("streamflow")
        selection.pop("window_start")
        selection.pop("window_end")
        selection["minerals"] = ["SOC(s)", "Calcite(s)"]
        selection["species"] = ["DIC"]
        selection["time_series"] = ["DIC", "DOC"]
        raw["selection"] = selection
        restored = GuiState.from_dict(raw)
        # The river species are the union of the two old lists, in
        # the panel's order (DOC before DIC).
        assert restored.selection.species == ["DOC", "DIC"]
        assert restored.selection.zones == selection["zones"]
        # The mineral is the first of the old list.
        assert restored.selection.mineral == "SOC(s)"
        # The streamflow axis is off: the old time-series list
        # carried no Streamflow entry.
        assert restored.selection.streamflow is False
        # The plotting window is the whole record.
        assert restored.selection.window_start == gui_data.DEFAULT_WINDOW_START
        assert restored.selection.window_end == gui_data.DEFAULT_WINDOW_END

    def test_legacy_selection_time_series_absent(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """An absent time-series list keeps every series on, as before."""
        raw: dict[str, object] = self._base(input_data)
        selection: dict[object, object] = _as_dict(raw, "selection")
        # The version-3 shape, as written by a state saved before the
        # time-series list existed: no mineral, no streamflow, no
        # water years, and no time-series list.
        selection.pop("mineral")
        selection.pop("streamflow")
        selection.pop("window_start")
        selection.pop("window_end")
        selection["minerals"] = ["SOC(s)"]
        raw["selection"] = selection
        restored = GuiState.from_dict(raw)
        assert restored.selection.species == list(RIVER_SPECIES)
        assert restored.selection.mineral == "SOC(s)"
        assert restored.selection.streamflow is True
        assert restored.selection.window_start == gui_data.DEFAULT_WINDOW_START
        assert restored.selection.window_end == gui_data.DEFAULT_WINDOW_END

    def test_legacy_minerals_not_strings(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        raw: dict[str, object] = self._base(input_data)
        selection: dict[object, object] = _as_dict(raw, "selection")
        selection.pop("mineral")
        selection.pop("streamflow")
        selection.pop("window_start")
        selection.pop("window_end")
        selection["minerals"] = "SOC(s)"
        raw["selection"] = selection
        with pytest.raises(ValueError) as excinfo:
            GuiState.from_dict(raw)
        assert (
            str(excinfo.value)
            == "gui_state.json selection field 'minerals' is not a list of strings."
        )

    def test_state_from_dict_objective_terms_validation(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """Absent or null terms restore as unset; wrong types error."""
        raw: dict[str, object] = self._base(input_data)
        del raw["objective_terms"]
        assert GuiState.from_dict(raw).objective_terms is None

        raw = self._base(input_data)
        raw["objective_terms"] = None
        assert GuiState.from_dict(raw).objective_terms is None

        raw = self._base(input_data)
        raw["objective_terms"] = "Streamflow: KGE"
        with pytest.raises(ValueError) as excinfo:
            GuiState.from_dict(raw)
        assert (
            str(excinfo.value)
            == "gui_state.json 'objective_terms' must be a list of "
            "objective terms, got 'Streamflow: KGE'."
        )

        raw = self._base(input_data)
        raw["objective_terms"] = [1]
        with pytest.raises(ValueError) as excinfo:
            GuiState.from_dict(raw)
        assert (
            str(excinfo.value)
            == "gui_state.json 'objective_terms' must be a list of "
            "objective terms, got [1]."
        )


class TestReadGuiState:
    """File-level reads: absent, malformed, and valid."""

    def test_absent_file(self, tmp_path: Path) -> None:
        assert read_gui_state(tmp_path) is None

    def test_malformed_file_prints_note_and_returns_none(
        self,
        tmp_path: Path,
        input_data: gui_data.GuiInputData,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        state_path: Path = tmp_path / "gui_state.json"
        state_path.write_text('{"version": 99}')
        assert read_gui_state(tmp_path) is None
        stderr: str = capsys.readouterr().err
        assert (
            stderr
            == f"Could not restore the GUI state from {state_path}: "
            f"Unsupported gui_state.json version 99 (expected [1, 2, 3, 4, 5, 6, 7]).\n"
        )

    def test_top_level_not_object(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        state_path: Path = tmp_path / "gui_state.json"
        state_path.write_text("[1, 2, 3]")
        assert read_gui_state(tmp_path) is None
        stderr: str = capsys.readouterr().err
        assert "the top-level JSON value is not an object" in stderr

    def test_valid_file(self, tmp_path: Path, input_data: gui_data.GuiInputData) -> None:
        state = _full_state(input_data)
        write_gui_state(tmp_path, state)
        assert read_gui_state(tmp_path) == state

    def test_write_gui_state_file(self, tmp_path: Path, input_data: gui_data.GuiInputData) -> None:
        path: Path = write_gui_state(tmp_path, _full_state(input_data))
        assert path == tmp_path / "gui_state.json"
        text: str = path.read_text()
        assert text.startswith("{\n  \"version\": 7,")
        assert text.endswith("\n")




class TestNetworkState:
    """The gui_state v7 network fingerprint round trip and recovery."""

    def test_fingerprint_round_trip(
        self, input_data: gui_data.GuiInputData, tmp_path: Path
    ) -> None:
        state = _full_state(input_data, network_fingerprint="9f86d081")
        write_gui_state(tmp_path, state)
        assert read_gui_state(tmp_path) == state

    def test_v6_state_reads_without_fingerprint(
        self, input_data: gui_data.GuiInputData, tmp_path: Path
    ) -> None:
        """A v6 state without the field reads with a null fingerprint."""
        raw: dict[str, object] = _full_state(input_data).to_dict()
        raw["version"] = 6
        raw.pop("network_fingerprint", None)
        (tmp_path / "gui_state.json").write_text(json.dumps(raw, indent=2))
        state = read_gui_state(tmp_path)
        assert state is not None
        assert state.network_fingerprint is None

    def test_fingerprint_non_string_raises(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        raw: dict[str, object] = _full_state(input_data).to_dict()
        raw["network_fingerprint"] = 3.0
        with pytest.raises(
            ValueError,
            match=r"network_fingerprint' must be a string or null, got float",
        ):
            GuiState.from_dict(raw)

    def test_saved_run_network_fingerprint(
        self,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
    ) -> None:
        assert gui_state.saved_run_network_fingerprint(tmp_path) is None
        state = _full_state(input_data, network_fingerprint="9f86d081")
        write_gui_state(tmp_path, state)
        assert gui_state.saved_run_network_fingerprint(tmp_path) == "9f86d081"


class TestFindLatestRunDir:
    """Discovery of the most recent saved run directory."""

    def test_missing_results_dir(self, tmp_path: Path) -> None:
        assert find_latest_run_dir(tmp_path) is None

    def test_no_run_entries(self, tmp_path: Path) -> None:
        (tmp_path / "results").mkdir()
        (tmp_path / "results" / "hydro").mkdir()
        assert find_latest_run_dir(tmp_path / "results") is None

    def test_latest_by_stamp(self, tmp_path: Path) -> None:
        results: Path = tmp_path / "results"
        older: Path = results / "run_20260101_000000"
        newest: Path = results / "run_20260102_000000"
        (older / "hydro").mkdir(parents=True)
        (newest / "hydro").mkdir(parents=True)
        assert find_latest_run_dir(results) == newest

    def test_counter_breaks_stamps(self, tmp_path: Path) -> None:
        results: Path = tmp_path / "results"
        first: Path = results / "run_20260101_000000"
        retry: Path = results / "run_20260101_000000_1"
        (first / "hydro").mkdir(parents=True)
        (retry / "hydro").mkdir(parents=True)
        assert find_latest_run_dir(results) == retry

    def test_non_run_entries_skipped(self, tmp_path: Path) -> None:
        results: Path = tmp_path / "results"
        other: Path = results / "some_directory"
        latest: Path = results / "run_20260101_000000"
        (other / "hydro").mkdir(parents=True)
        (latest / "hydro").mkdir(parents=True)
        assert find_latest_run_dir(results) == latest


class TestListRunDirs:
    """Discovery of every saved run directory, newest first."""

    def test_missing_results_dir(self, tmp_path: Path) -> None:
        assert list_run_dirs(tmp_path) == []

    def test_no_run_entries(self, tmp_path: Path) -> None:
        results: Path = tmp_path / "results"
        (results / "run_fake").mkdir(parents=True)
        (results / "some_directory" / "hydro").mkdir(parents=True)
        assert list_run_dirs(results) == []

    def test_newest_first(self, tmp_path: Path) -> None:
        results: Path = tmp_path / "results"
        older: Path = results / "run_20260101_000000"
        middle: Path = results / "run_20260101_120000"
        newest: Path = results / "run_20260102_000000"
        for run_dir in (older, middle, newest):
            (run_dir / "hydro").mkdir(parents=True)
        assert list_run_dirs(results) == [newest, middle, older]

    def test_counter_breaks_stamps(self, tmp_path: Path) -> None:
        results: Path = tmp_path / "results"
        first: Path = results / "run_20260101_000000"
        retry: Path = results / "run_20260101_000000_1"
        (first / "hydro").mkdir(parents=True)
        (retry / "hydro").mkdir(parents=True)
        assert list_run_dirs(results) == [retry, first]

    def test_latest_is_first_element(self, tmp_path: Path) -> None:
        results: Path = tmp_path / "results"
        older: Path = results / "run_20260101_000000"
        newest: Path = results / "run_20260102_000000"
        (older / "hydro").mkdir(parents=True)
        (newest / "hydro").mkdir(parents=True)
        assert list_run_dirs(results)[0] == find_latest_run_dir(results)


class TestRowKey:
    """The parameters.csv row → parameter-key mapping."""

    def test_model_row(self) -> None:
        assert (
            gui_state._row_key("model", "proportion.1", {})
            == "proportion.1"
        )

    def test_river_rows_are_dropped(self) -> None:
        assert gui_state._row_key("river", "DOC", {}) is None

    def test_bare_hydro_row(self) -> None:
        assert gui_state._row_key("snow", "tt", {}) == "snow.tt"

    def test_mineral_row(self) -> None:
        minerals: dict[str, int] = {"SOC(s)": 0, "SOC-lz(s)": 1, "Calcite(s)": 2}
        assert (
            gui_state._row_key("ground", "Calcite(s).k_a", minerals)
            == "ground.k_a[2]"
        )

    def test_unknown_mineral_is_dropped(self) -> None:
        minerals: dict[str, int] = {"SOC(s)": 0}
        assert gui_state._row_key("ground", "Bogus(s).k_a", minerals) is None




# ------------------------------------------------------------------ #
# potions.gui.state: restoring saved runs
# ------------------------------------------------------------------ #


class TestLoadSavedRunFullState:
    """A run saved by this GUI restores its state exactly."""

    def test_restore(self, tmp_path: Path, input_data: gui_data.GuiInputData) -> None:
        state = _full_state(input_data)
        run_dir = _write_saved_run(tmp_path, input_data, state)

        saved = load_saved_run(run_dir, input_data)
        assert saved.restore_note is None
        assert saved.param_values == state.param_values
        assert saved.param_bounds == state.param_bounds
        assert saved.selection == state.selection
        assert saved.window_geometry == state.window_geometry
        assert saved.splitter_sizes == state.splitter_sizes
        assert saved.save_history is False

    def test_results_reconstructed(self, tmp_path: Path, input_data: gui_data.GuiInputData) -> None:
        run_dir = _write_saved_run(tmp_path, input_data, _full_state(input_data))
        saved = load_saved_run(run_dir, input_data)

        hydro: pt.HydroModelResults = saved.results.hydro
        rt = saved.results.rt
        assert rt is not None
        assert hydro.saved_dir == str(run_dir)
        assert rt.saved_dir == str(run_dir)

        source_hydro, source_rt = _synthetic_results(input_data)
        pd.testing.assert_frame_equal(
            hydro.simulation,
            source_hydro.simulation,
            check_exact=False,
        )
        pd.testing.assert_frame_equal(
            rt.simulation,
            source_rt.simulation,
            check_exact=False,
        )

        # The zone layout round-trips: the reconstruction recovers it
        # from the hydrology frame's storage columns, so the per-zone
        # accessors keep working on the loaded run.
        assert hydro.zone_names == ZONE_ORDER
        assert rt.zone_names == ZONE_ORDER
        pd.testing.assert_frame_equal(
            hydro.storage,
            source_hydro.storage,
            check_exact=False,
        )

        hydro_objectives: pd.Series = hydro.objective_functions
        assert list(hydro_objectives) == pytest.approx(
            list(source_hydro.objective_functions)
        )
        rt_objectives: pd.DataFrame | None = rt.objective_functions
        source_rt_objectives: pd.DataFrame | None = source_rt.objective_functions
        assert rt_objectives is not None
        assert source_rt_objectives is not None
        pd.testing.assert_frame_equal(
            rt_objectives,
            source_rt_objectives,
            check_exact=False,
        )

    def test_restore_save_history_on(
        self, tmp_path: Path, input_data: gui_data.GuiInputData
    ) -> None:
        state = _full_state(input_data, save_history=True)
        run_dir = _write_saved_run(tmp_path, input_data, state)

        saved = load_saved_run(run_dir, input_data)
        assert saved.restore_note is None
        assert saved.save_history is True


class TestLoadSavedRunLegacy:
    """A run without gui_state.json restores best-effort values."""

    def test_restore(self, tmp_path: Path, input_data: gui_data.GuiInputData) -> None:
        run_dir = _write_saved_run(tmp_path, input_data, state=None)

        saved = load_saved_run(run_dir, input_data)
        assert saved.restore_note is not None
        assert (
            "parameters.csv (61 values, 4-decimal;"
            in saved.restore_note
        )
        assert saved.window_geometry is None
        assert saved.splitter_sizes is None
        assert saved.save_history is False
        assert saved.selection == gui_data.default_plot_selection()
        assert saved.param_bounds == dict(input_data.bounds)

        bounds: dict[str, tuple[float, float]] = input_data.bounds
        midpoints: dict[str, float] = {
            key: 0.5 * (lo + hi) for key, (lo, hi) in bounds.items()
        }

        # The rows the saved CSV does not map to a bounds key fall
        # back to the exact midpoint: the 18 hydrology zone keys (the
        # CSV names the hydrology layers, not the zones) and the four
        # initial-state keys.
        hydro_keys: list[str] = [
            f"{zone}.{param}" for zone in ZONE_ORDER for param in HYDRO_PARAMS[zone]
        ]
        initial_keys: list[str] = [
            f"initial_state.{zone}_x" for zone in SPECIATION_ZONES
        ]
        for key in hydro_keys + initial_keys:
            assert saved.param_values[key] == pytest.approx(midpoints[key])

        # The mapped rows carry the CSV's 4-decimal values: the
        # catchment proportion and every non-k_a parameter come back
        # within 1e-4 of their midpoint (the run used midpoints).
        assert saved.param_values["proportion.1"] == pytest.approx(0.6)
        k_a_keys: list[str] = [
            key
            for key in bounds
            if key.startswith(("surface", "ground"))
            and ".k_a[" in key
        ]
        assert len(k_a_keys) == 12
        for key, (lo, hi) in bounds.items():
            if key in k_a_keys:
                # The k_a values are saved on a natural scale in the
                # model's parameters.csv and come back as 0.0 (the
                # probed artifact); they are not at their midpoints.
                assert saved.param_values[key] == pytest.approx(0.0)
            elif key in hydro_keys + initial_keys:
                continue
            else:
                assert saved.param_values[key] == pytest.approx(
                    midpoints[key], abs=1e-4
                )

        # The results are still reconstructed from the saved CSVs.
        assert saved.results.hydro.saved_dir == str(run_dir)
        rt = saved.results.rt
        assert rt is not None
        assert rt.saved_dir == str(run_dir)

    def test_incomplete_run_raises(self, tmp_path: Path, input_data: gui_data.GuiInputData) -> None:
        # A complete saved run minus its parameters.csv files: the
        # simulation CSVs restore fine, but the parameter recovery
        # has nothing to read.
        run_dir = _write_saved_run(tmp_path, input_data, state=None)
        (run_dir / "hydro" / "parameters.csv").unlink()
        (run_dir / "rt" / "parameters.csv").unlink()
        with pytest.raises(ValueError) as excinfo:
            load_saved_run(run_dir, input_data)
        assert (
            str(excinfo.value)
            == f"Saved run is incomplete: no parameters.csv in {run_dir} "
            "(looked in the run's top level, hydro/, and rt/)."
        )



class TestLoadSavedRunHydro:
    """A hydrology-only saved run restores without a re-run."""

    def test_with_state(
        self, tmp_path: Path, input_data: gui_data.GuiInputData
    ) -> None:
        """The state file records the mode and the restore is exact."""
        state = _full_state(input_data, mode=MODE_HYDRO)
        run_dir = _write_saved_run(
            tmp_path, input_data, state, hydro_only=True
        )
        saved = load_saved_run(run_dir, input_data)

        assert saved.mode == MODE_HYDRO
        assert saved.restore_note is None
        assert saved.results.rt is None
        assert saved.results.hydro.saved_dir == str(run_dir)
        assert saved.param_values == state.param_values
        assert saved.selection == state.selection

    def test_without_state(
        self, tmp_path: Path, input_data: gui_data.GuiInputData
    ) -> None:
        """The mode is inferred from the flat layout and noted."""
        run_dir = _write_saved_run(
            tmp_path, input_data, None,
            hydro_only=True, with_state=False,
        )
        saved = load_saved_run(run_dir, input_data)

        assert saved.mode == MODE_HYDRO
        assert saved.results.rt is None
        assert saved.restore_note is not None
        assert (
            "(1 values, 4-decimal; hydro-zone and initial-state "
            "parameters at their defaults)"
            in saved.restore_note
        )
        # The flat parameters.csv is layer-named, so only the
        # model-level proportion row maps back onto a bound; every
        # other parameter holds its midpoint default.
        midpoints: dict[str, float] = {
            name: 0.5 * (lo + hi)
            for name, (lo, hi) in input_data.bounds.items()
        }
        assert saved.param_values["proportion.1"] == pytest.approx(0.6)
        for name in input_data.bounds:
            if name != "proportion.1":
                assert saved.param_values[name] == pytest.approx(
                    midpoints[name]
                )


# ------------------------------------------------------------------ #
# potions.gui.state: the abbreviated run history
# ------------------------------------------------------------------ #


def _small_history_entry(
    timestamp: str,
    model_key: str = model_specs.DEFAULT_MODEL_KEY,
) -> RunHistoryEntry:
    """A small run-history record for the append tests."""
    return RunHistoryEntry(
        timestamp=timestamp,
        mode="full",
        param_values={"proportion.1": 0.7},
        objectives={"Streamflow": {"nse": 0.85}},
        model_key=model_key,
    )


class TestAppendRunHistory:
    """``gui_run_history.jsonl``: one JSON object per finished run."""

    def test_appends_lines(self, tmp_path: Path) -> None:
        # The results directory does not exist yet: the first append
        # creates it (and the file) on demand.
        results_dir: Path = tmp_path / "does-not-exist" / "results"
        entry_one = _small_history_entry("2026-01-01T00:00:00+00:00")
        entry_two = _small_history_entry(
            "2026-01-02T00:00:00+00:00", model_key="hbv"
        )

        path = append_run_history(results_dir, entry_one)
        assert path == results_dir / RUN_HISTORY_FILENAME
        assert path.is_file()
        # A second append to the same directory returns the same file.
        assert append_run_history(results_dir, entry_two) == path

        lines: list[str] = path.read_text(encoding="utf-8").splitlines()
        assert [json.loads(line) for line in lines] == [
            entry_one.to_dict(),
            entry_two.to_dict(),
        ]
        # The model key is persisted with the run, so the history
        # dropdown can filter to the selected model.
        assert json.loads(lines[0])["model_key"] == "lateral_three_layer"
        assert json.loads(lines[1])["model_key"] == "hbv"

    def test_write_failure_prints_note_and_does_not_raise(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # A regular file where the results directory should be blocks
        # the append; the failure is reported on the standard error,
        # never raised.
        results_dir: Path = tmp_path / "results"
        results_dir.write_text("not a directory")
        entry = _small_history_entry("2026-01-01T00:00:00+00:00")

        path = append_run_history(results_dir, entry)
        assert path == results_dir / RUN_HISTORY_FILENAME
        assert not path.is_file()
        assert "Could not append to the run history" in capsys.readouterr().err


class TestObjectivesFromResults:
    """The per-target objective mapping of a finished run."""

    def test_scored_run(self, input_data: gui_data.GuiInputData) -> None:
        hydro, rt = _synthetic_results(input_data)
        objectives = objectives_from_results(pt.ModelResults(hydro, rt))

        assert list(objectives) == ["Streamflow", "DOC", "DIC", "Ca++"]
        assert objectives["Streamflow"] == {
            name: 0.9 * (i + 1) for i, name in enumerate(METRIC_NAMES)
        }
        for species in ["DOC", "DIC", "Ca++"]:
            assert objectives[species] == {
                name: 0.5 + 0.1 * i for i, name in enumerate(METRIC_NAMES)
            }

    def test_unscored_targets_are_omitted(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        hydro, rt = _synthetic_results(input_data)
        unscored_hydro: pt.HydroModelResults = pt.HydroModelResults(
            forcing=input_data.forc,
            simulation=hydro.simulation,
            objective_functions=pd.Series(dtype=float),
        )
        unscored_rt: pt.RtModelResults = pt.RtModelResults(
            simulation=rt.simulation,
            objective_functions=None,
        )
        assert (
            objectives_from_results(
                pt.ModelResults(unscored_hydro, unscored_rt)
            )
            == {}
        )
        # A run without reactive-transport results (the hydro-only
        # mode) has no species targets at all.  The dataclass types
        # the field non-optional, so `setattr` builds the None case
        # without the constructor's type check.
        no_rt: pt.ModelResults = pt.ModelResults(unscored_hydro, rt)
        setattr(no_rt, "reactive_transport", None)
        assert objectives_from_results(no_rt) == {}

    def test_hydro_only_wrapper(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """The `GuiRunResults` wrapper of a run scores it."""
        hydro, rt = _synthetic_results(input_data)
        full: dict[str, dict[str, float]] = objectives_from_results(
            pt.ModelResults(hydro, rt)
        )
        hydro_only: gui_data.GuiRunResults = gui_data.GuiRunResults(
            hydro, None
        )
        both: gui_data.GuiRunResults = gui_data.GuiRunResults(hydro, rt)

        assert objectives_from_results(hydro_only) == {
            "Streamflow": full["Streamflow"]
        }
        assert objectives_from_results(both) == full


# ------------------------------------------------------------------ #
# potions.gui.plots: the figure builders
# ------------------------------------------------------------------ #

from matplotlib import colors as mcolors
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PathCollection, PolyCollection
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox


def _day_number(timestamp: pd.Timestamp) -> float:
    """The matplotlib date number (days since 1970-01-01) of `timestamp`."""
    return float(timestamp.value / 1e9 / 86400)


def _figure_anchor(figure: Figure, x: float, y: float) -> list[float]:
    """A figure-fraction legend anchor as display extents."""
    width: float = figure.get_figwidth() * figure.dpi
    height: float = figure.get_figheight() * figure.dpi
    return [x * width, y * height, x * width, y * height]


def _axis_anchor(figure: Figure, axis: object, x: float, y: float) -> list[float]:
    """
    An axis-fraction legend anchor as display extents.

    `axis` is typed `object` because `Figure.axes` is declared as
    `FigureBase` in the matplotlib stubs; the assertion narrows it.
    """
    assert isinstance(axis, Axes)
    position = axis.get_position()
    width: float = figure.get_figwidth() * figure.dpi
    height: float = figure.get_figheight() * figure.dpi
    x0: float = (position.x0 + x * position.width) * width
    y0: float = (position.y0 + y * position.height) * height
    return [x0, y0, x0, y0]


def _axis_legends(axis: object) -> list[Legend]:
    """The legend artists kept on an axis, in creation order."""
    assert isinstance(axis, Axes)
    return [
        child
        for child in axis.get_children()
        if isinstance(child, Legend)
    ]


def _lines(ax: object) -> list[Line2D]:
    """The Line2D artists of an axes, in draw order."""
    assert isinstance(ax, Axes)
    return [artist for artist in ax.get_lines() if isinstance(artist, Line2D)]


def _alpha(line: Line2D) -> float:
    """
    A line's effective opacity.

    ``Line2D.get_alpha`` reports None for a line whose alpha was
    never set (this matplotlib version exposes no default); such a
    line draws fully opaque, so the missing value normalizes to 1.0.
    """
    value: float | None = line.get_alpha()
    return 1.0 if value is None else value


def _scatters(ax: object) -> list[PathCollection]:
    """The scatter PathCollections of an axes, in draw order."""
    assert isinstance(ax, Axes)
    return [
        artist
        for artist in ax.collections
        if isinstance(artist, PathCollection)
        and not isinstance(artist, PolyCollection)
    ]


def _fills(ax: object) -> list[PolyCollection]:
    """The fill_between PolyCollections of an axes, in draw order."""
    assert isinstance(ax, Axes)
    return [
        artist
        for artist in ax.collections
        if isinstance(artist, PolyCollection)
    ]


def _fill_band_y(fill: PolyCollection) -> tuple[np.ndarray, np.ndarray]:
    """
    The bottom and top y-values of a fill_between band, in x order.

    The band's path is a closed loop: the top's start point, the
    bottom curve, the top's end point, and the top curve in reverse
    (see `FillBetweenPolyCollection._make_verts_for_region`).  The
    synthetic and clipped frames have no gaps, so each band is one
    path.
    """
    verts: np.ndarray = np.asarray(fill.get_paths()[0].vertices)
    n: int = (len(verts) - 2) // 2
    bottom: np.ndarray = verts[1 : n + 1, 1]
    top: np.ndarray = verts[n + 2 : 2 * n + 2, 1][::-1]
    return bottom, top


class TestClipHelpers:
    """The window clipping every figure builder applies before drawing."""

    def test_frame_clipped_to_window(self) -> None:
        index: pd.Index = pd.date_range("2014-01-01", periods=10, freq="D")
        frame: pd.DataFrame = pd.DataFrame(
            {"a": np.arange(10, dtype=float)}, index=index
        )
        window: tuple[pd.Timestamp, pd.Timestamp] = (
            pd.Timestamp(2014, 1, 3),
            pd.Timestamp(2014, 1, 6),
        )
        clipped: pd.DataFrame = gui_plots._clip_frame(frame, window)
        assert clipped.index[0] == pd.Timestamp(2014, 1, 3)
        assert clipped.index[-1] == pd.Timestamp(2014, 1, 6)
        assert list(clipped["a"]) == [2.0, 3.0, 4.0, 5.0]

    def test_series_clipped_to_window(self) -> None:
        index: pd.Index = pd.date_range("2014-01-01", periods=10, freq="D")
        series: pd.Series = pd.Series(np.arange(10, dtype=float), index=index)
        window: tuple[pd.Timestamp, pd.Timestamp] = (
            pd.Timestamp(2014, 1, 3),
            pd.Timestamp(2014, 1, 6),
        )
        clipped: pd.Series = gui_plots._clip_series(series, window)
        assert list(clipped) == [2.0, 3.0, 4.0, 5.0]

    def test_clipping_keeps_nan_rows(self) -> None:
        """Missing observations survive the clip; the builders drop them."""
        index: pd.Index = pd.date_range("2014-01-01", periods=4, freq="D")
        series: pd.Series = pd.Series([1.0, np.nan, 3.0, 4.0], index=index)
        window: tuple[pd.Timestamp, pd.Timestamp] = (
            pd.Timestamp(2014, 1, 1),
            pd.Timestamp(2014, 1, 4),
        )
        clipped: pd.Series = gui_plots._clip_series(series, window)
        assert clipped.isna().tolist() == [False, True, False, False]


def _legend_extent_fractions(fig: Figure, legend: Legend) -> tuple[float, float]:
    """The drawn legend's top and bottom edges, in figure fractions."""
    extent = legend.get_window_extent()
    fig_height: float = fig.get_figheight() * fig.dpi
    return (float(extent.y1) / fig_height, float(extent.y0) / fig_height)


class TestLegendPlacement:
    """The shared frameless legend the figure builders attach below."""

    def test_legend_is_frameless_and_below_the_axes(self) -> None:
        fig = Figure(figsize=(15, 5))
        ax = fig.gca()
        ax.plot([0.0, 1.0], [0.0, 1.0])
        legend = gui_plots._attach_legend(
            fig,
            [Line2D([], [], color="k", linewidth=2)],
            ["a series"],
            title="Series",
        )
        # The plotting guidelines: the legend frame is off, and the
        # legend never overlaps the drawn series (it sits below the
        # axes, whose interior holds the data) and stays inside the
        # figure (the panel's page export saves at the figure size).
        assert legend.get_frame().get_visible() is False
        top, bottom = _legend_extent_fractions(fig, legend)
        assert top <= ax.get_position().y0
        assert bottom >= 0.0
        assert legend.get_title().get_text() == "Series"

    def test_second_row_stacks_above_the_first(self) -> None:
        fig = Figure(figsize=(15, 5))
        ax = fig.gca()
        ax.plot([0.0, 1.0], [0.0, 1.0])
        first = gui_plots._attach_legend(
            fig,
            [Line2D([], [], color="k", linewidth=2)],
            ["a"],
            title="First",
        )
        second = gui_plots._attach_legend(
            fig,
            [Line2D([], [], color="k", linewidth=2)],
            ["b"],
            title="Second",
            above=first,
        )
        first_top, _first_bottom = _legend_extent_fractions(fig, first)
        _second_top, second_bottom = _legend_extent_fractions(fig, second)
        # The second row's bottom edge sits at or above the first
        # row's top edge: the rows do not overlap.
        assert second_bottom >= first_top

    def test_relayout_keeps_rows_inside_after_a_resize(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """
        The panel rescales the figure to fit its canvas, and a
        point-based row occupies more of a shorter figure, so the
        construction-size anchor no longer holds; the relayout
        keeps the row inside the figure and the axes above it.
        """
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        fig: Figure = gui_plots.make_hydrology_figure(
            _synthetic_hydro_sim(index),
            input_data.q_meas,
            window,
        )
        # The construction layout holds: the row's bottom edge
        # sits above the figure's bottom edge.
        legend: Legend = fig.legends[0]
        _top, bottom = _legend_extent_fractions(fig, legend)
        assert bottom >= 0.0

        # The panel's rescale: the canvas is wide and short.
        fig.set_size_inches(13.33, 2.89)
        fig.canvas.draw()
        # The construction-size anchor no longer holds: the row
        # sinks past the figure's bottom edge.
        _top, bottom = _legend_extent_fractions(fig, legend)
        assert bottom < 0.0

        # The relayout re-anchors the row and restates the margin:
        # the row is inside the figure and the axes sit above it.
        gui_plots.relayout_below_axes_legends(fig)
        fig.canvas.draw()
        top, bottom = _legend_extent_fractions(fig, legend)
        assert bottom >= 0.0
        assert top <= 1.0
        assert fig.axes[0].get_position().y0 >= top

    def test_relayout_keeps_stacked_rows_inside_after_a_resize(
        self,
    ) -> None:
        """The stacked-row relayout keeps every row inside, in order."""
        fig = Figure(figsize=(15, 5))
        ax = fig.gca()
        ax.plot([0.0, 1.0], [0.0, 1.0])
        first = gui_plots._attach_legend(
            fig,
            [Line2D([], [], color="k", linewidth=2)],
            ["a"],
            title="First",
        )
        second = gui_plots._attach_legend(
            fig,
            [Line2D([], [], color="k", linewidth=2)],
            ["b"],
            title="Second",
            above=first,
        )
        # The builders' margin, as `make_conc_discharge_figure`
        # states it: the gap above the topmost row, grown to
        # clear the axes' tick labels.
        fig.subplots_adjust(
            bottom=gui_plots._legend_top_edge(fig, second)
            + gui_plots._axes_gap(fig)
        )
        fig.set_size_inches(13.33, 2.89)
        fig.canvas.draw()
        gui_plots.relayout_below_axes_legends(fig)
        fig.canvas.draw()
        first_top, first_bottom = _legend_extent_fractions(fig, first)
        second_top, second_bottom = _legend_extent_fractions(fig, second)
        # Every row is inside the figure, the rows do not overlap,
        # and the axes sit above the topmost row.
        assert first_bottom >= 0.0
        assert second_bottom >= first_top
        assert second_top <= 1.0
        assert ax.get_position().y0 >= second_top

    def test_relayout_clears_the_axes_tick_labels(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """
        The x tick labels hang below the axes' bottom edge in a
        point-based depth; after the panel's rescale that depth
        occupies more of the figure than the construction-size gap
        reserved, so the relayout grows the margin above the
        topmost row to clear the tick labels.

        The growth is measured against the figure's own renderer,
        which always carries the figure's current dpi: the gap
        must equal the labels' true current-dpi depth plus the
        margin (at least the builders' row gap), and this test
        fails both when a stale cached renderer over-reserves the
        gap (shrinking the drawn plot area) and when a too-small
        one lets the labels sit on the legend row.

        The figure is prepared the way the app prepares it: the
        entry point builds it under the publication rcParams
        (300 dpi), and the panel's canvas restates it at its
        canvas dpi and attaches a drawing canvas before the
        wide, short rescale.  The construction-time measurement
        and the relayout's measurement then happen at different
        dpi, as in the app.
        """
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        # The entry point's publication rcParam, restated only
        # around the build so the other figures' tests keep the
        # default.
        saved_dpi: float = matplotlib.rcParams["figure.dpi"]
        matplotlib.rcParams["figure.dpi"] = 300
        try:
            fig: Figure = gui_plots.make_hydrology_figure(
                _synthetic_hydro_sim(index),
                input_data.q_meas,
                window,
            )
        finally:
            matplotlib.rcParams["figure.dpi"] = saved_dpi
        fig.set_dpi(100)  # figure_panel.CANVAS_DPI
        canvas: FigureCanvasAgg = FigureCanvasAgg(fig)
        # The panel's rescale: the canvas is wide and short.
        fig.set_size_inches(13.33, 2.89)
        canvas.draw()  # type: ignore[no-untyped-call]
        gui_plots.relayout_below_axes_legends(fig)
        canvas.draw()  # type: ignore[no-untyped-call]
        legend: Legend = fig.legends[0]
        fig_height: float = float(fig.get_window_extent().height)
        legend_top: float = float(legend.get_window_extent().y1)
        # The figure's own renderer always carries the current
        # dpi, so the labels' true current-dpi depth is measured
        # against it.
        renderer = fig._get_renderer()  # type: ignore[attr-defined]
        ax: Axes = fig.axes[0]
        ax_bottom: float = float(ax.get_window_extent().y0)
        depth: float = 0.0
        for label in [*ax.get_xticklabels(), ax.xaxis.get_label()]:
            if not label.get_visible():
                continue
            extent: Bbox = label.get_window_extent(renderer)
            # A label with empty text occupies no space and keeps
            # no margin.
            if extent.height <= 0.0:
                continue
            # Every hanging label keeps the margin above the
            # topmost row.
            assert float(extent.y0) >= legend_top + (
                gui_plots._TICK_LABEL_MARGIN * fig_height
            ) - 0.5
            depth = max(depth, float(ax_bottom - extent.y0) / fig_height)
        # The relayout reserved exactly that depth plus the
        # margin, at least the builders' row gap.
        reserved: float = float(ax.get_position().y0) - legend_top / fig_height
        assert reserved == pytest.approx(
            max(
                gui_plots._LEGEND_ROW_GAP,
                depth + gui_plots._TICK_LABEL_MARGIN,
            ),
            abs=1e-5,
        )


class TestStorageFigure:
    """The per-zone water-storage figure on synthetic data."""

    def test_axes_and_labels(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        fig: Figure = gui_plots.make_storage_figure(
            _synthetic_hydro_sim(index),
            ["surface_hs", "ground_rp"],
            window,
        )
        assert len(fig.axes) == 1
        ax: Axes = fig.axes[0]
        assert ax.get_title() == "Storage"
        assert ax.get_ylabel() == "Storage (mm)"

    def test_one_line_per_zone_clipped_to_window(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        fig = gui_plots.make_storage_figure(
            _synthetic_hydro_sim(index),
            ["surface_hs", "ground_rp"],
            window,
        )
        ax = fig.axes[0]
        lines = _lines(ax)
        assert len(lines) == 2
        for line, zone in zip(lines, ["surface_hs", "ground_rp"]):
            assert mcolors.to_rgba(line.get_color()) == mcolors.to_rgba(
                ZONE_COLORS[zone]
            )
            assert line.get_linewidth() == 1
        # Clipped to the window: the full record minus the spin-up.
        assert len(np.asarray(lines[0].get_ydata())) == 2557
        # The record ends before the window's end (the partial 2025
        # water year), so the right limit is the record's last day.
        assert ax.get_xlim() == (
            _day_number(window[0]),
            _day_number(index[-1]),
        )

    def test_partial_window_limits(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(2017, 2020)
        fig = gui_plots.make_storage_figure(
            _synthetic_hydro_sim(index),
            ["surface_hs"],
            window,
        )
        # Both limits come from the window; the record spans it.
        assert fig.axes[0].get_xlim() == (
            _day_number(window[0]),
            _day_number(window[1]),
        )

    def test_legend_below_the_axes(self, input_data: gui_data.GuiInputData) -> None:
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        fig = gui_plots.make_storage_figure(
            _synthetic_hydro_sim(index),
            ["surface_hs", "ground_rp"],
            window,
        )
        assert len(fig.legends) == 1
        legend = fig.legends[0]
        assert legend.get_frame().get_visible() is False
        assert legend.get_title().get_text() == "Zone"
        assert [text.get_text() for text in legend.get_texts()] == [
            "Surface (HS)",
            "Ground (RP)",
        ]
        # The legend sits below the axes, with the axes' reserved
        # bottom margin keeping the two apart.
        top, bottom = _legend_extent_fractions(fig, legend)
        assert top <= fig.axes[0].get_position().y0
        assert bottom >= 0.0
        assert fig.subplotpars.bottom >= top

    def test_overlay_previous_run(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """The previous run's storages draw behind, faintly, out of the legend."""
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        hydro_sim = _synthetic_hydro_sim(index)
        overlay_hydro: pd.DataFrame = _synthetic_hydro_sim(index) + 10.0

        fig = gui_plots.make_storage_figure(
            hydro_sim,
            ["surface_hs", "ground_rp"],
            window,
            overlay_hydro_sim=overlay_hydro,
        )
        lines = _lines(fig.axes[0])
        # The overlay lines are drawn first (behind) and half
        # transparent; the current lines keep their full opacity.
        assert len(lines) == 4
        assert [line.get_alpha() for line in lines[:2]] == [0.5, 0.5]
        assert [_alpha(line) for line in lines[2:]] == [1.0, 1.0]
        expected: np.ndarray = np.asarray(
            overlay_hydro.loc[
                (overlay_hydro.index >= window[0])
                & (overlay_hydro.index <= window[1]),
                "s_surface_hs",
            ]
        )
        np.testing.assert_allclose(
            np.asarray(lines[0].get_ydata()), expected
        )

    def test_overlay_skips_missing_columns(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """A previous frame that lacks a zone's column contributes no line."""
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        hydro_sim = _synthetic_hydro_sim(index)
        overlay_hydro: pd.DataFrame = _synthetic_hydro_sim(index).drop(
            columns=["s_ground_rp"]
        )

        fig = gui_plots.make_storage_figure(
            hydro_sim,
            ["surface_hs", "ground_rp"],
            window,
            overlay_hydro_sim=overlay_hydro,
        )
        # One overlay line (surface_hs only), then the two current.
        assert len(_lines(fig.axes[0])) == 3


class TestPartitioningFigure:
    """The stacked discharge-partitioning figure on synthetic data."""

    # The stack order: ground zones under surface zones, ZONE_ORDER
    # within a family (the spec's ground-then-surface example).
    STACK_ZONES: tuple[str, ...] = (
        "ground_hs",
        "ground_rp",
        "surface_hs",
        "surface_rp",
    )

    def test_axes_and_labels(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        fig: Figure = gui_plots.make_partitioning_figure(
            _synthetic_hydro_sim(index),
            input_data.q_meas,
            window,
        )
        assert len(fig.axes) == 1
        ax: Axes = fig.axes[0]
        assert ax.get_title() == "Partitioning"
        assert ax.get_ylabel() == "Discharge (mm/d)"

    def test_stacked_layers(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """One filled band per river zone: cumulative components, ground first."""
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        frame: pd.DataFrame = _synthetic_hydro_sim(index)
        clipped: pd.DataFrame = gui_plots._clip_frame(frame, window)
        flow: pd.Series = clipped["streamflow_sim"]

        fig = gui_plots.make_partitioning_figure(
            frame,
            input_data.q_meas,
            window,
        )
        fills = _fills(fig.axes[0])
        assert len(fills) == 4
        # The fixture's per-zone streamflow shares (0.1 x zone index).
        shares: tuple[float, ...] = (0.5, 0.6, 0.3, 0.4)
        running: pd.Series = pd.Series(0.0, index=clipped.index)
        for fill, zone, share in zip(fills, self.STACK_ZONES, shares):
            assert np.allclose(
                fill.get_facecolor(), mcolors.to_rgba(ZONE_COLORS[zone])
            )
            band_bottom, band_top = _fill_band_y(fill)
            assert np.allclose(band_bottom, running.to_numpy())
            running = running + share * flow
            assert np.allclose(band_top, running.to_numpy())

    def test_measured_scatter(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """The measured streamflow plots as scatter points, never a line."""
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        fig = gui_plots.make_partitioning_figure(
            _synthetic_hydro_sim(index),
            input_data.q_meas,
            window,
        )
        scatters = _scatters(fig.axes[0])
        assert len(scatters) == 1
        assert np.allclose(
            scatters[0].get_facecolor(), mcolors.to_rgba(MEAS_Q_COLOR)
        )
        measured: pd.Series = (
            gui_plots._clip_series(input_data.q_meas, window).dropna()
        )
        offsets: np.ndarray = np.asarray(scatters[0].get_offsets())
        assert np.allclose(offsets[:, 1], measured.to_numpy())
        assert len(_lines(fig.axes[0])) == 0

    def test_window_clipping_and_limits(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        fig = gui_plots.make_partitioning_figure(
            _synthetic_hydro_sim(index),
            input_data.q_meas,
            window,
        )
        # The clipped record: the full record minus the spin-up.
        assert len(_fill_band_y(_fills(fig.axes[0])[0])[0]) == 2557
        # The record ends before the window's end (the partial 2025
        # water year), so the right limit is the record's last day.
        assert fig.axes[0].get_xlim() == (
            _day_number(window[0]),
            _day_number(index[-1]),
        )
        # A window the record spans keeps both limits from the window.
        inner: tuple[pd.Timestamp, pd.Timestamp] = water_year_window(2017, 2020)
        inner_fig = gui_plots.make_partitioning_figure(
            _synthetic_hydro_sim(index),
            input_data.q_meas,
            inner,
        )
        assert inner_fig.axes[0].get_xlim() == (
            _day_number(inner[0]),
            _day_number(inner[1]),
        )

    def test_legend_rows(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        fig = gui_plots.make_partitioning_figure(
            _synthetic_hydro_sim(index),
            input_data.q_meas,
            window,
        )
        assert len(fig.legends) == 2
        zone_row, discharge_row = fig.legends
        assert zone_row.get_frame().get_visible() is False
        assert zone_row.get_title().get_text() == "Zone"
        assert [text.get_text() for text in zone_row.get_texts()] == [
            ZONE_LABELS[zone] for zone in self.STACK_ZONES
        ]
        assert discharge_row.get_frame().get_visible() is False
        assert discharge_row.get_title().get_text() == "Discharge"
        assert [text.get_text() for text in discharge_row.get_texts()] == [
            "Discharge (meas)"
        ]
        # Both rows sit below the axes, the discharge row above the
        # zone row, without overlapping it.
        zone_top, _zone_bottom = _legend_extent_fractions(fig, zone_row)
        _discharge_top, discharge_bottom = _legend_extent_fractions(
            fig, discharge_row
        )
        assert zone_top <= fig.axes[0].get_position().y0
        assert discharge_bottom >= zone_top

    def test_overlay_previous_run(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """The previous run's stack draws behind, faintly, out of the legends."""
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        frame: pd.DataFrame = _synthetic_hydro_sim(index)
        # A previous run differs in the simulated discharge, not in
        # the zones' streamflow shares.
        overlay: pd.DataFrame = frame.copy()
        overlay["streamflow_sim"] = frame["streamflow_sim"] + 10.0

        fig = gui_plots.make_partitioning_figure(
            frame,
            input_data.q_meas,
            window,
            overlay_hydro_sim=overlay,
        )
        fills = _fills(fig.axes[0])
        # Four overlay bands first (half transparent), four current.
        assert len(fills) == 8
        assert [
            float(np.asarray(fill.get_facecolor())[0, 3]) for fill in fills
        ] == [0.5] * 4 + [1.0] * 4
        clipped_overlay: pd.DataFrame = gui_plots._clip_frame(overlay, window)
        overlay_flow: pd.Series = clipped_overlay["streamflow_sim"]
        shares: tuple[float, ...] = (0.5, 0.6, 0.3, 0.4)
        running: pd.Series = pd.Series(0.0, index=clipped_overlay.index)
        for fill, share in zip(fills[:4], shares):
            running = running + share * overlay_flow
            assert np.allclose(_fill_band_y(fill)[1], running.to_numpy())

    def test_overlay_skips_missing_columns(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """A previous frame that lacks a zone's column contributes no band."""
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        frame: pd.DataFrame = _synthetic_hydro_sim(index)
        overlay: pd.DataFrame = frame.drop(
            columns=[
                "prop_q_ground_rp",
                "q_lat_ground_rp",
                "s_ground_rp",
            ]
        )

        fig = gui_plots.make_partitioning_figure(
            frame,
            input_data.q_meas,
            window,
            overlay_hydro_sim=overlay,
        )
        # Three overlay bands (ground_rp missing), four current.
        assert len(_fills(fig.axes[0])) == 7


class TestRiverLoadsFigure:
    """The river loads (simulated concentration x discharge) figure."""

    SPECIES: tuple[str, ...] = ("DOC", "DIC", "Ca++")

    def test_axes_and_labels(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        fig: Figure = gui_plots.make_river_loads_figure(
            _synthetic_rt_sim(index, list(input_data.network.mineral_names)),
            _synthetic_hydro_sim(index),
            list(self.SPECIES),
            window,
        )
        assert len(fig.axes) == 1
        ax: Axes = fig.axes[0]
        assert ax.get_title() == "River loads"
        assert ax.get_ylabel() == "Load (mm·mol/L/d)"

    def test_one_line_per_species(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """Each river species' load: its concentration times the discharge."""
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        rt_sim = _synthetic_rt_sim(index, list(input_data.network.mineral_names))
        hydro_sim = _synthetic_hydro_sim(index)

        fig = gui_plots.make_river_loads_figure(
            rt_sim,
            hydro_sim,
            list(self.SPECIES),
            window,
        )
        lines = _lines(fig.axes[0])
        assert len(lines) == 3
        rt_plot: pd.DataFrame = gui_plots._clip_frame(rt_sim, window)
        hydro_plot: pd.DataFrame = gui_plots._clip_frame(hydro_sim, window)
        for line, species in zip(lines, self.SPECIES):
            assert mcolors.to_rgba(line.get_color()) == mcolors.to_rgba(
                RIVER_SPECIES_SIM_COLORS[species]
            )
            assert line.get_linewidth() == 1
            expected: np.ndarray = (
                rt_plot[f"{species}_riv"] * hydro_plot["streamflow_sim"]
            ).to_numpy()
            np.testing.assert_allclose(
                np.asarray(line.get_ydata()), expected
            )
        assert len(_scatters(fig.axes[0])) == 0

    def test_window_clipping_and_limits(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        fig = gui_plots.make_river_loads_figure(
            _synthetic_rt_sim(index, list(input_data.network.mineral_names)),
            _synthetic_hydro_sim(index),
            list(self.SPECIES),
            window,
        )
        # The clipped record: the full record minus the spin-up.
        assert len(np.asarray(_lines(fig.axes[0])[0].get_ydata())) == 2557
        assert fig.axes[0].get_xlim() == (
            _day_number(window[0]),
            _day_number(index[-1]),
        )
        # A window the record spans keeps both limits from the window.
        inner: tuple[pd.Timestamp, pd.Timestamp] = water_year_window(2017, 2020)
        inner_fig = gui_plots.make_river_loads_figure(
            _synthetic_rt_sim(index, list(input_data.network.mineral_names)),
            _synthetic_hydro_sim(index),
            list(self.SPECIES),
            inner,
        )
        assert inner_fig.axes[0].get_xlim() == (
            _day_number(inner[0]),
            _day_number(inner[1]),
        )

    def test_legend_below_the_axes(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        fig = gui_plots.make_river_loads_figure(
            _synthetic_rt_sim(index, list(input_data.network.mineral_names)),
            _synthetic_hydro_sim(index),
            list(self.SPECIES),
            window,
        )
        assert len(fig.legends) == 1
        legend = fig.legends[0]
        assert legend.get_frame().get_visible() is False
        assert legend.get_title().get_text() == "Species"
        assert [text.get_text() for text in legend.get_texts()] == list(
            self.SPECIES
        )
        top, bottom = _legend_extent_fractions(fig, legend)
        assert top <= fig.axes[0].get_position().y0
        assert bottom >= 0.0

    def test_overlay_previous_run(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """The previous run's loads draw behind, faintly, out of the legend."""
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        rt_sim = _synthetic_rt_sim(index, list(input_data.network.mineral_names))
        hydro_sim = _synthetic_hydro_sim(index)

        overlay_rt: pd.DataFrame = rt_sim.copy()
        overlay_hydro: pd.DataFrame = hydro_sim.copy()
        for species in self.SPECIES:
            overlay_rt[f"{species}_riv"] = (
                rt_sim[f"{species}_riv"] + 1.0
            )
        overlay_hydro["streamflow_sim"] = (
            hydro_sim["streamflow_sim"] + 10.0
        )

        fig = gui_plots.make_river_loads_figure(
            rt_sim,
            hydro_sim,
            list(self.SPECIES),
            window,
            overlay_rt_sim=overlay_rt,
            overlay_hydro_sim=overlay_hydro,
        )
        lines = _lines(fig.axes[0])
        # Three overlay lines first (half transparent), three current.
        assert len(lines) == 6
        assert [_alpha(line) for line in lines] == [0.5] * 3 + [1.0] * 3
        rt_plot: pd.DataFrame = gui_plots._clip_frame(rt_sim, window)
        hydro_plot: pd.DataFrame = gui_plots._clip_frame(hydro_sim, window)
        overlay_rt_plot: pd.DataFrame = gui_plots._clip_frame(
            overlay_rt, window
        )
        overlay_hydro_plot: pd.DataFrame = gui_plots._clip_frame(
            overlay_hydro, window
        )
        for line, species in zip(lines[:3], self.SPECIES):
            expected: np.ndarray = (
                overlay_rt_plot[f"{species}_riv"]
                * overlay_hydro_plot["streamflow_sim"]
            ).to_numpy()
            np.testing.assert_allclose(
                np.asarray(line.get_ydata()), expected
            )

    def test_overlay_skips_missing_columns(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """A previous frame that lacks a species contributes no line."""
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        rt_sim = _synthetic_rt_sim(index, list(input_data.network.mineral_names))
        hydro_sim = _synthetic_hydro_sim(index)
        overlay_rt: pd.DataFrame = rt_sim.drop(columns=["Ca++_riv"])
        overlay_hydro: pd.DataFrame = hydro_sim.copy()
        overlay_hydro["streamflow_sim"] = (
            hydro_sim["streamflow_sim"] + 10.0
        )

        fig = gui_plots.make_river_loads_figure(
            rt_sim,
            hydro_sim,
            list(self.SPECIES),
            window,
            overlay_rt_sim=overlay_rt,
            overlay_hydro_sim=overlay_hydro,
        )
        # Two overlay lines (Ca++ missing), three current.
        assert len(_lines(fig.axes[0])) == 5


class TestConcDischargeFigure:
    """The concentration-discharge scatter panels on synthetic data."""

    SPECIES: tuple[str, ...] = ("DOC", "DIC", "Ca++")

    def test_panels_and_axes(self, input_data: gui_data.GuiInputData) -> None:
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        fig: Figure = gui_plots.make_conc_discharge_figure(
            _synthetic_rt_sim(index, list(input_data.network.mineral_names)),
            _synthetic_hydro_sim(index),
            input_data.q_meas,
            input_data.conc_df,
            list(self.SPECIES),
            window,
        )
        assert len(fig.axes) == 3
        for ax, species in zip(fig.axes, self.SPECIES):
            assert isinstance(ax, Axes)
            assert ax.get_title() == species
            assert ax.get_xlabel() == "Discharge (mm/d)"
            assert ax.get_ylabel() == "Concentration (mol/L)"
            assert ax.get_xscale() == "log"
            assert ax.get_yscale() == "log"

    def test_sim_points(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """The simulated points pair the simulated q with the simulated conc."""
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        rt_sim = _synthetic_rt_sim(index, list(input_data.network.mineral_names))
        hydro_sim = _synthetic_hydro_sim(index)

        fig = gui_plots.make_conc_discharge_figure(
            rt_sim,
            hydro_sim,
            input_data.q_meas,
            input_data.conc_df,
            list(self.SPECIES),
            window,
        )
        rt_plot: pd.DataFrame = gui_plots._clip_frame(rt_sim, window)
        hydro_plot: pd.DataFrame = gui_plots._clip_frame(hydro_sim, window)
        for ax, species in zip(fig.axes, self.SPECIES):
            scatters = _scatters(ax)
            assert len(scatters) == 2
            sim = scatters[0]
            sim_face: np.ndarray = np.asarray(sim.get_facecolor())
            assert mcolors.to_rgba(sim_face[0]) == mcolors.to_rgba(
                RIVER_SPECIES_SIM_COLORS[species]
            )
            offsets: np.ndarray = np.asarray(sim.get_offsets())
            # The synthetic record is strictly positive in the window, so
            # every clipped day plots.
            assert np.allclose(
                offsets[:, 0], hydro_plot["streamflow_sim"].to_numpy()
            )
            assert np.allclose(
                offsets[:, 1], rt_plot[f"{species}_riv"].to_numpy()
            )
            assert len(_lines(ax)) == 0

    def test_meas_points(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """The observed points pair the measured q with the measured conc."""
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        fig = gui_plots.make_conc_discharge_figure(
            _synthetic_rt_sim(index, list(input_data.network.mineral_names)),
            _synthetic_hydro_sim(index),
            input_data.q_meas,
            input_data.conc_df,
            list(self.SPECIES),
            window,
        )
        q_plot: pd.Series = gui_plots._clip_series(input_data.q_meas, window)
        conc_plot: pd.DataFrame = gui_plots._clip_frame(
            input_data.conc_df, window
        )
        for ax, species in zip(fig.axes, self.SPECIES):
            scatters = _scatters(ax)
            meas = scatters[1]
            meas_face: np.ndarray = np.asarray(meas.get_facecolor())
            assert mcolors.to_rgba(meas_face[0]) == mcolors.to_rgba(
                RIVER_SPECIES_MEAS_COLORS[species]
            )
            conc: pd.Series = conc_plot[species]
            mask: pd.Series = (q_plot > 0) & (conc > 0)
            offsets = np.asarray(meas.get_offsets())
            assert np.allclose(offsets[:, 0], q_plot[mask].to_numpy())
            assert np.allclose(offsets[:, 1], conc[mask].to_numpy())

    def test_positive_values_only(self) -> None:
        """Zeros and negative values drop out of the log-log panels."""
        index = pd.date_range("2015-10-01", periods=6, freq="D")
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        hydro_sim = pd.DataFrame(
            {
                "streamflow_sim": np.array([0.0, 1.0, -2.0, 3.0, 4.0, 5.0])
            },
            index=index,
        )
        rt_sim = pd.DataFrame(
            {
                "DOC_riv": np.array([1.0, 0.0, -3.0, 4.0, 5.0, 6.0]),
            },
            index=index,
        )
        q_meas = pd.Series(
            [0.5, -1.0, 2.0, 3.0, 4.0, 5.0],
            index=index,
            name="q_meas",
        )
        conc_df = pd.DataFrame(
            {"DOC": np.array([1.0, 2.0, -3.0, 4.0, 5.0, 0.0])}, index=index
        )

        fig = gui_plots.make_conc_discharge_figure(
            rt_sim,
            hydro_sim,
            q_meas,
            conc_df,
            ["DOC"],
            window,
        )
        scatters = _scatters(fig.axes[0])
        # The first three days carry a zero or negative value on at
        # least one side of each pair: three points per cloud.
        sim_offsets: np.ndarray = np.asarray(scatters[0].get_offsets())
        assert np.allclose(
            sim_offsets, [[3.0, 4.0], [4.0, 5.0], [5.0, 6.0]]
        )
        meas_offsets: np.ndarray = np.asarray(scatters[1].get_offsets())
        assert np.allclose(
            meas_offsets, [[0.5, 1.0], [3.0, 4.0], [4.0, 5.0]]
        )

    def test_window_clipping(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        index: pd.Index = input_data.forc.precip.index
        inner: tuple[pd.Timestamp, pd.Timestamp] = water_year_window(2017, 2020)
        rt_sim = _synthetic_rt_sim(index, list(input_data.network.mineral_names))
        hydro_sim = _synthetic_hydro_sim(index)

        fig = gui_plots.make_conc_discharge_figure(
            rt_sim,
            hydro_sim,
            input_data.q_meas,
            input_data.conc_df,
            list(self.SPECIES),
            inner,
        )
        expected: int = int(((index >= inner[0]) & (index <= inner[1])).sum())
        # The bundled record carries a streamflow column, so the
        # measured overlay is present on every panel.
        assert input_data.q_meas is not None
        # The synthetic frames are strictly positive: every day in the
        # window plots, the record outside it does not.
        for ax, species in zip(fig.axes, self.SPECIES):
            sim, meas = _scatters(ax)
            assert len(np.asarray(sim.get_offsets())) == expected
            conc: pd.Series = input_data.conc_df[species]
            mask: pd.Series = (index >= inner[0]) & (index <= inner[1])
            positive: pd.Series = (
                (input_data.q_meas > 0) & (conc > 0)
            ) & mask
            assert len(np.asarray(meas.get_offsets())) == int(
                positive.sum()
            )

    def test_legends(self, input_data: gui_data.GuiInputData) -> None:
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        fig = gui_plots.make_conc_discharge_figure(
            _synthetic_rt_sim(index, list(input_data.network.mineral_names)),
            _synthetic_hydro_sim(index),
            input_data.q_meas,
            input_data.conc_df,
            list(self.SPECIES),
            window,
        )
        assert len(fig.legends) == 2
        sim_row, meas_row = fig.legends
        assert sim_row.get_frame().get_visible() is False
        assert sim_row.get_title().get_text() == "Simulated"
        assert [text.get_text() for text in sim_row.get_texts()] == list(
            self.SPECIES
        )
        assert meas_row.get_frame().get_visible() is False
        assert meas_row.get_title().get_text() == "Measured"
        assert [text.get_text() for text in meas_row.get_texts()] == list(
            self.SPECIES
        )
        # Both rows sit below the axes, the measured row above the
        # simulated row, without overlapping it.
        sim_top, _sim_bottom = _legend_extent_fractions(fig, sim_row)
        _meas_top, meas_bottom = _legend_extent_fractions(fig, meas_row)
        assert sim_top <= fig.axes[0].get_position().y0
        assert meas_bottom >= sim_top
        assert fig.subplotpars.bottom >= _meas_top


class TestZoneConcentrationsFigure:
    """The zone concentrations figure on synthetic data."""

    SPECIES: tuple[str, ...] = ("DOC", "DIC", "Ca++")
    # DIC is derived, so the frame carries no per-zone DIC column and
    # only the network species plot.
    NETWORK_SPECIES: tuple[str, ...] = ("DOC", "Ca++")
    ZONES: tuple[str, ...] = tuple(ZONE_ORDER)

    def test_axes_and_labels(self, input_data: gui_data.GuiInputData) -> None:
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        fig: Figure = gui_plots.make_zone_concentrations_figure(
            _synthetic_rt_sim(index, list(input_data.network.mineral_names)),
            list(self.SPECIES),
            list(self.ZONES),
            window,
        )
        assert len(fig.axes) == 1
        ax: Axes = fig.axes[0]
        assert ax.get_title() == "Concentration"
        assert ax.get_ylabel() == "Concentration (mol/L)"

    def test_lines_and_encoding(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """One line per zone-species pair: color is zone, linestyle species."""
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        rt_sim = _synthetic_rt_sim(index, list(input_data.network.mineral_names))

        fig = gui_plots.make_zone_concentrations_figure(
            rt_sim,
            list(self.SPECIES),
            list(self.ZONES),
            window,
        )
        lines = _lines(fig.axes[0])
        assert len(lines) == len(self.ZONES) * len(self.NETWORK_SPECIES)
        rt_plot: pd.DataFrame = gui_plots._clip_frame(rt_sim, window)
        for position, line in enumerate(lines):
            zone = self.ZONES[position // len(self.NETWORK_SPECIES)]
            species = self.NETWORK_SPECIES[
                position % len(self.NETWORK_SPECIES)
            ]
            assert mcolors.to_rgba(line.get_color()) == mcolors.to_rgba(
                ZONE_COLORS[zone]
            )
            assert line.get_linestyle() == LINESTYLE_CYCLE[
                self.SPECIES.index(species) % len(LINESTYLE_CYCLE)
            ]
            assert line.get_linewidth() == 1
            np.testing.assert_allclose(
                np.asarray(line.get_ydata()),
                rt_plot[f"{species}_{zone}"].to_numpy(),
            )
        assert len(_scatters(fig.axes[0])) == 0

    def test_window_clipping_and_limits(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        rt_sim = _synthetic_rt_sim(index, list(input_data.network.mineral_names))

        fig = gui_plots.make_zone_concentrations_figure(
            rt_sim,
            list(self.SPECIES),
            list(self.ZONES),
            window,
        )
        assert len(np.asarray(_lines(fig.axes[0])[0].get_ydata())) == 2557
        assert fig.axes[0].get_xlim() == (
            _day_number(window[0]),
            _day_number(index[-1]),
        )
        # A window the record spans keeps both limits from the window.
        inner: tuple[pd.Timestamp, pd.Timestamp] = water_year_window(2017, 2020)
        inner_fig = gui_plots.make_zone_concentrations_figure(
            rt_sim,
            list(self.SPECIES),
            list(self.ZONES),
            inner,
        )
        assert inner_fig.axes[0].get_xlim() == (
            _day_number(inner[0]),
            _day_number(inner[1]),
        )

    def test_legend_rows(self, input_data: gui_data.GuiInputData) -> None:
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        fig = gui_plots.make_zone_concentrations_figure(
            _synthetic_rt_sim(index, list(input_data.network.mineral_names)),
            list(self.SPECIES),
            list(self.ZONES),
            window,
        )
        assert len(fig.legends) == 2
        zone_row, species_row = fig.legends
        assert zone_row.get_frame().get_visible() is False
        assert zone_row.get_title().get_text() == "Zone"
        assert [text.get_text() for text in zone_row.get_texts()] == [
            ZONE_LABELS[zone] for zone in self.ZONES
        ]
        assert species_row.get_frame().get_visible() is False
        assert species_row.get_title().get_text() == "Species"
        assert [text.get_text() for text in species_row.get_texts()] == list(
            self.NETWORK_SPECIES
        )
        # The species row's handles carry the species' linestyles.
        for handle, species in zip(
            species_row.legend_handles, self.NETWORK_SPECIES
        ):
            assert isinstance(handle, Line2D)
            assert handle.get_linestyle() == LINESTYLE_CYCLE[
                self.SPECIES.index(species) % len(LINESTYLE_CYCLE)
            ]
        # Both rows sit below the axes, the species row above the zone
        # row, without overlapping it.
        zone_top, _zone_bottom = _legend_extent_fractions(fig, zone_row)
        _species_top, species_bottom = _legend_extent_fractions(fig, species_row)
        assert zone_top <= fig.axes[0].get_position().y0
        assert species_bottom >= zone_top

    def test_overlay_previous_run(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """The previous run's concentrations draw behind, faintly, unlabeled."""
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        rt_sim = _synthetic_rt_sim(index, list(input_data.network.mineral_names))

        overlay_rt: pd.DataFrame = rt_sim.copy()
        for species in self.NETWORK_SPECIES:
            for zone in self.ZONES:
                overlay_rt[f"{species}_{zone}"] = (
                    rt_sim[f"{species}_{zone}"] + 1.0
                )

        fig = gui_plots.make_zone_concentrations_figure(
            rt_sim,
            list(self.SPECIES),
            list(self.ZONES),
            window,
            overlay_rt_sim=overlay_rt,
        )
        lines = _lines(fig.axes[0])
        expected_count: int = len(self.ZONES) * len(self.NETWORK_SPECIES)
        # The overlay's lines first (half transparent), then the current.
        assert len(lines) == 2 * expected_count
        assert [_alpha(line) for line in lines] == [
            0.5
        ] * expected_count + [1.0] * expected_count
        overlay_plot: pd.DataFrame = gui_plots._clip_frame(
            overlay_rt, window
        )
        for position, line in enumerate(lines[:expected_count]):
            zone = self.ZONES[position // len(self.NETWORK_SPECIES)]
            species = self.NETWORK_SPECIES[
                position % len(self.NETWORK_SPECIES)
            ]
            np.testing.assert_allclose(
                np.asarray(line.get_ydata()),
                overlay_plot[f"{species}_{zone}"].to_numpy(),
            )

    def test_overlay_skips_missing_columns(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """A previous frame that lacks a column contributes no line for it."""
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        rt_sim = _synthetic_rt_sim(index, list(input_data.network.mineral_names))
        overlay_rt: pd.DataFrame = rt_sim.drop(columns=["DOC_snow_hs"])

        fig = gui_plots.make_zone_concentrations_figure(
            rt_sim,
            list(self.SPECIES),
            list(self.ZONES),
            window,
            overlay_rt_sim=overlay_rt,
        )
        expected_count: int = len(self.ZONES) * len(self.NETWORK_SPECIES)
        # One overlay line missing, all current lines present.
        assert len(_lines(fig.axes[0])) == (expected_count - 1) + expected_count

    def test_skips_derived_species_without_zone_columns(self) -> None:
        """A derived species carried only as ``_riv`` does not crash."""
        index = pd.date_range("2020-01-01", periods=3, freq="D")
        sim = pd.DataFrame(
            {
                "DOC_surface_hs": [1.0, 2.0, 3.0],
                "Ca++_surface_hs": [3.0, 2.0, 1.0],
                "DOC_riv": [1.5, 2.5, 3.5],
                "DIC_riv": [0.5, 1.5, 2.5],
                "Ca++_riv": [2.0, 2.0, 2.0],
            },
            index=index,
        )
        fig = gui_plots.make_zone_concentrations_figure(
            sim,
            ["DOC", "DIC", "Ca++"],
            ["surface_hs"],
            (index[0], index[-1]),
        )
        assert len(_lines(fig.axes[0])) == 2
        zone_row, species_row = fig.legends
        assert [text.get_text() for text in zone_row.get_texts()] == [
            "Surface (HS)"
        ]
        assert [text.get_text() for text in species_row.get_texts()] == [
            "DOC",
            "Ca++",
        ]

    def test_zone_legend_lists_only_plotted_zones(self) -> None:
        """A selected zone without any concentration column is omitted."""
        index = pd.date_range("2020-01-01", periods=3, freq="D")
        sim = pd.DataFrame(
            {"DOC_surface_hs": [1.0, 2.0, 3.0]},
            index=index,
        )
        fig = gui_plots.make_zone_concentrations_figure(
            sim,
            ["DOC"],
            ["snow_hs", "surface_hs"],
            (index[0], index[-1]),
        )
        assert len(_lines(fig.axes[0])) == 1
        zone_row, _species_row = fig.legends
        assert [text.get_text() for text in zone_row.get_texts()] == [
            "Surface (HS)"
        ]


class TestRateFigure:
    """The single-mineral reaction-rate figure on synthetic data."""

    MINERAL: str = "SOC(s)"
    # The model emits a mineral's rate columns only for the zones in
    # which its reactions run: here, all four non-snow zones.
    REACTION_ZONES: tuple[str, ...] = tuple(ZONE_ORDER[2:])

    def test_axes_and_labels(self, input_data: gui_data.GuiInputData) -> None:
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        fig: Figure = gui_plots.make_rate_figure(
            _synthetic_rt_sim(index, list(input_data.network.mineral_names)),
            self.MINERAL,
            window,
        )
        assert len(fig.axes) == 1
        ax: Axes = fig.axes[0]
        assert ax.get_title() == f"Reaction rates ({self.MINERAL})"
        assert ax.get_ylabel() == "Reaction rate (mol/L/day)"

    def test_one_line_per_reacting_zone(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """One zone-colored line per zone carrying the mineral's rate."""
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        rt_sim = _synthetic_rt_sim(index, list(input_data.network.mineral_names))

        fig = gui_plots.make_rate_figure(rt_sim, self.MINERAL, window)
        lines = _lines(fig.axes[0])
        assert len(lines) == len(self.REACTION_ZONES)
        rt_plot: pd.DataFrame = gui_plots._clip_frame(rt_sim, window)
        for line, zone in zip(lines, self.REACTION_ZONES):
            assert mcolors.to_rgba(line.get_color()) == mcolors.to_rgba(
                ZONE_COLORS[zone]
            )
            assert line.get_linewidth() == 1
            np.testing.assert_allclose(
                np.asarray(line.get_ydata()),
                rt_plot[f"{self.MINERAL}_rate_{zone}"].to_numpy(),
            )
        assert len(_scatters(fig.axes[0])) == 0

    def test_overlay_previous_run(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """The previous run's rates draw behind, faintly, out of the legend."""
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        rt_sim = _synthetic_rt_sim(index, list(input_data.network.mineral_names))
        # A previous run differs in the mineral's rates, not in the
        # rest of the frame.
        overlay_rt: pd.DataFrame = rt_sim.copy()
        for zone in self.REACTION_ZONES:
            overlay_rt[f"{self.MINERAL}_rate_{zone}"] = (
                rt_sim[f"{self.MINERAL}_rate_{zone}"] + 10.0
            )

        fig = gui_plots.make_rate_figure(
            rt_sim,
            self.MINERAL,
            window,
            overlay_rt_sim=overlay_rt,
        )
        lines = _lines(fig.axes[0])
        # The overlay's lines first (half transparent), then the
        # current ones.
        assert len(lines) == 2 * len(self.REACTION_ZONES)
        assert [_alpha(line) for line in lines] == [
            0.5
        ] * len(self.REACTION_ZONES) + [1.0] * len(self.REACTION_ZONES)
        overlay_plot: pd.DataFrame = gui_plots._clip_frame(
            overlay_rt, window
        )
        for line, zone in zip(
            lines[: len(self.REACTION_ZONES)], self.REACTION_ZONES
        ):
            np.testing.assert_allclose(
                np.asarray(line.get_ydata()),
                overlay_plot[f"{self.MINERAL}_rate_{zone}"].to_numpy(),
            )

    def test_window_clipping_and_limits(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        rt_sim = _synthetic_rt_sim(index, list(input_data.network.mineral_names))

        fig = gui_plots.make_rate_figure(rt_sim, self.MINERAL, window)
        assert len(np.asarray(_lines(fig.axes[0])[0].get_ydata())) == 2557
        assert fig.axes[0].get_xlim() == (
            _day_number(window[0]),
            _day_number(index[-1]),
        )
        # A window the record spans keeps both limits from the window.
        inner: tuple[pd.Timestamp, pd.Timestamp] = water_year_window(2017, 2020)
        inner_fig = gui_plots.make_rate_figure(rt_sim, self.MINERAL, inner)
        assert inner_fig.axes[0].get_xlim() == (
            _day_number(inner[0]),
            _day_number(inner[1]),
        )

    def test_zone_legend_below_the_axes(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        fig = gui_plots.make_rate_figure(
            _synthetic_rt_sim(index, list(input_data.network.mineral_names)),
            self.MINERAL,
            window,
        )
        assert len(fig.legends) == 1
        zone_row = fig.legends[0]
        assert zone_row.get_frame().get_visible() is False
        assert zone_row.get_title().get_text() == "Zone"
        assert [text.get_text() for text in zone_row.get_texts()] == [
            ZONE_LABELS[zone] for zone in self.REACTION_ZONES
        ]
        top, bottom = _legend_extent_fractions(fig, zone_row)
        assert top <= fig.axes[0].get_position().y0
        assert bottom >= 0.0

    def test_overlay_skips_missing_columns(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """A previous frame that lacks a zone's rate column contributes no line."""
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        rt_sim = _synthetic_rt_sim(index, list(input_data.network.mineral_names))
        # The overlay carries the first reacting zone's rate column
        # only.
        overlay_rt: pd.DataFrame = rt_sim.drop(
            columns=[
                f"{self.MINERAL}_rate_{zone}"
                for zone in self.REACTION_ZONES[1:]
            ]
        )

        fig = gui_plots.make_rate_figure(
            rt_sim,
            self.MINERAL,
            window,
            overlay_rt_sim=overlay_rt,
        )
        # One overlay line (the zone the frame carries), then the four
        # current ones.
        lines = _lines(fig.axes[0])
        assert len(lines) == len(self.REACTION_ZONES) + 1
        assert [_alpha(line) for line in lines] == [
            0.5
        ] + [1.0] * len(self.REACTION_ZONES)

    def test_mineral_without_rate_column(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """A mineral no zone carries plots no lines and no legend."""
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        rt_sim = _synthetic_rt_sim(index, [self.MINERAL]).drop(
            columns=[
                f"{self.MINERAL}_rate_{zone}" for zone in self.REACTION_ZONES
            ]
        )

        fig: Figure = gui_plots.make_rate_figure(rt_sim, self.MINERAL, window)
        assert len(fig.axes) == 1
        assert len(_lines(fig.axes[0])) == 0
        assert len(fig.legends) == 0


class TestHydrologyFigure:
    """The hydrology figure on synthetic data."""

    def test_axes_and_labels(self, input_data: gui_data.GuiInputData) -> None:
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        fig: Figure = gui_plots.make_hydrology_figure(
            _synthetic_hydro_sim(index),
            input_data.q_meas,
            window,
        )
        assert len(fig.axes) == 1
        ax: Axes = fig.axes[0]
        assert ax.get_title() == "Streamflow"
        assert ax.get_ylabel() == "Discharge (mm/d)"
        # The discharge axis is upright: the old figure inverted it to
        # hang from a shared top legend, which this standalone figure
        # no longer carries.
        lower, upper = ax.get_ylim()
        assert lower == 0.0
        assert upper > 0.0

    def test_sim_line_and_meas_scatter(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """The simulation is a line and the measurement is scatter."""
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        hydro_sim = _synthetic_hydro_sim(index)

        fig = gui_plots.make_hydrology_figure(
            hydro_sim, input_data.q_meas, window
        )
        ax = fig.axes[0]

        lines = _lines(ax)
        assert len(lines) == 1
        assert mcolors.to_rgba(lines[0].get_color()) == mcolors.to_rgba(
            SIM_Q_COLOR
        )
        assert lines[0].get_linewidth() == 1
        hydro_plot: pd.DataFrame = gui_plots._clip_frame(hydro_sim, window)
        np.testing.assert_allclose(
            np.asarray(lines[0].get_ydata()),
            hydro_plot["streamflow_sim"].to_numpy(),
        )

        scatters = _scatters(ax)
        assert len(scatters) == 1
        facecolors: np.ndarray = np.asarray(scatters[0].get_facecolor())
        assert mcolors.to_rgba(facecolors[0]) == mcolors.to_rgba(MEAS_Q_COLOR)
        assert list(scatters[0].get_sizes()) == [5]
        q_plot: pd.Series = (
            gui_plots._clip_series(input_data.q_meas, window).dropna()
        )
        offsets: np.ndarray = np.asarray(scatters[0].get_offsets())
        assert len(offsets) == len(q_plot)
        np.testing.assert_allclose(offsets[:, 1], q_plot.to_numpy())

    def test_window_clipping_and_limits(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        hydro_sim = _synthetic_hydro_sim(index)

        fig = gui_plots.make_hydrology_figure(
            hydro_sim, input_data.q_meas, window
        )
        assert len(np.asarray(_lines(fig.axes[0])[0].get_ydata())) == 2557
        assert fig.axes[0].get_xlim() == (
            _day_number(window[0]),
            _day_number(index[-1]),
        )
        # A window the record spans keeps both limits from the window.
        inner: tuple[pd.Timestamp, pd.Timestamp] = water_year_window(2017, 2020)
        inner_fig = gui_plots.make_hydrology_figure(
            hydro_sim, input_data.q_meas, inner
        )
        assert inner_fig.axes[0].get_xlim() == (
            _day_number(inner[0]),
            _day_number(inner[1]),
        )

    def test_legend_below_the_axes(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        fig = gui_plots.make_hydrology_figure(
            _synthetic_hydro_sim(index), input_data.q_meas, window
        )
        assert len(fig.legends) == 1
        row = fig.legends[0]
        assert row.get_frame().get_visible() is False
        assert row.get_title().get_text() == ""
        assert [text.get_text() for text in row.get_texts()] == [
            "Streamflow (sim)",
            "Streamflow (meas)",
        ]
        # The row sits below the axes, without overlapping them.
        top, bottom = _legend_extent_fractions(fig, row)
        assert top <= fig.axes[0].get_position().y0
        assert fig.subplotpars.bottom >= bottom

    def test_overlay_previous_run(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """The previous run's streamflow draws behind, faintly, unlabeled."""
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        hydro_sim = _synthetic_hydro_sim(index)

        overlay_hydro: pd.DataFrame = hydro_sim.copy()
        overlay_hydro["streamflow_sim"] = hydro_sim["streamflow_sim"] + 10.0

        fig = gui_plots.make_hydrology_figure(
            hydro_sim,
            input_data.q_meas,
            window,
            overlay_hydro_sim=overlay_hydro,
        )
        lines = _lines(fig.axes[0])
        # One overlay line (half transparent), then the current one.
        assert len(lines) == 2
        assert [_alpha(line) for line in lines] == [0.5, 1.0]
        overlay_plot: pd.DataFrame = gui_plots._clip_frame(
            overlay_hydro, window
        )
        np.testing.assert_allclose(
            np.asarray(lines[0].get_ydata()),
            overlay_plot["streamflow_sim"].to_numpy(),
        )


class TestRiverConcentrationsFigure:
    """The twin-axis streamflow / river-concentrations figure."""

    SPECIES: tuple[str, ...] = ("DOC", "DIC", "Ca++")

    def _fig(
        self,
        input_data: gui_data.GuiInputData,
        streamflow: bool,
        species: list[str] | None = None,
        **kwargs: object,
    ) -> Figure:
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        return gui_plots.make_river_concentrations_figure(
            _synthetic_rt_sim(index, list(input_data.network.mineral_names)),
            _synthetic_hydro_sim(index),
            input_data.q_meas,
            input_data.conc_df,
            list(self.SPECIES) if species is None else species,
            streamflow,
            window,
            **kwargs,
        )

    def test_twin_axes_and_labels(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        fig = self._fig(input_data, streamflow=True)
        assert len(fig.axes) == 2
        hydro_ax, species_ax = fig.axes
        assert hydro_ax.get_title() == "Streamflow"
        # The species title sits on the right, so it does not
        # overlap the streamflow title on the left.
        assert species_ax.get_title("right") == "River concentrations"
        assert species_ax.get_title() == ""
        assert hydro_ax.get_ylabel() == "Discharge (mm/d)"
        assert species_ax.get_ylabel() == "Concentration (mol/L)"
        # The twin axes share one panel.
        left = hydro_ax.get_position()
        right = species_ax.get_position()
        assert right.x0 == pytest.approx(left.x0)
        assert right.y0 == pytest.approx(left.y0)
        assert right.width == pytest.approx(left.width)
        assert right.height == pytest.approx(left.height)

    def test_left_axis_inverted_streamflow(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """The left axis carries the inverted streamflow series."""
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        hydro_sim = _synthetic_hydro_sim(index)

        fig = self._fig(input_data, streamflow=True)
        hydro_ax = fig.axes[0]

        # Inverted: the top of the axis is at zero.
        lower, upper = hydro_ax.get_ylim()
        assert lower > upper
        assert upper == 0.0

        lines = _lines(hydro_ax)
        assert len(lines) == 1
        assert mcolors.to_rgba(lines[0].get_color()) == mcolors.to_rgba(
            SIM_Q_COLOR
        )
        hydro_plot: pd.DataFrame = gui_plots._clip_frame(hydro_sim, window)
        np.testing.assert_allclose(
            np.asarray(lines[0].get_ydata()),
            hydro_plot["streamflow_sim"].to_numpy(),
        )

        scatters = _scatters(hydro_ax)
        assert len(scatters) == 1
        facecolors: np.ndarray = np.asarray(scatters[0].get_facecolor())
        assert mcolors.to_rgba(facecolors[0]) == mcolors.to_rgba(MEAS_Q_COLOR)
        q_plot: pd.Series = (
            gui_plots._clip_series(input_data.q_meas, window).dropna()
        )
        offsets: np.ndarray = np.asarray(scatters[0].get_offsets())
        assert len(offsets) == len(q_plot)
        np.testing.assert_allclose(offsets[:, 1], q_plot.to_numpy())

    def test_species_series(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """One sim line and one meas scatter per selected species."""
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        rt_sim = _synthetic_rt_sim(index, list(input_data.network.mineral_names))

        fig = self._fig(input_data, streamflow=True)
        species_ax = fig.axes[1]

        lines = _lines(species_ax)
        assert [line.get_label() for line in lines] == [
            f"{species} (sim)" for species in self.SPECIES
        ]
        rt_plot: pd.DataFrame = gui_plots._clip_frame(rt_sim, window)
        for line, species in zip(lines, self.SPECIES):
            assert mcolors.to_rgba(line.get_color()) == mcolors.to_rgba(
                RIVER_SPECIES_SIM_COLORS[species]
            )
            assert line.get_linewidth() == 1
            np.testing.assert_allclose(
                np.asarray(line.get_ydata()),
                rt_plot[f"{species}_riv"].to_numpy(),
            )

        scatters = _scatters(species_ax)
        assert [scatter.get_label() for scatter in scatters] == [
            f"{species} (meas)" for species in self.SPECIES
        ]
        conc_plot: pd.DataFrame = gui_plots._clip_frame(input_data.conc_df, window)
        for scatter, species in zip(scatters, self.SPECIES):
            facecolors: np.ndarray = np.asarray(scatter.get_facecolor())
            assert mcolors.to_rgba(facecolors[0]) == mcolors.to_rgba(
                RIVER_SPECIES_MEAS_COLORS[species]
            )
            assert list(scatter.get_sizes()) == [5]
            conc: pd.Series = conc_plot[species].dropna()
            offsets = np.asarray(scatter.get_offsets())
            assert len(offsets) == len(conc)
            np.testing.assert_allclose(offsets[:, 1], conc.to_numpy())

    def test_species_only_axis(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """With streamflow off, the figure carries the species axis alone."""
        fig = self._fig(input_data, streamflow=False)
        assert len(fig.axes) == 1
        ax: Axes = fig.axes[0]
        assert ax.get_title() == "River concentrations"
        assert ax.get_ylabel() == "Concentration (mol/L)"
        assert len(_lines(ax)) == len(self.SPECIES)
        assert len(_scatters(ax)) == len(self.SPECIES)

    def test_window_clipping_and_limits(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        hydro_sim = _synthetic_hydro_sim(index)
        rt_sim = _synthetic_rt_sim(index, list(input_data.network.mineral_names))

        fig = self._fig(input_data, streamflow=True)
        hydro_ax, species_ax = fig.axes
        assert len(np.asarray(_lines(hydro_ax)[0].get_ydata())) == 2557
        for ax in fig.axes:
            assert ax.get_xlim() == (
                _day_number(window[0]),
                _day_number(index[-1]),
            )
        # A window the record spans keeps both limits from the window.
        inner: tuple[pd.Timestamp, pd.Timestamp] = water_year_window(2017, 2020)
        inner_fig = gui_plots.make_river_concentrations_figure(
            rt_sim,
            hydro_sim,
            input_data.q_meas,
            input_data.conc_df,
            list(self.SPECIES),
            True,
            inner,
        )
        for ax in inner_fig.axes:
            assert ax.get_xlim() == (
                _day_number(inner[0]),
                _day_number(inner[1]),
            )

    def test_legend_below_the_axes(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        fig = self._fig(input_data, streamflow=True)
        assert len(fig.legends) == 1
        row = fig.legends[0]
        assert row.get_frame().get_visible() is False
        assert row.get_title().get_text() == ""
        expected: list[str] = ["Streamflow (sim)", "Streamflow (meas)"]
        for species in self.SPECIES:
            expected.extend([f"{species} (sim)", f"{species} (meas)"])
        assert [text.get_text() for text in row.get_texts()] == expected
        # The row sits below the axes, inside the figure's bounds.
        top, bottom = _legend_extent_fractions(fig, row)
        assert top <= fig.axes[0].get_position().y0
        assert fig.subplotpars.bottom >= bottom

    def test_overlay_previous_run(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """The previous run's series draw behind, faintly, unlabeled."""
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        rt_sim = _synthetic_rt_sim(index, list(input_data.network.mineral_names))
        hydro_sim = _synthetic_hydro_sim(index)

        overlay_rt: pd.DataFrame = rt_sim.copy()
        for species in self.SPECIES:
            overlay_rt[f"{species}_riv"] = rt_sim[f"{species}_riv"] + 1.0
        overlay_hydro: pd.DataFrame = hydro_sim.copy()
        overlay_hydro["streamflow_sim"] = hydro_sim["streamflow_sim"] + 10.0

        fig = gui_plots.make_river_concentrations_figure(
            rt_sim,
            hydro_sim,
            input_data.q_meas,
            input_data.conc_df,
            list(self.SPECIES),
            True,
            window,
            overlay_rt_sim=overlay_rt,
            overlay_hydro_sim=overlay_hydro,
        )
        hydro_ax, species_ax = fig.axes

        hydro_lines = _lines(hydro_ax)
        # One overlay streamflow line (half transparent), then the
        # current one.
        assert len(hydro_lines) == 2
        assert [_alpha(line) for line in hydro_lines] == [0.5, 1.0]
        overlay_hydro_plot: pd.DataFrame = gui_plots._clip_frame(
            overlay_hydro, window
        )
        np.testing.assert_allclose(
            np.asarray(hydro_lines[0].get_ydata()),
            overlay_hydro_plot["streamflow_sim"].to_numpy(),
        )

        species_lines = _lines(species_ax)
        # Three overlay lines (half transparent), then the current
        # three.
        assert len(species_lines) == 2 * len(self.SPECIES)
        assert [_alpha(line) for line in species_lines] == [
            0.5
        ] * len(self.SPECIES) + [1.0] * len(self.SPECIES)
        overlay_rt_plot: pd.DataFrame = gui_plots._clip_frame(
            overlay_rt, window
        )
        for line, species in zip(
            species_lines[: len(self.SPECIES)], self.SPECIES
        ):
            np.testing.assert_allclose(
                np.asarray(line.get_ydata()),
                overlay_rt_plot[f"{species}_riv"].to_numpy(),
            )

    def test_empty_selection(self, input_data: gui_data.GuiInputData) -> None:
        """Nothing selected: the figure carries no axes."""
        fig = self._fig(input_data, streamflow=False, species=[])
        assert len(fig.axes) == 0


class TestObjectiveTable:
    """The objective table's rows, labels, and empty cases."""

    def _hydro(self) -> pd.Series:
        return pd.Series(
            {name: 0.9 * (i + 1) for i, name in enumerate(METRIC_NAMES)},
            dtype=float,
        )

    def _rt(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                species: [0.5 + 0.1 * i for i in range(len(METRIC_NAMES))]
                for species in ["DOC", "DIC", "Ca++"]
            },
            index=METRIC_NAMES,
        ).T

    def test_full_table(self) -> None:
        table = gui_plots.build_objective_table(self._hydro(), self._rt())
        assert list(table.index) == [
            "KGE",
            "LogKGE",
            "NSE",
            "LogNSE",
            "relative bias",
            "Spearman \u03c1",
            "R\u00b2",
        ]
        assert list(table.columns) == ["Streamflow", "DOC", "DIC", "Ca++"]
        assert table.loc["KGE", "Streamflow"] == pytest.approx(0.9)
        assert table.loc["R\u00b2", "Ca++"] == pytest.approx(1.1)

    def test_hydro_only(self) -> None:
        table = gui_plots.build_objective_table(self._hydro(), None)
        assert list(table.index) == list(METRIC_LABELS.values())
        assert list(table.columns) == ["Streamflow"]

    def test_rt_only(self) -> None:
        table = gui_plots.build_objective_table(
            pd.Series(dtype=float), self._rt()
        )
        assert list(table.index) == list(METRIC_LABELS.values())
        assert list(table.columns) == ["DOC", "DIC", "Ca++"]

    def test_empty(self) -> None:
        table = gui_plots.build_objective_table(
            pd.Series(dtype=float), None
        )
        assert table.shape == (7, 0)
        assert list(table.index) == list(METRIC_LABELS.values())



# ------------------------------------------------------------------ #
# The full model run (marked slow: takes ~5 minutes)
# ------------------------------------------------------------------ #


@pytest.mark.slow
class TestEndToEnd:
    """``arr_to_results`` at the midpoint bounds completes and matches
    the model's documented output shapes."""

    def test_midpoint_run(self, input_data: gui_data.GuiInputData) -> None:
        results = gui_data.arr_to_results(_midpoint_vector(input_data), input_data)

        hydro: pt.HydroModelResults = results.hydro
        rt = results.rt
        assert rt is not None

        assert list(hydro.simulation.index) == list(input_data.forc.precip.index)
        assert hydro.simulation.shape == (3287, 53)
        assert hydro.simulation.columns[0] == "s_snow_hs"
        assert "streamflow_sim" in hydro.simulation.columns
        assert np.isfinite(hydro.objective_functions).all()

        assert rt.simulation.shape == (3287, 90)
        assert list(rt.simulation.index) == list(input_data.forc.precip.index)

        rt_objectives: pd.DataFrame | None = rt.objective_functions
        assert rt_objectives is not None
        assert rt_objectives.shape == (3, 7)
        assert list(rt_objectives.index) == ["DOC", "Ca++", "DIC"]
        assert list(rt_objectives.columns) == METRIC_NAMES

        # Every metric is finite, including the log-transformed
        # ones: at the midpoint parameters the simulated river DOC
        # is a real concentration (~3.5e-3 M), so its log series is
        # non-constant and defined.  (Before the task-11 re-record
        # the DOC series sat at the ~1e-40 numerical floor and its
        # log KGE was NaN.)
        for column in METRIC_NAMES:
            assert np.isfinite(rt_objectives[column]).all(), column

        baseline: Any = _load_baseline()
        # The hydrology is bitwise identical to the pre-change run:
        # the generalized initialization feeds the reactive
        # transport, never the hydrology (spec goal 6).
        assert (
            hashlib.sha256(hydro.simulation.to_csv().encode()).hexdigest()
            == baseline["hydro_sim_sha256"]
        )
        # The reactive objectives are pinned twice: against the
        # in-test re-record at rel=1e-9 (the run is deterministic, so
        # this is the effective regression guard) and against the
        # on-disk fixture, which guards the file itself (the fixture
        # was re-recorded with the same values by this task).
        for species in rt_objectives.index:
            for metric in rt_objectives.columns:
                species_name: str = str(species)
                metric_name: str = str(metric)
                value: float = float(
                    rt_objectives.loc[[species_name], [metric_name]]
                    .to_numpy()[0, 0]
                )
                old_value: float = float(
                    baseline["rt_objectives"][species_name][metric_name]
                )
                new_value: float = _RECORDED_RT_OBJECTIVES[
                    species_name
                ][metric_name]
                if np.isnan(old_value):
                    assert np.isnan(new_value)
                    assert np.isnan(value)
                else:
                    assert value == pytest.approx(new_value, rel=1e-9)
                    assert value == pytest.approx(
                        old_value,
                        abs=max(0.125, 0.05 * abs(old_value)),
                    )


# ------------------------------------------------------------------ #
# potions.gui.gradients: the finite-difference gradient calculator
# ------------------------------------------------------------------ #


def _step(key: str, h_plus: float, h_minus: float) -> gui_gradients._Perturbation:
    """A placeholder step: the given effective step lengths on each side."""
    return gui_gradients._Perturbation(
        key=key,
        x_plus=np.zeros(1),
        x_minus=np.zeros(1),
        h_plus=h_plus,
        h_minus=h_minus,
    )


class TestFreeParameterSteps:
    """``free_parameter_steps``: the ± perturbation of every free parameter."""

    def test_free_parameters_only(self) -> None:
        x: np.ndarray = np.array([0.5, 2.0, 25.0])
        bounds: dict[str, tuple[float, float]] = {
            "a": (0.0, 1.0),
            "b": (2.0, 2.0),
            "c": (10.0, 40.0),
        }
        steps = gui_gradients.free_parameter_steps(x, bounds)
        assert [step.key for step in steps] == ["a", "c"]

    def test_interior_steps(self) -> None:
        x: np.ndarray = np.array([0.5, 2.0, 25.0])
        bounds: dict[str, tuple[float, float]] = {
            "a": (0.0, 1.0),
            "b": (2.0, 2.0),
            "c": (10.0, 40.0),
        }
        a, c = gui_gradients.free_parameter_steps(x, bounds)
        assert a.x_plus.tolist() == pytest.approx([0.51, 2.0, 25.0])
        assert a.x_minus.tolist() == pytest.approx([0.49, 2.0, 25.0])
        assert a.h_plus == pytest.approx(0.01)
        assert a.h_minus == pytest.approx(0.01)
        assert c.x_plus.tolist() == pytest.approx([0.5, 2.0, 25.3])
        assert c.x_minus.tolist() == pytest.approx([0.5, 2.0, 24.7])
        assert c.h_plus == pytest.approx(0.3)
        assert c.h_minus == pytest.approx(0.3)

    def test_clamped_at_lower_bound(self) -> None:
        x: np.ndarray = np.array([0.0])
        bounds: dict[str, tuple[float, float]] = {"a": (0.0, 1.0)}
        (a,) = gui_gradients.free_parameter_steps(x, bounds)
        assert np.array_equal(a.x_minus, x)
        assert a.h_minus == 0.0
        assert a.h_plus == pytest.approx(0.01)

    def test_clamped_at_upper_bound(self) -> None:
        x: np.ndarray = np.array([1.0])
        bounds: dict[str, tuple[float, float]] = {"a": (0.0, 1.0)}
        (a,) = gui_gradients.free_parameter_steps(x, bounds)
        assert np.array_equal(a.x_plus, x)
        assert a.h_plus == 0.0
        assert a.h_minus == pytest.approx(0.01)

    def test_all_pinned(self) -> None:
        x: np.ndarray = np.array([0.0, 10.0])
        bounds: dict[str, tuple[float, float]] = {
            "a": (0.0, 0.0),
            "c": (10.0, 10.0),
        }
        assert gui_gradients.free_parameter_steps(x, bounds) == []


class TestObjectiveVector:
    """``objective_vector``: the flattened objective values, with labels."""

    def test_full_mode(self, input_data: gui_data.GuiInputData) -> None:
        hydro, rt = _synthetic_results(input_data)
        results: gui_data.GuiRunResults = gui_data.GuiRunResults(hydro, rt)
        labels, values = gui_gradients.objective_vector(results)
        expected: list[str] = [
            f"{target}: {METRIC_LABELS[metric]}"
            for target in ("Streamflow", "DOC", "DIC", "Ca++")
            for metric in METRIC_NAMES
        ]
        assert labels == expected
        for i in range(len(METRIC_NAMES)):
            assert values[i] == pytest.approx(0.9 * (i + 1))
        for block, _ in enumerate(("DOC", "DIC", "Ca++")):
            for i in range(len(METRIC_NAMES)):
                index = (block + 1) * len(METRIC_NAMES) + i
                assert values[index] == pytest.approx(0.5 + 0.1 * i)

    def test_hydro_only(self, input_data: gui_data.GuiInputData) -> None:
        hydro, rt = _synthetic_results(input_data)
        results: gui_data.GuiRunResults = gui_data.GuiRunResults(hydro, None)
        labels, values = gui_gradients.objective_vector(results)
        assert len(labels) == len(METRIC_NAMES)
        assert all(label.startswith("Streamflow: ") for label in labels)
        expected: list[float] = [
            0.9 * (i + 1) for i in range(len(METRIC_NAMES))
        ]
        assert values.tolist() == pytest.approx(expected)

    def test_missing_metric_is_nan(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        hydro, rt = _synthetic_results(input_data)
        hydro.objective_functions = hydro.objective_functions.drop("r_squared")
        results: gui_data.GuiRunResults = gui_data.GuiRunResults(hydro, rt)
        labels, values = gui_gradients.objective_vector(results)
        last_metric: int = len(METRIC_NAMES) - 1
        assert labels[last_metric].endswith(": R\u00b2")
        assert np.isnan(values[last_metric])
        assert np.isfinite(values[:last_metric]).all()


def _forcing_only_input_data(tmp_path: Path) -> gui_data.GuiInputData:
    """
    A forcing-only record, loaded through the public API.

    The record carries the three forcing columns (precipitation,
    temperature, potential evapotranspiration) and a daily index, and
    nothing else: no streamflow column and no concentration columns.
    `load_input_data` therefore yields ``q_meas=None`` and an empty
    ``conc_df``, and every figure the panel builds from it must
    degrade gracefully.
    """
    index: pd.DatetimeIndex = pd.date_range("2019-01-01", periods=10)
    frame: pd.DataFrame = pd.DataFrame(
        {
            "date": index.strftime("%Y-%m-%d"),
            "precip": np.ones(len(index)),
            "temp": np.full(len(index), 5.0),
            "pet": np.full(len(index), 2.0),
        }
    )
    path: Path = tmp_path / "forcing_only.csv"
    frame.to_csv(path, index=False)
    return gui_data.load_input_data(path)


class TestNoneSafeFigures:
    """The figure builders degrade gracefully when the record
    carries no measured series: the measured artists and legend
    entries are omitted, and the remaining panels, axes, series, and
    rows are complete.

    A record without a streamflow or concentration column loads with
    ``q_meas=None`` and an empty ``conc_df`` (Task 4), and every
    figure the panel builds from it must render.
    """

    SPECIES: tuple[str, ...] = ("DOC", "DIC", "Ca++")

    def test_hydrology_forcing_only(self, tmp_path: Path) -> None:
        """A forcing-only run's hydrology figure keeps its axis, the
        simulated line, and the (sim) legend entry, and omits the
        measured scatter and its legend entry."""
        input_data: gui_data.GuiInputData = _forcing_only_input_data(tmp_path)
        assert input_data.q_meas is None
        fig: Figure = gui_plots.make_hydrology_figure(
            _synthetic_hydro_sim(input_data.forc.precip.index),
            None,
            input_data.window,
        )
        assert len(fig.axes) == 1
        ax: Axes = fig.axes[0]
        assert ax.get_title() == "Streamflow"
        assert ax.get_ylabel() == "Discharge (mm/d)"
        assert len(_scatters(ax)) == 0
        assert len(fig.legends) == 1
        legend: Legend = fig.legends[0]
        assert [text.get_text() for text in legend.get_texts()] == [
            "Streamflow (sim)"
        ]

    def test_partitioning_forcing_only(self, tmp_path: Path) -> None:
        """A forcing-only run's partitioning figure keeps the zone
        stack and its legend row, and omits the measured-discharge
        scatter and its legend row."""
        input_data: gui_data.GuiInputData = _forcing_only_input_data(tmp_path)
        assert input_data.q_meas is None
        fig: Figure = gui_plots.make_partitioning_figure(
            _synthetic_hydro_sim(input_data.forc.precip.index),
            None,
            input_data.window,
        )
        assert len(fig.axes) == 1
        ax: Axes = fig.axes[0]
        assert ax.get_title() == "Partitioning"
        assert len(_fills(ax)) > 0
        assert len(_scatters(ax)) == 0
        assert len(fig.legends) == 1
        zone_row: Legend = fig.legends[0]
        assert zone_row.get_title().get_text() == "Zone"

    def test_conc_discharge_forcing_only(self, tmp_path: Path) -> None:
        """A forcing-only run's concentration-discharge figure keeps
        every panel with its simulated cloud only, and the measured
        legend row is gone."""
        input_data: gui_data.GuiInputData = _forcing_only_input_data(tmp_path)
        assert input_data.q_meas is None
        assert input_data.conc_df.empty
        index: pd.Index = input_data.forc.precip.index
        fig: Figure = gui_plots.make_conc_discharge_figure(
            _synthetic_rt_sim(index, list(input_data.network.mineral_names)),
            _synthetic_hydro_sim(index),
            None,
            input_data.conc_df,
            list(self.SPECIES),
            input_data.window,
        )
        assert len(fig.axes) == 3
        for ax, species in zip(fig.axes, self.SPECIES):
            assert isinstance(ax, Axes)
            assert ax.get_title() == species
            # One cloud only: the simulated points.
            assert len(_scatters(ax)) == 1
        assert len(fig.legends) == 1
        sim_row: Legend = fig.legends[0]
        assert sim_row.get_title().get_text() == "Simulated"
        assert [text.get_text() for text in sim_row.get_texts()] == list(
            self.SPECIES
        )

    def test_river_concentrations_forcing_only(self, tmp_path: Path) -> None:
        """A forcing-only run's twin-axis figure, with the streamflow
        axis on, keeps both axes and every simulated series, and
        omits every measured scatter and its legend entry."""
        input_data: gui_data.GuiInputData = _forcing_only_input_data(tmp_path)
        assert input_data.q_meas is None
        index: pd.Index = input_data.forc.precip.index
        fig: Figure = gui_plots.make_river_concentrations_figure(
            _synthetic_rt_sim(index, list(input_data.network.mineral_names)),
            _synthetic_hydro_sim(index),
            None,
            input_data.conc_df,
            list(self.SPECIES),
            True,
            input_data.window,
        )
        assert len(fig.axes) == 2
        left_ax: Axes = fig.axes[0]
        right_ax: Axes = fig.axes[1]
        assert left_ax.get_title() == "Streamflow"
        # The species title sits on the right so it does not overlap
        # the streamflow title on the left.
        assert right_ax.get_title("right") == "River concentrations"
        assert len(_scatters(left_ax)) == 0
        assert len(_scatters(right_ax)) == 0
        assert len(fig.legends) == 1
        legend: Legend = fig.legends[0]
        texts: list[str] = [text.get_text() for text in legend.get_texts()]
        assert "Streamflow (sim)" in texts
        assert "Streamflow (meas)" not in texts
        assert texts.count("DOC (sim)") == 1
        assert "DOC (meas)" not in texts

    def test_conc_discharge_partial_measurement(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """A record that measures only some of the selected species
        plots the measured cloud for the species it measures, and
        names exactly those in the measured row."""
        index: pd.Index = input_data.forc.precip.index
        window = water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
        conc: pd.DataFrame = input_data.conc_df.drop(columns=["Ca++"])
        fig: Figure = gui_plots.make_conc_discharge_figure(
            _synthetic_rt_sim(
                index, list(input_data.network.mineral_names)
            ),
            _synthetic_hydro_sim(index),
            input_data.q_meas,
            conc,
            list(self.SPECIES),
            window,
        )
        assert len(fig.axes) == 3
        doc_ax, dic_ax, ca_ax = fig.axes
        # The measured species carry both clouds; the unmeasured
        # species keeps its simulated cloud only.
        assert len(_scatters(doc_ax)) == 2
        assert len(_scatters(dic_ax)) == 2
        assert len(_scatters(ca_ax)) == 1
        sim_row, meas_row = fig.legends
        assert [text.get_text() for text in sim_row.get_texts()] == list(
            self.SPECIES
        )
        assert [text.get_text() for text in meas_row.get_texts()] == [
            "DOC",
            "DIC",
        ]


class TestGradientMatrix:
    """``gradient_matrix``: the central-difference entries."""

    def test_linear(self) -> None:
        x: float = 1.0
        h: float = 0.01
        f_plus: list[np.ndarray] = [np.array([x + h])]
        f_minus: list[np.ndarray] = [np.array([x - h])]
        matrix: np.ndarray = gui_gradients.gradient_matrix(
            f_plus, f_minus, [_step("a", h, h)]
        )
        assert matrix.flatten().tolist() == pytest.approx([1.0])

    def test_quadratic(self) -> None:
        x: float = 1.0
        h: float = 0.01
        f_plus = [np.array([(x + h) ** 2])]
        f_minus = [np.array([(x - h) ** 2])]
        matrix = gui_gradients.gradient_matrix(
            f_plus, f_minus, [_step("a", h, h)]
        )
        assert matrix.flatten().tolist() == pytest.approx([2.0])

    def test_one_sided_at_bound(self) -> None:
        x: float = 5.0
        h: float = 0.01
        f_plus = [np.array([x**2])]
        f_minus = [np.array([(x - h) ** 2])]
        matrix = gui_gradients.gradient_matrix(
            f_plus, f_minus, [_step("a", 0.0, h)]
        )
        assert matrix[0, 0] == pytest.approx(10.0 - h)

    def test_two_objectives_two_parameters(self) -> None:
        # f0 = a + 2c and f1 = 3a - c at (a, c) = (1, 1), with h = 1
        # for a and h = 2 for c on each side: the exact gradients.
        f_plus = [np.array([4.0, 5.0]), np.array([7.0, 0.0])]
        f_minus = [np.array([2.0, -1.0]), np.array([-1.0, 4.0])]
        steps = [_step("a", 1.0, 1.0), _step("c", 2.0, 2.0)]
        matrix = gui_gradients.gradient_matrix(f_plus, f_minus, steps)
        assert matrix.flatten().tolist() == pytest.approx(
            [1.0, 2.0, 3.0, -1.0]
        )

    def test_nan_propagates(self) -> None:
        f_plus = [np.array([np.nan, 2.0])]
        f_minus = [np.array([1.0, 1.0])]
        matrix = gui_gradients.gradient_matrix(
            f_plus, f_minus, [_step("a", 1.0, 1.0)]
        )
        assert np.isnan(matrix[0, 0])
        # (2.0 - 1.0) / (1.0 + 1.0): the entry is the slope of the
        # difference, not of any underlying function.
        assert matrix[1, 0] == pytest.approx(0.5)

    def test_length_mismatch(self) -> None:
        f_plus = [np.array([1.0]), np.array([2.0])]
        f_minus = [np.array([0.0])]
        with pytest.raises(ValueError, match="equal lengths"):
            gui_gradients.gradient_matrix(
                f_plus, f_minus, [_step("a", 1.0, 1.0)]
            )

    def test_no_steps(self) -> None:
        matrix = gui_gradients.gradient_matrix([], [], [])
        assert matrix.shape == (0, 0)


class TestWriteGradientCsv:
    """``write_gradient_csv``: the timestamped matrix file."""

    def test_path_and_round_trip(self, tmp_path: Path) -> None:
        table: pd.DataFrame = pd.DataFrame(
            [[1.0, -2.5], [np.nan, 0.0]],
            index=["Streamflow: KGE", "DOC: R\u00b2"],
            columns=["a", "c"],
        )
        path: Path = gui_gradients.write_gradient_csv(table, tmp_path)
        assert re.fullmatch(r"gradients_\d{8}_\d{6}\.csv", path.name) is not None
        assert path.parent == tmp_path
        loaded: pd.DataFrame = pd.read_csv(path, index_col="objective")
        assert list(loaded.index) == ["Streamflow: KGE", "DOC: R\u00b2"]
        assert list(loaded.columns) == ["a", "c"]
        values: np.ndarray = loaded.to_numpy(dtype=float)
        assert values[0, 0] == pytest.approx(1.0)
        assert values[0, 1] == pytest.approx(-2.5)
        assert np.isnan(values[1, 0])
        assert values[1, 1] == pytest.approx(0.0)

    def test_default_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        table: pd.DataFrame = pd.DataFrame(
            [[0.5]], index=["Streamflow: KGE"], columns=["a"]
        )
        path = gui_gradients.write_gradient_csv(table)
        # The default results directory is relative to the working
        # directory, so the path comes back relative too.
        assert path.parent == Path("results")
        assert (tmp_path / path).is_file()


class TestGradientFigure:
    """``make_gradient_figure``: the heatmap page of the figure panel."""

    def test_heatmap(self) -> None:
        table: pd.DataFrame = pd.DataFrame(
            [[5.0, -5.0], [np.nan, 2.0]],
            index=["Streamflow: KGE", "DOC: R\u00b2"],
            columns=["a", "c"],
        )
        note: str = "Central differences; saved to results/gradients_1.csv"
        fig: Figure = gui_plots.make_gradient_figure(table, note)
        assert len(fig.axes) == 2
        ax = fig.axes[0]
        assert ax.get_title() == "Gradients of the objective functions"
        image: AxesImage = ax.images[0]
        array: object = image.get_array()
        assert isinstance(array, np.ma.MaskedArray)
        assert array.mask.tolist() == [[False, False], [True, False]]
        assert image.get_clim() == pytest.approx((-5.0, 5.0))
        colorbar_ax = fig.axes[1]
        assert colorbar_ax.get_ylabel() == "gradient"
        x_labels: list[str] = [t.get_text() for t in ax.get_xticklabels()]
        assert x_labels == ["a", "c"]
        assert ax.get_xticklabels()[0].get_rotation() == 90
        y_labels: list[str] = [t.get_text() for t in ax.get_yticklabels()]
        assert y_labels == ["Streamflow: KGE", "DOC: R\u00b2"]
        assert note in [t.get_text() for t in fig.texts]

    def test_all_nan_scale(self) -> None:
        table: pd.DataFrame = pd.DataFrame(
            [[np.nan, np.nan]],
            index=["Streamflow: KGE"],
            columns=["a", "c"],
        )
        fig = gui_plots.make_gradient_figure(table, "note")
        ax = fig.axes[0]
        image = ax.images[0]
        assert image.get_clim() == pytest.approx((-1.0, 1.0))
        array = image.get_array()
        assert isinstance(array, np.ma.MaskedArray)
        assert array.mask.tolist() == [[True, True]]


class TestLineSearchFigure:
    """The line search's profile page, single term and sum of terms."""

    def test_line_search_figure_single_term(self) -> None:
        values: list[float] = [0.2, 0.8, 1.0, 0.6, 0.4]
        profile: pd.DataFrame = pd.DataFrame(
            {
                "value": values,
                "Streamflow: KGE": values,
            },
            index=np.linspace(0.0, 10.0, 5),
        )
        fig = gui_plots.make_line_search_figure(
            profile, ["Streamflow: KGE"], "the note"
        )
        ax = fig.axes[0]
        assert ax.get_title() == "Line search — Streamflow: KGE"
        assert ax.get_ylabel() == "KGE"
        lines = _lines(ax)
        assert [line.get_label() for line in lines] == [
            "Streamflow: KGE",
            "best",
        ]
        assert lines[1].get_marker() == "o"
        assert np.asarray(lines[1].get_xdata()).tolist() == [5.0]
        assert np.asarray(lines[1].get_ydata()).tolist() == [1.0]
        assert [t.get_text() for t in fig.legends[0].get_texts()] == [
            "Streamflow: KGE",
            "best",
        ]
        assert "the note" in [t.get_text() for t in fig.texts]

    def test_line_search_figure_sum_of_terms(self) -> None:
        kge: list[float] = [0.2, 0.8, 1.0, 0.6, 0.4]
        nse: list[float] = [0.1, 0.5, 0.9, 0.3, 0.2]
        profile: pd.DataFrame = pd.DataFrame(
            {
                "value": [a + b for a, b in zip(kge, nse)],
                "Streamflow: KGE": kge,
                "Streamflow: NSE": nse,
            },
            index=np.linspace(0.0, 10.0, 5),
        )
        fig = gui_plots.make_line_search_figure(
            profile, ["Streamflow: KGE", "Streamflow: NSE"], "the note"
        )
        ax = fig.axes[0]
        assert ax.get_title() == "Line search — sum of 2 objectives"
        assert ax.get_ylabel() == "combined value"
        lines = _lines(ax)
        assert [line.get_label() for line in lines] == [
            "Streamflow: KGE",
            "Streamflow: NSE",
            "sum",
            "best",
        ]
        assert lines[2].get_linewidth() == 2
        # The sum is the test's own per-row addition, so compare
        # at the precision the addition carries, not bitwise.
        assert np.asarray(lines[2].get_ydata()).tolist() == pytest.approx(
            [0.3, 1.3, 1.9, 0.9, 0.6]
        )
        assert np.asarray(lines[3].get_xdata()).tolist() == [5.0]
        assert np.asarray(lines[3].get_ydata()).tolist() == [1.9]
        assert [t.get_text() for t in fig.legends[0].get_texts()] == [
            "Streamflow: KGE",
            "Streamflow: NSE",
            "sum",
            "best",
        ]
        assert "the note" in [t.get_text() for t in fig.texts]


# ------------------------------------------------------------------ #
# model_pool
# ------------------------------------------------------------------ #


def _marker_results(
    input_data: gui_data.GuiInputData,
    x: np.ndarray,
    hydro_only: bool = False,
) -> gui_data.GuiRunResults:
    """
    Synthetic results whose first hydro objective encodes the point.

    The marker value — the sum of the point's elements, offset by
    1000 in hydro mode so the two modes are distinguishable — is
    placed in the first metric of the hydro objective series, so a
    test can tell which point produced which results object.
    """
    index: pd.Index = input_data.forc.precip.index
    marker: float = float(np.asarray(x).sum()) + (1000.0 if hydro_only else 0.0)
    values: list[float] = [marker] + [0.0] * (len(METRIC_NAMES) - 1)
    hydro: pt.HydroModelResults = pt.HydroModelResults(
        forcing=input_data.forc,
        simulation=_synthetic_hydro_sim(index),
        objective_functions=pd.Series(values, index=METRIC_NAMES, dtype=float),
        zone_names=list(ZONE_ORDER),
    )
    if hydro_only:
        return gui_data.GuiRunResults(hydro, None)
    rt: pt.RtModelResults = pt.RtModelResults(
        simulation=_synthetic_rt_sim(
            index, list(input_data.network.mineral_names)
        ),
        objective_functions=pd.DataFrame(
            {
                species: [0.5 + 0.1 * i for i in range(len(METRIC_NAMES))]
                for species in ["DOC", "DIC", "Ca++"]
            },
            index=METRIC_NAMES,
        ).T,
        saved_dir=None,
        zone_names=list(ZONE_ORDER),
    )
    return gui_data.GuiRunResults(hydro, rt)


class _PoolRecorder:
    """
    A stand-in for ``multiprocessing.Pool`` for the pool tests.

    It records each construction and, like the real pool, runs the
    initializer once per requested worker before any task runs.
    Tasks are executed in this process in reverse input order
    (one of the legitimate out-of-order completions of
    ``imap_unordered``) so an implementation that collected results
    in arrival order would come out in the wrong order.
    """

    instances: ClassVar["list[_PoolRecorder]"] = []

    def __init__(
        self,
        processes: int | None = None,
        initializer: Callable[..., object] | None = None,
        initargs: Iterable[Any] = (),
    ) -> None:
        self.processes: int = processes if processes is not None else 1
        self.initializer: Callable[..., object] | None = initializer
        self.initargs: Iterable[Any] = initargs
        self.closed: bool = False
        self.joined: bool = False
        _PoolRecorder.instances.append(self)
        for _worker in range(self.processes):
            if initializer is not None:
                initializer(*initargs)

    def imap_unordered(
        self,
        func: Callable[[Any], Any],
        iterable: Iterable[Any],
        chunksize: int = 1,
    ) -> list[Any]:
        return list(reversed([func(task) for task in iterable]))

    def close(self) -> None:
        self.closed = True

    def join(self) -> None:
        self.joined = True


@pytest.mark.skipif(
    multiprocessing.get_start_method() != "fork",
    reason="pool tests require the fork start method",
)
class TestModelPool:
    """``model_pool.map_model_runs``: the shared worker pool for the jobs."""

    @pytest.fixture(autouse=True)
    def clear_pool_record(self) -> None:
        _PoolRecorder.instances.clear()

    def test_map_model_runs_returns_results_in_input_order(
        self,
        input_data: gui_data.GuiInputData,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        points: list[np.ndarray] = [
            np.array([1.0, 2.0]),
            np.array([-1.0, 0.5]),
            np.array([0.0, 4.0]),
        ]

        def fake_load(
            csv_path: Path,
            model_key: str = model_specs.DEFAULT_MODEL_KEY,
            network: pt.ReactionNetwork | None = None,
        ) -> gui_data.GuiInputData:
            return input_data

        def fake_run(
            point: np.ndarray, data: gui_data.GuiInputData
        ) -> gui_data.GuiRunResults:
            return _marker_results(data, point)

        monkeypatch.setattr(gui_data, "load_input_data", fake_load)
        monkeypatch.setattr(gui_data, "arr_to_results", fake_run)
        monkeypatch.setattr(model_pool, "Pool", _PoolRecorder)

        results = model_pool.map_model_runs(points, MODE_FULL, "unused.csv")

        assert len(results) == 3
        markers: list[float] = [
            float(result.hydro.objective_functions.iloc[0])
            for result in results
            if not isinstance(result, str)
        ]
        assert len(markers) == 3
        # The points' sums are 3.0, -0.5, and 4.0: the results must
        # come back in input order, not in completion order.
        assert markers == [3.0, -0.5, 4.0]
        assert _PoolRecorder.instances[0].closed
        assert _PoolRecorder.instances[0].joined

    def test_map_model_runs_reports_worker_exceptions_as_strings(
        self,
        input_data: gui_data.GuiInputData,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        points: list[np.ndarray] = [
            np.array([1.0, 2.0]),
            np.array([-1.0, 0.5]),
            np.array([0.0, 4.0]),
        ]

        def fake_load(
            csv_path: Path,
            model_key: str = model_specs.DEFAULT_MODEL_KEY,
            network: pt.ReactionNetwork | None = None,
        ) -> gui_data.GuiInputData:
            return input_data

        def failing_run(
            point: np.ndarray, data: gui_data.GuiInputData
        ) -> gui_data.GuiRunResults:
            if point[0] < 0:
                raise ValueError("boom")
            return _marker_results(data, point)

        monkeypatch.setattr(gui_data, "load_input_data", fake_load)
        monkeypatch.setattr(gui_data, "arr_to_results", failing_run)
        monkeypatch.setattr(model_pool, "Pool", _PoolRecorder)

        results = model_pool.map_model_runs(points, MODE_FULL, "unused.csv")

        assert results[1] == "ValueError: boom"
        assert not isinstance(results[0], str)
        assert not isinstance(results[2], str)

    def test_map_model_runs_empty_points_returns_empty_without_pool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(model_pool, "Pool", _PoolRecorder)

        assert model_pool.map_model_runs([], MODE_FULL, "unused.csv") == []
        assert _PoolRecorder.instances == []

    def test_map_model_runs_initializer_failure_fails_cleanly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        points: list[np.ndarray] = [
            np.array([1.0]),
            np.array([2.0]),
            np.array([3.0]),
        ]

        def failing_load(
            csv_path: Path,
            model_key: str = model_specs.DEFAULT_MODEL_KEY,
            network: pt.ReactionNetwork | None = None,
        ) -> gui_data.GuiInputData:
            raise RuntimeError("missing csv")

        monkeypatch.setattr(gui_data, "load_input_data", failing_load)
        monkeypatch.setattr(model_pool, "Pool", _PoolRecorder)

        results = model_pool.map_model_runs(points, MODE_FULL, "unused.csv")

        assert len(results) == 3
        for result in results:
            assert isinstance(result, str)
            assert "missing csv" in result
        assert _PoolRecorder.instances[0].closed
        assert _PoolRecorder.instances[0].joined

    def test_map_model_runs_initializer_runs_per_worker(
        self,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        log_file: Path = tmp_path / "worker_init.log"
        points: list[np.ndarray] = [np.array([1.0]), np.array([2.0])]

        def logging_load(
            csv_path: Path,
            model_key: str = model_specs.DEFAULT_MODEL_KEY,
            network: pt.ReactionNetwork | None = None,
        ) -> gui_data.GuiInputData:
            with log_file.open("a", encoding="utf-8") as handle:
                handle.write(f"{csv_path}\n")
            return input_data

        def fake_run(
            point: np.ndarray, data: gui_data.GuiInputData
        ) -> gui_data.GuiRunResults:
            return _marker_results(data, point)

        monkeypatch.setattr(gui_data, "load_input_data", logging_load)
        monkeypatch.setattr(gui_data, "arr_to_results", fake_run)
        monkeypatch.setattr(model_pool, "Pool", _PoolRecorder)

        results = model_pool.map_model_runs(
            points, MODE_FULL, str(log_file), workers=2
        )

        assert len(results) == 2
        # Two workers means exactly two initializer calls, each
        # loading the CSV once.
        assert log_file.read_text(encoding="utf-8").splitlines() == [
            str(log_file),
            str(log_file),
        ]

    def test_map_model_runs_hydro_mode_dispatches_to_hydro_results(
        self,
        input_data: gui_data.GuiInputData,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        points: list[np.ndarray] = [np.array([1.0, 2.0])]

        def fake_load(
            csv_path: Path,
            model_key: str = model_specs.DEFAULT_MODEL_KEY,
            network: pt.ReactionNetwork | None = None,
        ) -> gui_data.GuiInputData:
            return input_data

        def full_run(
            point: np.ndarray, data: gui_data.GuiInputData
        ) -> gui_data.GuiRunResults:
            raise AssertionError(
                "full-mode conversion should not run in hydro mode"
            )

        def hydro_run(
            point: np.ndarray, data: gui_data.GuiInputData
        ) -> gui_data.GuiRunResults:
            return _marker_results(data, point, hydro_only=True)

        monkeypatch.setattr(gui_data, "load_input_data", fake_load)
        monkeypatch.setattr(gui_data, "arr_to_results", full_run)
        monkeypatch.setattr(gui_data, "arr_to_hydro_results", hydro_run)
        monkeypatch.setattr(model_pool, "Pool", _PoolRecorder)

        results = model_pool.map_model_runs(points, MODE_HYDRO, "unused.csv")

        assert len(results) == 1
        first = results[0]
        assert not isinstance(first, str)
        assert first.rt is None
        assert float(first.hydro.objective_functions.iloc[0]) == 1003.0

    def test_map_model_runs_forwards_model_key(
        self,
        input_data: gui_data.GuiInputData,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        seen: list[str] = []

        def fake_load(
            csv_path: Path,
            model_key: str = model_specs.DEFAULT_MODEL_KEY,
            network: pt.ReactionNetwork | None = None,
        ) -> gui_data.GuiInputData:
            seen.append(model_key)
            return input_data

        def fake_run(
            point: np.ndarray, data: gui_data.GuiInputData
        ) -> gui_data.GuiRunResults:
            return _marker_results(data, point)

        monkeypatch.setattr(gui_data, "load_input_data", fake_load)
        monkeypatch.setattr(gui_data, "arr_to_hydro_results", fake_run)
        monkeypatch.setattr(model_pool, "Pool", _PoolRecorder)

        model_pool.map_model_runs(
            [np.array([1.0])],
            MODE_HYDRO,
            "unused.csv",
            workers=1,
            model_key="hbv",
        )

        assert seen == ["hbv"]

    def test_map_model_runs_forwards_network(
        self,
        input_data: gui_data.GuiInputData,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        seen: list[pt.ReactionNetwork | None] = []

        def fake_load(
            csv_path: Path,
            model_key: str = model_specs.DEFAULT_MODEL_KEY,
            network: pt.ReactionNetwork | None = None,
        ) -> gui_data.GuiInputData:
            seen.append(network)
            return input_data

        def fake_run(
            point: np.ndarray, data: gui_data.GuiInputData
        ) -> gui_data.GuiRunResults:
            return _marker_results(data, point, hydro_only=True)

        monkeypatch.setattr(gui_data, "load_input_data", fake_load)
        monkeypatch.setattr(gui_data, "arr_to_hydro_results", fake_run)
        monkeypatch.setattr(model_pool, "Pool", _PoolRecorder)

        network = network_specs.get_network_spec("carbon").builder()
        model_pool.map_model_runs(
            [np.array([1.0])],
            MODE_HYDRO,
            "unused.csv",
            workers=1,
            network=network,
        )

        # `ReactionNetwork` defines no `__eq__`, so list equality is
        # identity: the exact object the caller passed arrived at the
        # worker's initializer.
        assert seen == [network]

    def test_map_model_runs_defaults_to_lateral_model(
        self,
        input_data: gui_data.GuiInputData,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        seen: list[str] = []

        def fake_load(
            csv_path: Path,
            model_key: str = model_specs.DEFAULT_MODEL_KEY,
            network: pt.ReactionNetwork | None = None,
        ) -> gui_data.GuiInputData:
            seen.append(model_key)
            return input_data

        def fake_run(
            point: np.ndarray, data: gui_data.GuiInputData
        ) -> gui_data.GuiRunResults:
            return _marker_results(data, point)

        monkeypatch.setattr(gui_data, "load_input_data", fake_load)
        monkeypatch.setattr(gui_data, "arr_to_hydro_results", fake_run)
        monkeypatch.setattr(model_pool, "Pool", _PoolRecorder)

        model_pool.map_model_runs(
            [np.array([1.0])], MODE_HYDRO, "unused.csv", workers=1
        )

        assert seen == [model_specs.DEFAULT_MODEL_KEY]


class TestModelSpecs:
    """The curated model registry and its derived zone metadata."""

    def test_registry_has_the_four_models_in_order(self) -> None:
        assert list(model_specs.MODEL_SPECS) == [
            "lateral_three_layer",
            "three_layer",
            "hbv",
            "hbv_lateral",
        ]
        assert model_specs.DEFAULT_MODEL_KEY == "lateral_three_layer"

    def test_display_names(self) -> None:
        assert {
            key: spec.display_name
            for key, spec in model_specs.MODEL_SPECS.items()
        } == {
            "lateral_three_layer": "Lateral three-layer",
            "three_layer": "Three-layer",
            "hbv": "HBV",
            "hbv_lateral": "HBV lateral",
        }

    @pytest.mark.parametrize(
        ("key", "zones"),
        [
            (
                "lateral_three_layer",
                [
                    "snow_hs",
                    "snow_rp",
                    "surface_hs",
                    "surface_rp",
                    "ground_hs",
                    "ground_rp",
                ],
            ),
            ("three_layer", ["snow", "surface", "ground"]),
            ("hbv", ["snow", "surface", "shallow", "deep"]),
            (
                "hbv_lateral",
                [
                    "snow_hs",
                    "snow_rp",
                    "surface_hs",
                    "surface_rp",
                    "shallow_hs",
                    "shallow_rp",
                    "deep_hs",
                    "deep_rp",
                ],
            ),
        ],
    )
    def test_zones_follow_model_order(
        self, key: str, zones: list[str]
    ) -> None:
        spec = model_specs.get_model_spec(key)
        assert spec.zones == zones
        assert list(spec.zone_configs) == zones
        assert set(spec.zone_labels) == set(zones)
        assert set(spec.zone_colors) == set(zones)

    def test_chemistry_configs(self) -> None:
        for spec in model_specs.MODEL_SPECS.values():
            for zone in spec.zones:
                cfg = spec.zone_configs[zone]
                if zone in spec.snow_zones:
                    assert (cfg.do_reactions, cfg.do_speciation) == (
                        False,
                        False,
                    )
                else:
                    assert (cfg.do_reactions, cfg.do_speciation) == (
                        True,
                        True,
                    )

    def test_snow_reactive_and_speciation_zones(self) -> None:
        hbv = model_specs.get_model_spec("hbv")
        assert hbv.snow_zones == ["snow"]
        assert hbv.reactive_zones == ["surface", "shallow", "deep"]
        assert hbv.speciation_zones == ["surface", "shallow", "deep"]

    def test_stack_order_is_deepest_first(self) -> None:
        assert model_specs.get_model_spec(
            "lateral_three_layer"
        ).stack_order == [
            "ground_hs",
            "ground_rp",
            "surface_hs",
            "surface_rp",
            "snow_hs",
            "snow_rp",
        ]
        assert model_specs.get_model_spec("hbv").stack_order == [
            "deep",
            "shallow",
            "surface",
            "snow",
        ]

    def test_zone_layer_index(self) -> None:
        spec = model_specs.get_model_spec("hbv")
        assert spec.zone_layer_index == {
            "snow": 0,
            "surface": 1,
            "shallow": 2,
            "deep": 3,
        }

    def test_unknown_key_raises(self) -> None:
        with pytest.raises(ValueError, match="lateral_three_layer"):
            model_specs.get_model_spec("nope")

    def test_class_name_maps_to_key(self) -> None:
        assert (
            model_specs.model_key_for_class_name(
                "potions.common_models.LateralThreeLayerModel"
            )
            == "lateral_three_layer"
        )
        assert (
            model_specs.model_key_for_class_name(
                "potions.common_models.HbvLateralModel"
            )
            == "hbv_lateral"
        )
        assert model_specs.model_key_for_class_name("UnknownModel") is None


# ------------------------------------------------------------------ #
# Model-specific zone metadata on the plot builders
# ------------------------------------------------------------------ #


def test_storage_figure_accepts_zone_metadata() -> None:
    """The storage figure uses the given zone labels and colors."""
    index = pd.date_range("2020-01-01", periods=3, freq="D")
    sim = pd.DataFrame(
        {"s_a": [1.0, 2.0, 3.0], "s_b": [3.0, 2.0, 1.0]}, index=index
    )
    fig = gui_plots.make_storage_figure(
        sim,
        ["a", "b"],
        (index[0], index[-1]),
        zone_labels={"a": "Alpha", "b": "Beta"},
        zone_colors={"a": "#111111", "b": "#222222"},
    )
    ax = fig.axes[0]
    assert [line.get_color() for line in _lines(ax)] == [
        "#111111",
        "#222222",
    ]
    legend: Legend = fig.legends[0]
    assert [text.get_text() for text in legend.get_texts()] == [
        "Alpha",
        "Beta",
    ]


def test_zone_concentrations_figure_accepts_zone_metadata() -> None:
    """The zone-concentration figure uses the given labels and colors."""
    index = pd.date_range("2020-01-01", periods=3, freq="D")
    sim = pd.DataFrame(
        {
            "DOC_a": [1.0, 2.0, 3.0],
            "DOC_b": [3.0, 2.0, 1.0],
        },
        index=index,
    )
    fig = gui_plots.make_zone_concentrations_figure(
        sim,
        ["DOC"],
        ["a", "b"],
        (index[0], index[-1]),
        zone_labels={"a": "Alpha", "b": "Beta"},
        zone_colors={"a": "#111111", "b": "#222222"},
    )
    ax = fig.axes[0]
    assert [line.get_color() for line in _lines(ax)] == [
        "#111111",
        "#222222",
    ]
    zone_row: Legend = fig.legends[0]
    assert [text.get_text() for text in zone_row.get_texts()] == [
        "Alpha",
        "Beta",
    ]


def test_rate_figure_accepts_zone_order_and_metadata() -> None:
    """The rate figure iterates the zones in the given order."""
    index = pd.date_range("2020-01-01", periods=3, freq="D")
    sim = pd.DataFrame(
        {
            "SOC(s)_rate_a": [1.0, 2.0, 3.0],
            "SOC(s)_rate_b": [3.0, 2.0, 1.0],
        },
        index=index,
    )
    fig = gui_plots.make_rate_figure(
        sim,
        "SOC(s)",
        (index[0], index[-1]),
        zone_labels={"a": "Alpha", "b": "Beta"},
        zone_colors={"a": "#111111", "b": "#222222"},
        zone_order=["b", "a"],
    )
    ax = fig.axes[0]
    assert [line.get_color() for line in _lines(ax)] == [
        "#222222",
        "#111111",
    ]
    zone_row: Legend = fig.legends[0]
    assert [text.get_text() for text in zone_row.get_texts()] == [
        "Beta",
        "Alpha",
    ]


def test_partitioning_figure_accepts_stack_order() -> None:
    """The partitioning figure stacks the zones in the given order."""
    index = pd.date_range("2020-01-01", periods=3, freq="D")
    sim = pd.DataFrame(
        {
            "streamflow_sim": [1.0, 2.0, 3.0],
            "prop_q_a": [0.2, 0.2, 0.2],
            "prop_q_b": [0.8, 0.8, 0.8],
        },
        index=index,
    )
    q_meas = pd.Series([np.nan] * 3, index=index)
    fig = gui_plots.make_partitioning_figure(
        sim,
        q_meas,
        (index[0], index[-1]),
        zone_labels={"a": "Alpha", "b": "Beta"},
        zone_colors={"a": "#111111", "b": "#222222"},
        stack_order=["b", "a"],
    )
    zone_row: Legend = fig.legends[0]
    assert [text.get_text() for text in zone_row.get_texts()] == [
        "Beta",
        "Alpha",
    ]


# ------------------------------------------------------------------ #
# The YAML network file (potions.core NetworkFile)
# ------------------------------------------------------------------ #


class TestNetworkFile:
    """The network file format: fingerprints, round-trip, section
    order, and the file path's error and validation contract."""

    def test_fingerprint_stable(self) -> None:
        """The same network always serializes to the same bytes; a
        different network to different bytes."""

        def fingerprint(network: pt.ReactionNetwork) -> str:
            return hashlib.sha256(
                pt.NetworkFile.from_network(network).to_bytes()
            ).hexdigest()

        soc_first: str = fingerprint(get_soc_calcite_network())
        soc_second: str = fingerprint(get_soc_calcite_network())
        carbon: str = fingerprint(get_carbon_network())
        assert soc_first == soc_second
        assert carbon != soc_first

    def test_to_file_from_file_round_trip(
        self,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
    ) -> None:
        """The file view of the default network written to disk and
        read back yields the same species and identical bytes."""
        network: pt.ReactionNetwork = input_data.network
        path: Path = tmp_path / "network.yaml"
        pt.NetworkFile.from_network(network).to_file(str(path))

        rebuilt: pt.ReactionNetwork = pt.ReactionNetwork.from_file(str(path))
        # The file's BTreeMap sections rebuild the species in key
        # order, not the constructor's, so compare the set, not the
        # order (order is canonicalized by the file format).
        assert set(rebuilt.species_names) == set(network.species_names)
        assert (
            pt.NetworkFile.from_network(rebuilt).to_bytes()
            == pt.NetworkFile.from_network(network).to_bytes()
        )

    def test_section_order_in_bytes(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """The six section headers appear in canonical order."""
        text: str = (
            pt.NetworkFile.from_network(input_data.network).to_bytes()
        ).decode("utf-8")
        sections: tuple[str, ...] = (
            "primary_species:",
            "secondary_species:",
            "mineral_species:",
            "exchange_reactions:",
            "tst_reactions:",
            "monod_reactions:",
        )
        positions: list[int] = [text.find(section) for section in sections]
        for section, position in zip(sections, positions):
            assert position >= 0, f"section header {section!r} missing"
        assert positions == sorted(positions)

    def test_from_file_error_paths(self, tmp_path: Path) -> None:
        """The file path's error contract: a missing file, invalid
        YAML, and a V-gate violation."""
        missing: Path = tmp_path / "missing.yaml"
        with pytest.raises(FileNotFoundError, match="does not exist"):
            pt.NetworkFile.from_file(str(missing))

        garbage: Path = tmp_path / "garbage.yaml"
        garbage.write_text("a: [1, 2\n")
        with pytest.raises(ValueError, match="Invalid network YAML"):
            pt.NetworkFile.from_file(str(garbage))

        bad: Path = tmp_path / "bad.yaml"
        bad.write_text(
            "primary_species:\n"
            "  DOC:\n"
            "    name: DOC\n"
            "    molar_mass: 300.0\n"
            "    charge: 0.0\n"
            "    dh_size_param: 4.0\n"
            "secondary_species:\n"
            "  S:\n"
            "    name: S\n"
            "    stoichiometry:\n"
            "      S: -1.0\n"
            "      Unknown: 0.5\n"
            "    eq_consts:\n"
            "    - 1.0\n"
            "    - 2.0\n"
            "    dh_size_param: 3.0\n"
            "    charge: 0.0\n"
            "    molar_mass: 100.0\n"
            "mineral_species: {}\n"
            "exchange_reactions: {}\n"
            "tst_reactions: {}\n"
            "monod_reactions: {}\n"
        )
        with pytest.raises(ValueError, match="V3:"):
            pt.NetworkFile.from_file(str(bad))

    def test_validate_issue_repr(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """The default network validates clean; a finding is a
        ``NetworkValidationIssue`` whose repr carries both fields."""
        assert input_data.network.validate() == []

        issues: list[pt.NetworkValidationIssue] = pt.NetworkFile.from_network(
            get_chloride_network()
        ).validate()
        assert [issue.code for issue in issues] == ["W1"]
        issue: pt.NetworkValidationIssue = issues[0]
        assert isinstance(issue.code, str)
        assert isinstance(issue.message, str)
        assert "W1" in repr(issue)
        assert issue.message in repr(issue)


# ------------------------------------------------------------------ #
# Network-driven data binding (measured columns, spin-up, primary
# initialization)
# ------------------------------------------------------------------ #


class TestNetworkDataBinding:
    """The GUI data binding derives the measured series and the
    primary-species initialization from the run's network."""

    def test_measured_columns_default(self) -> None:
        """The default network's measured columns: its aqueous species
        present in the frame, in species order, then DIC."""
        df: pd.DataFrame = pd.DataFrame(
            {"DOC": [1.0], "Ca++": [2.0], "DIC": [3.0]}
        )
        assert (
            gui_data._measured_columns(get_soc_calcite_network(), df)
            == ["DOC", "Ca++", "DIC"]
        )

    def test_measured_columns_dic_requires_trio(self) -> None:
        """DIC is appended only when the network carries the whole
        carbonate trio: the chloride network leaves it out."""
        df: pd.DataFrame = pd.DataFrame({"Cl-": [1.0], "DIC": [2.0]})
        assert gui_data._measured_columns(get_chloride_network(), df) == ["Cl-"]

    def test_spinup_nans_only_measured(self) -> None:
        """The spin-up NaNs ``q_meas`` and the measured columns only;
        other columns keep their copied values."""
        n_days: int = gui_data.NUM_SPINUP_DAYS + 1
        df: pd.DataFrame = pd.DataFrame(
            {
                "q_meas": np.arange(n_days, dtype=float),
                "DOC": np.full(n_days, 2.0),
                "X": np.full(n_days, 3.0),
            },
            index=pd.date_range("2016-10-01", periods=n_days, freq="D"),
        )
        result: pd.DataFrame = gui_data._with_spinup(df, ["DOC"])
        spinup_rows: pd.DataFrame = result.iloc[: gui_data.NUM_SPINUP_DAYS]
        assert spinup_rows["q_meas"].isna().all()
        assert spinup_rows["DOC"].isna().all()
        assert spinup_rows["X"].notna().all()
        assert result.iloc[gui_data.NUM_SPINUP_DAYS :].equals(df)

    def test_primary_initial_concentrations(self) -> None:
        """A primary species with a measured column starts at that
        column's mean; the others take the scalar fallback."""
        conc_df: pd.DataFrame = pd.DataFrame(
            {"DOC": [1.0, 3.0], "Ca++": [0.5, 1.5]}
        )
        inits: dict[str, float] = gui_data._primary_initial_concentrations(
            get_soc_calcite_network(), conc_df, 1e-3
        )
        assert inits["DOC"] == 2.0
        assert inits["Ca++"] == 1.0
        assert inits["HCO3-"] == 1e-3
        assert inits["H+"] == 1e-3
        assert set(inits) == set(get_soc_calcite_network().primary_names)

    def test_load_input_data_network_param(self) -> None:
        """`load_input_data` binds the given network's measured
        columns; the default call keeps the default network."""
        carbon: pt.ReactionNetwork = get_carbon_network()
        result: gui_data.GuiInputData = gui_data.load_input_data(
            gui_data.DEFAULT_CSV_PATH, network=carbon
        )
        assert list(result.network.species_names) == list(carbon.species_names)
        assert list(result.conc_df.columns) == ["DOC", "DIC"]

        default: gui_data.GuiInputData = gui_data.load_input_data(
            gui_data.DEFAULT_CSV_PATH
        )
        assert list(default.conc_df.columns) == ["DOC", "Ca++", "DIC"]


# ------------------------------------------------------------------ #
# Generalized exchange initialization
# ------------------------------------------------------------------ #


class TestExchangeInit:
    """The exchange-site initialization solves any set of exchange
    reactions against the fixed aqueous initial concentrations,
    staying identity-preserving for the 1:1 XDOC network."""

    def test_identity_one_to_one(self) -> None:
        """One X- + DOC <-> XDOC reaction: the solved root equals the
        old closed form k * tot / (k + doc)."""
        reaction: pt.core.ExchangeReaction = pt.core.ExchangeReaction(
            name="XDOC",
            stoichiometry={"X-": 1.0, "DOC": 1.0, "XDOC": -1.0},
            log10_k_eq=-2.75,
            charge=0.0,
        )
        k: float = 10.0 ** -2.75
        tot: float = 1.0
        result: dict[str, float] = gui_data._solve_exchange_initial_state(
            tot, [reaction], {"DOC": 1.12e-3}
        )
        assert result["X-"] == pytest.approx(k * tot / (k + 1.12e-3), rel=1e-9)

    def test_fallback_all_free(self) -> None:
        """A product that does not consume X- and whose constant
        adsorption already reaches tot leaves every site free."""
        reaction: pt.core.ExchangeReaction = pt.core.ExchangeReaction(
            name="P",
            stoichiometry={"DOC": 1.0, "P": -1.0},
            log10_k_eq=0.0,
            charge=0.0,
        )
        tot: float = 1.0
        result: dict[str, float] = gui_data._solve_exchange_initial_state(
            tot, [reaction], {"DOC": 1.0}
        )
        assert result["X-"] == tot

    def test_divalent_x(self) -> None:
        """A 2 X- + S <-> P reaction: the solved state satisfies the
        mass balance, and the product matches the direct formula."""
        reaction: pt.core.ExchangeReaction = pt.core.ExchangeReaction(
            name="P",
            stoichiometry={"X-": 2.0, "S": 1.0, "P": -1.0},
            log10_k_eq=0.0,
            charge=0.0,
        )
        s: float = 1e-3
        tot: float = 0.1
        result: dict[str, float] = gui_data._solve_exchange_initial_state(
            tot, [reaction], {"S": s}
        )
        x: float = result["X-"]
        expected: float = 10.0 ** (
            -0.0 + 2.0 * math.log10(x) + math.log10(s)
        )
        assert x + result["P"] == pytest.approx(tot, rel=1e-9)
        assert result["P"] == pytest.approx(expected, rel=1e-9)

    def test_missing_reactant_zero(self) -> None:
        """A product with a reactant at zero initial concentration has
        exactly zero concentration (no log-of-zero)."""
        reaction: pt.core.ExchangeReaction = pt.core.ExchangeReaction(
            name="XDOC",
            stoichiometry={"X-": 1.0, "DOC": 1.0, "XDOC": -1.0},
            log10_k_eq=-2.75,
            charge=0.0,
        )
        product: gui_data._ExchangeProduct = gui_data._exchange_product_terms(
            reaction
        )
        assert (
            gui_data._exchange_product_concentration(0.5, product, {"DOC": 0.0})
            == 0.0
        )

    def test_multi_product(self) -> None:
        """Two products sharing the X- pool: the solved state satisfies
        the full mass balance and both direct formulas."""
        reactions: list[pt.core.ExchangeReaction] = [
            pt.core.ExchangeReaction(
                name="E1",
                stoichiometry={"X-": 1.0, "A": 1.0, "E1": -1.0},
                log10_k_eq=-2.0,
                charge=0.0,
            ),
            pt.core.ExchangeReaction(
                name="E2",
                stoichiometry={"X-": 1.0, "B": 1.0, "E2": -1.0},
                log10_k_eq=-1.0,
                charge=0.0,
            ),
        ]
        species_init: dict[str, float] = {"A": 1e-3, "B": 2e-3}
        tot: float = 1.0
        result: dict[str, float] = gui_data._solve_exchange_initial_state(
            tot, reactions, species_init
        )
        x: float = result["X-"]
        e1: float = x * 1e-3 / 10.0 ** -2.0
        e2: float = x * 2e-3 / 10.0 ** -1.0
        assert x + result["E1"] + result["E2"] == pytest.approx(tot, rel=1e-9)
        assert result["E1"] == pytest.approx(e1, rel=1e-9)
        assert result["E2"] == pytest.approx(e2, rel=1e-9)

    def test_zero_total(self) -> None:
        """No exchange sites: the cation and every product are exactly
        zero."""
        reaction: pt.core.ExchangeReaction = pt.core.ExchangeReaction(
            name="XDOC",
            stoichiometry={"X-": 1.0, "DOC": 1.0, "XDOC": -1.0},
            log10_k_eq=-2.75,
            charge=0.0,
        )
        result: dict[str, float] = gui_data._solve_exchange_initial_state(
            0.0, [reaction], {"DOC": 1.12e-3}
        )
        assert result == {"X-": 0.0, "XDOC": 0.0}


class TestNetworkSpecs:
    """The curated reaction-network registry."""

    def test_registry_has_the_five_networks_in_order(self) -> None:
        assert list(network_specs.NETWORK_SPECS) == [
            "soc_calcite",
            "carbon",
            "simple_carbon",
            "simple_carbon_adsorption",
            "chloride",
        ]
        assert network_specs.DEFAULT_NETWORK_KEY == "soc_calcite"
        assert (
            network_specs.DEFAULT_NETWORK_SPEC
            is network_specs.NETWORK_SPECS[network_specs.DEFAULT_NETWORK_KEY]
        )

    def test_display_names(self) -> None:
        assert {
            key: spec.display_name
            for key, spec in network_specs.NETWORK_SPECS.items()
        } == {
            "soc_calcite": "East River SOC / DOC / calcite",
            "carbon": "Carbon (SOC, no calcite)",
            "simple_carbon": "Simple carbon (DOC + SOC)",
            "simple_carbon_adsorption": "Simple carbon with adsorption",
            "chloride": "Chloride (no speciation)",
        }

    def test_builders_are_the_common_network_factories(self) -> None:
        from potions.reactive_transport.common_networks import (
            get_carbon_network,
            get_chloride_network,
            get_simple_carbon_adsorption_network,
            get_simple_carbon_network,
            get_soc_calcite_network,
        )

        builders = {
            key: spec.builder
            for key, spec in network_specs.NETWORK_SPECS.items()
        }
        assert builders == {
            "soc_calcite": get_soc_calcite_network,
            "carbon": get_carbon_network,
            "simple_carbon": get_simple_carbon_network,
            "simple_carbon_adsorption": get_simple_carbon_adsorption_network,
            "chloride": get_chloride_network,
        }

    def test_get_network_spec_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown network key"):
            network_specs.get_network_spec("nope")


class TestNetworkFingerprint:
    """The stable, order-independent network fingerprint."""

    def test_stable_across_fresh_builds(self) -> None:
        spec = network_specs.get_network_spec(
            network_specs.DEFAULT_NETWORK_KEY
        )
        assert gui_data.fingerprint_network(
            spec.builder()
        ) == gui_data.fingerprint_network(spec.builder())

    def test_order_insensitive(self) -> None:
        network = network_specs.get_network_spec(
            network_specs.DEFAULT_NETWORK_KEY
        ).builder()
        fingerprint = gui_data.fingerprint_network(network)
        # The canonical YAML sorts every map key, so reordering the
        # primary species does not change the fingerprint.
        reordered = pt.ReactionNetwork(
            primary_aqueous=list(reversed(network.primary_aqueous)),
            mineral=network.mineral,
            secondary=network.secondary,
            mineral_kinetics=network.mineral_kinetics,
            exchange_species=network.exchange_species,
        )
        assert gui_data.fingerprint_network(reordered) == fingerprint

    def test_distinct_networks_differ(self) -> None:
        soc = network_specs.get_network_spec("soc_calcite").builder()
        carbon = network_specs.get_network_spec("carbon").builder()
        assert gui_data.fingerprint_network(
            soc
        ) != gui_data.fingerprint_network(carbon)


class TestNetworkCli:
    """The ``--network`` option of `main` (headless: it fails before
    any Qt object is created)."""

    def test_rejects_non_yaml_network(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as excinfo:
            gui_main(["--network", "nope.txt"])
        assert excinfo.value.code == 2
        assert "must be a .yaml or .yml" in capsys.readouterr().err


class TestNetworkDraft:
    """
    The T9 draft dataclasses: N1 normalization at seed time,
    lossless seed → build round-trip, order-preserving upsert,
    mineral-removal cascade, and the two-pass validation (raw
    V3/V7 first, then the network pass's V1–V9/W1/W2).
    """

    W1_MESSAGE: ClassVar[str] = (
        "no H+ primary species: the speciation solve has no "
        "charge-balance row"
    )

    @staticmethod
    def _seed_draft() -> gui_network_editor.NetworkDraft:
        network: pt.ReactionNetwork = get_soc_calcite_network()
        return gui_network_editor.NetworkDraft.from_network(network)

    def test_seed_drops_uncarried_stoich_keys(self) -> None:
        draft: gui_network_editor.NetworkDraft = self._seed_draft()
        co2: gui_network_editor.DraftSecondary = next(
            entry
            for entry in draft.secondary
            if entry.name == "CO2(aq)"
        )
        keys: set[str] = {key for key, _ in co2.stoichiometry}
        assert keys == {"CO2(aq)", "H+", "HCO3-"}
        xdoc: gui_network_editor.DraftExchange = draft.exchange[0]
        exchange_keys: set[str] = {
            key for key, _ in xdoc.stoichiometry
        }
        assert "X-" in exchange_keys
        assert "XDOC" in exchange_keys

    def test_seed_to_network_round_trip_is_lossless(self) -> None:
        network: pt.ReactionNetwork = get_soc_calcite_network()
        draft: gui_network_editor.NetworkDraft = (
            gui_network_editor.NetworkDraft.from_network(network)
        )
        rebuilt: pt.ReactionNetwork = draft.to_network()
        assert rebuilt.species_names == network.species_names
        carried: set[str] = set(network.species_names) | {"X-"}

        def kept(stoichiometry: dict[str, float]) -> dict[str, float]:
            return {
                key: value
                for key, value in stoichiometry.items()
                if key in carried
            }

        for primary, new_primary in zip(
            network.primary_aqueous, rebuilt.primary_aqueous
        ):
            assert primary.name == new_primary.name
            assert primary.molar_mass == new_primary.molar_mass
            assert primary.charge == new_primary.charge
            assert primary.dh_size_param == new_primary.dh_size_param
        for secondary, new_secondary in zip(
            network.secondary, rebuilt.secondary
        ):
            assert secondary.name == new_secondary.name
            assert secondary.molar_mass == new_secondary.molar_mass
            assert secondary.charge == new_secondary.charge
            assert secondary.dh_size_param == new_secondary.dh_size_param
            assert (
                tuple(secondary.eq_consts) == tuple(new_secondary.eq_consts)
            )
            assert kept(secondary.stoichiometry) == new_secondary.stoichiometry
        for mineral, new_mineral in zip(network.mineral, rebuilt.mineral):
            assert mineral.name == new_mineral.name
            assert mineral.molar_mass == new_mineral.molar_mass
            assert mineral.molar_volume == new_mineral.molar_volume
            assert (
                tuple(mineral.eq_consts) == tuple(new_mineral.eq_consts)
            )
            assert kept(mineral.stoichiometry) == new_mineral.stoichiometry
        for exchange, new_exchange in zip(
            network.exchange_species, rebuilt.exchange_species
        ):
            assert exchange.name == new_exchange.name
            assert exchange.log10_k_eq == new_exchange.log10_k_eq
            assert exchange.charge == new_exchange.charge
            assert kept(exchange.stoichiometry) == new_exchange.stoichiometry
        for mineral_name, original_tst in (
            network.mineral_kinetics.tst_reactions.items()
        ):
            match_tst: pt.TstReaction | None = (
                rebuilt.mineral_kinetics.tst_reactions.get(mineral_name)
            )
            assert match_tst is not None
            assert match_tst.label == original_tst.label
            assert match_tst.rate_constant == original_tst.rate_constant
            assert dict(match_tst.dependence) == dict(original_tst.dependence)
        for mineral_name, original_monod in (
            network.mineral_kinetics.monod_reactions.items()
        ):
            match_monod: pt.MonodReaction | None = (
                rebuilt.mineral_kinetics.monod_reactions.get(mineral_name)
            )
            assert match_monod is not None
            assert match_monod.label == original_monod.label
            assert match_monod.rate_constant == original_monod.rate_constant
            assert (
                dict(match_monod.monod_terms)
                == dict(original_monod.monod_terms)
            )
            assert (
                dict(match_monod.inhib_terms)
                == dict(original_monod.inhib_terms)
            )

    def test_seed_draft_is_clean(self) -> None:
        draft: gui_network_editor.NetworkDraft = self._seed_draft()
        errors, warnings = gui_network_editor.draft_validation(draft)
        assert errors == []
        assert warnings == []

    def test_upsert_replaces_in_place_and_appends(self) -> None:
        draft: gui_network_editor.NetworkDraft = self._seed_draft()
        replaced: gui_network_editor.DraftSecondary = (
            gui_network_editor.DraftSecondary(
                name="CO2(aq)",
                stoichiometry=(("CO2(aq)", 1.0),),
                eq_consts=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0),
                molar_mass=44.0,
                charge=0.0,
                dh_size_param=0.0,
            )
        )
        updated: gui_network_editor.NetworkDraft = (
            draft.upsert_secondary(replaced)
        )
        assert [entry.name for entry in updated.secondary] == [
            "CO2(aq)",
            "CO3--",
        ]
        co2 = next(
            entry for entry in updated.secondary
            if entry.name == "CO2(aq)"
        )
        assert co2.eq_consts == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
        added: gui_network_editor.DraftPrimary = (
            gui_network_editor.DraftPrimary(
                name="NewIon",
                molar_mass=1.0,
                charge=1.0,
                dh_size_param=0.0,
            )
        )
        with_new: gui_network_editor.NetworkDraft = (
            draft.upsert_primary(added)
        )
        assert with_new.primary[-1].name == "NewIon"

    def test_remove_mineral_cascades_into_kinetics(self) -> None:
        draft: gui_network_editor.NetworkDraft = self._seed_draft()
        removed: gui_network_editor.NetworkDraft = (
            draft.remove_mineral("SOC(s)")
        )
        assert "SOC(s)" not in [entry.name for entry in removed.mineral]
        assert not any(
            entry.mineral_name == "SOC(s)" for entry in removed.monod
        )
        assert not any(
            entry.mineral_name == "SOC(s)" for entry in removed.tst
        )

    def test_upsert_tst_replaces_the_minerals_monod(self) -> None:
        draft: gui_network_editor.NetworkDraft = self._seed_draft()
        entry: gui_network_editor.DraftTst = (
            gui_network_editor.DraftTst(
                mineral_name="SOC-lz(s)",
                label="second",
                log10_k=-1.0,
                dependence=(),
            )
        )
        updated: gui_network_editor.NetworkDraft = draft.upsert_tst(entry)
        assert [t.mineral_name for t in updated.tst] == [
            "Calcite(s)",
            "SOC-lz(s)",
        ]
        assert [m.mineral_name for m in updated.monod] == ["SOC(s)"]

    def test_dangling_hplus_surfaces_as_v3_and_w1(self) -> None:
        draft: gui_network_editor.NetworkDraft = self._seed_draft()
        updated: gui_network_editor.NetworkDraft = (
            draft.remove_primary("H+")
        )
        errors, warnings = gui_network_editor.draft_validation(updated)
        assert errors == [
            "V3: stoichiometry of secondary species 'CO2(aq)' "
            "references unknown species 'H+'",
            "V3: stoichiometry of secondary species 'CO3--' "
            "references unknown species 'H+'",
        ]
        assert warnings == [f"W1: {self.W1_MESSAGE}"]

    def test_v5_both_and_neither_kinds_block(self) -> None:
        draft: gui_network_editor.NetworkDraft = self._seed_draft()
        both: gui_network_editor.NetworkDraft = replace(
            draft,
            tst=draft.tst
            + (
                gui_network_editor.DraftTst(
                    mineral_name="SOC(s)",
                    label="extra",
                    log10_k=0.0,
                    dependence=(),
                ),
            ),
        )
        errors, _ = gui_network_editor.draft_validation(both)
        assert errors == [
            "V5: mineral 'SOC(s)' has both TST and Monod kinetics "
            "entries",
        ]
        neither: gui_network_editor.NetworkDraft = (
            both.remove_mineral_kinetics("SOC(s)")
        )
        errors, _ = gui_network_editor.draft_validation(neither)
        assert errors == [
            "V5: mineral 'SOC(s)' has no kinetics entry",
        ]

    def test_v8_requires_two_eq_consts(self) -> None:
        draft: gui_network_editor.NetworkDraft = self._seed_draft()
        section: tuple[gui_network_editor.DraftSecondary, ...] = tuple(
            replace(entry, eq_consts=(0.0,))
            if entry.name == "CO2(aq)"
            else entry
            for entry in draft.secondary
        )
        short: gui_network_editor.NetworkDraft = replace(
            draft,
            secondary=section,
        )
        errors, _ = gui_network_editor.draft_validation(short)
        assert (
            "V8: secondary species 'CO2(aq)' carries fewer than two "
            "eq_consts"
        ) in errors

    def test_chloride_draft_warns_w1_only(self) -> None:
        network: pt.ReactionNetwork = get_chloride_network()
        draft: gui_network_editor.NetworkDraft = (
            gui_network_editor.NetworkDraft.from_network(network)
        )
        errors, warnings = gui_network_editor.draft_validation(draft)
        assert errors == []
        assert warnings == [f"W1: {self.W1_MESSAGE}"]
