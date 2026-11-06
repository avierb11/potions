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
* `potions.gui.plots` — the four figure builders and the objective
  table, asserted on the figures' artists (titles, axes, series
  counts, legend entries, colors, linestyles, clipping).

The Qt-layer tests live in `test_gui_qt.py`.
"""

import json
import os
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
matplotlib.use("Agg")

import potions as pt
from potions.hydro_model import HydrologicalModel
from potions.result_saving import (
    DEFAULT_RESULTS_DIR,
    save_hydro_run,
    save_model_run,
)

from ..gui import data as gui_data
from ..gui import gradients as gui_gradients
from ..gui import plots as gui_plots
from ..gui import state as gui_state
from ..gui.common import (
    HYDRO_PARAMS,
    METRIC_LABELS,
    METRIC_NAMES,
    PARAM_LABELS,
    PLOT_START,
    PLOT_START_DATE,
    RIVER_SPECIES_SIM_COLORS,
    SPECIATION_ZONES,
    TIME_SERIES_OPTIONS,
    ZONE_COLORS,
    ZONE_LABELS,
    ZONE_ORDER,
    bound_decimals,
    param_label,
    param_symbol,
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
    write_gui_state,
)

# ------------------------------------------------------------------ #
# Fixtures and helpers
# ------------------------------------------------------------------ #


@pytest.fixture(scope="session")
def input_data() -> gui_data.GuiInputData:
    """The package's own East River input data (the app's default)."""
    return gui_data.load_input_data(gui_data.DEFAULT_CSV_PATH)


def _synthetic_hydro_sim(index: pd.Index) -> pd.DataFrame:
    """A synthetic hydrology simulation frame with the figure columns."""
    data: dict[str, np.ndarray] = {
        "streamflow_sim": np.arange(len(index), dtype=float)
    }
    for zone in ZONE_ORDER:
        data[f"s_{zone}"] = np.arange(len(index), dtype=float)
    return pd.DataFrame(data, index=index)


def _synthetic_rt_sim(index: pd.Index, minerals: list[str]) -> pd.DataFrame:
    """A synthetic reactive-transport frame with every plotted column."""
    species: list[str] = ["DOC", "DIC", "Ca++"]
    data: dict[str, np.ndarray] = {}
    for species_name in species:
        data[f"{species_name}_riv"] = np.arange(len(index), dtype=float)
        for zone in ZONE_ORDER:
            data[f"{species_name}_{zone}"] = np.arange(len(index), dtype=float)
    for mineral in minerals:
        for zone in ZONE_ORDER[2:]:
            data[f"{mineral}_conc_{zone}"] = np.arange(len(index), dtype=float)
            data[f"{mineral}_rate_{zone}"] = np.arange(len(index), dtype=float)
    return pd.DataFrame(data, index=index)


def _synthetic_results(
    input_data: gui_data.GuiInputData,
) -> tuple[pt.HydroModelResults, pt.RtModelResults]:
    """Synthetic hydrology and reactive-transport results objects."""
    index: pd.Index = input_data.forc.precip.index
    hydro: pt.HydroModelResults = pt.HydroModelResults(
        forcing=input_data.forc,
        simulation=_synthetic_hydro_sim(index),
        objective_functions=pd.Series(
            {name: 0.9 * (i + 1) for i, name in enumerate(METRIC_NAMES)},
            dtype=float,
        ),
    )
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
        num_hydro: int = pt.LateralThreeLayerModel.get_num_hydro_parameters()
        hydro_model: HydrologicalModel = (
            pt.LateralThreeLayerModel.hydro_from_array(
                _midpoint_vector(input_data)[:num_hydro],
                natural_scales=True,
            )
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


def _full_state(
    input_data: gui_data.GuiInputData,
    save_history: bool = False,
    mode: str = MODE_FULL,
) -> GuiState:
    """A complete gui_state matching the loaded input data."""
    bounds: dict[str, tuple[float, float]] = dict(input_data.bounds)
    values: dict[str, float] = {
        key: 0.5 * (lo + hi) for key, (lo, hi) in bounds.items()
    }
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
        assert list(conc.columns) == ["DOC", "DIC", "Ca++"]
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
        assert bounds["ground_hs.alpha"] == (1.0, 1.0)
        assert bounds["ground_rp.k_a[1]"] == (-15.0, -5.0)


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


class TestDefaultPlotSelection:
    """The panel's starting selection matches the app's defaults."""

    def test_defaults(self, input_data: gui_data.GuiInputData) -> None:
        selection = gui_data.default_plot_selection()
        assert selection.time_series == [
            "Streamflow",
            "DOC",
            "DIC",
            "Ca++",
        ]
        assert selection.species == ["DOC"]
        assert selection.zones == [
            "surface_hs",
            "surface_rp",
            "ground_hs",
            "ground_rp",
        ]
        assert selection.minerals == ["SOC(s)"]



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
        assert GUI_STATE_VERSION == 2
        assert state.to_dict()["version"] == 2

    def test_to_dict_shape(self, input_data: gui_data.GuiInputData) -> None:
        raw: dict[str, object] = _full_state(input_data).to_dict()
        assert raw["version"] == 2
        assert raw["csv_path"] == str(gui_data.DEFAULT_CSV_PATH)
        param_values = _as_dict(raw, "param_values")
        assert param_values["proportion.1"] == 0.7
        param_bounds = _as_dict(raw, "param_bounds")
        assert param_bounds["proportion.1"] == [0.25, 0.95]
        selection = _as_dict(raw, "selection")
        assert selection["species"] == ["DOC"]
        assert selection["time_series"] == ["Streamflow", "DOC", "DIC", "Ca++"]
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


class TestFromDictErrors:
    """Each malformed gui_state field produces its documented message."""

    def _base(self, input_data: gui_data.GuiInputData) -> dict[str, object]:
        return _full_state(input_data).to_dict()

    @pytest.mark.parametrize(
        ("patch", "message"),
        [
            (
                {"version": 99},
                "Unsupported gui_state.json version 99 (expected [1, 2]).",
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

    def test_time_series_field_not_strings(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        raw: dict[str, object] = self._base(input_data)
        selection: dict[object, object] = _as_dict(raw, "selection")
        selection["time_series"] = ["Streamflow", 1]
        raw["selection"] = selection
        with pytest.raises(ValueError) as excinfo:
            GuiState.from_dict(raw)
        assert (
            str(excinfo.value)
            == "gui_state.json selection field 'time_series' "
            "is not a list of strings."
        )

    def test_time_series_absent_keeps_all_on(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """A state saved before the checkboxes existed draws every series."""
        raw: dict[str, object] = self._base(input_data)
        selection: dict[object, object] = _as_dict(raw, "selection")
        del selection["time_series"]
        raw["selection"] = selection
        restored = GuiState.from_dict(raw)
        assert restored.selection.time_series == list(TIME_SERIES_OPTIONS)


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
            f"Unsupported gui_state.json version 99 (expected [1, 2]).\n"
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
        assert text.startswith("{\n  \"version\": 2,")
        assert text.endswith("\n")




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


def _small_history_entry(timestamp: str) -> RunHistoryEntry:
    """A small run-history record for the append tests."""
    return RunHistoryEntry(
        timestamp=timestamp,
        mode="full",
        param_values={"proportion.1": 0.7},
        objectives={"Streamflow": {"nse": 0.85}},
    )


class TestAppendRunHistory:
    """``gui_run_history.jsonl``: one JSON object per finished run."""

    def test_appends_lines(self, tmp_path: Path) -> None:
        # The results directory does not exist yet: the first append
        # creates it (and the file) on demand.
        results_dir: Path = tmp_path / "does-not-exist" / "results"
        entry_one = _small_history_entry("2026-01-01T00:00:00+00:00")
        entry_two = _small_history_entry("2026-01-02T00:00:00+00:00")

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
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.legend import Legend
from matplotlib.lines import Line2D


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
    """The PathCollection artists of an axes, in draw order."""
    assert isinstance(ax, Axes)
    return [
        artist for artist in ax.collections if isinstance(artist, PathCollection)
    ]


class TestTimeSeriesFigure:
    """The streamflow / river-concentration figure on synthetic data."""

    def test_axes_and_labels(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        index: pd.Index = input_data.forc.precip.index
        fig: Figure = gui_plots.make_time_series_figure(
            _synthetic_hydro_sim(index),
            _synthetic_rt_sim(index, list(input_data.network.mineral_names)),
            input_data.q_meas,
            input_data.conc_df,
        )
        assert len(fig.axes) == 2
        hydro_ax = fig.axes[0]
        species_ax = fig.axes[1]
        assert hydro_ax.get_title() == "Streamflow"
        # The species title sits on the right; the center slot is empty.
        assert species_ax.get_title("right") == "River concentrations"
        assert species_ax.get_title() == ""
        assert hydro_ax.get_ylabel() == "Discharge (mm/d)"
        assert species_ax.get_ylabel() == "Concentration (mol/L)"

    def test_streamflow_series(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        index: pd.Index = input_data.forc.precip.index
        fig: Figure = gui_plots.make_time_series_figure(
            _synthetic_hydro_sim(index),
            _synthetic_rt_sim(index, list(input_data.network.mineral_names)),
            input_data.q_meas,
            input_data.conc_df,
        )
        hydro_ax = fig.axes[0]

        sim = _lines(hydro_ax)
        assert len(sim) == 1
        assert sim[0].get_label() == "Streamflow (sim)"
        assert mcolors.to_rgba(sim[0].get_color()) == mcolors.to_rgba(
            "cornflowerblue"
        )
        assert sim[0].get_linewidth() == 1
        # Clipped to PLOT_START: the full record minus the spin-up.
        # (The artists' data getters are typed with the broad ArrayLike
        # union, so the numpy conversion is for mypy's benefit.)
        assert len(np.asarray(sim[0].get_xdata())) == 2557

        meas = _scatters(hydro_ax)
        assert len(meas) == 1
        assert meas[0].get_label() == "Streamflow (meas)"
        facecolors: np.ndarray = np.asarray(meas[0].get_facecolor())
        assert mcolors.to_rgba(facecolors[0]) == mcolors.to_rgba("black")
        assert list(meas[0].get_sizes()) == [5]
        expected_meas = int(
            input_data.q_meas.loc[input_data.q_meas.index >= PLOT_START]
            .dropna()
            .size
        )
        offsets: np.ndarray = np.asarray(meas[0].get_offsets())
        assert len(offsets) == expected_meas

    def test_river_species_series(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        index: pd.Index = input_data.forc.precip.index
        fig: Figure = gui_plots.make_time_series_figure(
            _synthetic_hydro_sim(index),
            _synthetic_rt_sim(index, list(input_data.network.mineral_names)),
            input_data.q_meas,
            input_data.conc_df,
        )
        species_ax = fig.axes[1]

        sim = _lines(species_ax)
        assert [line.get_label() for line in sim] == [
            "DOC (sim)",
            "DIC (sim)",
            "Ca++ (sim)",
        ]
        for line, species in zip(sim, ["DOC", "DIC", "Ca++"]):
            assert mcolors.to_rgba(line.get_color()) == mcolors.to_rgba(
                RIVER_SPECIES_SIM_COLORS[species]
            )

        meas = _scatters(species_ax)
        assert [scatter.get_label() for scatter in meas] == [
            "DOC (meas)",
            "DIC (meas)",
            "Ca++ (meas)",
        ]
        for scatter, species in zip(meas, ["DOC", "DIC", "Ca++"]):
            expected = int(
                input_data.conc_df[species]
                .loc[input_data.conc_df.index >= PLOT_START]
                .dropna()
                .size
            )
            scatter_offsets: np.ndarray = np.asarray(scatter.get_offsets())
            assert len(scatter_offsets) == expected

    def test_limits_and_clipping(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        index: pd.Index = input_data.forc.precip.index
        fig: Figure = gui_plots.make_time_series_figure(
            _synthetic_hydro_sim(index),
            _synthetic_rt_sim(index, list(input_data.network.mineral_names)),
            input_data.q_meas,
            input_data.conc_df,
        )
        last_day: float = _day_number(index[-1])
        assert fig.axes[0].get_xlim() == (PLOT_START_DATE, last_day)
        assert fig.axes[1].get_xlim() == (PLOT_START_DATE, last_day)
        # The streamflow axis is inverted (discharge grows downward)
        # and spans twice the data maximum (plus the autoscale margin).
        top: float = fig.axes[0].get_ylim()[0]
        bottom: float = fig.axes[0].get_ylim()[1]
        assert bottom == 0.0
        data_max: float = len(index) - 1
        assert 2 * data_max < top < 3 * data_max

    def test_single_figure_legend(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        index: pd.Index = input_data.forc.precip.index
        fig: Figure = gui_plots.make_time_series_figure(
            _synthetic_hydro_sim(index),
            _synthetic_rt_sim(index, list(input_data.network.mineral_names)),
            input_data.q_meas,
            input_data.conc_df,
        )
        assert len(fig.legends) == 1
        legend = fig.legends[0]
        assert legend.get_title().get_text() == ""
        assert [text.get_text() for text in legend.get_texts()] == [
            "Streamflow (sim)",
            "Streamflow (meas)",
            "DOC (sim)",
            "DOC (meas)",
            "DIC (sim)",
            "DIC (meas)",
            "Ca++ (sim)",
            "Ca++ (meas)",
        ]
        # `_ncols` is a private matplotlib attribute, hence getattr.
        assert getattr(legend, "_ncols") == 5
        # One row, above the top edge of the figure, horizontally centered.
        np.testing.assert_allclose(
            legend.get_bbox_to_anchor().extents,
            _figure_anchor(fig, 0.5, 1.02),
        )

    def test_subset_selection(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """The selection draws only the chosen series."""
        index: pd.Index = input_data.forc.precip.index
        fig: Figure = gui_plots.make_time_series_figure(
            _synthetic_hydro_sim(index),
            _synthetic_rt_sim(index, list(input_data.network.mineral_names)),
            input_data.q_meas,
            input_data.conc_df,
            ["Streamflow", "DOC"],
        )
        assert len(fig.axes) == 2
        assert [line.get_label() for line in _lines(fig.axes[0])] == [
            "Streamflow (sim)",
        ]
        assert [
            scatter.get_label() for scatter in _scatters(fig.axes[0])
        ] == ["Streamflow (meas)"]
        assert [line.get_label() for line in _lines(fig.axes[1])] == [
            "DOC (sim)",
        ]
        assert [
            scatter.get_label() for scatter in _scatters(fig.axes[1])
        ] == ["DOC (meas)"]
        legend = fig.legends[0]
        assert [text.get_text() for text in legend.get_texts()] == [
            "Streamflow (sim)",
            "Streamflow (meas)",
            "DOC (sim)",
            "DOC (meas)",
        ]

    def test_streamflow_only(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """Without any river species, the right axis is not created."""
        index: pd.Index = input_data.forc.precip.index
        fig: Figure = gui_plots.make_time_series_figure(
            _synthetic_hydro_sim(index),
            _synthetic_rt_sim(index, list(input_data.network.mineral_names)),
            input_data.q_meas,
            input_data.conc_df,
            ["Streamflow"],
        )
        assert len(fig.axes) == 1
        assert fig.axes[0].get_title() == "Streamflow"
        assert len(_lines(fig.axes[0])) == 1
        assert len(_scatters(fig.axes[0])) == 1
        legend = fig.legends[0]
        assert [text.get_text() for text in legend.get_texts()] == [
            "Streamflow (sim)",
            "Streamflow (meas)",
        ]

    def test_no_series(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """With nothing selected, the figure is empty and legend-less."""
        index: pd.Index = input_data.forc.precip.index
        fig: Figure = gui_plots.make_time_series_figure(
            _synthetic_hydro_sim(index),
            _synthetic_rt_sim(index, list(input_data.network.mineral_names)),
            input_data.q_meas,
            input_data.conc_df,
            [],
        )
        assert len(fig.axes) == 1
        assert fig.axes[0].get_title() == ""
        assert len(_lines(fig.axes[0])) == 0
        assert len(_scatters(fig.axes[0])) == 0
        assert len(fig.legends) == 0

    def test_no_rt_results(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """A hydrology-only run: one axis even with species selected."""
        index: pd.Index = input_data.forc.precip.index
        fig: Figure = gui_plots.make_time_series_figure(
            _synthetic_hydro_sim(index),
            None,
            input_data.q_meas,
            input_data.conc_df,
            ["Streamflow", "DOC", "DIC", "Ca++"],
        )
        assert len(fig.axes) == 1
        assert fig.axes[0].get_title() == "Streamflow"
        assert [line.get_label() for line in _lines(fig.axes[0])] == [
            "Streamflow (sim)",
        ]
        assert [
            scatter.get_label() for scatter in _scatters(fig.axes[0])
        ] == ["Streamflow (meas)"]

    def test_overlay_previous_run(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """The previous run's series draw behind, faintly, out of the legend."""
        index: pd.Index = input_data.forc.precip.index
        hydro_sim = _synthetic_hydro_sim(index)
        rt_sim = _synthetic_rt_sim(index, list(input_data.network.mineral_names))
        # Distinguishable overlay frames (same columns, shifted values).
        overlay_hydro: pd.DataFrame = _synthetic_hydro_sim(index) + 10.0
        overlay_rt: pd.DataFrame = (
            _synthetic_rt_sim(index, list(input_data.network.mineral_names))
            + 10.0
        )

        fig = gui_plots.make_time_series_figure(
            hydro_sim,
            rt_sim,
            input_data.q_meas,
            input_data.conc_df,
            overlay_hydro_sim=overlay_hydro,
            overlay_rt_sim=overlay_rt,
        )
        hydro_ax = fig.axes[0]
        species_ax = fig.axes[1]

        # The overlay streamflow line is drawn first (behind) and
        # half transparent; the current line keeps its label and its
        # full opacity, and the overlay data survives the clipping.
        hydro_lines = _lines(hydro_ax)
        assert len(hydro_lines) == 2
        assert hydro_lines[0].get_alpha() == 0.5
        assert _alpha(hydro_lines[1]) == 1.0
        assert hydro_lines[1].get_label() == "Streamflow (sim)"
        expected: np.ndarray = np.asarray(
            overlay_hydro.loc[
                overlay_hydro.index >= PLOT_START, "streamflow_sim"
            ]
        )
        np.testing.assert_allclose(
            np.asarray(hydro_lines[0].get_ydata()), expected
        )

        # The right axis carries the overlay species lines first, in
        # species order, then the current ones with their labels.
        species_lines = _lines(species_ax)
        assert len(species_lines) == 6
        assert [line.get_alpha() for line in species_lines[:3]] == [0.5] * 3
        assert [_alpha(line) for line in species_lines[3:]] == [1.0] * 3
        assert [line.get_label() for line in species_lines[3:]] == [
            "DOC (sim)",
            "DIC (sim)",
            "Ca++ (sim)",
        ]

        # The legend still lists only the current run's series.
        legend = fig.legends[0]
        assert [text.get_text() for text in legend.get_texts()] == [
            "Streamflow (sim)",
            "Streamflow (meas)",
            "DOC (sim)",
            "DOC (meas)",
            "DIC (sim)",
            "DIC (meas)",
            "Ca++ (sim)",
            "Ca++ (meas)",
        ]

    def test_overlay_skips_missing_columns(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """A previous frame that lacks a column contributes no line for it."""
        index: pd.Index = input_data.forc.precip.index
        hydro_sim = _synthetic_hydro_sim(index)
        rt_sim = _synthetic_rt_sim(index, list(input_data.network.mineral_names))
        # The overlay carries streamflow but no river-species columns.
        overlay_hydro: pd.DataFrame = _synthetic_hydro_sim(index)

        fig = gui_plots.make_time_series_figure(
            hydro_sim,
            rt_sim,
            input_data.q_meas,
            input_data.conc_df,
            overlay_hydro_sim=overlay_hydro,
        )
        hydro_lines = _lines(fig.axes[0])
        assert len(hydro_lines) == 2
        assert hydro_lines[0].get_alpha() == 0.5
        # No overlay species lines on the right axis.
        assert len(_lines(fig.axes[1])) == 3

    def test_hydro_only_overlay(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """A hydrology-only figure can still carry the streamflow overlay."""
        index: pd.Index = input_data.forc.precip.index
        hydro_sim = _synthetic_hydro_sim(index)
        overlay_hydro: pd.DataFrame = _synthetic_hydro_sim(index) + 10.0

        fig = gui_plots.make_time_series_figure(
            hydro_sim,
            None,
            input_data.q_meas,
            input_data.conc_df,
            overlay_hydro_sim=overlay_hydro,
        )
        assert len(fig.axes) == 1
        lines = _lines(fig.axes[0])
        assert len(lines) == 2
        assert lines[0].get_alpha() == 0.5
        assert lines[1].get_label() == "Streamflow (sim)"
        legend = fig.legends[0]
        assert [text.get_text() for text in legend.get_texts()] == [
            "Streamflow (sim)",
            "Streamflow (meas)",
        ]


class TestZoneFigure:
    """The zone concentrations / storage figure on synthetic data."""

    def test_axes_and_labels(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        index: pd.Index = input_data.forc.precip.index
        fig: Figure = gui_plots.make_zone_figure(
            _synthetic_rt_sim(index, list(input_data.network.mineral_names)),
            _synthetic_hydro_sim(index),
            ["DOC"],
            ["surface_hs", "ground_rp"],
        )
        assert len(fig.axes) == 2
        assert fig.axes[0].get_title() == "Concentration"
        assert fig.axes[0].get_ylabel() == "Concentration (mol/L)"
        assert fig.axes[1].get_title() == "Storage"
        assert fig.axes[1].get_ylabel() == "Storage (mm)"

    def test_series_colors_and_linestyles(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        index: pd.Index = input_data.forc.precip.index
        fig: Figure = gui_plots.make_zone_figure(
            _synthetic_rt_sim(index, list(input_data.network.mineral_names)),
            _synthetic_hydro_sim(index),
            ["DOC", "DIC"],
            ["surface_hs", "ground_rp"],
        )
        conc_ax = fig.axes[0]
        conc_lines = _lines(conc_ax)
        # One line per (zone, species) pair, zones in selection order.
        assert len(conc_lines) == 4
        for line, zone in zip(
            conc_lines, ["surface_hs", "surface_hs", "ground_rp", "ground_rp"]
        ):
            assert mcolors.to_rgba(line.get_color()) == mcolors.to_rgba(
                ZONE_COLORS[zone]
            )
        # The species index drives the linestyle.
        assert [line.get_linestyle() for line in conc_lines] == [
            "-",
            "--",
            "-",
            "--",
        ]

        storage_lines = _lines(fig.axes[1])
        assert len(storage_lines) == 2
        for line, zone in zip(
            storage_lines, ["surface_hs", "ground_rp"]
        ):
            assert mcolors.to_rgba(line.get_color()) == mcolors.to_rgba(
                ZONE_COLORS[zone]
            )

    def test_legends_and_layout(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        index: pd.Index = input_data.forc.precip.index
        fig: Figure = gui_plots.make_zone_figure(
            _synthetic_rt_sim(index, list(input_data.network.mineral_names)),
            _synthetic_hydro_sim(index),
            ["DOC", "DIC"],
            ["surface_hs", "ground_rp"],
        )
        assert len(fig.legends) == 2
        zone_legend, species_legend = fig.legends
        assert zone_legend.get_title().get_text() == "Zone"
        assert [t.get_text() for t in zone_legend.get_texts()] == [
            "Surface (HS)",
            "Ground (RP)",
        ]
        assert species_legend.get_title().get_text() == "Species"
        assert [t.get_text() for t in species_legend.get_texts()] == [
            "DOC",
            "DIC",
        ]
        # The zone row sits above the species row, both above the panels.
        np.testing.assert_allclose(
            zone_legend.get_bbox_to_anchor().extents,
            _figure_anchor(fig, 0.5, 0.93),
        )
        np.testing.assert_allclose(
            species_legend.get_bbox_to_anchor().extents,
            _figure_anchor(fig, 0.5, 0.83),
        )

        pars = fig.subplotpars
        assert (pars.left, pars.right, pars.bottom, pars.top) == (
            0.07,
            0.96,
            0.07,
            0.76,
        )

    def test_overlay_previous_run(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """The previous run's series draw behind, faintly, out of the legend."""
        index: pd.Index = input_data.forc.precip.index
        rt_sim = _synthetic_rt_sim(index, list(input_data.network.mineral_names))
        hydro_sim = _synthetic_hydro_sim(index)
        overlay_rt: pd.DataFrame = (
            _synthetic_rt_sim(index, list(input_data.network.mineral_names))
            + 10.0
        )
        overlay_hydro: pd.DataFrame = _synthetic_hydro_sim(index) + 10.0

        fig = gui_plots.make_zone_figure(
            rt_sim,
            hydro_sim,
            ["DOC", "DIC"],
            ["surface_hs", "ground_rp"],
            overlay_rt_sim=overlay_rt,
            overlay_hydro_sim=overlay_hydro,
        )

        # Both panels: the overlay lines (4 concentration, 2 storage)
        # draw first and half transparent, in the same zone-species
        # order as the current lines that follow.
        conc_lines = _lines(fig.axes[0])
        assert len(conc_lines) == 8
        assert [line.get_alpha() for line in conc_lines[:4]] == [0.5] * 4
        assert [_alpha(line) for line in conc_lines[4:]] == [1.0] * 4
        assert [line.get_linestyle() for line in conc_lines[:4]] == [
            "-",
            "--",
            "-",
            "--",
        ]
        storage_lines = _lines(fig.axes[1])
        assert len(storage_lines) == 4
        assert [line.get_alpha() for line in storage_lines[:2]] == [0.5] * 2
        assert [_alpha(line) for line in storage_lines[2:]] == [1.0] * 2

        # The legends still describe only the current run.
        zone_legend, species_legend = fig.legends
        assert [t.get_text() for t in zone_legend.get_texts()] == [
            "Surface (HS)",
            "Ground (RP)",
        ]
        assert [t.get_text() for t in species_legend.get_texts()] == [
            "DOC",
            "DIC",
        ]


class TestRateFigure:
    """The mineral reaction-rate figure on synthetic data."""

    def test_series_and_legends(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        index: pd.Index = input_data.forc.precip.index
        minerals: list[str] = ["SOC(s)", "Calcite(s)"]
        fig: Figure = gui_plots.make_rate_figure(
            _synthetic_rt_sim(index, list(input_data.network.mineral_names)),
            minerals,
        )
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert ax.get_title() == "Mineral reaction rates"
        assert ax.get_ylabel() == "Reaction rate (mol/L/day)"
        assert ax.get_xlim() == (PLOT_START_DATE, _day_number(index[-1]))

        # The synthetic frame carries rate columns in all four
        # non-snow zones, so both selected minerals plot 4 lines each.
        lines = _lines(ax)
        assert len(lines) == 8
        assert [line.get_linestyle() for line in lines] == [
            "-",
            "-",
            "-",
            "-",
            "--",
            "--",
            "--",
            "--",
        ]

        legends = _axis_legends(ax)
        assert len(legends) == 2
        zone_legend, mineral_legend = legends
        assert zone_legend.get_title().get_text() == "Zone"
        assert [t.get_text() for t in zone_legend.get_texts()] == [
            "Surface (HS)",
            "Surface (RP)",
            "Ground (HS)",
            "Ground (RP)",
        ]
        # The zone legend hangs below the axes.
        np.testing.assert_allclose(
            zone_legend.get_bbox_to_anchor().extents,
            _axis_anchor(fig, ax, 0.5, -0.18),
        )
        assert mineral_legend.get_title().get_text() == "Mineral"
        assert [t.get_text() for t in mineral_legend.get_texts()] == minerals
        # The mineral legend anchors to the axes' upper-right corner.
        mineral_box = mineral_legend.get_bbox_to_anchor().extents
        position = ax.get_position()
        width = fig.get_figwidth() * fig.dpi
        height = fig.get_figheight() * fig.dpi
        assert mineral_box[2] == pytest.approx((position.x0 + position.width) * width)
        assert mineral_box[3] == pytest.approx((position.y0 + position.height) * height)

    def test_overlay_previous_run(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """The previous run's rates draw behind, faintly, out of the legend."""
        index: pd.Index = input_data.forc.precip.index
        minerals: list[str] = ["SOC(s)", "Calcite(s)"]
        rt_sim = _synthetic_rt_sim(index, list(input_data.network.mineral_names))
        overlay_rt: pd.DataFrame = (
            _synthetic_rt_sim(index, list(input_data.network.mineral_names))
            + 10.0
        )

        fig = gui_plots.make_rate_figure(
            rt_sim, minerals, overlay_rt_sim=overlay_rt
        )
        ax = fig.axes[0]
        lines = _lines(ax)
        # The overlay (4 zones x 2 minerals) draws first and half
        # transparent, in the same mineral-zone order as the current
        # lines that follow.
        assert len(lines) == 16
        assert [line.get_alpha() for line in lines[:8]] == [0.5] * 8
        assert [_alpha(line) for line in lines[8:]] == [1.0] * 8
        assert [line.get_linestyle() for line in lines[:8]] == [
            "-",
            "-",
            "-",
            "-",
            "--",
            "--",
            "--",
            "--",
        ]

        # The legends still describe only the current run.
        zone_legend, mineral_legend = _axis_legends(ax)
        assert [t.get_text() for t in mineral_legend.get_texts()] == minerals

    def test_overlay_skips_missing_columns(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """A previous frame that lacks a rate column contributes no line."""
        index: pd.Index = input_data.forc.precip.index
        rt_sim = _synthetic_rt_sim(index, list(input_data.network.mineral_names))
        # The overlay carries the first non-snow zone's rate column
        # only.
        overlay_rt: pd.DataFrame = _synthetic_rt_sim(
            index, ["SOC(s)"]
        ).drop(
            columns=[
                f"SOC(s)_rate_{zone}" for zone in ZONE_ORDER[3:]
            ]
        )

        fig = gui_plots.make_rate_figure(rt_sim, ["SOC(s)"], overlay_rt)
        ax = fig.axes[0]
        lines = _lines(ax)
        # One overlay line (the zone the frame carries), then the four
        # current ones.
        assert len(lines) == 5
        assert lines[0].get_alpha() == 0.5
        assert [_alpha(line) for line in lines[1:]] == [1.0] * 4

    def test_mineral_without_rate_column_is_skipped(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        index: pd.Index = input_data.forc.precip.index
        rt_sim = _synthetic_rt_sim(index, ["SOC(s)"]).drop(
            columns=[f"SOC(s)_rate_{zone}" for zone in ZONE_ORDER[2:]]
        )
        fig: Figure = gui_plots.make_rate_figure(rt_sim, ["SOC(s)"])
        ax = fig.axes[0]
        assert len(_lines(ax)) == 0
        legends = _axis_legends(ax)
        assert [t.get_text() for t in legends[0].get_texts()] == []
        assert [t.get_text() for t in legends[1].get_texts()] == ["SOC(s)"]

    def test_single_mineral_legend(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        index: pd.Index = input_data.forc.precip.index
        fig: Figure = gui_plots.make_rate_figure(
            _synthetic_rt_sim(index, ["SOC(s)"]), ["SOC(s)"]
        )
        mineral_legend = _axis_legends(fig.axes[0])[1]
        assert [t.get_text() for t in mineral_legend.get_texts()] == ["SOC(s)"]
        # `_ncols` is a private matplotlib attribute, hence getattr.
        assert getattr(mineral_legend, "_ncols") == 1


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
        assert list(rt_objectives.index) == ["DOC", "DIC", "Ca++"]
        assert list(rt_objectives.columns) == METRIC_NAMES

        # Every metric is finite except DOC's log KGE: at the midpoint
        # parameters the simulated river DOC is ~1e-40 (numerically
        # zero), so its log-transformed series is constant and its
        # correlation is undefined.  The log NSE is sum-based and stays
        # finite for a constant series.
        for column in ("kge", "nse", "bias", "spearman_rho", "r_squared"):
            assert np.isfinite(rt_objectives[column]).all(), column
        assert np.isfinite(
            rt_objectives.loc[["DIC", "Ca++"], ["log_kge", "log_nse"]].to_numpy()
        ).all()
        # The scalar .loc access is typed with pandas' scalar union, so
        # the values come back through a row-plus-columns selection
        # (columns in the order given: log_kge, then log_nse).
        doc_log_metrics: np.ndarray = rt_objectives.loc[
            ["DOC"], ["log_kge", "log_nse"]
        ].to_numpy()
        assert np.isnan(doc_log_metrics[0, 0])
        assert np.isfinite(doc_log_metrics[0, 1])


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
