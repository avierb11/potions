"""
Non-Qt tests for the potions.gui package.

Covers the pure-Python layer of the GUI:

* `potions.gui.data` — the input loader and the parameter-array
  conversions (``arr_to_model`` / ``arr_to_results``; the real
  model run is the ``slow``-marked end-to-end test).
* `potions.gui.common` — the parameter-label and slider-mapping
  helpers and the module constants the panel and the figures use.
* `potions.gui.state` — the ``gui_state.json`` round trip, its
  error paths, the saved-run discovery, and the restore of full and
  legacy (pre-GUI) saved runs.
* `potions.gui.plots` — the three figure builders and the objective
  table, asserted on the figures' artists (titles, axes, series
  counts, legend entries, colors, linestyles, clipping).

The Qt-layer tests live in `test_gui_qt.py`.
"""

import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
matplotlib.use("Agg")

import potions as pt
from potions.result_saving import save_model_run

from ..gui import data as gui_data
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
    ZONE_COLORS,
    ZONE_LABELS,
    ZONE_ORDER,
    format_slider_value,
    param_label,
    slider_decimals,
    step_to_value,
    value_to_step,
)
from ..gui.state import (
    GUI_STATE_VERSION,
    GuiState,
    find_latest_run_dir,
    load_saved_run,
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
) -> Path:
    """
    Write a saved run (via the real `save_model_run`) with synthetic
    results, optionally with a gui_state.json, into `base_dir`.

    Returns:
        The created run directory.
    """
    model, _init_arr = gui_data.arr_to_model(
        _midpoint_vector(input_data), input_data
    )
    hydro, rt = _synthetic_results(input_data)
    run_dir = save_model_run(
        model, hydro, rt, results_dir=base_dir / "results"
    )
    if state is not None:
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


def _full_state(input_data: gui_data.GuiInputData) -> GuiState:
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
        splitter_sizes=[300, 200, 400],
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


class TestSliderMath:
    """The integer-step slider mapping and its display precision."""

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
    def test_slider_decimals(
        self, lo: float, hi: float, decimals: int
    ) -> None:
        assert slider_decimals(lo, hi) == decimals

    def test_step_to_value(self) -> None:
        assert step_to_value(0.0, 1.0, 500) == pytest.approx(0.5)
        assert step_to_value(0.0, 1.0, 0) == 0.0
        assert step_to_value(0.0, 1.0, 1000) == pytest.approx(1.0)
        assert step_to_value(-1.0, 1.0, 250) == pytest.approx(-0.5)
        # A degenerate range collapses to its single value at any step.
        assert step_to_value(3.0, 3.0, 777) == pytest.approx(3.0)

    def test_value_to_step(self) -> None:
        assert value_to_step(0.0, 1.0, 0.5) == 500
        assert value_to_step(0.0, 1.0, 0.0) == 0
        assert value_to_step(0.0, 1.0, 1.0) == 1000
        # Out-of-range values clamp to the ends of the slider.
        assert value_to_step(0.0, 1.0, 2.0) == 1000
        assert value_to_step(0.0, 1.0, -2.0) == 0
        # A degenerate range is always at step 0.
        assert value_to_step(3.0, 3.0, 3.0) == 0

    def test_format_slider_value(self) -> None:
        # The display precision follows the bounds' span.
        assert format_slider_value(0.25, 0.95, 0.6) == "0.60000"
        assert format_slider_value(0.0, 1.0, 0.5) == "0.5000"
        assert format_slider_value(150.0, 1500.0, 825.0) == "825.0"
        assert format_slider_value(-15.0, -5.0, -10.0) == "-10.000"


class TestDefaultPlotSelection:
    """The panel's starting selection matches the app's defaults."""

    def test_defaults(self, input_data: gui_data.GuiInputData) -> None:
        selection = gui_data.default_plot_selection()
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
        assert GUI_STATE_VERSION == 1
        assert state.to_dict()["version"] == 1

    def test_to_dict_shape(self, input_data: gui_data.GuiInputData) -> None:
        raw: dict[str, object] = _full_state(input_data).to_dict()
        assert raw["version"] == 1
        assert raw["csv_path"] == str(gui_data.DEFAULT_CSV_PATH)
        param_values = _as_dict(raw, "param_values")
        assert param_values["proportion.1"] == 0.7
        param_bounds = _as_dict(raw, "param_bounds")
        assert param_bounds["proportion.1"] == [0.25, 0.95]
        selection = _as_dict(raw, "selection")
        assert selection["species"] == ["DOC"]

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


class TestFromDictErrors:
    """Each malformed gui_state field produces its documented message."""

    def _base(self, input_data: gui_data.GuiInputData) -> dict[str, object]:
        return _full_state(input_data).to_dict()

    @pytest.mark.parametrize(
        ("patch", "message"),
        [
            (
                {"version": 99},
                "Unsupported gui_state.json version 99 (expected 1).",
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
            f"Unsupported gui_state.json version 99 (expected 1).\n"
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
        assert text.startswith("{\n  \"version\": 1,")
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

    def test_results_reconstructed(self, tmp_path: Path, input_data: gui_data.GuiInputData) -> None:
        run_dir = _write_saved_run(tmp_path, input_data, _full_state(input_data))
        saved = load_saved_run(run_dir, input_data)

        hydro: pt.HydroModelResults = saved.results.hydro
        rt: pt.RtModelResults = saved.results.reactive_transport
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
        assert saved.results.reactive_transport.saved_dir == str(run_dir)

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
            "(looked in hydro/ and rt/)."
        )



# ------------------------------------------------------------------ #
# potions.gui.plots: the figure builders
# ------------------------------------------------------------------ #

from matplotlib import colors as mcolors
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
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
        assert list(table.columns) == list(METRIC_LABELS.values())
        assert list(table.index) == ["Streamflow", "DOC", "DIC", "Ca++"]
        assert table.loc["Streamflow", "KGE"] == pytest.approx(0.9)
        assert table.loc["Ca++", "R\u00b2"] == pytest.approx(1.1)

    def test_hydro_only(self) -> None:
        table = gui_plots.build_objective_table(self._hydro(), None)
        assert list(table.index) == ["Streamflow"]
        assert list(table.columns) == list(METRIC_LABELS.values())

    def test_rt_only(self) -> None:
        table = gui_plots.build_objective_table(
            pd.Series(dtype=float), self._rt()
        )
        assert list(table.index) == ["DOC", "DIC", "Ca++"]

    def test_empty(self) -> None:
        table = gui_plots.build_objective_table(
            pd.Series(dtype=float), None
        )
        assert table.shape == (0, 7)
        assert list(table.columns) == list(METRIC_LABELS.values())



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
        rt: pt.RtModelResults = results.reactive_transport

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
