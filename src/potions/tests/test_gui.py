"""
Tests for the East River lateral-model calibration GUI package
(`potions.gui`).

Covered here:

* the East River series CSV that ships with the package (the GUI's
  sole data source);
* `potions.gui.data.load_input_data` (the record assembly and the
  East River parameter bounds, including the per-layer overrides and
  the trailing initial-state entries);
* the label and slider helpers of `potions.gui.common`;
* the figure builders and the objective table of
  `potions.gui.plots`, on synthetic simulation frames;
* and, marked `slow`, a full end-to-end smoke run of the entry
  script under streamlit's `AppTest`.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from streamlit.testing.v1 import AppTest

from .. import gui
from ..gui import common as gui_common
from ..gui import data as gui_data
from ..gui import plots as gui_plots


# ==== Fixtures / helpers ==== #
def _synthetic_rt_sim(
    species: list[str], zones: list[str], minerals: list[str]
) -> pd.DataFrame:
    """
    Build a synthetic reactive-transport simulation frame.

    The columns follow the layout the reactive transport model
    produces: ``{species}_{zone}`` concentrations, ``{species}_riv``
    river concentrations, and ``{mineral}_rate_{zone}`` rates for the
    zones in which reactions run only (the model emits no rate column
    for a zone with reactions disabled).  The index starts a couple
    of months before ``gui_common.PLOT_START`` and runs past it, so
    the builders' clipping is exercised and never left empty.

    Args:
        species: The species whose concentration and river columns
            are created.
        zones: The zones whose per-species concentration columns are
            created.
        minerals: The minerals whose per-zone rate columns are
            created (in every reacting zone).

    Returns:
        The synthetic simulation frame.
    """
    index: pd.DatetimeIndex = pd.date_range("2014-08-01", periods=90, freq="D")
    columns: dict[str, pd.Series] = {
        f"{sp}_riv": pd.Series(1e-4, index=index) for sp in species
    }
    for sp in species:
        for zone in zones:
            columns[f"{sp}_{zone}"] = pd.Series(1e-4, index=index)
    for mineral in minerals:
        for zone in gui_common.ZONE_ORDER:
            if zone not in gui_common.SNOW_ZONES:
                columns[f"{mineral}_rate_{zone}"] = pd.Series(1e-6, index=index)
    return pd.DataFrame(columns)


def _synthetic_hydro_sim(zones: list[str]) -> pd.DataFrame:
    """
    Build a synthetic hydrology simulation frame.

    The columns follow the layout the hydrology model produces: one
    per-zone storage column (``s_{zone}``, in millimeters) plus the
    simulated streamflow, matching the zone set of the reactive
    transport frame the figure tests pair it with.

    Args:
        zones: The zones whose storage (``s_{zone}``) columns are
            created.

    Returns:
        The synthetic simulation frame.
    """
    index: pd.DatetimeIndex = pd.date_range("2014-08-01", periods=90, freq="D")
    columns: dict[str, pd.Series] = {
        f"s_{zone}": pd.Series(10.0, index=index) for zone in zones
    }
    columns["streamflow_sim"] = pd.Series(10.0, index=index)
    return pd.DataFrame(columns)


# ==== Package contents ==== #
def test_shipped_series_csv() -> None:
    """
    The East River series record that ships with the package is
    present and carries the raw columns the loader expects.
    """
    csv_path: Path = gui_data.DEFAULT_CSV_PATH
    assert csv_path.is_file()
    assert csv_path.stat().st_size > 0

    df: pd.DataFrame = pd.read_csv(csv_path, index_col="date")
    assert list(df.columns) == [
        "ppt",
        "temp",
        "pet",
        "q_meas",
        "doc_molL",
        "dic_molL",
        "ca_molL",
    ]


# ==== Loader and bounds ==== #
def test_load_input_data() -> None:
    """
    The loader assembles the forcing window plus the two-year
    spin-up, with the East River overrides applied to the generic
    bounds and the exchange-site states as the trailing entries.
    """
    input_data: gui_data.GuiInputData = gui_data.load_input_data()

    # The full record: the forcing window (2014-10-01 to 2021-09-30,
    # both ends inclusive) plus the two-year spin-up prepended by the
    # loader.
    window_days: int = len(pd.date_range("2014-10-01", "2021-09-30", freq="D"))
    assert len(input_data.q_meas) == window_days + gui_data.NUM_SPINUP_DAYS
    assert len(input_data.conc_df) == window_days + gui_data.NUM_SPINUP_DAYS

    # The spin-up (the first two years, shifted back in time) carries
    # no measurements, and the record starts two years before the
    # forcing window.
    assert input_data.q_meas.index[0] == pd.Timestamp("2012-10-01")
    assert input_data.q_meas.index[-1] == pd.Timestamp("2021-09-30")
    assert input_data.q_meas.iloc[: gui_data.NUM_SPINUP_DAYS].isna().all()
    assert input_data.q_meas.iloc[gui_data.NUM_SPINUP_DAYS :].notna().all()
    assert input_data.conc_df.iloc[: gui_data.NUM_SPINUP_DAYS].isna().all().all()

    # The East River overrides on top of the generic bounds.
    bounds: dict[str, tuple[float, float]] = input_data.bounds
    assert bounds["proportion.1"] == (0.25, 0.95)
    assert bounds["ground_hs.alpha"] == (1.0, 1.0)
    assert bounds["ground_rp.alpha"] == (1.0, 1.0)

    # The per-layer reactive water storage bounds.
    assert bounds["snow_hs.max_water"] == (5000.0, 5000.0)
    assert bounds["snow_rp.max_water"] == (5000.0, 5000.0)
    assert bounds["surface_hs.max_water"] == (150.0, 1500.0)
    assert bounds["surface_rp.max_water"] == (150.0, 1500.0)
    assert bounds["ground_hs.max_water"] == (100.0, 1000.0)
    assert bounds["ground_rp.max_water"] == (100.0, 1000.0)

    # Per-mineral kinetics: the ground layer pins the soil-moisture
    # and temperature dependencies and calibrates only k_a, while the
    # surface layer calibrates all four fields.
    num_minerals: int = input_data.network.num_minerals
    for zone in ("ground_hs", "ground_rp"):
        for mineral_id in range(num_minerals):
            assert bounds[f"{zone}.sw_threshold[{mineral_id}]"] == (0.0, 0.0)
            assert bounds[f"{zone}.sw_exp[{mineral_id}]"] == (0.0, 0.0)
            assert bounds[f"{zone}.q_10[{mineral_id}]"] == (1.0, 1.0)
            assert bounds[f"{zone}.k_a[{mineral_id}]"] == (-15.0, -5.0)
    for zone in ("surface_hs", "surface_rp"):
        for mineral_id in range(num_minerals):
            assert bounds[f"{zone}.sw_threshold[{mineral_id}]"] == (0.0, 1.0)
            assert bounds[f"{zone}.sw_exp[{mineral_id}]"] == (0.5, 2.5)
            assert bounds[f"{zone}.q_10[{mineral_id}]"] == (1.0, 5.0)
            assert bounds[f"{zone}.k_a[{mineral_id}]"] == (-15.0, -5.0)

    # The calibrated exchange-site states are the trailing entries of
    # the parameter array, one per speciation zone, in model order.
    assert list(bounds)[-4:] == [
        "initial_state.surface_hs_x",
        "initial_state.surface_rp_x",
        "initial_state.ground_hs_x",
        "initial_state.ground_rp_x",
    ]
    assert bounds["initial_state.surface_hs_x"] == (-5.0, -1.0)
    assert bounds["initial_state.surface_rp_x"] == (-5.0, -1.0)
    assert bounds["initial_state.ground_hs_x"] == (-7.0, -3.0)
    assert bounds["initial_state.ground_rp_x"] == (-7.0, -3.0)

    # The network is the SOC / DOC / calcite network of the site
    # model, and the precipitation concentration array carries one
    # value per top-layer zone, day, and species.
    assert input_data.network.mineral_names == ["SOC(s)", "SOC-lz(s)", "Calcite(s)"]
    assert input_data.precip_conc.shape == (
        2,
        len(input_data.q_meas),
        input_data.network.num_species,
    )


# ==== Labels and sliders ==== #
def test_param_label() -> None:
    """
    The label helper resolves each of the three key forms (and
    rejects a key of an unrecognized form).
    """
    mineral_names: list[str] = ["SOC(s)", "SOC-lz(s)", "Calcite(s)"]

    assert (
        gui_common.param_label("proportion.1", mineral_names)
        == "Distal proportion of catchment"
    )
    assert gui_common.param_label("snow_hs.tt", mineral_names) == "threshold temperature [°C]"
    assert (
        gui_common.param_label("surface_hs.k_a[1]", mineral_names)
        == "SOC-lz(s) dissolution rate constant (k_a) [1/m³/day]"
    )
    assert (
        gui_common.param_label("initial_state.ground_rp_x", mineral_names)
        == "Ground (RP) total sorption sites"
    )

    with pytest.raises(ValueError):
        gui_common.param_label("bogus.key", mineral_names)

    with pytest.raises(ValueError):
        gui_common.param_label("no_dot_key", mineral_names)


def test_slider_value() -> None:
    """
    A fixed parameter is pinned to its bound, and a free parameter
    defaults to the midpoint of its bounds.
    """
    mineral_names: list[str] = ["SOC(s)", "SOC-lz(s)", "Calcite(s)"]

    fixed_bounds: dict[str, tuple[float, float]] = {"snow_hs.fmax": (2.5, 2.5)}
    assert gui_common.slider_value("snow_hs.fmax", fixed_bounds, mineral_names) == 2.5

    free_bounds: dict[str, tuple[float, float]] = {"surface_hs.k_a[0]": (0.0, 2.0)}
    assert gui_common.slider_value("surface_hs.k_a[0]", free_bounds, mineral_names) == 1.0


# ==== Figures and the objective table ==== #
def test_make_time_series_figure() -> None:
    """The streamflow / river-concentration figure builds on a
    synthetic simulation."""
    index: pd.DatetimeIndex = pd.date_range("2014-08-01", periods=90, freq="D")
    hydro_sim: pd.DataFrame = pd.DataFrame({"streamflow_sim": 10.0}, index=index)
    rt_sim: pd.DataFrame = _synthetic_rt_sim(["DOC", "DIC", "Ca++"], [], [])
    q_meas: pd.Series = pd.Series(10.0, index=index)
    conc_df: pd.DataFrame = pd.DataFrame(
        {"DOC": 1e-4, "DIC": 1e-3, "Ca++": 1e-3}, index=index
    )

    fig: Figure = gui_plots.make_time_series_figure(
        hydro_sim, rt_sim, q_meas, conc_df
    )
    assert isinstance(fig, Figure)


def test_make_zone_figure() -> None:
    """The zone concentration / storage figure builds on a
    synthetic simulation."""
    zones: list[str] = ["surface_hs", "ground_rp"]
    rt_sim: pd.DataFrame = _synthetic_rt_sim(["DOC", "DIC"], zones, [])
    hydro_sim: pd.DataFrame = _synthetic_hydro_sim(zones)

    fig: Figure = gui_plots.make_zone_figure(
        rt_sim, hydro_sim, ["DOC", "DIC"], zones
    )
    assert isinstance(fig, Figure)
    # One concentration line per species per zone on top, one
    # storage line per zone below.
    assert len(fig.axes[0].lines) == 2 * len(zones)
    assert len(fig.axes[1].lines) == len(zones)


def test_make_rate_figure() -> None:
    """The mineral reaction-rate figure builds on a synthetic
    simulation."""
    minerals: list[str] = ["SOC(s)", "Calcite(s)"]
    rt_sim: pd.DataFrame = _synthetic_rt_sim([], [], minerals)

    fig: Figure = gui_plots.make_rate_figure(rt_sim, minerals)
    assert isinstance(fig, Figure)
    # The model emits a rate column only for the zones in which
    # reactions run, so the figure carries one line per mineral per
    # reacting zone (not per zone in the model's zone order).
    num_reacting_zones: int = len(
        [zone for zone in gui_common.ZONE_ORDER if zone not in gui_common.SNOW_ZONES]
    )
    assert len(fig.axes[0].lines) == len(minerals) * num_reacting_zones


def test_build_objective_table() -> None:
    """
    The per-run table combines the streamflow row and the species
    rows under the display labels, and degrades gracefully when
    objectives are missing (RT-only and none at all).
    """
    metric_names: list[str] = gui_common.METRIC_NAMES
    metric_labels: list[str] = [
        gui_common.METRIC_LABELS[name] for name in metric_names
    ]

    hydro: pd.Series = pd.Series(
        {name: 1.0 + 0.1 * i for i, name in enumerate(metric_names)},
        dtype=float,
    )
    rt: pd.DataFrame = pd.DataFrame(
        0.5, index=["DOC", "DIC", "Ca++"], columns=metric_names
    )

    # A fully-scored run: the streamflow row plus one row per measured
    # species, in that order, under the display labels.
    table: pd.DataFrame = gui_plots.build_objective_table(hydro, rt)
    assert list(table.columns) == metric_labels
    assert list(table.index) == ["Streamflow", "DOC", "DIC", "Ca++"]
    for i, name in enumerate(metric_names):
        assert (
            table.loc["Streamflow", gui_common.METRIC_LABELS[name]]
            == pytest.approx(1.0 + 0.1 * i)
        )
    assert (
        table.loc["Ca++", gui_common.METRIC_LABELS[metric_names[0]]] == 0.5
    )

    # No measured streamflow: the streamflow row is omitted.
    empty_hydro: pd.Series = pd.Series(dtype=float)
    table_rt_only: pd.DataFrame = gui_plots.build_objective_table(
        empty_hydro, rt
    )
    assert list(table_rt_only.index) == ["DOC", "DIC", "Ca++"]

    # No objectives at all: an empty table that still carries the
    # display labels as columns.
    table_empty: pd.DataFrame = gui_plots.build_objective_table(
        empty_hydro, None
    )
    assert len(table_empty) == 0
    assert list(table_empty.columns) == metric_labels


# ==== End-to-end ==== #
@pytest.mark.slow
def test_app_smoke() -> None:
    """
    The entry script runs end-to-end under streamlit's AppTest: the
    sidebar renders, the models run on the default (midpoint)
    parameters, and the objective table and title appear.

    The first run of the East River model over the full
    (spin-up-inclusive) record is the slow step of the app: it takes
    several minutes, so the timeout below carries ample headroom.
    """
    app_script: Path = Path(gui_data.__file__).with_name("app.py")
    # Ample headroom: the full East River model run alone takes several
    # minutes (the reactive-transport integration is the slow step).
    at: AppTest = AppTest.from_file(app_script, default_timeout=900)

    at.run()

    assert not at.exception
    assert "East River lateral model calibration" in [
        title.value for title in at.title
    ]
    # The sidebar rendered its controls (free parameters as sliders,
    # the three plot selections as multiselects).
    assert len(at.sidebar.slider) > 0
    assert len(at.sidebar.multiselect) == 3
    # The per-run objective table was produced by a successful model
    # run.
    assert len(at.dataframe) == 1
