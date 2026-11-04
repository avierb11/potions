"""
East River lateral-model calibration GUI (Streamlit entry point).

An interactive GUI for calibrating the East River three-layer
lateral (snow / surface / ground) hydrology–reactive-transport model
against the site's forcing-and-chemistry series record
(``potions/gui/data/east_river_series.csv``, the sole data source of
the GUI):

* the sidebar exposes one slider per model parameter that varies
  within its bounds (fixed parameters are not shown) — hydrology
  (per zone), the catchment area proportion, reactive transport (per
  zone and mineral), and the initial exchange-site states — plus the
  plot options;
* every rerun re-runs both models from the flat parameter array
  assembled in the bounds' key order and shows a per-run table of
  objective-function values — with a button that saves the run via
  potions' built-in run saver to a fresh timestamped directory under
  the ``results`` directory of the current working directory —
  followed by the streamflow / river-concentration figure, the zone
  concentration / storage figure, and the mineral reaction-rate
  figure.

The implementation is split across the modules in this package:
`potions.gui.data` (the series-CSV loader and the run functions),
`potions.gui.common` (shared constants, labels, and color schemes),
`potions.gui.sidebar` (the sidebar controls), and `potions.gui.plots`
(the figures and the objective table).  Start the GUI with the
console script installed with the package (see `potions.gui.main`)::

    potions-gui

or directly with::

    streamlit run <path to this file>
"""

from pathlib import Path

import numpy as np
import pandas as pd
import potions as pt
import streamlit as st
from potions.result_saving import save_model_run

from potions.gui.common import METRIC_NOTES, set_rc_params
from potions.gui.data import (
    GuiInputData,
    arr_to_model,
    arr_to_results,
    load_input_data,
)
from potions.gui.plots import (
    build_objective_table,
    make_rate_figure,
    make_time_series_figure,
    make_zone_figure,
)
from potions.gui.sidebar import render_sidebar


# ==== Functions ==== #
def main() -> None:
    """
    Build the sidebar, run the models, and render the results.

    The app script is executed by Streamlit as the ``__main__``
    module and re-executed on every rerun, so this function runs
    once per user interaction; the guard at the bottom of the module
    keeps importing this module side-effect free.
    """
    set_rc_params()
    st.set_page_config(
        page_title="East River lateral model calibration", layout="wide"
    )

    input_data: GuiInputData = load_input_data()

    st.title("East River lateral model calibration")
    st.caption(
        "Adjust any parameter in the sidebar; the hydrology and "
        "reactive-transport models re-run on every change and the "
        "objective table and figures update in place."
    )

    # ==== Sidebar ==== #
    param_values, selection = render_sidebar(input_data)

    # ==== Run ==== #
    x: np.ndarray = np.array(
        [param_values[name] for name in input_data.bounds], dtype=np.float64
    )

    try:
        model_results: pt.ModelResults = arr_to_results(x, input_data)
    except (pt.WaterVolumeError, pt.RtNumericalError) as error:
        st.error(f"Model run failed: {error}")
        st.stop()
        # Reachable only if st.stop() does not terminate the script:
        # let the original solver exception propagate instead.
        raise

    hydro_sim: pd.DataFrame = model_results.hydro.simulation
    rt_sim: pd.DataFrame = model_results.reactive_transport.simulation

    # ==== Objectives ==== #
    st.subheader("Objective function values")
    rt_objectives: pd.DataFrame | None = (
        model_results.reactive_transport.objective_functions
    )
    st.dataframe(
        build_objective_table(
            model_results.hydro.objective_functions, rt_objectives
        ).round(4)
    )
    st.caption(METRIC_NOTES)

    # ==== Save ==== #
    # ModelResults keeps no reference to the model that produced it,
    # and potions' built-in run saver needs one (for the saved
    # parameter table and the run metadata), so on click the model is
    # rebuilt from the same parameter array: arr_to_model is
    # construction only (no integration) and uses the same defaults
    # as the arr_to_results call above, so it is the exact model that
    # produced this run's results.
    #
    # The run is saved with the run saver's built-in default: a
    # fresh ``run_<timestamp>`` directory under the ``results``
    # directory of the current working directory (results_dir=None
    # falls through to that default).
    if st.button("Save run"):
        model: pt.Model
        _init_state_params: np.ndarray
        model, _init_state_params = arr_to_model(x, input_data)
        saved_dir: Path = save_model_run(
            model,
            model_results.hydro,
            model_results.reactive_transport,
            results_dir=None,
        )
        st.success(f"Saved run to {saved_dir}")

    # ==== Plots ==== #
    st.subheader("Streamflow and river concentrations")
    st.pyplot(
        make_time_series_figure(
            hydro_sim, rt_sim, input_data.q_meas, input_data.conc_df
        )
    )

    st.subheader("Zone concentrations and storage")
    st.pyplot(
        make_zone_figure(rt_sim, hydro_sim, selection.species, selection.zones)
    )

    st.subheader("Mineral reaction rates")
    st.pyplot(make_rate_figure(rt_sim, selection.minerals))
# =================== #


if __name__ == "__main__":
    main()
