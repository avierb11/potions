"""
The sidebar of the East River lateral-model calibration GUI.

Renders the parameter controls grouped into five expanders:

* *Hydrology* — one slider per zone parameter, in the zone order of
  ``potions.gui.common.ZONE_ORDER``;
* *Catchment proportion* — the single area-fraction parameter that
  splits the catchment between the distal and riparian columns;
* *Reactive transport* — the per-zone storage parameters, plus one
  group of kinetic sliders per mineral in every non-snow zone;
* *Initial state* — the calibrated exchange-site state per zone;
* *Plots* — the species / zone / mineral selections for the figures.

Free parameters (bounds with distinct min and max) are sliders whose
default is the midpoint of the bounds and whose value persists across
reruns through an explicit session key; fixed parameters (identical
bounds) are not shown and are pinned to their value.
"""

from dataclasses import dataclass

import streamlit as st

from potions.gui.common import (
    HYDRO_PARAMS,
    MINERAL_PARAMS,
    RT_DIM_PARAMS,
    SPECIATION_ZONES,
    SNOW_ZONES,
    ZONE_LABELS,
    ZONE_ORDER,
    slider_value,
)
from potions.gui.data import GuiInputData


@dataclass(frozen=True)
class PlotSelection:
    """
    The user's choices of which series to draw.

    Attributes:
        species: The network species to plot in the zone figure.
        zones: The zones to plot in the zone figure.
        minerals: The minerals to plot in the rate figure.
    """

    species: list[str]
    zones: list[str]
    minerals: list[str]


def render_sidebar(
    input_data: GuiInputData,
) -> tuple[dict[str, float], PlotSelection]:
    """
    Render the whole sidebar and collect its values.

    Args:
        input_data: The loaded calibration inputs (parameter bounds
            and the reaction network).

    Returns:
        A mapping of flat parameter name to the value chosen (or
        pinned) on this run, in the bounds' key order, and the plot
        selection.
    """
    bounds: dict[str, tuple[float, float]] = input_data.bounds
    mineral_names: list[str] = input_data.network.mineral_names
    param_values: dict[str, float] = {}

    with st.sidebar.expander("Hydrology", expanded=True):
        for zone in ZONE_ORDER:
            st.subheader(ZONE_LABELS[zone])
            for param in HYDRO_PARAMS[zone]:
                hydro_name: str = f"{zone}.{param}"
                param_values[hydro_name] = slider_value(hydro_name, bounds, mineral_names)

    with st.sidebar.expander("Catchment proportion", expanded=True):
        param_values["proportion.1"] = slider_value("proportion.1", bounds, mineral_names)
        st.caption(
            "Fraction of the catchment area in the distal (hillslope) "
            "column; the riparian column receives the remaining fraction."
        )

    with st.sidebar.expander("Reactive transport", expanded=True):
        for zone in ZONE_ORDER:
            st.subheader(ZONE_LABELS[zone])
            for param in RT_DIM_PARAMS:
                rt_name: str = f"{zone}.{param}"
                param_values[rt_name] = slider_value(rt_name, bounds, mineral_names)

            if zone not in SNOW_ZONES:
                for mineral_id, mineral_name in enumerate(mineral_names):
                    st.markdown(f"**{mineral_name}**")
                    for field in MINERAL_PARAMS:
                        mineral_key: str = f"{zone}.{field}[{mineral_id}]"
                        param_values[mineral_key] = slider_value(mineral_key, bounds, mineral_names)

    with st.sidebar.expander("Initial state", expanded=True):
        for zone in SPECIATION_ZONES:
            init_name: str = f"initial_state.{zone}_x"
            param_values[init_name] = slider_value(init_name, bounds, mineral_names)

    # The plottable species are the network's species, in the
    # network's order: every one of them has per-zone concentration
    # and storage columns in the simulation.
    species_options: list[str] = list(input_data.network.species_names)
    zone_options: list[str] = [ZONE_LABELS[zone] for zone in ZONE_ORDER]
    label_to_zone: dict[str, str] = {
        ZONE_LABELS[zone]: zone for zone in ZONE_ORDER
    }

    with st.sidebar.expander("Plots", expanded=True):
        species_selection: list[str] = st.multiselect(
            "Species (zone figure)",
            options=species_options,
            default=["DOC"],
        )
        selected_zone_labels: list[str] = st.multiselect(
            "Zones (zone figure)",
            options=zone_options,
            default=[ZONE_LABELS[zone] for zone in ZONE_ORDER if zone not in SNOW_ZONES],
        )
        mineral_selection: list[str] = st.multiselect(
            "Minerals (rate figure)",
            options=mineral_names,
            default=["SOC(s)"],
        )

    zone_selection: list[str] = [
        label_to_zone[label] for label in selected_zone_labels
    ]

    return (
        param_values,
        PlotSelection(
            species=species_selection,
            zones=zone_selection,
            minerals=mineral_selection,
        ),
    )
