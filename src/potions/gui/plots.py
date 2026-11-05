"""
The figures and the objective table of the East River
lateral-model calibration GUI.

Three figures:

1. *Streamflow and river concentrations* — the simulated streamflow
   as a line and the measured streamflow as scatter points (size 5)
   on the left axis (inverted, so the two panels hang down from the
   shared top legend) and, on the right axis, the simulated river
   concentrations (the ``{species}_riv`` column) as lines and the
   measured concentrations as scatter points (size 5) of DOC, DIC,
   and Ca++: the color encodes the species family and the simulated
   lines carry the lighter shade of it, the measured scatter points
   the darker one (see `potions.gui.common` for the scheme).  The two
   axis titles share one panel, so they are anchored at the left and
   right edges and do not overlap.  The measured streamflow scatter
   is the CSV's ``q_meas`` column (a catchment-mean depth, NaN over
   the spin-up), and the measured concentrations are the sparse
   observations of the CSV record
   (`potions.gui.data.conc_df` of the loaded inputs).

2. *Zone concentrations and storage* — the concentration (mol/L) of
   every selected species in every selected zone in the top panel and
   the selected zones' water storages (the hydro state ``s``, in
   millimeters — the same quantity and units as the calibrated
   ``max_water`` bounds) in the bottom panel, stacked vertically.  The
   zone and species legends are stacked above the figure.  The color
   encodes the zone (the project's lateral-model palette) and the
   linestyle encodes the species.

3. *Mineral reaction rates* — the simulated reaction rate of every
   selected mineral in every zone that carries the mineral's rate
   columns (mol/L/day): the model emits a ``{mineral}_rate_{zone}``
   column only for zones in which the mineral reactions run.  The
   color encodes the zone and the linestyle encodes the mineral.

All three figures show the record from the start of water year 2015
(``potions.gui.common.PLOT_START``) onward, so the two-year spin-up
the simulation prepends to the site record is not plotted.

plus a per-run table of objective-function values for each measured
species, with a note explaining the metrics.
"""

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.lines import Line2D

from potions.gui.common import (
    LINESTYLE_CYCLE,
    Linestyle,
    MEAS_Q_COLOR,
    METRIC_LABELS,
    METRIC_NAMES,
    PLOT_START,
    PLOT_START_DATE,
    RIVER_SPECIES_MEAS_COLORS,
    RIVER_SPECIES_SIM_COLORS,
    SIM_Q_COLOR,
    ZONE_COLORS,
    ZONE_LABELS,
    ZONE_ORDER,
)


# ==== Functions ==== #
def make_time_series_figure(
    hydro_sim: pd.DataFrame,
    rt_sim: pd.DataFrame,
    q_meas: pd.Series,
    conc_df: pd.DataFrame,
) -> Figure:
    """
    Build the streamflow and river-concentration figure.

    The left axis shows the simulated streamflow as a line and the
    measured streamflow as scatter points (size 5), in millimeters per
    day; its title is anchored at the left edge of the panel.  The
    right axis, sharing the left axis's panel, shows the simulated
    river concentrations (the ``{species}_riv`` columns) as lines and
    the measured concentrations as scatter points (size 5), in moles
    per liter, for each measured species (DOC, DIC, and calcium); its
    title is anchored at the right edge so the two titles do not
    overlap, and the color encodes the species family with the
    simulated lines in the lighter shade and the measured points in
    the darker one (`potions.gui.common`).  All series are clipped to
    ``potions.gui.common.PLOT_START`` (the start of water year 2015),
    so the two-year spin-up the simulation prepends is not shown, and
    the measured series are the observations of the East River series
    CSV record.

    Args:
        hydro_sim: The hydrology simulation DataFrame.
        rt_sim: The reactive transport simulation DataFrame.
        q_meas: The measured streamflow over the full record, in
            millimeters per day (catchment-mean depth); NaN during
            the spin-up.
        conc_df: The measured river concentrations over the full
            record, one column per species.

    Returns:
        The completed figure.
    """
    fig: Figure = Figure(figsize=(15, 7))
    hydro_ax: Axes = fig.gca()
    species_ax: Axes = hydro_ax.twinx()

    # Clip to the first plotted water year; see potions.gui.common.PLOT_START.
    hydro_plot: pd.DataFrame = hydro_sim.loc[hydro_sim.index >= PLOT_START]
    rt_plot: pd.DataFrame = rt_sim.loc[rt_sim.index >= PLOT_START]
    q_meas_plot: pd.Series = q_meas.loc[q_meas.index >= PLOT_START].dropna()
    conc_plot: pd.DataFrame = conc_df.loc[conc_df.index >= PLOT_START]

    hydro_ax.plot(
        hydro_plot.index,
        hydro_plot["streamflow_sim"],
        color=SIM_Q_COLOR,
        label="Streamflow (sim)",
        linewidth=1,
    )
    hydro_ax.scatter(
        q_meas_plot.index,
        q_meas_plot,
        s=5,
        color=MEAS_Q_COLOR,
        label="Streamflow (meas)",
    )
    hydro_ax.set_title("Streamflow")
    hydro_ax.set_xlim(PLOT_START_DATE, hydro_plot.index.max())
    hydro_ax.set_ylabel("Discharge (mm/d)")
    hydro_ax.set_ylim(0, 2 * hydro_ax.get_ylim()[1])
    hydro_ax.invert_yaxis()

    for species, sim_color in RIVER_SPECIES_SIM_COLORS.items():
        species_ax.plot(
            rt_plot.index,
            rt_plot[f"{species}_riv"],
            color=sim_color,
            label=f"{species} (sim)",
            linewidth=1,
        )
        measured_sp: pd.Series = conc_plot[species].dropna()
        species_ax.scatter(
            measured_sp.index,
            measured_sp,
            s=5,
            color=RIVER_SPECIES_MEAS_COLORS[species],
            label=f"{species} (meas)",
        )
    species_ax.set_title("River concentrations", loc="right")
    species_ax.set_xlim(PLOT_START_DATE, rt_plot.index.max())
    species_ax.set_ylim(0, 1.5 * species_ax.get_ylim()[1])
    species_ax.set_ylabel("Concentration (mol/L)")

    hydro_handles, hydro_labels = hydro_ax.get_legend_handles_labels()
    species_handles, species_labels = species_ax.get_legend_handles_labels()
    fig.legend(
        hydro_handles + species_handles,
        hydro_labels + species_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=5,
        fontsize=9,
    )

    return fig


def make_zone_figure(
    rt_sim: pd.DataFrame,
    hydro_sim: pd.DataFrame,
    species: list[str],
    zones: list[str],
) -> Figure:
    """
    Build the zone concentrations and storages figure.

    The top panel draws the concentration (mol/L) of every selected
    species in every selected zone, and the bottom panel draws the
    selected zones' water storages: the hydro state ``s`` in
    millimeters, the same quantity and units as the calibrated
    ``max_water`` bounds.  The zone and species legends are stacked
    above the figure, and the record is clipped to
    ``potions.gui.common.PLOT_START`` (the start of water year 2015).
    The color encodes the zone (the project's lateral-model palette)
    and the linestyle encodes the species (the storage panel carries
    one line per zone, so it uses a single linestyle).

    Args:
        rt_sim: The reactive transport simulation DataFrame.
        hydro_sim: The hydrology simulation DataFrame, for the
            per-zone storage columns (``s_{zone}``, in millimeters).
        species: The network species to plot.
        zones: The zones to plot.

    Returns:
        The completed figure.
    """
    fig: Figure = Figure(figsize=(15, 9))
    conc_ax: Axes = fig.add_subplot(2, 1, 1)
    storage_ax: Axes = fig.add_subplot(2, 1, 2)

    # Clip to the first plotted water year; see potions.gui.common.PLOT_START.
    rt_plot: pd.DataFrame = rt_sim.loc[rt_sim.index >= PLOT_START]
    hydro_plot: pd.DataFrame = hydro_sim.loc[hydro_sim.index >= PLOT_START]

    for zone in zones:
        for i, sp in enumerate(species):
            linestyle: Linestyle = LINESTYLE_CYCLE[i % len(LINESTYLE_CYCLE)]
            conc_ax.plot(
                rt_plot.index,
                rt_plot[f"{sp}_{zone}"],
                color=ZONE_COLORS[zone],
                linestyle=linestyle,
                linewidth=1,
            )
        storage_ax.plot(
            hydro_plot.index,
            hydro_plot[f"s_{zone}"],
            color=ZONE_COLORS[zone],
            linewidth=1,
        )

    conc_ax.set_title("Concentration")
    conc_ax.set_ylabel("Concentration (mol/L)")
    conc_ax.set_xlim(PLOT_START_DATE, rt_plot.index.max())
    storage_ax.set_title("Storage")
    storage_ax.set_ylabel("Storage (mm)")
    storage_ax.set_xlim(PLOT_START_DATE, hydro_plot.index.max())

    # Reserve the top of the figure for two stacked figure-level
    # legends (a zone row above a species row).  Stacked panels put a
    # per-axis bottom legend between the panels or below the figure,
    # so the legends are anchored to the figure instead.
    fig.subplots_adjust(left=0.07, right=0.96, bottom=0.07, top=0.76)

    zone_handles: list[Line2D] = [
        Line2D([], [], color=ZONE_COLORS[zone], linewidth=2) for zone in zones
    ]
    fig.legend(
        zone_handles,
        [ZONE_LABELS[zone] for zone in zones],
        title="Zone",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=max(len(zones), 1),
        fontsize=9,
    )

    species_handles: list[Line2D] = [
        Line2D(
            [],
            [],
            color="k",
            linestyle=LINESTYLE_CYCLE[i % len(LINESTYLE_CYCLE)],
            linewidth=2,
        )
        for i in range(len(species))
    ]
    fig.legend(
        species_handles,
        list(species),
        title="Species",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.83),
        ncol=max(len(species), 1),
        fontsize=9,
    )

    return fig


def make_rate_figure(rt_sim: pd.DataFrame, minerals: list[str]) -> Figure:
    """
    Build the mineral reaction-rate figure.

    Every selected mineral is drawn, in moles per liter per day, for
    every zone that carries one of the selected minerals' rate
    columns on shared axes: the model emits a
    ``{mineral}_rate_{zone}`` column only for zones in which
    reactions run, so zones without a column are omitted from both
    the plot and the zone legend.  The color encodes the zone (the
    project's lateral-model palette) and the linestyle encodes the
    mineral, and the record is clipped to
    ``potions.gui.common.PLOT_START`` (the start of water year 2015).

    Args:
        rt_sim: The reactive transport simulation DataFrame.
        minerals: The minerals to plot.

    Returns:
        The completed figure.
    """
    fig: Figure = Figure(figsize=(15, 5))
    ax: Axes = fig.gca()

    # Clip to the first plotted water year; see potions.gui.common.PLOT_START.
    rt_plot: pd.DataFrame = rt_sim.loc[rt_sim.index >= PLOT_START]

    # The model emits a rate column only for zones in which reactions
    # run; plot (and legend) exactly the zones that carry one of the
    # selected minerals' columns.
    plot_zones: list[str] = [
        zone
        for zone in ZONE_ORDER
        if any(
            f"{mineral}_rate_{zone}" in rt_plot.columns for mineral in minerals
        )
    ]

    for i, mineral in enumerate(minerals):
        linestyle: Linestyle = LINESTYLE_CYCLE[i % len(LINESTYLE_CYCLE)]
        for zone in plot_zones:
            col: str = f"{mineral}_rate_{zone}"
            if col in rt_plot.columns:
                ax.plot(
                    rt_plot.index,
                    rt_plot[col],
                    color=ZONE_COLORS[zone],
                    linestyle=linestyle,
                    linewidth=1,
                )

    ax.set_title("Mineral reaction rates")
    ax.set_ylabel("Reaction rate (mol/L/day)")
    ax.set_xlim(PLOT_START_DATE, rt_plot.index.max())

    zone_handles: list[Line2D] = [
        Line2D([], [], color=ZONE_COLORS[zone], linewidth=2) for zone in plot_zones
    ]
    zone_legend: Legend = ax.legend(
        zone_handles,
        [ZONE_LABELS[zone] for zone in plot_zones],
        title="Zone",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=max(len(plot_zones), 1),
        fontsize=8,
    )
    # Keep the zone legend while adding the mineral legend on top.
    ax.add_artist(zone_legend)

    mineral_handles: list[Line2D] = [
        Line2D(
            [],
            [],
            color="k",
            linestyle=LINESTYLE_CYCLE[i % len(LINESTYLE_CYCLE)],
            linewidth=2,
        )
        for i in range(len(minerals))
    ]
    ax.legend(
        mineral_handles,
        list(minerals),
        title="Mineral",
        loc="upper right",
        fontsize=8,
    )

    return fig


def build_objective_table(
    hydro_objectives: pd.Series,
    rt_objectives: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Build the per-run objective-function table.

    The rows are the streamflow and the measured river species and
    the columns are the default objective-function metrics
    (`potions.gui.common.METRIC_NAMES`), renamed to
    `potions.gui.common.METRIC_LABELS` for display.  A row is omitted
    when its objectives are unavailable: the streamflow row when no
    measured streamflow was provided to the model (the hydro
    objectives come back empty), and the species rows when the
    reactive transport model reported none.

    Args:
        hydro_objectives: The streamflow objective-function values;
            an empty Series when no measured streamflow was provided
            to the model.
        rt_objectives: The reactive transport objective-function
            values, one row per measured species; None when the
            reactive transport model reported none.

    Returns:
        The table with the metric display labels as columns; an empty
        table carrying the metric display labels when no objectives
        were reported at all.
    """
    rows: list[pd.DataFrame] = []

    if len(hydro_objectives) > 0:
        hydro_row: pd.DataFrame = pd.DataFrame(
            [[hydro_objectives[name] for name in METRIC_NAMES]],
            index=["Streamflow"],
            columns=METRIC_NAMES,
        )
        rows.append(hydro_row)

    if rt_objectives is not None and len(rt_objectives) > 0:
        rows.append(rt_objectives[METRIC_NAMES])

    if rows:
        table: pd.DataFrame = pd.concat(rows)
    else:
        table = pd.DataFrame(columns=METRIC_NAMES)

    return table.rename(columns=METRIC_LABELS)
# =================== #
