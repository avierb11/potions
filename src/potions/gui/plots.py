"""
The figures and the objective table of the East River
lateral-model calibration GUI.

Five figures:

1. *Streamflow and river concentrations* — whichever of the
   streamflow and the river concentrations the selection asks for
   (the time-series checkboxes of the parameter panel; by default
   all of them): the simulated streamflow as a line and the measured
   streamflow as scatter points (size 5) on the left axis (inverted,
   so the two panels hang down from the shared top legend) and, on
   the right axis (created only when the run has
   reactive-transport results and at least one river species is
   selected), the simulated river concentrations (the
   ``{species}_riv`` column) as lines and the measured concentrations
   as scatter points (size 5) of DOC, DIC, and Ca++: the color
   encodes the species family and the simulated lines carry the
   lighter shade of it, the measured scatter points the darker one
   (see `potions.gui.common` for the scheme).  The two axis titles
   share one panel, so they are anchored at the left and right edges
   and do not overlap.  The measured streamflow scatter is the CSV's
   ``q_meas`` column (a catchment-mean depth, NaN over the spin-up),
   and the measured concentrations are the sparse observations of
   the CSV record
   (`potions.gui.data.conc_df` of the loaded inputs).

2. *Zone concentrations and storage* — the concentration (mol/L) of
   every selected species in every selected zone in the top panel and
   the selected zones' water storages (the hydro state ``s``, in
   millimeters — the same quantity and units as the calibrated
   ``max_water`` bounds) in the bottom panel, stacked vertically.  The
   zone and species legends are stacked above the figure.  The color
   encodes the zone (the project's lateral-model palette) and the
   linestyle encodes the species.  A hydrology-only run (no
   reactive-transport results) carries the storage panel alone.

3. *Mineral reaction rates* — the simulated reaction rate of every
   selected mineral in every zone that carries the mineral's rate
   columns (mol/L/day): the model emits a ``{mineral}_rate_{zone}``
   column only for zones in which the mineral reactions run.  The
   color encodes the zone and the linestyle encodes the mineral.

4. *Gradients* — the finite-difference gradient matrix of a
   finished gradient job (the window's *Gradients* button): the
   central difference of every objective function with respect to
   every free parameter, drawn as a heatmap (one row per objective
   function, one column per parameter, blue below zero and red
   above, on a scale symmetric about the largest absolute finite
   value) with a color bar and a caption noting the difference
   method and the CSV the matrix was saved to.

5. *Line search* — the value profile of a finished line search
   job (the window's *Line search* button): the searched
   objective's value at every evaluated step ``t`` of the ascent
   line (one marker per point), with the best point (the largest
   finite value) marked in red, and a caption noting the search
   and the CSV the profile was saved to.

The first three figures show the record from the start of water
year 2015 (``potions.gui.common.PLOT_START``) onward, so the
two-year spin-up the simulation prepends to the site record is not
plotted.  Each of them can also carry a previous run's results,
when the window hands its frames in: the previous run's lines are
drawn faintly (half transparent) behind the current run's and stay
out of the legends, so the two can be compared at a glance.

plus a per-run table of objective-function values: one row per
metric and one column per measured target (the streamflow and the
river species), with a note explaining the metrics.
"""

from typing import Final

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.artist import Artist
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.legend import Legend
from matplotlib.lines import Line2D

# ==== Constants ==== #
# The alpha of the faint lines that overlay a previous run's results
# on the current figures (the parameter panel's *History* group,
# *Overlay previous results* checkbox).
_OVERLAY_ALPHA: Final[float] = 0.5
# =================== #

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
    TIME_SERIES_OPTIONS,
    ZONE_COLORS,
    ZONE_LABELS,
    ZONE_ORDER,
)


# ==== Functions ==== #
def make_time_series_figure(
    hydro_sim: pd.DataFrame,
    rt_sim: pd.DataFrame | None,
    q_meas: pd.Series,
    conc_df: pd.DataFrame,
    time_series: list[str] | None = None,
    overlay_hydro_sim: pd.DataFrame | None = None,
    overlay_rt_sim: pd.DataFrame | None = None,
) -> Figure:
    """
    Build the streamflow and river-concentration figure, drawing
    whichever of the selectable series the selection asks for.

    The left axis shows the simulated streamflow as a line and the
    measured streamflow as scatter points (size 5), in millimeters per
    day; its title is anchored at the left edge of the panel.  The
    right axis, sharing the left axis's panel and created only when
    `rt_sim` is not None and at least one river species is selected,
    shows the simulated river concentrations (the
    ``{species}_riv`` columns) as lines and the measured
    concentrations as scatter points (size 5), in moles per liter,
    for each selected species (DOC, DIC, and calcium); its title is
    anchored at the right edge so the two titles do not overlap, and
    the color encodes the species family with the simulated lines in
    the lighter shade and the measured points in the darker one
    (`potions.gui.common`).  All series are clipped to
    ``potions.gui.common.PLOT_START`` (the start of water year 2015),
    so the two-year spin-up the simulation prepends is not shown, and
    the measured series are the observations of the East River series
    CSV record.  The top legend lists the drawn series and is omitted
    when nothing is drawn.

    A previous run's results, when the overlay frames are given, are
    drawn first and faintly (half transparent, unlabeled), so the
    current run's series stay on top and in the legend.

    Args:
        hydro_sim: The hydrology simulation DataFrame.
        rt_sim: The reactive transport simulation DataFrame, or None
            for a hydrology-only run (in which case the right axis is
            not created, even when species are selected).
        q_meas: The measured streamflow over the full record, in
            millimeters per day (catchment-mean depth); NaN during
            the spin-up.
        conc_df: The measured river concentrations over the full
            record, one column per species.
        time_series: The series to draw, by the names of
            `potions.gui.common.TIME_SERIES_OPTIONS` ("Streamflow"
            and the river species); by default every series.  Names
            outside that list are ignored.
        overlay_hydro_sim: The previous run's hydrology simulation,
            for the faint streamflow overlay; None for no overlay.
        overlay_rt_sim: The previous run's reactive-transport
            simulation, for the faint river-concentration overlay;
            None for no overlay (a species it does not carry is
            skipped).

    Returns:
        The completed figure.
    """
    options: list[str] = (
        list(TIME_SERIES_OPTIONS) if time_series is None else time_series
    )
    fig: Figure = Figure(figsize=(15, 7))
    hydro_ax: Axes = fig.gca()

    # Clip to the first plotted water year; see potions.gui.common.PLOT_START.
    hydro_plot: pd.DataFrame = hydro_sim.loc[hydro_sim.index >= PLOT_START]
    rt_plot: pd.DataFrame | None = (
        rt_sim.loc[rt_sim.index >= PLOT_START] if rt_sim is not None else None
    )
    q_meas_plot: pd.Series = q_meas.loc[q_meas.index >= PLOT_START].dropna()
    conc_plot: pd.DataFrame = conc_df.loc[conc_df.index >= PLOT_START]
    overlay_hydro_plot: pd.DataFrame | None = (
        overlay_hydro_sim.loc[overlay_hydro_sim.index >= PLOT_START]
        if overlay_hydro_sim is not None
        else None
    )
    overlay_rt_plot: pd.DataFrame | None = (
        overlay_rt_sim.loc[overlay_rt_sim.index >= PLOT_START]
        if overlay_rt_sim is not None
        else None
    )

    # The previous run's series, drawn first (behind) and faintly,
    # without labels (they stay out of the legend).
    if overlay_hydro_plot is not None:
        hydro_ax.plot(
            overlay_hydro_plot.index,
            overlay_hydro_plot["streamflow_sim"],
            color=SIM_Q_COLOR,
            alpha=_OVERLAY_ALPHA,
            linewidth=1,
        )

    if "Streamflow" in options:
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

    active_species: list[str] = (
        []
        if rt_plot is None
        else [
            species
            for species in RIVER_SPECIES_SIM_COLORS
            if species in options
        ]
    )
    species_ax: Axes | None = hydro_ax.twinx() if active_species else None
    if species_ax is not None and rt_plot is not None:
        if overlay_rt_plot is not None:
            for species in active_species:
                overlay_column: str = f"{species}_riv"
                if overlay_column in overlay_rt_plot.columns:
                    species_ax.plot(
                        overlay_rt_plot.index,
                        overlay_rt_plot[overlay_column],
                        color=RIVER_SPECIES_SIM_COLORS[species],
                        alpha=_OVERLAY_ALPHA,
                        linewidth=1,
                    )
        for species in active_species:
            species_ax.plot(
                rt_plot.index,
                rt_plot[f"{species}_riv"],
                color=RIVER_SPECIES_SIM_COLORS[species],
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
    if species_ax is None:
        species_handles: list[Artist] = []
        species_labels: list[str] = []
    else:
        species_handles, species_labels = species_ax.get_legend_handles_labels()
    if hydro_handles or species_handles:
        fig.legend(
            hydro_handles + species_handles,
            hydro_labels + species_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=5,
        )

    return fig


def make_zone_figure(
    rt_sim: pd.DataFrame | None,
    hydro_sim: pd.DataFrame,
    species: list[str],
    zones: list[str],
    overlay_rt_sim: pd.DataFrame | None = None,
    overlay_hydro_sim: pd.DataFrame | None = None,
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

    With `rt_sim` as None (a hydrology-only run), the figure
    carries the storage panel alone, in its own smaller figure.

    A previous run's results, when the overlay frames are given, are
    drawn first and faintly (half transparent, unlabeled), so the
    current run's series stay on top; a zone or species the previous
    simulation does not carry is skipped.

    Args:
        rt_sim: The reactive transport simulation DataFrame; None
            for a hydrology-only run (the storage panel alone).
        hydro_sim: The hydrology simulation DataFrame, for the
            per-zone storage columns (``s_{zone}``, in millimeters).
        species: The network species to plot.
        zones: The zones to plot.
        overlay_rt_sim: The previous run's reactive-transport
            simulation, for the faint concentration overlay; None for
            no overlay.
        overlay_hydro_sim: The previous run's hydrology simulation,
            for the faint storage overlay; None for no overlay.

    Returns:
        The completed figure.
    """
    # Clip to the first plotted water year; see potions.gui.common.PLOT_START.
    hydro_plot: pd.DataFrame = hydro_sim.loc[hydro_sim.index >= PLOT_START]
    overlay_hydro_plot: pd.DataFrame | None = (
        overlay_hydro_sim.loc[overlay_hydro_sim.index >= PLOT_START]
        if overlay_hydro_sim is not None
        else None
    )

    if rt_sim is None:
        # A hydrology-only run has no reactive-transport
        # simulation: the figure carries the storage panel alone.
        fig: Figure = Figure(figsize=(15, 5))
        storage_ax: Axes = fig.gca()

        # The previous run's storages, drawn first (behind) and
        # faintly, without labels (they stay out of the legend).
        if overlay_hydro_plot is not None:
            for zone in zones:
                overlay_storage: str = f"s_{zone}"
                if overlay_storage in overlay_hydro_plot.columns:
                    storage_ax.plot(
                        overlay_hydro_plot.index,
                        overlay_hydro_plot[overlay_storage],
                        color=ZONE_COLORS[zone],
                        alpha=_OVERLAY_ALPHA,
                        linewidth=1,
                    )

        for zone in zones:
            storage_ax.plot(
                hydro_plot.index,
                hydro_plot[f"s_{zone}"],
                color=ZONE_COLORS[zone],
                linewidth=1,
            )

        storage_ax.set_title("Storage")
        storage_ax.set_ylabel("Storage (mm)")
        storage_ax.set_xlim(PLOT_START_DATE, hydro_plot.index.max())
        fig.subplots_adjust(left=0.07, right=0.96, bottom=0.11, top=0.8)

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
        )
        return fig

    # The reactive-transport record, clipped to the same start.
    rt_plot: pd.DataFrame = rt_sim.loc[rt_sim.index >= PLOT_START]
    overlay_rt_plot: pd.DataFrame | None = (
        overlay_rt_sim.loc[overlay_rt_sim.index >= PLOT_START]
        if overlay_rt_sim is not None
        else None
    )

    fig = Figure(figsize=(15, 9))
    conc_ax: Axes = fig.add_subplot(2, 1, 1)
    storage_ax = fig.add_subplot(2, 1, 2)

    # The previous run's series, drawn first (behind) and faintly,
    # without labels (they stay out of the legend).
    if overlay_rt_plot is not None:
        for zone in zones:
            for i, sp in enumerate(species):
                overlay_column: str = f"{sp}_{zone}"
                if overlay_column in overlay_rt_plot.columns:
                    overlay_linestyle: Linestyle = (
                        LINESTYLE_CYCLE[i % len(LINESTYLE_CYCLE)]
                    )
                    conc_ax.plot(
                        overlay_rt_plot.index,
                        overlay_rt_plot[overlay_column],
                        color=ZONE_COLORS[zone],
                        linestyle=overlay_linestyle,
                        alpha=_OVERLAY_ALPHA,
                        linewidth=1,
                    )
    if overlay_hydro_plot is not None:
        for zone in zones:
            overlay_storage = f"s_{zone}"
            if overlay_storage in overlay_hydro_plot.columns:
                storage_ax.plot(
                    overlay_hydro_plot.index,
                    overlay_hydro_plot[overlay_storage],
                    color=ZONE_COLORS[zone],
                    alpha=_OVERLAY_ALPHA,
                    linewidth=1,
                )

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

    zone_handles = [
        Line2D([], [], color=ZONE_COLORS[zone], linewidth=2) for zone in zones
    ]
    fig.legend(
        zone_handles,
        [ZONE_LABELS[zone] for zone in zones],
        title="Zone",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=max(len(zones), 1),
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
    )

    return fig


def make_rate_figure(
    rt_sim: pd.DataFrame,
    minerals: list[str],
    overlay_rt_sim: pd.DataFrame | None = None,
) -> Figure:
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

    A previous run's results, when the overlay frame is given, is
    drawn first and faintly (half transparent, unlabeled), so the
    current run's series stay on top; a rate column the previous
    simulation does not carry is skipped.

    Args:
        rt_sim: The reactive transport simulation DataFrame.
        minerals: The minerals to plot.
        overlay_rt_sim: The previous run's reactive-transport
            simulation, for the faint rate overlay; None for no
            overlay.

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

    # The previous run's series, drawn first (behind) and faintly,
    # without labels (they stay out of the legend).
    overlay_rt_plot: pd.DataFrame | None = (
        overlay_rt_sim.loc[overlay_rt_sim.index >= PLOT_START]
        if overlay_rt_sim is not None
        else None
    )
    if overlay_rt_plot is not None:
        for i, mineral in enumerate(minerals):
            overlay_linestyle: Linestyle = (
                LINESTYLE_CYCLE[i % len(LINESTYLE_CYCLE)]
            )
            for zone in plot_zones:
                overlay_column: str = f"{mineral}_rate_{zone}"
                if overlay_column in overlay_rt_plot.columns:
                    ax.plot(
                        overlay_rt_plot.index,
                        overlay_rt_plot[overlay_column],
                        color=ZONE_COLORS[zone],
                        linestyle=overlay_linestyle,
                        alpha=_OVERLAY_ALPHA,
                        linewidth=1,
                    )

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
    )

    return fig


def build_objective_table(
    hydro_objectives: pd.Series,
    rt_objectives: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Build the per-run objective-function table.

    The rows are the default objective-function metrics
    (`potions.gui.common.METRIC_NAMES`), renamed to
    `potions.gui.common.METRIC_LABELS` for display, and the columns
    are the measured targets: the streamflow and the river species
    (DOC, DIC, Ca++).  A column is omitted when its objectives are
    unavailable: the streamflow column when no measured streamflow
    was provided to the model (the hydro objectives come back
    empty), and the species columns when the reactive transport
    model reported none.

    Args:
        hydro_objectives: The streamflow objective-function values;
            an empty Series when no measured streamflow was provided
            to the model.
        rt_objectives: The reactive transport objective-function
            values, one row per measured species; None when the
            reactive transport model reported none.

    Returns:
        The table with the metric display labels as rows and the
        target names as columns; an empty table carrying the metric
        display labels when no objectives were reported at all.
    """
    targets: list[str] = []
    columns: dict[str, pd.Series] = {}

    if len(hydro_objectives) > 0:
        targets.append("Streamflow")
        columns["Streamflow"] = hydro_objectives[METRIC_NAMES]

    if rt_objectives is not None and len(rt_objectives) > 0:
        for species in rt_objectives.index:
            name: str = str(species)
            targets.append(name)
            columns[name] = rt_objectives.loc[species].reindex(METRIC_NAMES)

    if targets:
        table: pd.DataFrame = pd.DataFrame(columns, index=METRIC_NAMES)
    else:
        table = pd.DataFrame(index=METRIC_NAMES)

    return table.rename(index=METRIC_LABELS)


# ==== Gradient figure ==== #


def make_gradient_figure(table: pd.DataFrame, note: str) -> Figure:
    """
    The gradient job's matrix, as a color-shaded page of the figure
    panel.

    The page draws the matrix as a heatmap: one row per objective
    function (the row labels of `table`), one column per free
    parameter, each cell shaded by the blue-to-red diverging
    `RdBu_r` colormap on a scale symmetric about zero (the largest
    absolute finite value, or 1.0 when every entry is missing, so
    an all-NaN matrix still shades a neutral color bar).  A
    missing (NaN) entry is left unshaded.  A color bar carries the
    scale, and the job's note is printed under the panel.

    Args:
        table: The gradient matrix, one row per objective function
            and one column per parameter.
        note: The caption printed under the panel (the difference
            method and the path of the saved CSV).

    Returns:
        The figure with the shaded matrix and its color bar.
    """
    fig: Figure = Figure(figsize=(15, 9))
    # The wide left margin leaves room for the 12-point objective
    # row labels ("Streamflow: LogNSE" and its kin) without clipping
    # the figure's left edge; the 55 rotated column labels still fit
    # at this width with no adjacent overlap.
    fig.subplots_adjust(left=0.16, bottom=0.16, right=0.85, top=0.9)
    ax: Axes = fig.gca()

    values: np.ndarray = table.to_numpy(dtype=float)
    finite: np.ndarray = values[np.isfinite(values)]
    scale: float = float(np.max(np.abs(finite))) if finite.size else 1.0

    image: AxesImage = ax.imshow(
        np.ma.masked_invalid(values),
        cmap="RdBu_r",
        aspect="auto",
        vmin=-scale,
        vmax=scale,
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(values.shape[1]))
    ax.set_xticklabels([str(column) for column in table.columns], rotation=90)
    ax.set_yticks(np.arange(values.shape[0]))
    ax.set_yticklabels([str(row) for row in table.index])
    ax.set_title("Gradients of the objective functions")

    colorbar: Colorbar = fig.colorbar(image, ax=ax, fraction=0.06, pad=0.02)
    colorbar.set_label("gradient")

    fig.text(
        0.01,
        0.01,
        note,
        ha="left",
        va="bottom",
        color="gray",
    )

    return fig


# ==== Line search figure ==== #


def make_line_search_figure(
    profile: pd.DataFrame,
    target: str,
    metric: str,
    note: str,
) -> Figure:
    """
    The line search's value profile, as a page of the figure panel.

    The page draws the searched objective's value at every
    evaluated point of the ascent line: the profile's ``value``
    column against the step ``t`` on the horizontal axis, one
    marker per point.  The point with the largest finite value is
    marked in red as the best one, and the job's note is printed
    under the panel.

    Args:
        profile: The value profile, one row per evaluated point:
            the step ``t`` as index, the objective's value in the
            ``value`` column.
        target: The objective's target (measured series) name, for
            the title.
        metric: The objective's metric display label, for the
            title and the vertical axis.
        note: The caption printed under the panel (the search and
            the path of the saved CSV).

    Returns:
        The figure with the profile, its best point, and its note.
    """
    fig: Figure = Figure(figsize=(15, 9))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.14, top=0.9)
    ax: Axes = fig.gca()

    t: np.ndarray = profile.index.to_numpy()
    values: np.ndarray = profile["value"].to_numpy(dtype=np.float64)
    best_index: int = int(np.nanargmax(values))

    ax.plot(t, values, color="k", linewidth=1, marker="o", markersize=3)
    ax.plot(
        [t[best_index]],
        [values[best_index]],
        color="r",
        marker="o",
        markersize=7,
        label="best",
    )
    ax.set_title(f"Line search — {target}: {metric}")
    ax.set_xlabel("t")
    ax.set_ylabel(metric)
    ax.legend(loc="best")

    fig.text(
        0.01,
        0.01,
        note,
        ha="left",
        va="bottom",
        color="gray",
    )

    return fig
# =================== #
