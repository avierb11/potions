"""
The figures and the objective table of the East River
lateral-model calibration GUI.

The eight figure pages of the full run mode, in page order (the
hydrology-only run mode shows the first three):

1. *Hydrology* — the simulated river discharge (mm/day) as a line
   and the measured river discharge as scatter points (size 5), per
   the plotting guidelines: observations are always scatter,
   simulations are always lines.

2. *Storage* — every selected zone's hydro state ``s`` (in
   millimeters of water, the same quantity and units as the
   calibrated ``max_water`` bounds), as zone-colored lines.

3. *Partitioning* — the simulated streamflow as the stack of the
   lateral components of every river zone discharging to the river,
   as filled bands between the zones' cumulative discharges (ground
   under surface), with the measured streamflow as scatter points on
   the same axis.

4. *River concentrations* — the simulated concentrations and the
   measured scatter points (size 5) of every selected river species
   on the right axis, and, when the selection's streamflow toggle is
   on, the simulated and measured streamflow on an inverted left
   axis.  The right axis's colors encode the species family and the
   simulated lines carry the lighter shade of it, the measured
   scatter points the darker one (see `potions.gui.common` for the
   scheme).

5. *River loads* — the time series of each selected river species'
   mass load through the river: the simulated concentration times
   the river discharge, simulated values only (see the builder's
   docstring for why there are no observed loads).

6. *Concentration vs. discharge* — the log-log scatter of
   concentration against discharge for every selected river species:
   one panel per species, the simulated points paired with the
   simulated discharge and the measured points with the measured
   discharge.

7. *Zone concentrations* — the concentration (mol/L) of every
   selected species in every selected zone: the color encodes the
   zone and the linestyle encodes the species.

8. *Reaction rates* — the simulated reaction rate (mol/L/day) of the
   single mineral the panel selects, in every zone that carries the
   mineral's rate columns (the model emits those only for zones in
   which the mineral's reactions run).

Every figure is clipped to the plotting window — the span between
the panel's two water-year pickers, October 1 to September 30 per
the plotting guidelines — so a range that ends after the record's
last day (the partial 2025 water year) still plots only the days the
record carries.  Each can also carry a previous run's results, when
the window hands its frames in: the previous run's lines are drawn
first and faintly (half transparent) behind the current run's
series and stay out of the legends, so the two can be compared at a
glance.

plus two per-run diagnostic pages, shown after the mode's own
figures, once their jobs have finished:

* *Gradients* — the finite-difference gradient matrix of a
  finished gradient job (the window's *Gradients* button): the
  central difference of every objective function with respect to
  every free parameter, drawn as a heatmap (one row per objective
  function, one column per parameter, blue below zero and red
  above, on a scale symmetric about the largest absolute finite
  value) with a color bar and a caption noting the difference
  method and the CSV the matrix was saved to.

* *Line search* — the value profile of a finished line search
  job (the window's *Line search* button): the searched
  objective's value at every evaluated step ``t`` of the ascent
  line (one marker per point), with the best point (the largest
  finite value) marked in red, and a caption noting the search
  and the CSV the profile was saved to.

and a per-run table of objective-function values: one row per
metric and one column per measured target (the streamflow and the
river species), with a note explaining the metrics.
"""

from collections.abc import Sequence
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
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.transforms import Bbox

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
    RIVER_SPECIES_MEAS_COLORS,
    RIVER_SPECIES_SIM_COLORS,
    SIM_Q_COLOR,
    ZONE_COLORS,
    ZONE_LABELS,
    ZONE_ORDER,
    ZONE_STACK_ORDER,
)


# ==== Functions ==== #
def _day_number(timestamp: pd.Timestamp) -> float:
    """
    The matplotlib date number (days since 1970-01-01) of `timestamp`.

    ``Axes.set_xlim`` is typed to accept floats only, so the figure
    builders convert their plot-window boundaries with this helper
    instead of passing Timestamps.
    """
    return float(timestamp.value / 1e9 / 86400)


def _clip_frame(
    frame: pd.DataFrame, window: tuple[pd.Timestamp, pd.Timestamp]
) -> pd.DataFrame:
    """
    `frame` restricted to the plotting window's rows.

    The window is intersected with the record: rows before its start
    or after its end are dropped, and rows whose index falls outside
    the record (a range ending after the record's last day) simply do
    not exist.  Missing (NaN) values inside the window are kept; the
    builders decide what to do with them.
    """
    start, end = window
    return frame.loc[(frame.index >= start) & (frame.index <= end)]


def _clip_series(
    series: pd.Series, window: tuple[pd.Timestamp, pd.Timestamp]
) -> pd.Series:
    """
    `series` restricted to the plotting window's rows.

    See `_clip_frame` for the intersection and NaN semantics.
    """
    start, end = window
    return series.loc[(series.index >= start) & (series.index <= end)]


# The figure-fraction gaps that keep the legend rows inside the
# figure and out of each other: the floor of the lowest legend row
# above the figure's bottom edge, the spacing between stacked
# rows, and the margin the axes' bottom edge keeps beyond the
# deepest label hanging below it.
_LEGEND_BOTTOM_GAP: Final[float] = 0.02
_LEGEND_ROW_GAP: Final[float] = 0.03
_TICK_LABEL_MARGIN: Final[float] = 0.015


def _legend_top_edge(fig: Figure, legend: Legend) -> float:
    """
    The top edge of a drawn legend, in figure fractions.

    The extent is measured with the figure's own canvas renderer, so
    the result is a pure figure fraction: the panel's later dpi
    restatement rescales both the extent and the figure size by the
    same factor, and the placement stays stable.
    """
    extent: Bbox = legend.get_window_extent()
    fig_height: float = fig.get_figheight() * fig.dpi
    return float(extent.y1) / fig_height


def _axes_tick_depth(fig: Figure) -> float:
    """
    The deepest label hanging below the figure's axes' bottom
    edges, in figure fractions.

    The x tick labels — and an axis's x label, when it carries
    one — hang below the axes' bottom edge in a point-based depth,
    while the builders design the layout in figure fractions: the
    panel's rescale changes the figure's inches per fraction, so a
    fraction reserved at the construction size does not survive it.
    The depth is measured against the figure's current window
    extent, after a draw at that size (the same contract
    `relayout_below_axes_legends` already has).

    Every extent call is passed the figure's own renderer: a
    `Text` caches the renderer of its last draw, and a tick the
    axis skips (its position falls outside the drawn window)
    never draws, keeping the throwaway renderer of the
    construction.  A label's layout cache is keyed on the
    figure's dpi but filled in the cached renderer's, so a
    measurement against a stale renderer reports a wrong-dpi
    box, and the deepest stale one would over-reserve the gap
    the builders and the relayout compute from this depth,
    shrinking the drawn plot area.  The figure's own renderer
    always carries the figure's current dpi, and passing it
    explicitly restates each label's layout at that dpi (healing
    the stale caches as a side effect).

    Args:
        fig: The figure whose axes to measure.

    Returns:
        The largest distance, in figure fractions, from an axes'
        bottom edge to the lowest edge of a label hanging below
        it; zero when no axes carries a label hanging below it.
    """
    fig_height: float = float(fig.get_window_extent().height)
    # The figure's own canvas renderer, whose dpi always equals
    # the figure's current dpi (the canvas makes a fresh one
    # whenever the last drawn size or dpi changed), or a
    # throwaway at the same dpi for a figure without a drawing
    # canvas.  The accessor is private, hence the ignore.
    renderer = fig._get_renderer()  # type: ignore[attr-defined]
    depth: float = 0.0
    for ax in fig.axes:
        ax_bbox: Bbox = ax.get_window_extent()
        labels: list[Text] = [*ax.get_xticklabels(), ax.xaxis.get_label()]
        for label in labels:
            if not label.get_visible():
                continue
            extent: Bbox = label.get_window_extent(renderer)
            # A label with empty text carries a zero-height box
            # and no depth; a non-empty one, drawn or not,
            # carries a valid current-dpi box, which is exactly
            # what the margin needs.
            if extent.height > 0.0:
                depth = max(depth, float(ax_bbox.y0 - extent.y0) / fig_height)
    return depth


def _axes_gap(fig: Figure) -> float:
    """
    The figure-fraction gap the axes' bottom edge keeps above the
    topmost below-axes legend row.

    At least `_LEGEND_ROW_GAP`, the builders' margin, grown to
    clear the labels that hang below the axes' bottom edge: a
    fixed gap does not survive the panel's rescale of the
    point-based tick labels, and one too small for them lets a
    tick label sit on the legend row beneath the axes.

    Args:
        fig: The figure whose axes' hanging labels to clear.

    Returns:
        The gap, in figure fractions, for the builders'
        `Figure.subplots_adjust` reservation and the relayout's
        restated margin.
    """
    return max(
        _LEGEND_ROW_GAP,
        _axes_tick_depth(fig) + _TICK_LABEL_MARGIN,
    )


def _attach_legend(
    fig: Figure,
    handles: Sequence[Artist],
    labels: Sequence[str],
    *,
    title: str | None = None,
    ncol: int = 1,
    above: Legend | None = None,
) -> Legend:
    """
    A frameless legend row placed below the drawn axes.

    The row obeys the project's plotting guidelines: the legend
    frame is off, and the legend never overlaps the drawn series.
    It is created once and drawn (with the figure's canvas) to
    measure its display height, then anchored so that its bottom
    edge sits `_LEGEND_BOTTOM_GAP` above the figure's bottom edge —
    or `_LEGEND_ROW_GAP` above the `above` row, when stacking a
    second row (the builders reserve the room with
    `Figure.subplots_adjust` after placing their rows).  Because the
    series are drawn inside the axes and the legend below them, they
    cannot overlap, and because the legend stays inside the figure
    bounds, the figure panel's page export (a fixed-size `savefig`,
    no tight bounding box) keeps it in the saved PNG.

    Args:
        fig: The figure to attach the legend to.
        handles: The legend handles, in display order.
        labels: The legend labels, one per handle.
        title: The legend's title, or None for none.
        ncol: The number of columns of the legend.
        above: The legend row this one stacks on, or None for the
            lowest row.

    Returns:
        The attached legend, its top edge at
        `_legend_top_edge(fig, ...)` for the builders' layout math.
    """
    legend: Legend = fig.legend(
        handles,
        labels,
        title=title,
        ncol=ncol,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.5),
    )
    # The tag marks this as a below-axes row: when the panel
    # rescales the figure to fit its canvas, it re-lays out the
    # tagged rows (`relayout_below_axes_legends`) and leaves the
    # figure's other legends (the line search page's title legend)
    # where the builders placed them.  Private attribute, since
    # matplotlib has no flag of its own for it.
    legend._below_axes_row = True  # type: ignore[attr-defined]
    fig.canvas.draw()
    if above is not None:
        floor: float = _legend_top_edge(fig, above) + _LEGEND_ROW_GAP
    else:
        floor = _LEGEND_BOTTOM_GAP
    height: float = (
        float(legend.get_window_extent().height)
        / (fig.get_figheight() * fig.dpi)
    )
    # Re-anchor at the measured position: the anchor is the row's
    # top edge, `_LEGEND_*_GAP` above the floor.
    legend.set_bbox_to_anchor((0.5, floor + height))
    return legend


def relayout_below_axes_legends(fig: Figure) -> None:
    """
    Re-place the figure's below-axes legend rows for its current
    size, and restate the axes' bottom margin to match.

    The builders anchor the rows and reserve the margin in figure
    fractions measured at the construction size, but the panel
    rescales the figure to fit its canvas, and a point-based row
    occupies a different fraction after the rescale: left alone,
    the lowest row sinks past the figure's bottom edge (and out of
    the panel's fixed-size page export).  The relayout re-measures
    each tagged row in stacking order, from the bottom up,
    re-anchors it to the floor the builders designed, and restates
    the axes' bottom margin — `_axes_gap`: the builders' row gap,
    grown to clear the axes' point-based hanging labels — so the
    rows stay inside the figure, out of the tick labels, at any
    size the panel gives it.

    Matplotlib applies a row's anchor to its outer box, which sits
    a small constant gap above the row's drawn extent; the gap is
    measured against the row's current anchor and carried into the
    new one, so each row's bottom edge lands exactly on the floor.

    The caller must have drawn the figure at its current size
    first, so the rows' and the axes' labels' extents are measured
    against a canvas laid out at that size.

    Args:
        fig: The figure whose rows to re-layout.
    """
    rows: list[Legend] = [
        legend
        for legend in fig.legends
        if getattr(legend, "_below_axes_row", None) is True
    ]
    if not rows:
        return
    # Both extents come from the figure's renderer, so their ratio
    # is a pure figure fraction at whatever size the figure now is.
    fig_height: float = float(fig.get_window_extent().height)
    floor: float = _LEGEND_BOTTOM_GAP
    for legend in rows:
        extent: Bbox = legend.get_window_extent()
        height: float = float(extent.height) / fig_height
        # The two-tuple anchor is a figure fraction; the drawn
        # extent's top edge sits the constant outer-box gap below
        # it, measured here at the row's current anchor.
        anchor: float = float(legend.get_bbox_to_anchor().y0) / fig_height
        gap: float = anchor - float(extent.y1) / fig_height
        # The anchor is the row's top edge, `_LEGEND_*_GAP` above
        # the floor, as at construction.
        legend.set_bbox_to_anchor((0.5, floor + height + gap))
        floor = floor + height + _LEGEND_ROW_GAP
    # The builders' margin above the topmost row, restated at the
    # measured position: at least `_LEGEND_ROW_GAP`, grown to
    # clear the axes' hanging labels (`_axes_gap`).
    fig.subplots_adjust(bottom=floor - _LEGEND_ROW_GAP + _axes_gap(fig))


def make_hydrology_figure(
    hydro_sim: pd.DataFrame,
    q_meas: pd.Series,
    window: tuple[pd.Timestamp, pd.Timestamp],
    overlay_hydro_sim: pd.DataFrame | None = None,
) -> Figure:
    """
    Build the hydrology figure.

    The simulated river discharge (``streamflow_sim``) is drawn as a
    line and the measured river discharge (``q_meas``) as scatter
    points (size 5), in millimeters per day, on a single upright
    axis: the observations are always scatter points and the
    simulation is always a line.  The record is clipped to the
    plotting `window`, so a range that ends after the record's last
    day (the partial 2025 water year) still plots only the days the
    record carries.

    A previous run's results, when the overlay frame is given, are
    drawn first and faintly (half transparent, unlabeled), so the
    current run's streamflow stays on top.

    The frameless streamflow legend sits below the axes, inside the
    figure, so it never overlaps the drawn series and stays in the
    figure panel's page export.

    Args:
        hydro_sim: The hydrology simulation DataFrame, for the
            simulated river discharge (``streamflow_sim``).
        q_meas: The measured river discharge over the full record, in
            millimeters per day (catchment-mean depth); NaN values
            do not plot.
        window: The plotting window, as
            `potions.gui.common.water_year_window` returns it.
        overlay_hydro_sim: The previous run's hydrology simulation,
            for the faint streamflow overlay; None for no overlay.

    Returns:
        The completed figure.
    """
    hydro_plot: pd.DataFrame = _clip_frame(hydro_sim, window)
    q_meas_plot: pd.Series = _clip_series(q_meas, window).dropna()
    overlay_hydro_plot: pd.DataFrame | None = (
        _clip_frame(overlay_hydro_sim, window)
        if overlay_hydro_sim is not None
        else None
    )

    fig: Figure = Figure(figsize=(15, 5))
    ax: Axes = fig.gca()

    # The previous run's streamflow, drawn first (behind) and
    # faintly, without a label (it stays out of the legend).
    if overlay_hydro_plot is not None:
        ax.plot(
            overlay_hydro_plot.index,
            overlay_hydro_plot["streamflow_sim"],
            color=SIM_Q_COLOR,
            alpha=_OVERLAY_ALPHA,
            linewidth=1,
        )

    ax.plot(
        hydro_plot.index,
        hydro_plot["streamflow_sim"],
        color=SIM_Q_COLOR,
        linewidth=1,
    )
    ax.scatter(
        q_meas_plot.index,
        q_meas_plot,
        s=5,
        color=MEAS_Q_COLOR,
    )

    ax.set_title("Streamflow")
    ax.set_ylabel("Discharge (mm/d)")
    ax.set_xlim(
        _day_number(window[0]),
        _day_number(hydro_plot.index.max()),
    )
    ax.set_ylim(0, 2 * ax.get_ylim()[1])
    fig.subplots_adjust(left=0.07, right=0.96, top=0.88)

    legend: Legend = _attach_legend(
        fig,
        [
            Line2D([], [], color=SIM_Q_COLOR, linewidth=2),
            Line2D(
                [],
                [],
                marker="o",
                color=MEAS_Q_COLOR,
                linestyle="none",
                markersize=3,
            ),
        ],
        ["Streamflow (sim)", "Streamflow (meas)"],
    )
    fig.subplots_adjust(bottom=_legend_top_edge(fig, legend) + _axes_gap(fig))

    return fig


def make_storage_figure(
    hydro_sim: pd.DataFrame,
    zones: list[str],
    window: tuple[pd.Timestamp, pd.Timestamp],
    overlay_hydro_sim: pd.DataFrame | None = None,
    zone_labels: dict[str, str] = ZONE_LABELS,
    zone_colors: dict[str, str] = ZONE_COLORS,
) -> Figure:
    """
    Build the per-zone water-storage figure.

    Every selected zone's hydro state ``s`` is drawn, in millimeters
    of water (the same quantity and units as the calibrated
    ``max_water`` bounds), as a line on a single shared axis: the
    color encodes the zone (the project's lateral-model palette),
    and the record is clipped to the plotting `window`, so a range
    that ends after the record's last day (the partial 2025 water
    year) still plots only the days the record carries.

    A previous run's results, when the overlay frame is given, are
    drawn first and faintly (half transparent, unlabeled), so the
    current run's series stay on top; a zone the previous simulation
    does not carry is skipped.

    The frameless zone legend sits below the axes, inside the
    figure, so it never overlaps the drawn series and stays in the
    figure panel's page export.

    Args:
        hydro_sim: The hydrology simulation DataFrame, for the
            per-zone storage columns (``s_{zone}``, in millimeters).
        zones: The zones to plot.
        window: The plotting window, as
            `potions.gui.common.water_year_window` returns it.
        overlay_hydro_sim: The previous run's hydrology simulation,
            for the faint storage overlay; None for no overlay.
        zone_labels: The display name of every zone; defaults to the
            lateral model's labels.
        zone_colors: The line color of every zone; defaults to the
            lateral model's palette.

    Returns:
        The completed figure.
    """
    hydro_plot: pd.DataFrame = _clip_frame(hydro_sim, window)
    overlay_hydro_plot: pd.DataFrame | None = (
        _clip_frame(overlay_hydro_sim, window)
        if overlay_hydro_sim is not None
        else None
    )

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
                    color=zone_colors[zone],
                    alpha=_OVERLAY_ALPHA,
                    linewidth=1,
                )

    for zone in zones:
        storage_ax.plot(
            hydro_plot.index,
            hydro_plot[f"s_{zone}"],
            color=zone_colors[zone],
            linewidth=1,
        )

    storage_ax.set_title("Storage")
    storage_ax.set_ylabel("Storage (mm)")
    storage_ax.set_xlim(
        _day_number(window[0]),
        _day_number(hydro_plot.index.max()),
    )
    fig.subplots_adjust(left=0.07, right=0.96, top=0.88)

    legend: Legend = _attach_legend(
        fig,
        [Line2D([], [], color=zone_colors[zone], linewidth=2) for zone in zones],
        [zone_labels[zone] for zone in zones],
        title="Zone",
        ncol=max(len(zones), 1),
    )
    fig.subplots_adjust(bottom=_legend_top_edge(fig, legend) + _axes_gap(fig))

    return fig


def _river_zone_stack_order(
    hydro_sim: pd.DataFrame,
    stack_order: list[str] = ZONE_STACK_ORDER,
) -> list[str]:
    """
    The river zones a hydrology frame partitions, in stack order.

    A zone partitions the streamflow when the frame carries its
    ``prop_q_{zone}`` column (the same discovery the model's
    `river_zone_fractions` accessor uses).  The stack order puts the
    deepest zones under the shallower ones, in ``stack_order``'s
    order, so the stack reads bottom-up the way the figure spec
    describes it (the ground components first, then the surface).
    A discovered zone the order does not name goes last.

    Args:
        hydro_sim: The hydrology simulation frame.
        stack_order: The zones deepest-layer first; defaults to the
            lateral model's order.

    Returns:
        The river zones, in stack order.
    """
    river_zones: list[str] = [
        name.removeprefix("prop_q_")
        for name in hydro_sim.columns
        if name.startswith("prop_q_")
    ]
    rank: dict[str, int] = {
        zone: index for index, zone in enumerate(stack_order)
    }
    return sorted(
        river_zones,
        key=lambda zone: rank.get(zone, len(stack_order)),
    )


def _draw_river_stack(
    ax: Axes,
    frame: pd.DataFrame,
    zones: list[str],
    alpha: float | None,
    zone_colors: dict[str, str] = ZONE_COLORS,
) -> None:
    """
    Draw a frame's stacked river-zone streamflow components on `ax`.

    Each zone the frame carries (its ``prop_q_{zone}`` column) adds
    a filled band between the cumulative discharge so far and the
    cumulative discharge including the zone's component (its share
    of the streamflow, ``prop_q_{zone} × streamflow_sim``, equal to
    the zone's lateral discharge to the river).  The bands are
    drawn in the zones' stack order, so the ground components fill
    the bottom of the stack and the surface components the top.  A
    zone the frame does not carry is skipped.

    Args:
        ax: The axes to draw on.
        frame: A hydrology simulation frame (clipped to the plot
            window), carrying ``streamflow_sim`` and the zones'
            ``prop_q_{zone}`` columns.
        zones: The river zones to stack, in stack order.
        alpha: The band transparency; None for opaque bands.
        zone_colors: The band color of every zone; defaults to the
            lateral model's palette.
    """
    bottom: pd.Series = pd.Series(0.0, index=frame.index)
    for zone in zones:
        component: str = f"prop_q_{zone}"
        if component not in frame.columns:
            continue
        share: pd.Series = frame[component] * frame["streamflow_sim"]
        ax.fill_between(
            frame.index,
            bottom,
            bottom + share,
            color=zone_colors[zone],
            alpha=alpha,
        )
        bottom = bottom + share


def make_partitioning_figure(
    hydro_sim: pd.DataFrame,
    q_meas: pd.Series,
    window: tuple[pd.Timestamp, pd.Timestamp],
    overlay_hydro_sim: pd.DataFrame | None = None,
    zone_labels: dict[str, str] = ZONE_LABELS,
    zone_colors: dict[str, str] = ZONE_COLORS,
    stack_order: list[str] = ZONE_STACK_ORDER,
) -> Figure:
    """
    Build the discharge-partitioning figure.

    The simulated streamflow is drawn as the stack of the lateral
    components of every river zone discharging to the river, as
    filled bands between the zones' cumulative discharges (ground
    under surface, per the figure spec).  The measured streamflow
    is drawn as scatter points on the same axis, as all
    observations in the panel are, and the record is clipped to the
    plotting `window`, so a range that ends after the record's last
    day (the partial 2025 water year) still plots only the days the
    record carries.

    The river zones are discovered from the ``prop_q_`` columns the
    hydrology simulation carries (the same scan the model's
    `river_zone_fractions` accessor uses), so the figure tracks
    whichever zones the model partitions.

    A previous run's results, when the overlay frame is given, are
    drawn first and faintly (half transparent, unlabeled) behind
    the current stack; a zone the previous simulation does not
    carry is skipped.

    The frameless zone and discharge legend rows sit below the
    axes, inside the figure, so they never overlap the drawn
    series and stay in the figure panel's page export.

    Args:
        hydro_sim: The hydrology simulation DataFrame.
        q_meas: The measured streamflow over the full record, in
            millimeters per day; missing (NaN) rows are dropped
            before the scatter is drawn.
        window: The plotting window, as
            `potions.gui.common.water_year_window` returns it.
        overlay_hydro_sim: The previous run's hydrology simulation,
            for the faint stack overlay; None for no overlay.
        zone_labels: The display name of every zone; defaults to the
            lateral model's labels.
        zone_colors: The band color of every zone; defaults to the
            lateral model's palette.
        stack_order: The zones deepest-layer first, which sets the
            stack order of the bands; defaults to the lateral model's
            order.

    Returns:
        The completed figure.
    """
    hydro_plot: pd.DataFrame = _clip_frame(hydro_sim, window)
    q_meas_plot: pd.Series = _clip_series(q_meas, window).dropna()
    overlay_hydro_plot: pd.DataFrame | None = (
        _clip_frame(overlay_hydro_sim, window)
        if overlay_hydro_sim is not None
        else None
    )

    zones: list[str] = _river_zone_stack_order(hydro_plot, stack_order)

    fig: Figure = Figure(figsize=(15, 5))
    ax: Axes = fig.gca()

    # The previous run's stack, drawn first (behind) and faintly,
    # without labels (they stay out of the legends).
    if overlay_hydro_plot is not None:
        _draw_river_stack(
            ax,
            overlay_hydro_plot,
            zones,
            alpha=_OVERLAY_ALPHA,
            zone_colors=zone_colors,
        )

    _draw_river_stack(ax, hydro_plot, zones, alpha=None, zone_colors=zone_colors)

    # The measured streamflow: scatter points, never a line.
    ax.scatter(
        q_meas_plot.index,
        q_meas_plot,
        s=5,
        color=MEAS_Q_COLOR,
    )

    ax.set_title("Partitioning")
    ax.set_ylabel("Discharge (mm/d)")
    ax.set_xlim(
        _day_number(window[0]),
        _day_number(hydro_plot.index.max()),
    )
    ax.set_ylim(bottom=0.0)
    fig.subplots_adjust(left=0.07, right=0.96, top=0.88)

    zone_row: Legend | None = None
    if zones:
        zone_row = _attach_legend(
            fig,
            [
                Patch(facecolor=zone_colors[zone], edgecolor="none")
                for zone in zones
            ],
            [zone_labels[zone] for zone in zones],
            title="Zone",
            ncol=max(len(zones), 1),
        )
    discharge_row: Legend = _attach_legend(
        fig,
        [
            Line2D(
                [],
                [],
                marker="o",
                color=MEAS_Q_COLOR,
                linestyle="none",
                markersize=3,
            )
        ],
        ["Discharge (meas)"],
        title="Discharge",
        above=zone_row,
    )
    fig.subplots_adjust(
        bottom=_legend_top_edge(fig, discharge_row) + _axes_gap(fig)
    )

    return fig


def make_river_loads_figure(
    rt_sim: pd.DataFrame,
    hydro_sim: pd.DataFrame,
    species: list[str],
    window: tuple[pd.Timestamp, pd.Timestamp],
    overlay_rt_sim: pd.DataFrame | None = None,
    overlay_hydro_sim: pd.DataFrame | None = None,
) -> Figure:
    """
    Build the river loads figure.

    Each selected river species' mass load through the river — its
    simulated concentration (the ``{species}_riv`` column) times the
    simulated river discharge (``streamflow_sim``, the model's
    ``doc_load`` convention: load = concentration x discharge) — is
    drawn as a line, in the species' color.

    The figure carries the simulated loads only: the river species
    have no continuous observed concentration record at the site, so
    an observed load would not be a measurement (an approved
    deviation from the figure spec, which calls for both).

    The record is clipped to the plotting `window`, so a range that
    ends after the record's last day (the partial 2025 water year)
    still plots only the days the record carries.

    A previous run's results, when the overlay frames are given, are
    drawn first and faintly (half transparent, unlabeled) behind the
    current run's loads; a species the previous simulation does not
    carry is skipped.

    The frameless species legend sits below the axes, inside the
    figure, so it never overlaps the drawn series and stays in the
    figure panel's page export.

    Args:
        rt_sim: The reactive-transport simulation DataFrame, for the
            river concentrations (``{species}_riv``).
        hydro_sim: The hydrology simulation DataFrame, for the
            simulated river discharge (``streamflow_sim``).
        species: The river species to plot.
        window: The plotting window, as
            `potions.gui.common.water_year_window` returns it.
        overlay_rt_sim: The previous run's reactive-transport
            simulation, for the faint loads overlay; None for no
            overlay.
        overlay_hydro_sim: The previous run's hydrology simulation,
            for the faint loads overlay; None for no overlay.

    Returns:
        The completed figure.
    """
    rt_plot: pd.DataFrame = _clip_frame(rt_sim, window)
    hydro_plot: pd.DataFrame = _clip_frame(hydro_sim, window)
    overlay_rt_plot: pd.DataFrame | None = (
        _clip_frame(overlay_rt_sim, window)
        if overlay_rt_sim is not None
        else None
    )
    overlay_hydro_plot: pd.DataFrame | None = (
        _clip_frame(overlay_hydro_sim, window)
        if overlay_hydro_sim is not None
        else None
    )

    fig: Figure = Figure(figsize=(15, 5))
    ax: Axes = fig.gca()

    # The previous run's loads, drawn first (behind) and faintly,
    # without labels (they stay out of the legend).
    if overlay_rt_plot is not None and overlay_hydro_plot is not None:
        for species_name in species:
            load_column: str = f"{species_name}_riv"
            if load_column not in overlay_rt_plot.columns:
                continue
            ax.plot(
                overlay_hydro_plot.index,
                overlay_rt_plot[load_column]
                * overlay_hydro_plot["streamflow_sim"],
                color=RIVER_SPECIES_SIM_COLORS[species_name],
                alpha=_OVERLAY_ALPHA,
                linewidth=1,
            )

    for species_name in species:
        load_column = f"{species_name}_riv"
        if load_column not in rt_plot.columns:
            continue
        ax.plot(
            hydro_plot.index,
            rt_plot[load_column] * hydro_plot["streamflow_sim"],
            color=RIVER_SPECIES_SIM_COLORS[species_name],
            linewidth=1,
        )

    ax.set_title("River loads")
    ax.set_ylabel("Load (mm·mol/L/d)")
    ax.set_xlim(
        _day_number(window[0]),
        _day_number(hydro_plot.index.max()),
    )
    fig.subplots_adjust(left=0.07, right=0.96, top=0.88)

    legend: Legend = _attach_legend(
        fig,
        [
            Line2D([], [], color=RIVER_SPECIES_SIM_COLORS[name], linewidth=2)
            for name in species
        ],
        list(species),
        title="Species",
        ncol=max(len(species), 1),
    )
    fig.subplots_adjust(bottom=_legend_top_edge(fig, legend) + _axes_gap(fig))

    return fig


def make_conc_discharge_figure(
    rt_sim: pd.DataFrame,
    hydro_sim: pd.DataFrame,
    q_meas: pd.Series,
    conc_df: pd.DataFrame,
    species: list[str],
    window: tuple[pd.Timestamp, pd.Timestamp],
) -> Figure:
    """
    Build the concentration-discharge figure.

    One log-log panel per selected river species, showing the
    relationship between the river discharge and the species'
    concentration.  The simulated points pair the simulated
    discharge (``streamflow_sim``) with the simulated river
    concentration (the ``{species}_riv`` column), and the observed
    points pair the measured discharge (``q_meas``) with the measured
    concentration (the species' column of `conc_df`): each cloud
    compares like with like, so the panel shows the relationship the
    model produces next to the one the record shows.  The simulated
    points carry the lighter species shade and the observed points
    the darker one, as all other figures do.

    Both axes are log-scaled, so only strictly positive pairs plot:
    a pair with a zero or negative discharge or concentration is
    dropped, along with every missing (NaN) observation.  The record
    is clipped to the plotting `window` first.

    This figure carries no previous-run overlay: it is a diagnostic
    comparison of one run's simulated cloud against the record's
    observed cloud, and a second run's cloud would only clutter the
    relationship.

    The frameless simulated and measured legend rows sit below the
    panels, inside the figure, so they never overlap the drawn
    points and stay in the figure panel's page export.

    Args:
        rt_sim: The reactive-transport simulation DataFrame, for the
            river concentrations (``{species}_riv``).
        hydro_sim: The hydrology simulation DataFrame, for the
            simulated river discharge (``streamflow_sim``).
        q_meas: The measured river discharge over the full record, in
            millimeters per day.
        conc_df: The measured river concentrations over the full
            record, one column per species.
        species: The river species to plot, one panel each.
        window: The plotting window, as
            `potions.gui.common.water_year_window` returns it.

    Returns:
        The completed figure.
    """
    rt_plot: pd.DataFrame = _clip_frame(rt_sim, window)
    hydro_plot: pd.DataFrame = _clip_frame(hydro_sim, window)
    q_meas_plot: pd.Series = _clip_series(q_meas, window)
    conc_plot: pd.DataFrame = _clip_frame(conc_df, window)

    fig: Figure = Figure(figsize=(15, 5))
    if not species:
        # No species selected: the figure carries no panels.
        return fig

    for i, species_name in enumerate(species):
        ax: Axes = fig.add_subplot(1, len(species), i + 1)
        ax.set_title(species_name)
        ax.set_xlabel("Discharge (mm/d)")
        ax.set_ylabel("Concentration (mol/L)")
        ax.set_xscale("log")
        ax.set_yscale("log")

        # The simulated cloud: the run's own discharge and
        # concentration.  Zero, negative, and missing values do not
        # plot on a log axis.
        sim_x: pd.Series = hydro_plot["streamflow_sim"]
        sim_y: pd.Series = rt_plot[f"{species_name}_riv"]
        sim_mask: pd.Series = (sim_x > 0) & (sim_y > 0)
        ax.scatter(
            sim_x[sim_mask],
            sim_y[sim_mask],
            s=5,
            color=RIVER_SPECIES_SIM_COLORS[species_name],
        )

        # The observed cloud: the record's discharge and
        # concentrations, the observations' scatter as always.
        meas_x: pd.Series = q_meas_plot
        meas_y: pd.Series = conc_plot[species_name]
        meas_mask: pd.Series = (meas_x > 0) & (meas_y > 0)
        ax.scatter(
            meas_x[meas_mask],
            meas_y[meas_mask],
            s=5,
            color=RIVER_SPECIES_MEAS_COLORS[species_name],
        )

    fig.subplots_adjust(left=0.07, right=0.96, top=0.88)

    sim_row: Legend = _attach_legend(
        fig,
        [
            Line2D(
                [],
                [],
                marker="o",
                color=RIVER_SPECIES_SIM_COLORS[name],
                linestyle="none",
                markersize=3,
            )
            for name in species
        ],
        list(species),
        title="Simulated",
        ncol=max(len(species), 1),
    )
    meas_row: Legend = _attach_legend(
        fig,
        [
            Line2D(
                [],
                [],
                marker="o",
                color=RIVER_SPECIES_MEAS_COLORS[name],
                linestyle="none",
                markersize=3,
            )
            for name in species
        ],
        list(species),
        title="Measured",
        above=sim_row,
    )
    fig.subplots_adjust(bottom=_legend_top_edge(fig, meas_row) + _axes_gap(fig))

    return fig


def make_zone_concentrations_figure(
    rt_sim: pd.DataFrame,
    species: list[str],
    zones: list[str],
    window: tuple[pd.Timestamp, pd.Timestamp],
    overlay_rt_sim: pd.DataFrame | None = None,
    zone_labels: dict[str, str] = ZONE_LABELS,
    zone_colors: dict[str, str] = ZONE_COLORS,
) -> Figure:
    """
    Build the zone concentrations figure.

    The concentration (mol/L) of every selected species in every
    selected zone is drawn as a line on a single shared axis: the
    color encodes the zone (the project's lateral-model palette) and
    the linestyle encodes the species (cycled through
    `potions.gui.common.LINESTYLE_CYCLE`).  The record is clipped to
    the plotting `window`, so a range that ends after the record's
    last day (the partial 2025 water year) still plots only the days
    the record carries.

    The model emits a ``{species}_{zone}`` column for every zone
    (each zone keeps the full state vector of every species), so all
    selected zone-species pairs plot.

    A previous run's results, when the overlay frame is given, are
    drawn first and faintly (half transparent, unlabeled), so the
    current run's series stay on top; a zone or species the previous
    simulation does not carry is skipped.

    The frameless zone and species legend rows sit below the axes,
    inside the figure, so they never overlap the drawn series and
    stay in the figure panel's page export.

    Args:
        rt_sim: The reactive-transport simulation DataFrame, for the
            per-zone concentrations (``{species}_{zone}``).
        species: The network species to plot.
        zones: The zones to plot.
        window: The plotting window, as
            `potions.gui.common.water_year_window` returns it.
        overlay_rt_sim: The previous run's reactive-transport
            simulation, for the faint concentration overlay; None for
            no overlay.
        zone_labels: The display name of every zone; defaults to the
            lateral model's labels.
        zone_colors: The line color of every zone; defaults to the
            lateral model's palette.

    Returns:
        The completed figure.
    """
    rt_plot: pd.DataFrame = _clip_frame(rt_sim, window)
    overlay_rt_plot: pd.DataFrame | None = (
        _clip_frame(overlay_rt_sim, window)
        if overlay_rt_sim is not None
        else None
    )

    fig: Figure = Figure(figsize=(15, 5))
    ax: Axes = fig.gca()

    # The previous run's series, drawn first (behind) and faintly,
    # without labels (they stay out of the legends).
    if overlay_rt_plot is not None:
        for zone in zones:
            for i, species_name in enumerate(species):
                overlay_column: str = f"{species_name}_{zone}"
                if overlay_column in overlay_rt_plot.columns:
                    overlay_linestyle: Linestyle = (
                        LINESTYLE_CYCLE[i % len(LINESTYLE_CYCLE)]
                    )
                    ax.plot(
                        overlay_rt_plot.index,
                        overlay_rt_plot[overlay_column],
                        color=zone_colors[zone],
                        linestyle=overlay_linestyle,
                        alpha=_OVERLAY_ALPHA,
                        linewidth=1,
                    )

    for zone in zones:
        for i, species_name in enumerate(species):
            linestyle: Linestyle = LINESTYLE_CYCLE[i % len(LINESTYLE_CYCLE)]
            ax.plot(
                rt_plot.index,
                rt_plot[f"{species_name}_{zone}"],
                color=zone_colors[zone],
                linestyle=linestyle,
                linewidth=1,
            )

    ax.set_title("Concentration")
    ax.set_ylabel("Concentration (mol/L)")
    ax.set_xlim(
        _day_number(window[0]),
        _day_number(rt_plot.index.max()),
    )
    fig.subplots_adjust(left=0.07, right=0.96, top=0.88)

    zone_row: Legend = _attach_legend(
        fig,
        [Line2D([], [], color=zone_colors[zone], linewidth=2) for zone in zones],
        [zone_labels[zone] for zone in zones],
        title="Zone",
        ncol=max(len(zones), 1),
    )
    species_row: Legend = _attach_legend(
        fig,
        [
            Line2D(
                [],
                [],
                color="k",
                linestyle=LINESTYLE_CYCLE[i % len(LINESTYLE_CYCLE)],
                linewidth=2,
            )
            for i in range(len(species))
        ],
        list(species),
        title="Species",
        above=zone_row,
    )
    fig.subplots_adjust(
        bottom=_legend_top_edge(fig, species_row) + _axes_gap(fig)
    )

    return fig


def make_river_concentrations_figure(
    rt_sim: pd.DataFrame,
    hydro_sim: pd.DataFrame,
    q_meas: pd.Series,
    conc_df: pd.DataFrame,
    species: list[str],
    streamflow: bool,
    window: tuple[pd.Timestamp, pd.Timestamp],
    overlay_rt_sim: pd.DataFrame | None = None,
    overlay_hydro_sim: pd.DataFrame | None = None,
) -> Figure:
    """
    Build the twin-axis streamflow / river-concentrations figure.

    When `streamflow` is on, the left axis carries the river
    discharge in millimeters per day, inverted (zero at the top):
    the simulated ``streamflow_sim`` as a line and the measured
    ``q_meas`` as scatter points (size 5), so rising events sweep
    down against the concentrations above them.  The right axis,
    created as the left axis's twin, carries the river
    concentrations in moles per liter of every selected species: the
    simulated ``{species}_riv`` as a line and the measured
    concentration as scatter points (size 5); the color encodes the
    species family, with the simulated lines in the lighter shade
    and the measured points in the darker one
    (`potions.gui.common`).

    With the streamflow off, the species axis owns the panel alone
    (centered title), and a selection that carries neither family
    yields a figure with no axes.  All series are clipped to the
    plotting `window`, so a range that ends after the record's last
    day (the partial 2025 water year) still plots only the days the
    record carries.

    A previous run's results, when the overlay frames are given, are
    drawn first and faintly (half transparent, unlabeled), so the
    current run's series stay on top; a column the previous
    simulation does not carry is skipped.

    The single frameless legend sits below the axes, inside the
    figure, so it never overlaps the drawn series and stays in the
    figure panel's page export.

    Args:
        rt_sim: The reactive-transport simulation DataFrame, for the
            simulated river concentrations (``{species}_riv``).
        hydro_sim: The hydrology simulation DataFrame, for the
            simulated river discharge (``streamflow_sim``); used
            only when `streamflow` is on.
        q_meas: The measured river discharge over the full record, in
            millimeters per day; NaN values do not plot; used only
            when `streamflow` is on.
        conc_df: The measured river concentrations (DOC, DIC, and
            calcium) over the full record, in moles per liter.
        species: The river species to plot; a species whose
            ``{species}_riv`` column the simulation does not carry
            is skipped.
        streamflow: Whether to draw the inverted streamflow axis.
        window: The plotting window, as
            `potions.gui.common.water_year_window` returns it.
        overlay_rt_sim: The previous run's reactive-transport
            simulation, for the faint species overlay; None for no
            overlay.
        overlay_hydro_sim: The previous run's hydrology simulation,
            for the faint streamflow overlay; None for no overlay.

    Returns:
        The completed figure.
    """
    if not streamflow and not species:
        return Figure(figsize=(15, 5))

    rt_plot: pd.DataFrame = _clip_frame(rt_sim, window)
    hydro_plot: pd.DataFrame = _clip_frame(hydro_sim, window)
    q_meas_plot: pd.Series = _clip_series(q_meas, window).dropna()
    conc_plot: pd.DataFrame = _clip_frame(conc_df, window)
    overlay_rt_plot: pd.DataFrame | None = (
        _clip_frame(overlay_rt_sim, window)
        if overlay_rt_sim is not None
        else None
    )
    overlay_hydro_plot: pd.DataFrame | None = (
        _clip_frame(overlay_hydro_sim, window)
        if overlay_hydro_sim is not None
        else None
    )

    fig: Figure = Figure(figsize=(15, 5))
    handles: list[Line2D] = []
    labels: list[str] = []

    # The left axis carries the inverted streamflow panel only when
    # selected; otherwise the species axis owns the panel alone.
    left_ax: Axes | None = None
    if streamflow:
        left_ax = fig.gca()

        if (
            overlay_hydro_plot is not None
            and "streamflow_sim" in overlay_hydro_plot.columns
        ):
            # The previous run's streamflow, drawn first (behind)
            # and faintly, without a label (out of the legend).
            left_ax.plot(
                overlay_hydro_plot.index,
                overlay_hydro_plot["streamflow_sim"],
                color=SIM_Q_COLOR,
                alpha=_OVERLAY_ALPHA,
                linewidth=1,
            )
        left_ax.plot(
            hydro_plot.index,
            hydro_plot["streamflow_sim"],
            color=SIM_Q_COLOR,
            label="Streamflow (sim)",
            linewidth=1,
        )
        left_ax.scatter(
            q_meas_plot.index,
            q_meas_plot,
            s=5,
            color=MEAS_Q_COLOR,
            label="Streamflow (meas)",
        )
        left_ax.set_ylim(0, 2 * float(left_ax.get_ylim()[1]))
        left_ax.invert_yaxis()
        left_ax.set_title("Streamflow")
        left_ax.set_ylabel("Discharge (mm/d)")
        left_ax.set_xlim(
            _day_number(window[0]), _day_number(hydro_plot.index.max())
        )

        handles.extend(
            [
                Line2D([], [], color=SIM_Q_COLOR, linewidth=2),
                Line2D(
                    [],
                    [],
                    marker="o",
                    color=MEAS_Q_COLOR,
                    linestyle="none",
                    markersize=3,
                ),
            ]
        )
        labels.extend(["Streamflow (sim)", "Streamflow (meas)"])

    species_ax: Axes = left_ax.twinx() if left_ax is not None else fig.gca()

    # The previous run's species lines, drawn first (behind) and
    # faintly, without labels (they stay out of the legend).
    if overlay_rt_plot is not None:
        for name in species:
            overlay_column: str = f"{name}_riv"
            if (
                overlay_column in rt_plot.columns
                and overlay_column in overlay_rt_plot.columns
            ):
                species_ax.plot(
                    overlay_rt_plot.index,
                    overlay_rt_plot[overlay_column],
                    color=RIVER_SPECIES_SIM_COLORS[name],
                    alpha=_OVERLAY_ALPHA,
                    linewidth=1,
                )

    for name in species:
        column: str = f"{name}_riv"
        if column not in rt_plot.columns:
            continue
        species_ax.plot(
            rt_plot.index,
            rt_plot[column],
            color=RIVER_SPECIES_SIM_COLORS[name],
            label=f"{name} (sim)",
            linewidth=1,
        )
        if name in conc_plot.columns:
            measured: pd.Series = conc_plot[name].dropna()
            species_ax.scatter(
                measured.index,
                measured,
                s=5,
                color=RIVER_SPECIES_MEAS_COLORS[name],
                label=f"{name} (meas)",
            )
        handles.extend(
            [
                Line2D(
                    [],
                    [],
                    color=RIVER_SPECIES_SIM_COLORS[name],
                    linewidth=2,
                ),
                Line2D(
                    [],
                    [],
                    marker="o",
                    color=RIVER_SPECIES_MEAS_COLORS[name],
                    linestyle="none",
                    markersize=3,
                ),
            ]
        )
        labels.extend([f"{name} (sim)", f"{name} (meas)"])

    species_ax.set_ylim(0, 1.5 * float(species_ax.get_ylim()[1]))
    if left_ax is not None:
        # The species title sits on the right, so it does not
        # overlap the streamflow title on the left.
        species_ax.set_title("River concentrations", loc="right")
    else:
        species_ax.set_title("River concentrations")
    species_ax.set_ylabel("Concentration (mol/L)")
    species_ax.set_xlim(
        _day_number(window[0]), _day_number(rt_plot.index.max())
    )
    fig.subplots_adjust(left=0.07, right=0.96, top=0.88)

    if handles:
        legend: Legend = _attach_legend(
            fig, handles, labels, ncol=max(len(handles), 1)
        )
        fig.subplots_adjust(
            bottom=_legend_top_edge(fig, legend) + _axes_gap(fig)
        )

    return fig




def make_rate_figure(
    rt_sim: pd.DataFrame,
    mineral: str,
    window: tuple[pd.Timestamp, pd.Timestamp],
    overlay_rt_sim: pd.DataFrame | None = None,
    zone_labels: dict[str, str] = ZONE_LABELS,
    zone_colors: dict[str, str] = ZONE_COLORS,
    zone_order: list[str] = ZONE_ORDER,
) -> Figure:
    """
    Build the reaction-rate figure for a single mineral.

    The mineral's reaction rate (mol/L/day) is drawn as a
    zone-colored line for every zone that carries one of the
    mineral's rate columns: the model emits a
    ``{mineral}_rate_{zone}`` column only for zones in which
    reactions run, so zones without the column are omitted from both
    the plot and the zone legend.  The record is clipped to the
    plotting `window`, so a range that ends after the record's last
    day (the partial 2025 water year) still plots only the days the
    record carries.

    A previous run's results, when the overlay frame is given, are
    drawn first and faintly (half transparent, unlabeled), so the
    current run's series stay on top; a zone whose rate column the
    previous simulation does not carry is skipped.

    The frameless zone legend sits below the axes, inside the
    figure, so it never overlaps the drawn series and stays in the
    figure panel's page export.

    Args:
        rt_sim: The reactive-transport simulation DataFrame, for the
            rates (``{mineral}_rate_{zone}`).
        mineral: The mineral whose reaction rate to plot.
        window: The plotting window, as
            `potions.gui.common.water_year_window` returns it.
        overlay_rt_sim: The previous run's reactive-transport
            simulation, for the faint rate overlay; None for no
            overlay.
        zone_labels: The display name of every zone; defaults to the
            lateral model's labels.
        zone_colors: The line color of every zone; defaults to the
            lateral model's palette.
        zone_order: The zones to consider, in plot order; defaults to
            the lateral model's zone order.

    Returns:
        The completed figure.
    """
    rt_plot: pd.DataFrame = _clip_frame(rt_sim, window)
    overlay_rt_plot: pd.DataFrame | None = (
        _clip_frame(overlay_rt_sim, window)
        if overlay_rt_sim is not None
        else None
    )

    # The model emits a rate column only for zones in which reactions
    # run; plot (and legend) exactly the zones that carry the
    # mineral's column.
    plot_zones: list[str] = [
        zone
        for zone in zone_order
        if f"{mineral}_rate_{zone}" in rt_plot.columns
    ]

    fig: Figure = Figure(figsize=(15, 5))
    ax: Axes = fig.gca()

    # The previous run's rates, drawn first (behind) and faintly,
    # without labels (they stay out of the legend).
    if overlay_rt_plot is not None:
        for zone in plot_zones:
            overlay_column: str = f"{mineral}_rate_{zone}"
            if overlay_column in overlay_rt_plot.columns:
                ax.plot(
                    overlay_rt_plot.index,
                    overlay_rt_plot[overlay_column],
                    color=zone_colors[zone],
                    alpha=_OVERLAY_ALPHA,
                    linewidth=1,
                )

    for zone in plot_zones:
        ax.plot(
            rt_plot.index,
            rt_plot[f"{mineral}_rate_{zone}"],
            color=zone_colors[zone],
            linewidth=1,
        )

    ax.set_title(f"Reaction rates ({mineral})")
    ax.set_ylabel("Reaction rate (mol/L/day)")
    ax.set_xlim(
        _day_number(window[0]),
        _day_number(rt_plot.index.max()),
    )
    fig.subplots_adjust(left=0.07, right=0.96, top=0.88)

    if plot_zones:
        zone_row: Legend = _attach_legend(
            fig,
            [Line2D([], [], color=zone_colors[zone], linewidth=2) for zone in plot_zones],
            [zone_labels[zone] for zone in plot_zones],
            title="Zone",
            ncol=max(len(plot_zones), 1),
        )
        fig.subplots_adjust(
            bottom=_legend_top_edge(fig, zone_row) + _axes_gap(fig)
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
    terms: list[str],
    note: str,
) -> Figure:
    """
    The line search's value profile, as a page of the figure panel.

    The page draws the searched value at every evaluated point of
    the ascent line, against the step ``t`` on the horizontal axis,
    one marker per point.  With one searched term, it draws the
    profile's ``value`` column (the term's value) and labels the
    title and the vertical axis after the term; with several, it
    draws each term's value in the terms' order, the searched sum
    (the ``value`` column) in bold, and labels the vertical axis
    after the sum.  The point with the largest finite value is
    marked in red as the best one, and the job's note is printed
    under the panel.

    Args:
        profile: The value profile, one row per evaluated point:
            the step ``t`` as index, the searched value in the
            ``value`` column, and one column per searched term.
        terms: The searched objective terms, as ``"target:
            metric"`` labels, in the profile's column order.
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

    if len(terms) == 1:
        metric: str = terms[0].rsplit(": ", 1)[1]
        ax.plot(
            t,
            values,
            color="k",
            linewidth=1,
            marker="o",
            markersize=3,
            label=terms[0],
        )
        ax.set_title(f"Line search — {terms[0]}")
        ax.set_ylabel(metric)
    else:
        for term in terms:
            ax.plot(
                t,
                profile[term].to_numpy(dtype=np.float64),
                marker="o",
                markersize=3,
                label=term,
            )
        ax.plot(t, values, color="k", linewidth=2, label="sum")
        ax.set_title(f"Line search — sum of {len(terms)} objectives")
        ax.set_ylabel("combined value")
    ax.plot(
        [t[best_index]],
        [values[best_index]],
        color="r",
        marker="o",
        markersize=7,
        label="best",
    )
    ax.set_xlabel("t")
    # The figure, not the axis, carries the legend: the panel's
    # pages read the figure's legends, and in this matplotlib the
    # axes' own legend does not register there.
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # A figure-level legend must give an explicit location
        # (there is no "best" for the whole figure); the time
        # series page's anchor above the axes is the house
        # pattern.
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(labels),
        )

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
