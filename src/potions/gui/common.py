"""
Shared constants and helpers for the East River lateral-model
calibration GUI.

The GUI is split across the modules in this package:

* ``potions.gui.window`` (the main window) wires everything together;
* ``potions.gui.data`` loads the record spreadsheet (the East River
  series by default) and runs the models;
* ``potions.gui.param_panel`` renders the parameter controls;
* ``potions.gui.plots`` builds the figures and the objective table;
* ``potions.gui.state`` persists and restores the GUI state.

This module holds what all of them share:

* the data provenance (the site and the water years of the site's
  original calibration design);
* the flat parameter layout — which zones carry which hydrology,
  reactive-transport, and initial-state parameters, in the order
  ``potions.gui.data`` lists their bounds;
* the human-readable label and the short display symbol for every
  flat parameter key;
* the color schemes, including the blue / red parameter-group themes;
* the rcParams setup and the bound-precision rule shared by the
  parameter panel and its tests.

Color schemes
-------------
Zones: a copy of the project's established lateral-model palette
(the hue encodes the layer (snow: blue, surface: red, ground: green)
and the lightness encodes the region (hillslope "HS" is
the lighter shade, riparian "RP" the darker one), so related zones
form an obvious visual family in every figure that draws zones.

River species: the hue encodes the chemical family (DOC brown,
dissolved organic carbon; DIC teal, the inorganic-carbon family
HCO3- + CO2(aq) + CO3--; Ca++ crimson, the dominant divalent cation),
and the lightness encodes the series — the simulated lines are the
lighter shade of each hue and the measured scatter points the darker
one, following the project's published convention that simulated
series are drawn lighter than measured ones.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

import matplotlib.pyplot as plt
import pandas as pd
import potions as pt
import potions.objective_functions as obj

# ==== Constants ==== #

# The site the East River series record belongs to.
SITE: Final[str] = "EastRiver"

# The water years of the site's testing period and the single
# validation water year, kept here as provenance of the site's
# original calibration design.
TEST_WYS: Final[list[int]] = [2017, 2020]
VAL_WY: Final[int] = 2018

# The water years the site record contains, in order: the plotting
# range the figure panel's from / to pickers offer.  The bundled
# record (the East River series) runs 2014-05-02 to 2025-09-18, so the
# first full water year is 2015 and the last, 2025, is partial — it
# ends with the record rather than on September 30.
WATER_YEARS: Final[list[int]] = list(range(2015, 2026))

# The six lateral zones, in model order.
ZONE_ORDER: Final[list[str]] = [
    "snow_hs",
    "snow_rp",
    "surface_hs",
    "surface_rp",
    "ground_hs",
    "ground_rp",
]

# The lateral model's zones deepest-layer first, for the partitioning
# figure's stack order.
ZONE_STACK_ORDER: Final[list[str]] = [
    "ground_hs",
    "ground_rp",
    "surface_hs",
    "surface_rp",
    "snow_hs",
    "snow_rp",
]

# Human-readable zone label: the layer name plus the region tag.
# HS = hillslope (upland, distal column), RP = riparian (near-channel,
# proximal column).
ZONE_LABELS: Final[dict[str, str]] = {
    "snow_hs": "Snow (HS)",
    "snow_rp": "Snow (RP)",
    "surface_hs": "Surface (HS)",
    "surface_rp": "Surface (RP)",
    "ground_hs": "Ground (HS)",
    "ground_rp": "Ground (RP)",
}

# Zone colors: a copy of the project's established lateral-model
# palette.  See the module docstring for the encoding.
ZONE_COLORS: Final[dict[str, str]] = {
    "snow_hs": "#598DB1",
    "surface_hs": "#F64740",
    "ground_hs": "#76B041",
    "snow_rp": "#2E4D61",
    "surface_rp": "#750906",
    "ground_rp": "#466827",
}

# The snow zones carry no reactive chemistry beyond water storage.
SNOW_ZONES: Final[list[str]] = ["snow_hs", "snow_rp"]

# The zones with a calibrated exchange-site initial state, in the
# order of the ``initial_state.*`` bounds.
SPECIATION_ZONES: Final[list[str]] = [
    "surface_hs",
    "surface_rp",
    "ground_hs",
    "ground_rp",
]

# The hydrology parameters each zone carries, in bounds order.
HYDRO_PARAMS: Final[dict[str, tuple[str, ...]]] = {
    "snow_hs": ("tt", "fmax"),
    "snow_rp": ("tt", "fmax"),
    "surface_hs": ("fc", "lp", "beta", "k0", "thr"),
    "surface_rp": ("fc", "lp", "beta", "k0", "thr"),
    "ground_hs": ("k", "alpha"),
    "ground_rp": ("k", "alpha"),
}

# The storage parameters every reactive-transport zone carries.
RT_DIM_PARAMS: Final[tuple[str, ...]] = ("max_water", "passive_water_storage")

# The kinetic parameters carried per mineral in each reactive zone.
MINERAL_PARAMS: Final[tuple[str, ...]] = (
    "sw_threshold",
    "sw_exp",
    "q_10",
    "k_a",
)

# Human-readable labels for the bare parameter names (no zone prefix).
# The zone context comes from the sidebar's subheaders and the
# mineral context from the bold header above each mineral group, so
# only the parameter itself is named here.
PARAM_LABELS: Final[dict[str, str]] = {
    # Snow zone: precipitation falls as snow at T <= tt, and meltout is
    # min(s, fmax * (T - tt)).
    "tt": "threshold temperature [°C]",
    "fmax": "melt rate coefficient [mm/d/K]",
    # Surface zone: ET = pet * min(s/(fc*lp), 1); recharge =
    # q_in * (s/fc)^beta; lateral outflow = k0 * (s - thr).
    "fc": "field capacity [mm]",
    "lp": "saturation capacity fraction (LP)",
    "beta": "recharge response exponent (β)",
    "k0": "lateral outflow coefficient (K0) [1/d]",
    "thr": "lateral outflow threshold [mm]",
    # Ground zone: lateral outflow = k * s^alpha.
    "k": "discharge coefficient",
    "alpha": "discharge exponent (α)",
    # HBV subsurface zone: percolation to the next layer.
    "perc": "percolation rate [mm/d]",
    # Reactive-transport zone storage.
    "max_water": "max reactive water storage [mm]",
    "passive_water_storage": "passive water storage [mm]",
    # Per-mineral reaction parameters.
    "sw_threshold": "saturation threshold",
    "sw_exp": "saturation exponent",
    "q_10": "Q₁₀ temperature sensitivity [1/°C]",
    "k_a": "dissolution rate constant (k_a) [1/m³/day]",
}

# The river species, in the panel's checkbox order: DOC first, then
# the inorganic-carbon family, then calcium.  The color maps below
# are keyed by this order.
RIVER_SPECIES: Final[tuple[str, ...]] = ("DOC", "DIC", "Ca++")

# The river species and their colors: the hue encodes the chemical
# family (brown for DOC, teal for the inorganic-carbon family, crimson
# for Ca++) and the lightness encodes the series — the simulated lines
# are drawn in the lighter shade of each hue and the measured scatter
# points in the darker one, following the project's published
# convention that simulated series are lighter than measured ones.
RIVER_SPECIES_SIM_COLORS: Final[dict[str, str]] = {
    "DOC": "sandybrown",
    "DIC": "turquoise",
    "Ca++": "lightcoral",
}
RIVER_SPECIES_MEAS_COLORS: Final[dict[str, str]] = {
    "DOC": "peru",
    "DIC": "teal",
    "Ca++": "crimson",
}
# Fixed palettes for measured species beyond the legacy trio: eight
# hues each, the simulated (lighter) and measured (darker) variants of
# the project's published color families.
SPECIES_SIM_PALETTE: Final[tuple[str, ...]] = (
    "sandybrown", "turquoise", "lightcoral", "goldenrod",
    "slateblue", "darkorange", "olive", "seagreen",
)
SPECIES_MEAS_PALETTE: Final[tuple[str, ...]] = (
    "peru", "teal", "crimson", "darkgoldenrod",
    "darkslateblue", "saddlebrown", "olivedrab", "forestgreen",
)
# Immutable snapshots of the legacy colors: the session maps above
# are cleared and refilled by `set_species_colors`, so anything that
# needs the original DOC / DIC / Ca++ colors reads these instead.
_LEGACY_SIM: Final[dict[str, str]] = dict(RIVER_SPECIES_SIM_COLORS)
_LEGACY_MEAS: Final[dict[str, str]] = dict(RIVER_SPECIES_MEAS_COLORS)
_LEGACY_NAMES: Final[frozenset[str]] = frozenset(RIVER_SPECIES)

# Streamflow colors, matching the project's published figures.
SIM_Q_COLOR: Final[str] = "cornflowerblue"
MEAS_Q_COLOR: Final[str] = "black"

# ==== Types ==== #
@dataclass(frozen=True, slots=True)
class Theme:
    """
    A color theme for a parameter group.

    The accent color tints the group's title and top border, and the
    light shade is the background of the group container.  The two
    themes give the hydrology and reactive-transport panels their
    distinct blue / red character (the accents reuse the snow-zone
    blue and surface-zone red of the figure palette, so the panel and
    the figures stay visually related).

    Attributes:
        name: The display name of the themed group.
        accent: The accent color (hex) for the group title and border.
        light: The light background color (hex) of the group.
    """

    name: str
    accent: str
    light: str


# The two parameter-group themes (todo: blues vs reds).
HYDRO_THEME: Final[Theme] = Theme(
    name="hydrology", accent="#598DB1", light="#EAF1F8"
)
RT_THEME: Final[Theme] = Theme(name="reactive transport", accent="#F64740", light="#FDEDEB")
# The window background of the reactive-transport mode, deliberately
# distinct from RT_THEME's light tint, which the panel groups keep.
RT_MODE_TINT: Final[str] = "#FBEEDB"
# =============== #

# The objective-function metrics shown in the per-run table, in the
# order of potions' default objective functions
# (potions.objective_functions.DEFAULT_OBJECTIVE_FUNCTIONS).
METRIC_NAMES: Final[list[str]] = [name for name, _ in obj.DEFAULT_OBJECTIVE_FUNCTIONS]

# Display labels for the metric columns of the per-run table.  The
# "log" metrics are evaluated on log10(x + 0.1), with values below the
# -0.1 lower bound floored at the zero-equivalent.
METRIC_LABELS: Final[dict[str, str]] = {
    "kge": "KGE",
    "log_kge": "LogKGE",
    "nse": "NSE",
    "log_nse": "LogNSE",
    "bias": "relative bias",
    "spearman_rho": "Spearman ρ",
    "r_squared": "R²",
}

# The note rendered under the per-run objective table.
METRIC_NOTES: Final[str] = (
    "KGE = Kling–Gupta efficiency; NSE = Nash–Sutcliffe efficiency; "
    "LogKGE and LogNSE are the KGE and NSE evaluated on log₁₀(x + 0.1), "
    "with values below -0.1 floored at the zero-equivalent; relative "
    "bias = mean(sim − meas) / mean(meas) (0 is unbiased, negative is "
    "an underprediction).  Higher is better for KGE, NSE, LogKGE, "
    "LogNSE, Spearman ρ, and R²; bias closest to 0 is best."
)

# The narrow set of linestyles the matplotlib type stubs accept for
# `Line2D`, cycled to distinguish species or minerals on shared axes.
Linestyle = Literal["-", "--", ":", "-."]
LINESTYLE_CYCLE: Final[tuple[Linestyle, ...]] = ("-", "--", ":", "-.")
# =================== #


# ==== Functions ==== #
def water_year_window(
    from_year: int,
    to_year: int,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    The plotting window spanning water years `from_year` through
    `to_year`, inclusive.

    A water year runs from October 1 of the previous calendar year to
    September 30 of the year it is labelled by, so the full 2015 to
    2025 range starts on 2014-10-01 and ends on 2025-09-30.  The
    window is a requested span, not a clip: the figure builders
    intersect it with each record, so a range that ends after the
    record's last day (the partial 2025 water year) still plots only
    the days the record carries.

    Args:
        from_year: The first water year to include.
        to_year: The last water year to include.

    Returns:
        The window's start and end as Timestamps: October 1 of
        ``from_year - 1`` and September 30 of ``to_year``.

    Raises:
        ValueError:
            Raised when either year lies outside
            `WATER_YEARS`, or when `from_year` is after `to_year`.
    """
    if from_year not in WATER_YEARS:
        raise ValueError(
            f"Water year {from_year} is outside the site record's "
            f"plotting range, {WATER_YEARS[0]} to {WATER_YEARS[-1]}."
        )
    if to_year not in WATER_YEARS:
        raise ValueError(
            f"Water year {to_year} is outside the site record's "
            f"plotting range, {WATER_YEARS[0]} to {WATER_YEARS[-1]}."
        )
    if from_year > to_year:
        raise ValueError(
            f"The first water year ({from_year}) is after the last "
            f"({to_year})."
        )
    return (
        pd.Timestamp(from_year - 1, 10, 1),
        pd.Timestamp(to_year, 9, 30),
    )


def set_rc_params() -> None:
    """
    Configure the matplotlib rcParams for the GUI figures.

    Removes the axes spines and left-aligns the titles so the figures
    match the project's published-figure style, and sets a 12-point
    base font: the workbench's smallest text is the base font itself,
    so no figure text (tick labels, titles, legends, notes) is ever
    smaller than 12 point.
    """
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.spines.bottom"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.left"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.titlelocation"] = "left"


def build_species_colors(
    measured: Sequence[str],
) -> tuple[dict[str, str], dict[str, str]]:
    """
    The sim and measured color maps for a run's measured species.

    Legacy species (DOC / DIC / Ca++) keep their published colors;
    every other measured species takes the next palette hue in the
    measured list's order (wrapping at eight).

    Args:
        measured: The measured species, in plot order.

    Returns:
        The (sim_colors, measured_colors) dicts keyed by species name.
    """
    sim: dict[str, str] = {}
    meas: dict[str, str] = {}
    for index, name in enumerate(measured):
        if name in _LEGACY_NAMES:
            sim[name] = _LEGACY_SIM[name]
            meas[name] = _LEGACY_MEAS[name]
        else:
            sim[name] = SPECIES_SIM_PALETTE[index % len(SPECIES_SIM_PALETTE)]
            meas[name] = SPECIES_MEAS_PALETTE[index % len(SPECIES_MEAS_PALETTE)]
    return sim, meas


def set_species_colors(measured: Sequence[str]) -> None:
    """
    Refill the session species-color maps for ``measured`` (in place).

    The maps are the ``RIVER_SPECIES_SIM_COLORS`` and
    ``RIVER_SPECIES_MEAS_COLORS`` dicts the figures read; they are
    cleared and refilled from `build_species_colors`.  An empty
    sequence clears them, and a caller that plots species must pass
    the species the figures plot (the run's network's measured
    species), not a subset a record happens to carry.

    Args:
        measured: The species the figures plot, in plot order.
    """
    sim, meas = build_species_colors(measured)
    RIVER_SPECIES_SIM_COLORS.clear()
    RIVER_SPECIES_MEAS_COLORS.clear()
    RIVER_SPECIES_SIM_COLORS.update(sim)
    RIVER_SPECIES_MEAS_COLORS.update(meas)


def param_label(
    name: str,
    mineral_names: Sequence[str],
    zone_labels: Mapping[str, str] = ZONE_LABELS,
) -> str:
    """
    The human-readable label for a flat parameter key.

    The keys come in three forms: ``{zone}.{parameter}`` (hydrology
    and reactive-transport dimensions), ``{zone}.{field}[{mineral_id}]``
    (per-mineral kinetics), and the special ``proportion.1`` and
    ``initial_state.{zone}_x`` keys.  The zone- and mineral-prefixed
    keys are labeled with the parameter part only (plus the mineral
    name for the bracketed keys), since the zone and mineral context
    is already shown by the parameter panel's zone subheaders and
    mineral headers.  The proportion and initial-state keys (which
    have no subheader above them) are labeled in full.

    Args:
        name: The parameter key, as a key of the bounds mapping.
        mineral_names: The mineral names of the network, in the
            network's order, for labeling the bracketed mineral keys.
        zone_labels: The display name of every zone, for labeling the
            initial-state keys; defaults to the lateral model's labels.

    Returns:
        The label to display for the parameter.

    Raises:
        ValueError: Raised for a key of an unrecognized form.
    """
    if name == "proportion.1":
        return "Distal proportion of catchment"

    if name.startswith("initial_state."):
        zone: str = name.removeprefix("initial_state.").removesuffix("_x")
        return f"{zone_labels[zone]} total sorption sites"

    if "." not in name:
        raise ValueError(f"Unrecognized parameter key: {name!r}")

    _zone_part, rest = name.split(".", 1)
    if "[" in rest:  # mineral parameter: {zone}.{field}[{mineral_id}]
        field, id_part = rest.split("[", 1)
        mineral_id: int = int(id_part.rstrip("]"))
        return f"{mineral_names[mineral_id]} {PARAM_LABELS[field]}"

    label: str | None = PARAM_LABELS.get(rest)
    if label is None:
        raise ValueError(f"Unrecognized parameter key: {name!r}")
    return label


def param_symbol(name: str) -> str:
    """
    The short symbol shown in place of a parameter's full label.

    The symbol is the parameter's field name ("proportion" for the
    catchment-proportion key, the field for zone-prefixed keys, and a
    bare "x" for the initial-state keys); the parameter's full label
    is available as the cell's tooltip.

    Args:
        name: The parameter key, as a key of the bounds mapping.

    Returns:
        The short symbol for the parameter.

    Raises:
        ValueError: Raised for a key of an unrecognized form.
    """
    if name == "proportion.1":
        return "proportion"

    if name.startswith("initial_state."):
        return "x"

    if "." not in name:
        raise ValueError(f"Unrecognized parameter key: {name!r}")

    _zone_part, rest = name.split(".", 1)
    field: str = rest.split("[", 1)[0]
    if field not in PARAM_LABELS:
        raise ValueError(f"Unrecognized parameter key: {name!r}")
    return field


def bound_decimals(lo: float, hi: float) -> int:
    """
    The number of decimals a value in the bounds needs to display.

    Just enough decimals to distinguish values a thousandth of the
    span apart, with at least one so integral values still read as
    numbers.  Degenerate bounds (a fixed parameter) fall back to four
    decimals, the precision the saved ``parameters.csv`` files carry.

    Args:
        lo: The lower bound.
        hi: The upper bound.

    Returns:
        The number of decimals to format a value in the bounds with.
    """
    span: float = float(hi) - float(lo)
    if span <= 0:
        return 4
    step: float = span / 1000.0
    return max(1, int(math.ceil(-math.log10(step))) + 1)
# =================== #
