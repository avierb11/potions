"""
Shared constants and helpers for the East River lateral-model
calibration GUI.

The GUI is split across the modules in this package:

* ``potions.gui.app`` (the entry script) wires everything together;
* ``potions.gui.data`` loads the East River series CSV and runs the
  models;
* ``potions.gui.sidebar`` renders the sidebar controls;
* ``potions.gui.plots`` builds the figures and the objective table.

This module holds what all three share:

* the data provenance (the site and the water years of the site's
  original calibration design);
* the flat parameter layout — which zones carry which hydrology,
  reactive-transport, and initial-state parameters, in the order
  ``potions.gui.data`` lists their bounds;
* the human-readable label for every flat parameter key;
* the color schemes;
* the rcParams setup and the slider control.

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
from collections.abc import Sequence
from typing import Final, Literal

import matplotlib.pyplot as plt
import pandas as pd
import potions as pt
import potions.objective_functions as obj
import streamlit as st

# ==== Constants ==== #

# The site the East River series record belongs to.
SITE: Final[str] = "EastRiver"

# The water years of the site's testing period and the single
# validation water year, kept here as provenance of the site's
# original calibration design.
TEST_WYS: Final[list[int]] = [2017, 2020]
VAL_WY: Final[int] = 2018

# The first day plotted by the figures: the start of the forcing
# window the loader keeps from the CSV record (the start of water
# year 2015; a water year runs from October 1 to September 30 and is
# labelled by the year it ends in).  The simulation prepends a
# two-year spin-up before this date, so clipping the figures at this
# date shows the site record exactly and drops the spin-up.
PLOT_START: Final[pd.Timestamp] = pd.Timestamp(2014, 10, 1)
# The same boundary as a matplotlib date number: days since
# 1970-01-01, the unit matplotlib stores for datetime axes (the
# ``Timestamp.value`` nanoseconds since that epoch, converted to
# days).  ``Axes.set_xlim`` is typed to accept floats only, so the
# figure builders pass this pre-converted value instead of the
# Timestamp.
PLOT_START_DATE: Final[float] = float(PLOT_START.value / 1e9 / 86400)

# The six lateral zones, in model order.
ZONE_ORDER: Final[list[str]] = [
    "snow_hs",
    "snow_rp",
    "surface_hs",
    "surface_rp",
    "ground_hs",
    "ground_rp",
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
    # Reactive-transport zone storage.
    "max_water": "max reactive water storage [mm]",
    "passive_water_storage": "passive water storage [mm]",
    # Per-mineral reaction parameters.
    "sw_threshold": "saturation threshold",
    "sw_exp": "saturation exponent",
    "q_10": "Q₁₀ temperature sensitivity [1/°C]",
    "k_a": "dissolution rate constant (k_a) [1/m³/day]",
}

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

# Streamflow colors, matching the project's published figures.
SIM_Q_COLOR: Final[str] = "cornflowerblue"
MEAS_Q_COLOR: Final[str] = "black"

# The objective-function metrics shown in the per-run table, in the
# order of potions' default objective functions
# (potions.objective_functions.DEFAULT_OBJECTIVE_FUNCTIONS).
METRIC_NAMES: Final[list[str]] = [name for name, _ in obj.DEFAULT_OBJECTIVE_FUNCTIONS]

# Display labels for the metric columns of the per-run table.  The
# "log" metrics are evaluated on log10(x + 0.1).
METRIC_LABELS: Final[dict[str, str]] = {
    "kge": "KGE",
    "log_kge": "KGE (log₁₀)",
    "nse": "NSE",
    "log_nse": "NSE (log₁₀)",
    "bias": "relative bias",
    "spearman_rho": "Spearman ρ",
    "r_squared": "R²",
}

# The note rendered under the per-run objective table.
METRIC_NOTES: Final[str] = (
    "KGE = Kling–Gupta efficiency; NSE = Nash–Sutcliffe efficiency; the "
    "log₁₀ metrics are evaluated on log₁₀(x + 0.1); relative bias = "
    "mean(sim − meas) / mean(meas) (0 is unbiased, negative is an "
    "underprediction).  Higher is better for KGE, NSE, Spearman ρ, and R²; "
    "bias closest to 0 is best."
)

# The narrow set of linestyles the matplotlib type stubs accept for
# `Line2D`, cycled to distinguish species or minerals on shared axes.
Linestyle = Literal["-", "--", ":", "-."]
LINESTYLE_CYCLE: Final[tuple[Linestyle, ...]] = ("-", "--", ":", "-.")
# =================== #


# ==== Functions ==== #
def set_rc_params() -> None:
    """
    Configure the matplotlib rcParams for the GUI figures.

    Removes the axes spines and left-aligns the titles so the figures
    match the project's published-figure style.
    """
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["axes.spines.bottom"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.left"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.titlelocation"] = "left"


def param_label(name: str, mineral_names: Sequence[str]) -> str:
    """
    The human-readable label for a flat parameter key.

    The keys come in three forms: ``{zone}.{parameter}`` (hydrology
    and reactive-transport dimensions), ``{zone}.{field}[{mineral_id}]``
    (per-mineral kinetics), and the special ``proportion.1`` and
    ``initial_state.{zone}_x`` keys.  The zone- and mineral-prefixed
    keys are labeled with the parameter part only (plus the mineral
    name for the bracketed keys), since the zone and mineral context
    is already shown by the sidebar's subheaders and bold headers.
    The proportion and initial-state keys (which have no subheader
    above them) are labeled in full.

    Args:
        name: The parameter key, as a key of the bounds mapping.
        mineral_names: The mineral names of the network, in the
            network's order, for labeling the bracketed mineral keys.

    Returns:
        The label to display for the parameter.

    Raises:
        ValueError: Raised for a key of an unrecognized form.
    """
    if name == "proportion.1":
        return "Distal proportion of catchment"

    if name.startswith("initial_state."):
        zone: str = name.removeprefix("initial_state.").removesuffix("_x")
        return f"{ZONE_LABELS[zone]} total sorption sites"

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


def slider_value(
    name: str,
    bounds: dict[str, tuple[float, float]],
    mineral_names: Sequence[str],
) -> float:
    """
    Render the sidebar control for a single parameter and return its
    value.

    Free parameters (distinct bounds) get a slider whose default is
    the midpoint of the bounds and which carries an explicit session
    key, so a changed value persists across reruns regardless of the
    sidebar layout.  Fixed parameters (identical bounds) are not
    shown and are pinned to their value.

    The slider step is a thousandth of the parameter's range, and the
    displayed value carries just enough decimals to show that step, so
    both wide and tight parameter ranges are resolvable and the label
    is not rounded to a fixed two decimals.

    Args:
        name: The parameter name, as a key of ``bounds``.
        bounds: The mapping of parameter name to (min, max) bound.
        mineral_names: The mineral names of the network, in the
            network's order, for labeling the bracketed mineral keys.

    Returns:
        The value to use for the parameter on this run.
    """
    lo, hi = bounds[name]
    if abs(lo - hi) <= 1e-5:
        return lo

    span: float = float(hi) - float(lo)
    step: float = span / 1000.0
    # Just enough decimals to distinguish adjacent steps, with at
    # least one so integral values still read as numbers.
    decimals: int = max(1, int(math.ceil(-math.log10(step))) + 1)

    return st.slider(
        param_label(name, mineral_names),
        min_value=float(lo),
        max_value=float(hi),
        value=float(0.5 * (lo + hi)),
        step=step,
        format=f"%.{decimals}f",
        key=f"param:{name}",
    )
# =================== #
