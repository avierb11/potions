"""
The data loader and run machinery of the East River calibration GUI.

Loads a record spreadsheet — a ``.csv`` or ``.xlsx`` file; the
record shipped with the package is the East River series
(``potions/gui/data/east_river_series.csv``) — and assembles
everything a run needs: the run's reaction network, the model's
parameter bounds, the reactive-transport zone configurations, the
forcing, the precipitation concentrations, and the measured series.
The model structure itself is selected by a
`potions.gui.model_specs.ModelSpec` key; every spec runs over the
loaded record.

The record is a daily table of the site's measurements.  The
bundled record is the East River series over its forcing span
(2014-10-01 to 2021-09-30): the meteorological forcing (``ppt``,
``temp``, ``pet``, in mm/day and degrees Celsius), the measured
streamflow (``q_meas``, in millimeters per day), and the measured
river concentrations (``doc_molL``, ``dic_molL``, ``ca_molL``, in
moles per liter).  A different record is accepted in the same shape:
its columns must resolve to the record roles — automatically, by
the canonical names or the East River aliases (case-insensitive), or
through an explicit role → column map — and its units are taken as
the model's, with no conversion.  The loader trims the record to its
forcing span (the meteorological forcing is daily only over that
span), rejecting interior forcing NaNs, missing dates, and
non-daily cadence; rows outside the span are the only rows ever
dropped.  The record is a single continuous run: the loader
prepends a spin-up (a copy of the record's leading days — the
requested length, capped at half the record — shifted back in time
with its measurements set to NaN).

The run functions (`arr_to_model`, `arr_to_results`, and
`arr_to_hydro_results`) interpret a flat parameter array in the
bounds' key order: the hydro parameters, the per-zone
reactive-transport parameters, and the trailing per-speciation-zone
log10 total exchange-site concentrations.  They construct the model
and run it, the same way the showcase's calibration pipeline did, so
the GUI can re-run the models on every parameter change:
`arr_to_results` runs both models together, and
`arr_to_hydro_results` runs only the hydrology model (the GUI's
hydrology-only mode).
"""

import gc
import hashlib
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from xml.etree.ElementTree import ParseError

import numpy as np
import pandas as pd
import potions as pt
from potions.calibration import build_rt_bounds
from potions.gui.common import (
    RIVER_SPECIES,
    WATER_YEARS,
    set_species_colors,
    water_year_window,
)
from potions.gui.model_specs import (
    DEFAULT_MODEL_KEY,
    DEFAULT_MODEL_SPEC,
    ModelSpec,
    get_model_spec,
)
from potions.gui.network_specs import DEFAULT_NETWORK_KEY, get_network_spec
from potions.hydro_model import HydrologicalModel
from scipy.optimize import root_scalar

# ==== Constants ==== #
_PACKAGE_DIR: Final[Path] = Path(__file__).resolve().parent

# The East River forcing-and-chemistry series record, shipped with
# the package (a copy of the site record in the repo's ``input``
# directory) so the GUI works out of the box from an installed copy
# of potions.
DEFAULT_CSV_PATH: Final[Path] = (
    _PACKAGE_DIR / "data" / "east_river_series.csv"
)

# The number of days of the two-year spin-up prepended to the record
# so the model reaches a representative state before the record's
# first day.
NUM_SPINUP_DAYS: Final[int] = 2 * 365

# The (frozen) initial concentration of every mineral in a reactive
# zone, in mol/L: the mineral state does not vary over the simulation,
# so a single reference value is enough; it is absorbed into the
# per-zone k_a parameter.
MINERAL_CONCENTRATION: Final[float] = 1.0

# The aqueous carbonate species that sum to DIC: the measured DIC
# column is derived from them, so it is appended to a network's
# measured columns only when the network carries all three.
CARBONATE_TRIO: Final[tuple[str, ...]] = ("CO2(aq)", "CO3--", "HCO3-")

# The record's column roles: each role's canonical column name and
# its aliases, in descending priority order (the canonical name
# first, then the aliases in table order).  Matching is
# case-insensitive and trims surrounding whitespace.  The ``date``
# role owns the record's index (the reader resolves it); the other
# data roles are resolved by `resolve_columns`.
BASE_COLUMN_ROLES: Final[dict[str, tuple[str, tuple[str, ...]]]] = {
    "date": ("date", ("datetime", "timestamp", "time")),
    "precip": ("precip", ("precipitation", "ppt", "rainfall", "rain")),
    "temp": ("temp", ("temperature", "tair", "air_temperature")),
    "pet": ("pet", ("etp", "et0", "epot", "potential_evapotranspiration")),
    "q_meas": ("q_meas", ("streamflow", "discharge", "flow")),
}

# The default plotting window: the first and last day of the site
# record's water years, as ISO dates.
_DEFAULT_WINDOW: Final[tuple[pd.Timestamp, pd.Timestamp]] = (
    water_year_window(WATER_YEARS[0], WATER_YEARS[-1])
)
DEFAULT_WINDOW_START: Final[str] = _DEFAULT_WINDOW[0].strftime("%Y-%m-%d")
DEFAULT_WINDOW_END: Final[str] = _DEFAULT_WINDOW[1].strftime("%Y-%m-%d")
# =================== #


# ==== Types ==== #
@dataclass(frozen=True)
class PlotSelection:
    """
    The user's choices of which series to draw and over which period.

    The species set is shared by the river-concentration and
    zone-concentration figures, and the zone set by the storage and
    zone-concentration figures; the remaining fields select the
    river-concentration figure's streamflow axis, the
    reaction-rate figure's mineral, and the date window every figure
    is clipped to.

    Attributes:
        species: The river species to plot in the river- and
            zone-concentration figures.
        zones: The zones to plot in the storage and
            zone-concentration figures.
        mineral: The mineral to plot in the reaction-rate figure.
        streamflow: Whether the river-concentration figure's inverted
            streamflow axis is on.
        window_start: The plotting window's first day, as an ISO
            date (``YYYY-MM-DD``).
        window_end: The plotting window's last day, as an ISO date.
    """

    species: list[str]
    zones: list[str]
    mineral: str
    streamflow: bool = True
    window_start: str = DEFAULT_WINDOW_START
    window_end: str = DEFAULT_WINDOW_END

    def plot_window(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """
        The plotting window as timestamps.

        Returns:
            The window's start and end timestamps.

        Raises:
            ValueError: Raised when the dates are malformed or the
                window starts after it ends.
        """
        start: pd.Timestamp = pd.Timestamp(self.window_start)
        end: pd.Timestamp = pd.Timestamp(self.window_end)
        if start > end:
            raise ValueError(
                f"Plot window start {self.window_start!r} is after its "
                f"end {self.window_end!r}."
            )
        return (start, end)


@dataclass(frozen=True)
class GuiInputData:
    """
    All of the inputs a run of the East River GUI needs.

    Assembled once by `load_input_data` and cached for the session.

    Attributes:
        spec: The selected model's spec.
        network: The run's reaction network.
        bounds: The flat parameter bounds, in the order of the run's
            parameter array: the hydro parameters, the per-zone
            reactive-transport parameters, and the trailing
            initial-state entries.
        zone_configs: The reactive-transport configuration of each
            model zone.
        forc: The forcing data over the full (spin-up-inclusive)
            record.
        precip_conc: The precipitation concentration array, one value
            per surface zone, day, and species.
        q_meas: The measured streamflow over the full record, in
            millimeters per day (catchment-mean depth); NaN during
            the spin-up; None when the record carries no streamflow
            column.
        conc_df: The measured river concentrations — the network's
            measured columns, in species order with the derived DIC
            last — over the full record, in moles per liter; NaN
            during the spin-up and on days without a sample; zero
            columns when the record carries no concentrations.
        window: The record's forcing span: the first and last date
            of the trimmed record (before the spin-up is prepended).
    """

    spec: ModelSpec
    network: pt.ReactionNetwork
    bounds: dict[str, tuple[float, float]]
    zone_configs: dict[str, pt.RtZoneConfiguration]
    forc: pt.ForcingData
    precip_conc: np.ndarray
    q_meas: pd.Series | None
    conc_df: pd.DataFrame
    window: tuple[pd.Timestamp, pd.Timestamp]


@dataclass(frozen=True)
class GuiRunResults:
    """
    The results of one finished GUI run.

    The hydrology results are always present; the reactive-transport
    results are None when the run was done in hydrology-only mode
    (`arr_to_hydro_results`).

    Attributes:
        hydro: The hydrology model's results.
        rt: The reactive-transport model's results, or None when the
            run was hydrology-only.
    """

    hydro: pt.HydroModelResults
    rt: pt.RtModelResults | None
# =============== #


# ==== Functions ==== #
def _max_water_bounds(zone_class: type) -> tuple[float, float]:
    """
    The reactive water-storage bounds of a zone class.

    Args:
        zone_class: The zone's hydrology class.

    Returns:
        The lower and upper bounds of the zone's ``max_water``
        parameter, in millimeters.

    Raises:
        ValueError:
            Raised for a zone class the GUI has no storage bounds for.
    """
    if issubclass(zone_class, pt.SnowZone):
        return (5000.0, 5000.0)
    if issubclass(zone_class, pt.SurfaceZone):
        return (150.0, 1500.0)
    if issubclass(zone_class, pt.SubsurfaceZone):
        return (150.0, 1500.0)
    if issubclass(zone_class, pt.SubsurfaceZoneB):
        return (100.0, 5000.0)
    raise ValueError(
        f"No max_water bounds for zone class {zone_class.__name__}."
    )


def _initial_state_bounds(zone_class: type) -> tuple[float, float]:
    """
    The exchange-site bounds of a speciation zone class.

    The deep, slow-draining subsurface zone holds a lower exchange-site
    concentration than the surface and shallow zones.

    Args:
        zone_class: The zone's hydrology class.

    Returns:
        The lower and upper bounds of the zone's log10 total
        exchange-site concentration, in mol/L.
    """
    if issubclass(zone_class, pt.SubsurfaceZoneB):
        return (-7.0, -3.0)
    return (-5.0, -1.0)


def _bounds(
    spec: ModelSpec, network: pt.ReactionNetwork
) -> dict[str, tuple[float, float]]:
    """
    Assemble the flat parameter bounds of a model spec.

    Starts from the generic bounds of `build_rt_bounds` (the hydro
    parameters in the model's order, then each zone's storage and
    per-mineral kinetic parameters, in the model's per-zone order) and
    applies the GUI's East River overrides on the same keys: the
    per-zone-class storage bounds, the pinned per-column drainage
    alphas, the calibrated catchment proportion, and the trailing
    per-speciation-zone initial-state (exchange-site) bounds.

    The key order matches the run's flat parameter array
    (`arr_to_model` / `arr_to_results`).

    Args:
        spec: The selected model's spec.
        network: The reaction network of the run.

    Returns:
        The bounds of every flat parameter of the run, in array order.
    """
    bounds: dict[str, tuple[float, float]] = build_rt_bounds(
        spec.model_class,
        # build_rt_bounds is annotated with the legacy Python-side
        # ReactionNetwork, but it only reads the network's mineral
        # count, so the core network it actually receives at runtime
        # (and the one from_array consumes) is accepted here.
        network,  # type: ignore[arg-type]
        spec.zone_configs,
    )
    zone_classes: dict[str, type] = {
        zone: spec.model_class.get_zone_type(zone) for zone in spec.zones
    }

    for zone in spec.zones:
        bounds[f"{zone}.max_water"] = _max_water_bounds(zone_classes[zone])

    # A model with more than one surface column splits the catchment
    # between them, and pins the per-column drainage-fraction alphas.
    if "proportion.1" in bounds:
        bounds["proportion.1"] = (0.25, 0.95)
        for zone in spec.zones:
            alpha_key: str = f"{zone}.alpha"
            if alpha_key in bounds and issubclass(
                zone_classes[zone],
                (pt.SubsurfaceZone, pt.SubsurfaceZoneB),
            ):
                bounds[alpha_key] = (1.0, 1.0)

    if network.has_exchange:
        for zone in spec.speciation_zones:
            bounds[f"initial_state.{zone}_x"] = _initial_state_bounds(
                zone_classes[zone]
            )

    return bounds


def _measured_columns(
    network: pt.ReactionNetwork, df: pd.DataFrame
) -> list[str]:
    """
    The measured columns a network reads from the East River CSV.

    The network's primary and secondary species that have a column in
    the frame, in the network's species order, followed by ``"DIC"``
    when the network carries the whole carbonate trio and the frame
    has the column (DIC is derived, as today, from the trio).

    Args:
        network: The run's reaction network.
        df: The CSV frame as loaded (before the spin-up NaN-ing).

    Returns:
        The column names, e.g. ``["DOC", "Ca++", "DIC"]`` for the
        default network.
    """
    carried: set[str] = set(network.species_names)
    aqueous: set[str] = set(network.primary_names) | set(network.secondary_names)
    columns: list[str] = [
        name
        for name in network.species_order
        if name in aqueous and name in df.columns
    ]
    if set(CARBONATE_TRIO) <= carried and "DIC" in df.columns:
        columns.append("DIC")
    return columns


def _species_alias(species: str) -> str:
    """
    The spreadsheet column alias for a measured-species record role.

    The species name lowercased with every non-alphanumeric character
    removed, followed by ``"_molL"`` — the bundled record's naming
    convention (``"DOC"`` → ``"doc_molL"``, ``"Ca++"`` →
    ``"ca_molL"``, ``"HCO3-"`` → ``"hco3_molL"``).

    Args:
        species: The measured-species name.

    Returns:
        The alias column name.
    """
    stem: str = "".join(ch for ch in species.lower() if ch.isalnum())
    return stem + "_molL"


def _role_names(role: str) -> tuple[str, ...]:
    """
    The column names that resolve a record role, in priority order.

    The canonical name first, then the aliases in table order (the
    canonical name wins a tie).  A measured-species role's names are
    the species name itself, then its `_species_alias` form.

    Args:
        role: A record role key: a key of `BASE_COLUMN_ROLES` or a
            measured-species name from `measured_roles`.

    Returns:
        The candidate column names, in resolution order.
    """
    if role in BASE_COLUMN_ROLES:
        canonical, aliases = BASE_COLUMN_ROLES[role]
        return (canonical, *aliases)
    return (role, _species_alias(role))


def _auto_match(lowered: dict[str, str], role: str) -> str | None:
    """
    The record column matching a role's names, or None.

    The role's names — its canonical name first, then its aliases —
    are looked up in `lowered` in resolution order; the first hit
    wins.  `lowered` maps each column name (lowercased, trimmed) to
    the column's actual name.

    Args:
        lowered: The column-name lookup table, as built by
            `resolve_columns` and `prefill_mapping`.
        role: A record role key: a key of `BASE_COLUMN_ROLES` or a
            measured-species name from `measured_roles`.

    Returns:
        The matched column's actual name, or None when the role
        matches no column.
    """
    for candidate in _role_names(role):
        match: str | None = lowered.get(candidate.strip().lower())
        if match is not None:
            return match
    return None


def measured_roles(network: pt.ReactionNetwork) -> list[str]:
    """
    The measured-species record roles a network can carry, in plot
    order.

    The network's primary and secondary species that the GUI plots
    as river concentrations (its intersection with `RIVER_SPECIES`),
    in the network's species order, followed by ``"DIC"`` when the
    network carries the whole carbonate trio (DIC is derived from the
    trio, as the figures derive it).

    Args:
        network: The run's reaction network.

    Returns:
        The species names, e.g. ``["DOC", "Ca++", "DIC"]`` for the
        default network; empty for a network without river species.
    """
    aqueous: set[str] = set(network.primary_names) | set(network.secondary_names)
    roles: list[str] = [
        name
        for name in network.species_order
        if name in aqueous and name in RIVER_SPECIES
    ]
    if "DIC" not in roles and set(CARBONATE_TRIO) <= set(network.species_names):
        roles.append("DIC")
    return roles


def resolve_columns(
    df: pd.DataFrame,
    network: pt.ReactionNetwork,
    column_map: dict[str, str | None] | None = None,
) -> dict[str, str]:
    """
    Resolve a record's data columns to its record roles.

    Each role resolves, in order: the explicit `column_map` value
    (a `None` value marks an optional role deliberately absent; a
    string value must name an existing column), then the role's
    canonical column name, then its aliases — the matching
    case-insensitive and trims surrounding whitespace.  The ``date``
    role is excluded: the reader owns the record's index.  The
    resolved columns keep the record's original spelling.

    Args:
        df: The loaded record frame (any columns).
        network: The run's reaction network: it selects the
            measured-species roles to resolve.
        column_map: The explicit role → column override, or None for
            pure auto-detection.

    Returns:
        The resolved roles (``precip``, ``temp``, ``pet``,
        ``q_meas``, then the measured species in plot order) as
        role key → actual column name, in that key order.  Optional
        roles that did not resolve are absent from the result.

    Raises:
        ValueError:
            Raised when an explicit map value names a column the
            record does not have (the message lists the columns that
            do), when two roles resolve to the same column (the
            message names the column and both roles), or when a
            required role (precipitation, temperature, PET) does not
            resolve (the message lists the missing roles and the
            columns that exist).
    """
    columns: list[str] = [str(name) for name in df.columns]
    lowered: dict[str, str] = {
        str(name).strip().lower(): str(name) for name in df.columns
    }
    roles: list[str] = [
        "precip",
        "temp",
        "pet",
        "q_meas",
        *measured_roles(network),
    ]
    resolved: dict[str, str] = {}
    for role in roles:
        if column_map is not None and role in column_map:
            name: str | None = column_map[role]
            if name is None:
                continue
            if name not in df.columns:
                raise ValueError(
                    f"The column map points the role {role!r} at the "
                    f"column {name!r}, which the record does not have. "
                    f"Available columns: {', '.join(columns)}."
                )
            resolved[role] = name
            continue
        match: str | None = _auto_match(lowered, role)
        if match is not None:
            resolved[role] = match
    for column, role_names in _roles_by_column(resolved).items():
        if len(role_names) > 1:
            raise ValueError(
                f"Two record roles resolve to the same column {column!r}: "
                f"{', '.join(role_names)}. Map them to different columns."
            )
    missing: list[str] = [
        role for role in ("precip", "temp", "pet") if role not in resolved
    ]
    if missing:
        raise ValueError(
            f"Could not resolve the required record role(s) "
            f"{', '.join(missing)} from the available columns: "
            f"{', '.join(columns)}."
        )
    return resolved


def _roles_by_column(
    resolved: dict[str, str],
) -> dict[str, list[str]]:
    """
    Invert a role resolution: actual column → the roles on it.

    Args:
        resolved: The role → column resolution from `resolve_columns`.

    Returns:
        The column → roles mapping, in resolution order.
    """
    by_column: dict[str, list[str]] = {}
    for role, column in resolved.items():
        by_column.setdefault(column, []).append(role)
    return by_column


def _datetime_column(
    df: pd.DataFrame,
    column_map: dict[str, str | None] | None = None,
) -> str:
    """
    The name of the record's datetime column.

    An explicit map value for the ``date`` role wins (it must name
    an existing column); otherwise the column is auto-detected from
    the date role's names — the canonical name, then the aliases —
    matched case-insensitively with surrounding whitespace trimmed.

    Args:
        df: The loaded record frame.
        column_map: The explicit role → column override, or None for
            pure auto-detection.

    Returns:
        The actual column name of the record's dates.

    Raises:
        ValueError:
            Raised when the map points the date role at a column the
            record does not have (the message lists the columns that
            do) or when no date-like column exists (the message lists
            the columns that do).
    """
    if column_map is not None and "date" in column_map:
        name: str | None = column_map["date"]
        if name is None:
            raise ValueError(
                "The column map marks the date role absent, but a "
                "record needs a date column."
            )
        if name not in df.columns:
            raise ValueError(
                f"The column map points the date role at the column "
                f"{name!r}, which the record does not have. "
                f"Available columns: "
                f"{', '.join(str(column) for column in df.columns)}."
            )
        return str(name)
    lowered: dict[str, str] = {
        str(name).strip().lower(): str(name) for name in df.columns
    }
    for candidate in _role_names("date"):
        match: str | None = lowered.get(candidate.strip().lower())
        if match is not None:
            return match
    raise ValueError(
        "No date column found in the record. Available columns: "
        f"{', '.join(str(column) for column in df.columns)}."
    )


def _float_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert every column of a record frame to float64.

    Args:
        df: The record frame whose data columns should be numeric.

    Returns:
        A new frame, same index, every column float64.

    Raises:
        ValueError:
            Raised when a column holds a non-numeric value; the
            message names the column and the offending value.
    """
    converted: dict[str, pd.Series] = {}
    for column in df.columns:
        series: pd.Series = df[column]
        numeric: pd.Series = pd.to_numeric(series, errors="coerce")
        bad_mask: pd.Series = numeric.isna() & series.notna()
        if bad_mask.any():
            bad_value: object = series[bad_mask].iloc[0]
            raise ValueError(
                f"The column {str(column)!r} is not numeric: the "
                f"value {bad_value!r} is not a number."
            )
        converted[str(column)] = numeric.astype("float64")
    return pd.DataFrame(converted, index=df.index)


def _read_raw(path: Path) -> pd.DataFrame:
    """
    Read a record spreadsheet into a raw frame by its suffix.

    ``.csv`` files are read with `pandas.read_csv`, ``.xlsx`` files
    with `pandas.read_excel` and the openpyxl engine.  The frame is
    as the file holds it: no index, no typing (the caller parses
    the dates and coerces the data columns).

    Args:
        path: The record file.

    Returns:
        The spreadsheet as read: one row per record row, the file's
        header as column names.

    Raises:
        ValueError:
            Raised when the file has no supported extension, or the
            spreadsheet cannot be parsed (a non-zip file or a zip
            whose XML is malformed — the message includes the
            parser's error).

        OSError:
            Raised when the file cannot be read or holds no valid
            workbook part.
    """
    suffix: str = path.suffix.lower()
    if suffix not in {".csv", ".xlsx"}:
        raise ValueError(
            f"Unsupported record file {path.name!r}: the record must "
            "be a CSV (.csv) or XLSX (.xlsx) spreadsheet."
        )
    try:
        if suffix == ".csv":
            raw: pd.DataFrame = pd.read_csv(path)
        else:
            raw = pd.read_excel(path, engine="openpyxl")
    except (zipfile.BadZipFile, ParseError) as err:
        raise ValueError(
            f"Could not read the record spreadsheet {path.name!r}: "
            f"{err}."
        ) from err
    return raw


def read_spreadsheet(
    path: Path,
    column_map: dict[str, str | None] | None = None,
) -> pd.DataFrame:
    """
    Read a CSV or XLSX record into an indexed frame.

    The file is read by `_read_raw` (``.csv`` → `pandas.read_csv`,
    ``.xlsx`` → `pandas.read_excel` with the openpyxl engine).  The
    record's date column — the map's ``date`` value when present,
    otherwise auto-detected by `_datetime_column` — is parsed with
    `pandas.to_datetime` (real date cells and text dates alike
    become timestamps) and set as the index; the rows come back in
    date order and the remaining columns as float64.

    Args:
        path: The record file: a ``.csv`` or ``.xlsx`` spreadsheet.
        column_map: The explicit role → column override; only its
            ``date`` value is consulted here (the data roles are
            resolved later by `resolve_columns`).

    Returns:
        The record indexed by a sorted `pandas.DatetimeIndex`
        (named ``"date"``), all data columns float64.

    Raises:
        ValueError:
            Raised when the file has no supported extension, the
            spreadsheet cannot be parsed (a non-zip file or a zip
            whose XML is malformed — the message includes the
            parser's error), the date column is missing (the message
            lists the available columns), two rows share a date (the
            message names the duplicated date), or a data column is
            not numeric (the message names the column and the
            offending value).
    """
    raw: pd.DataFrame = _read_raw(path)
    date_column: str = _datetime_column(raw, column_map)
    dates: pd.Series = pd.to_datetime(raw[date_column])
    if dates.duplicated().any():
        first_duplicate: pd.Timestamp = dates[
            dates.duplicated(keep="first")
        ].iloc[0]
        raise ValueError(
            f"Duplicate dates in the record: {first_duplicate.date()}. "
            "The record must have at most one row per date."
        )
    out: pd.DataFrame = raw.drop(columns=[date_column]).set_index(dates)
    out.index.name = "date"
    return _float_frame(out.sort_index())


def trim_record(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trim a loaded record to its forcing span and validate it.

    The forcing span is the first-to-last date over which all three
    forcing columns (``precip``, ``temp``, ``pet``) are non-NaN.
    Rows outside the span — observations that extend beyond the
    forcing period — are the only rows the loader ever drops.  The
    surviving record must then be exactly daily with no forcing NaN
    or gaps inside the span; measured columns (``q_meas`` and
    concentrations) may be NaN anywhere, including the whole span.

    Args:
        df: The record indexed by a sorted `pandas.DatetimeIndex`
            with its forcing columns renamed to the internal names
            (the `read_spreadsheet` + `resolve_columns` output).

    Returns:
        The record restricted to the forcing span, all forcing
        complete and exactly daily.

    Raises:
        ValueError:
            Raised when no row carries complete forcing, when a
            forcing column has a missing value inside the span (the
            message names the column, the missing count, and the
            first affected date), when a date is missing inside the
            span (the message names it), when the surviving cadence
            is not daily (the message states the observed step), or
            when no rows remain at all.
    """
    forcing: list[str] = ["precip", "temp", "pet"]
    forcing_rows: pd.Series = df[forcing].notna().all(axis=1)
    span_dates: pd.DatetimeIndex = df.index[forcing_rows]
    if len(span_dates) == 0:
        raise ValueError(
            "The record has no forcing: no row where precipitation, "
            "temperature, and PET are all present."
        )
    span_start: pd.Timestamp = span_dates.min()
    span_end: pd.Timestamp = span_dates.max()
    out: pd.DataFrame = df.loc[span_start:span_end]
    for column in forcing:
        nans: pd.Series = out[column].isna()
        if nans.any():
            first_date: pd.Timestamp = out.index[nans].min()
            raise ValueError(
                f"The forcing column {column!r} has "
                f"{int(nans.sum())} missing value(s) inside the "
                f"forcing span; first at {first_date.date()}."
            )
    expected: pd.DatetimeIndex = pd.date_range(span_start, span_end, freq="D")
    missing: pd.DatetimeIndex = expected.difference(out.index)
    if len(missing) > 0:
        raise ValueError(
            f"Missing date(s) inside the forcing span: "
            f"{missing[0].date()} ({len(missing)} missing day(s) of "
            f"{len(expected)})."
        )
    if len(out) != len(expected):
        steps: pd.Series = out.index.to_series().diff().dropna()
        median_step: pd.Timedelta = steps.median()
        raise ValueError(
            f"The record is not daily inside the forcing span: "
            f"{len(out)} rows with a median step of {median_step} "
            f"where {len(expected)} daily dates were expected."
        )
    return out


def read_record(
    path: Path,
    network: pt.ReactionNetwork,
    column_map: dict[str, str | None] | None = None,
) -> pd.DataFrame:
    """
    Load a record file into a runnable record frame.

    The file is read by `read_spreadsheet` (CSV / XLSX, sorted
    datetime index, float64 columns), its columns resolved to the
    record roles by `resolve_columns`, renamed to the internal names
    (``precip``, ``temp``, ``pet``, ``q_meas``, species names), and
    trimmed to the forcing span by `trim_record`.  Spreadsheet
    columns that resolve to no role are ignored.

    Args:
        path: The record file (a ``.csv`` or ``.xlsx`` spreadsheet).
        network: The run's reaction network: it selects the
            measured-species roles to resolve.
        column_map: The explicit role → column override; None lets
            `resolve_columns` auto-detect the roles from the column
            names.

    Returns:
        The record indexed by a sorted `pandas.DatetimeIndex` and
        restricted to the forcing span: ``precip``, ``temp``,
        ``pet``, then ``q_meas`` and the resolved measured species
        (in plot order) when present — the forcing complete and
        exactly daily, the measured columns allowed to be NaN.

    Raises:
        ValueError:
            Raised by the underlying reader, role resolver, or span
            trim: an unsupported file extension, a missing or
            duplicated date column, a non-numeric data column, an
            unresolved required role, a role collision, a map value
            pointing at a missing column, an interior forcing NaN, a
            missing date, or a non-daily cadence.
    """
    raw: pd.DataFrame = read_spreadsheet(path, column_map)
    resolved: dict[str, str] = resolve_columns(raw, network, column_map)
    rename: dict[str, str] = {
        column: role for role, column in resolved.items()
    }
    out: pd.DataFrame = raw.rename(columns=rename)[list(resolved.keys())]
    return trim_record(out)


def prefill_mapping(
    path: Path,
    network: pt.ReactionNetwork,
) -> tuple[list[str], dict[str, str | None]]:
    """
    The dialog prefill of the Load data flow: the record's columns
    and the auto-detected role → column mapping.

    The file is read by `_read_raw` (file-level errors — an
    unsupported suffix, an unreadable file — propagate to the
    caller), its date column is auto-detected by `_datetime_column`
    (a missing date column is a file-level error, reported before
    the dialog opens), and the data roles — streamflow and the
    network's measured species — are resolved leniently: each
    role's names are matched against the columns exactly as
    `resolve_columns` matches them, but an unmatched role yields
    None instead of an error and no collision is checked (the
    dialog enforces the constraints and reports them live).

    Args:
        path: The record file (a ``.csv`` or ``.xlsx`` spreadsheet).
        network: The run's reaction network: it selects the
            measured-species roles to prefill.

    Returns:
        ``(columns, prefill)``: the file's column names in file
        order (the date column included) and the prefill mapping
        with a ``date`` entry plus one entry per data role (a
        matched column name or None).

    Raises:
        ValueError:
            Raised when the file has no supported extension, the
            spreadsheet cannot be parsed (a non-zip file or a zip
            whose XML is malformed — the message includes the
            parser's error), or no date column (the message lists
            the available columns).

        OSError:
            Raised when the file cannot be read.
    """
    raw: pd.DataFrame = _read_raw(path)
    date_column: str = _datetime_column(raw)
    columns: list[str] = [str(name) for name in raw.columns]
    lowered: dict[str, str] = {
        str(name).strip().lower(): str(name) for name in raw.columns
    }
    prefill: dict[str, str | None] = {"date": date_column}
    for role in ("precip", "temp", "pet", "q_meas", *measured_roles(network)):
        prefill[role] = _auto_match(lowered, role)
    return columns, prefill


def _with_spinup(
    df: pd.DataFrame,
    measured_columns: list[str],
    spinup_days: int = NUM_SPINUP_DAYS,
) -> pd.DataFrame:
    """
    Prepend the spin-up to the record.

    The first ``spinup_days`` rows of the record are copied, their
    index shifted back by the same number of days, and their measured
    values (``q_meas`` and the run's measured concentration columns,
    whichever exist) set to NaN, so the model integrates a spin-up
    period before the first day of the record and the measured
    series are sparse over it.

    Args:
        df: The record indexed by daily dates, sorted ascending.
        measured_columns: The run's measured concentration columns
            (from `_measured_columns` or the resolved species
            roles).
        spinup_days: The spin-up length in days; the caller caps it
            to at most half the record (`load_input_data`).

    Returns:
        The spin-up-inclusive record (the shifted copy, then the
        record itself); a copy of the record itself when
        ``spinup_days`` is zero.
    """
    if spinup_days <= 0:
        return df.copy()
    spinup: pd.DataFrame = df.iloc[:spinup_days].copy()
    spinup.index = pd.DatetimeIndex(spinup.index) - pd.Timedelta(
        days=spinup_days
    )
    for column in ("q_meas",) + tuple(measured_columns):
        if column in spinup.columns:
            spinup[column] = np.nan
    return pd.concat([spinup, df])


def load_input_data(
    path: Path = DEFAULT_CSV_PATH,
    model_key: str = DEFAULT_MODEL_KEY,
    network: pt.ReactionNetwork | None = None,
    column_map: dict[str, str | None] | None = None,
    spinup_days: int = NUM_SPINUP_DAYS,
) -> GuiInputData:
    """
    Load and assemble the GUI inputs from a record spreadsheet.

    Builds the selected model's bounds, loads the record through
    `read_record` (CSV or XLSX; the roles auto-detected from the
    column names or overridden by `column_map`; trimmed to the
    forcing span), prepends the spin-up, and creates the forcing,
    the precipitation concentrations, and the measured series.  The
    measured series are the record's resolved species columns in
    plot order (the default network's record yields "DOC", "Ca++",
    "DIC" — species order, DIC last); a record without a streamflow
    column loads with ``q_meas = None`` and one without
    concentrations with an empty ``conc_df``.

    Args:
        path: The record file (a ``.csv`` or ``.xlsx`` spreadsheet);
            defaults to the record shipped with the package.
        model_key: The key of the selected model's spec; defaults to
            the lateral three-layer model.
        network: The run's reaction network; defaults to the SOC /
            DOC / calcite network.
        column_map: The explicit role → column override; None lets
            the loader auto-detect the roles from the column names.
        spinup_days: The spin-up length in days, capped at half the
            record; defaults to `NUM_SPINUP_DAYS`.

    Returns:
        The assembled `GuiInputData`; its ``window`` is the
        record's forcing span.

    Raises:
        ValueError: Raised for an unknown model key or by the
            record loader: an unsupported file extension, a missing
            or duplicated date column, a non-numeric data column, an
            unresolved required role, a role collision, a map value
            pointing at a missing column, an interior forcing NaN,
            a missing date, or a non-daily cadence.
    """
    spec: ModelSpec = get_model_spec(model_key)
    network = network if network is not None else default_network()
    bounds: dict[str, tuple[float, float]] = _bounds(spec, network)

    df: pd.DataFrame = read_record(path, network, column_map)
    window: tuple[pd.Timestamp, pd.Timestamp] = (df.index[0], df.index[-1])
    measured: list[str] = [
        column
        for column in df.columns
        if column not in ("precip", "temp", "pet", "q_meas")
    ]
    df = _with_spinup(df, measured, max(0, min(spinup_days, len(df) // 2)))

    precip_conc: np.ndarray = np.full(
        (
            len(spec.model_class.structure[0]),
            len(df),
            network.num_species,
        ),
        fill_value=1e-20,
        dtype=np.float64,
    )
    forc: pt.ForcingData = pt.ForcingData(
        precip=df["precip"],
        temp=df["temp"],
        pet=df["pet"],
    )
    q_meas: pd.Series | None = (
        df["q_meas"] if "q_meas" in df.columns else None
    )
    conc_df: pd.DataFrame = df[measured]
    # The palette is keyed by the species the figures plot — the
    # network's measured species, a superset of the record's columns —
    # so a record that measures fewer species (or none) still colors
    # every series a figure can draw.
    set_species_colors(network_measured_species(network))

    return GuiInputData(
        spec=spec,
        network=network,
        bounds=bounds,
        zone_configs=spec.zone_configs,
        forc=forc,
        precip_conc=precip_conc,
        q_meas=q_meas,
        conc_df=conc_df,
        window=window,
    )


def _split_arr(
    arr: np.ndarray, input_data: GuiInputData
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a flat parameter array into model and initial-state arrays.

    The initial-state parameters are the trailing entries of the
    array, one log10 total exchange-site concentration per
    speciation zone, and only exist when the network has an exchange
    species.  The split point is computed as an explicit index rather
    than by negative slicing, so a network without exchange (no
    trailing initial-state entries) yields the whole array as model
    parameters and an empty initial-state array.

    Args:
        arr: The flat parameter array of model parameters followed by
            initial-state parameters.
        input_data: The assembled GUI inputs.

    Returns:
        The model parameter array and the initial-state parameter
        array.
    """
    num_spec_zones: int = sum(
        1 for cfg in input_data.zone_configs.values() if cfg.do_speciation
    )
    num_init_states: int = num_spec_zones * int(input_data.network.has_exchange)
    split_index: int = arr.size - num_init_states

    return arr[:split_index], arr[split_index:]


def arr_to_model(arr: np.ndarray, input_data: GuiInputData) -> tuple[pt.Model, np.ndarray]:
    """
    Construct the East River model from a flat parameter array.

    Splits the array into the model parameters and the trailing
    initial-state entries and constructs the model from the model
    parameters.  If the measured river concentration data contains a
    DIC column, the model is declared with DIC as a derived species
    (the sum of the aqueous carbonate species) so that the reactive
    transport model computes and scores the simulated DIC river series
    itself.

    Args:
        arr: The flat parameter array of model parameters followed by
            initial-state parameters.
        input_data: The assembled GUI inputs.

    Returns:
        The constructed model and the initial-state parameter array.
    """
    arr = arr.copy()
    model_params, init_state_params = _split_arr(arr, input_data)

    # DIC is not a species of the network: it is the sum of the
    # aqueous carbonate species, so it is declared as a derived
    # measurement on the model.
    derived_species: list[pt.DerivedSpecies] = (
        [pt.DerivedSpecies(name="DIC", components=("CO2(aq)", "HCO3-", "CO3--"))]
        if "DIC" in input_data.conc_df.columns
        else []
    )

    model: pt.Model = input_data.spec.model_class.from_array(
        model_params,
        network=input_data.network,
        config=input_data.zone_configs,
        natural_scales=True,
        derived_species=derived_species,
    )

    return model, init_state_params


@dataclass(frozen=True)
class _ExchangeProduct:
    """
    One adsorbed exchange product and its equilibrium terms.

    The parsed form of one exchange reaction's product side, used by
    the exchange-site initialization solver: the product species, the
    ``X-`` coefficient, the other reactants with their coefficients,
    the product's (positive) stoichiometric magnitude, and the stored
    ``log10_k_eq`` (the log10 of the inverse thermodynamic K).

    Attributes:
        name: The product species' name.
        x_coeff: The ``X-`` stoichiometric coefficient.
        reactants: The remaining reactant species and their
            (positive) stoichiometric coefficients.
        product_coeff: The product's stoichiometric magnitude.
        log10_k_eq: The reaction's stored ``log10_k_eq``.
    """

    name: str
    x_coeff: float
    reactants: tuple[tuple[str, float], ...]
    product_coeff: float
    log10_k_eq: float


def _exchange_product_terms(reaction: pt.core.ExchangeReaction) -> _ExchangeProduct:
    """
    Parse an exchange reaction into its equilibrium-product terms.

    The product is the single non-``X-`` species with a negative
    stoichiometric coefficient (its positive magnitude is
    ``product_coeff``); ``x_coeff`` is the ``X-`` coefficient and
    ``reactants`` the remaining positive-coefficient species.

    Args:
        reaction: The network's exchange reaction to parse.

    Returns:
        The parsed product terms.
    """
    stoich: dict[str, float] = dict(reaction.stoichiometry)
    product_name: str = ""
    product_coeff: float = 0.0
    for name, coeff in stoich.items():
        if name != "X-" and coeff < 0.0:
            product_name, product_coeff = name, -coeff
    reactants: tuple[tuple[str, float], ...] = tuple(
        (name, coeff)
        for name, coeff in stoich.items()
        if name != "X-" and name != product_name and coeff > 0.0
    )
    return _ExchangeProduct(
        name=product_name,
        x_coeff=stoich.get("X-", 0.0),
        reactants=reactants,
        product_coeff=product_coeff,
        log10_k_eq=reaction.log10_k_eq,
    )


def _exchange_product_concentration(
    x: float, product: _ExchangeProduct, species_init: dict[str, float]
) -> float:
    """
    The equilibrium concentration of one exchange product.

    ``[E](x) = 10**(( -log10_k_eq + a·log10(x) + Σ b·log10([S]_0) ) / c)``
    where ``a`` is the ``X-`` coefficient, ``b`` the other reactants'
    coefficients, ``c`` the product's (positive) coefficient, and
    ``[S]_0`` the reactants' initial concentrations.  An absent
    reactant (``x <= 0`` with ``a > 0``, or ``[S]_0 <= 0`` with
    ``b > 0``) makes the product concentration exactly zero.

    Args:
        x: The candidate ``X-`` concentration.
        product: The parsed product terms.
        species_init: The initial concentration of every aqueous
            species (the fixed reactant concentrations).

    Returns:
        The product's equilibrium concentration at ``x`` (mol/L).
    """
    if product.x_coeff > 0.0 and x <= 0.0:
        return 0.0
    exponent: float = -product.log10_k_eq
    if product.x_coeff > 0.0:
        exponent += product.x_coeff * math.log10(x)
    for name, coeff in product.reactants:
        value: float = species_init.get(name, 0.0)
        if coeff > 0.0 and value <= 0.0:
            return 0.0
        if coeff != 0.0:
            exponent += coeff * math.log10(value)
    concentration: float = 10.0 ** (exponent / product.product_coeff)
    return concentration


def _solve_exchange_initial_state(
    tot: float,
    exchange_species: list[pt.core.ExchangeReaction],
    species_init: dict[str, float],
) -> dict[str, float]:
    """
    Solve the exchange-site mass balance for the initial state.

    Finds ``x = [X-]`` such that the free cation plus every adsorbed
    product occupy exactly ``tot`` exchange sites, then returns the
    initial ``"X-"`` and per-product concentrations.  For the 1:1 XDOC
    network this root equals the old closed form ``k·tot/(k+doc)``.

    Args:
        tot: The total exchange-site concentration.
        exchange_species: The network's exchange reactions.
        species_init: The initial concentration of every aqueous
            species (the fixed reactant concentrations).

    Returns:
        A mapping ``{"X-": x, <product>: [product], ...}``.
    """
    products: list[_ExchangeProduct] = [
        _exchange_product_terms(r) for r in exchange_species
    ]

    def residual(x: float) -> float:
        return x + sum(
            _exchange_product_concentration(x, p, species_init) for p in products
        ) - tot

    if tot <= 0.0 or residual(0.0) >= 0.0:
        # Degenerate: no root in (0, tot) — the adsorbed products
        # alone reach the capacity — so keep every site free.
        x: float = tot
    else:
        # residual(tot) = sum [E_i](tot) >= 0, so [0, tot] brackets.
        x = float(root_scalar(residual, bracket=[0.0, tot], xtol=1e-14).root)

    return {
        "X-": x,
        **{
            p.name: _exchange_product_concentration(x, p, species_init)
            for p in products
        },
    }


def _primary_initial_concentrations(
    network: pt.ReactionNetwork,
    conc_df: pd.DataFrame,
    init_aq_conc: float,
) -> dict[str, float]:
    """
    The initial aqueous concentration of each primary species.

    A primary species with a measured concentration column starts at
    that column's mean; every other primary species starts at the
    scalar fallback ``init_aq_conc``.  This generalizes the previous
    behavior, where only DOC started at its column's mean and every
    other primary species started at the scalar.

    Args:
        network: The run's reaction network.
        conc_df: The measured-concentration frame (columns are the
            network's measured columns).
        init_aq_conc: Fallback initial concentration for a primary
            species with no measured column.

    Returns:
        A mapping from each primary species name to its initial
        aqueous concentration (mol/L).
    """
    return {
        name: (
            float(conc_df[name].mean())
            if name in conc_df.columns
            else init_aq_conc
        )
        for name in network.primary_names
    }


def default_network() -> pt.ReactionNetwork:
    """
    Build the GUI's default reaction network.

    The zero-argument builder of the default preset (the East River
    SOC / DOC / calcite network); the fallback for callers and
    persisted state that predate the network selector.

    Returns:
        A freshly built default network.
    """
    return get_network_spec(DEFAULT_NETWORK_KEY).builder()


def fingerprint_network(network: pt.ReactionNetwork) -> str:
    """
    Compute the stable fingerprint of a reaction network.

    The fingerprint is the SHA-256 hex digest of the network's
    canonical YAML bytes (`NetworkFile.from_network(network).
    to_bytes()`).  The canonical form fixes the section order and
    sorts every map key, so the fingerprint is independent of the
    order the species were added in: two networks share a
    fingerprint exactly when they carry the same entries.  That is
    what lets the window match a file-loaded network (species in
    sorted file order) against the built-in preset (database order)
    the file came from.

    Args:
        network: The network to fingerprint.

    Returns:
        The 64-character lowercase hex digest.
    """
    return hashlib.sha256(
        pt.NetworkFile.from_network(network).to_bytes()
    ).hexdigest()


def arr_to_results(
    arr: np.ndarray,
    input_data: GuiInputData,
    init_aq_conc: float = 1e-3,
    verbose: bool = False,
) -> GuiRunResults:
    """
    Run the hydrology and reactive transport models for a parameter
    array.

    Builds the model and the reactive zone initial states (fixed
    mineral concentrations and the calibrated exchange-site
    concentrations) from the parameters and runs the hydrology and
    reactive transport models over the full (spin-up-inclusive)
    record.

    Args:
        arr: The flat parameter array of model parameters followed by
            initial-state parameters.
        input_data: The assembled GUI inputs.
        init_aq_conc: The initial aqueous concentration of each zone's
            primary species without a measured column, and of all
            secondary species.
        verbose: Whether to print verbose output while running the
            models.

    Returns:
        The run's `GuiRunResults`, with the reactive-transport
        results present.
    """
    model: pt.Model
    init_state_params: np.ndarray
    model, init_state_params = arr_to_model(arr, input_data)

    network: pt.ReactionNetwork = input_data.network
    primary_init: dict[str, float] = _primary_initial_concentrations(
        network, input_data.conc_df, init_aq_conc
    )
    species_init: dict[str, float] = {
        **primary_init,
        **{name: init_aq_conc for name in network.secondary_names},
    }

    # Construct the initial state
    init_state: dict[str, pt.ChemicalState] = {}
    init_idx: int = 0
    for zone_name, zone_config in input_data.zone_configs.items():
        init_i: pt.ChemicalState = pt.ChemicalState()

        for prim_name in network.primary_names:
            init_i.primary[prim_name] = primary_init[prim_name]

        for sec_name in network.secondary_names:
            init_i.secondary[sec_name] = init_aq_conc

        if zone_config.do_reactions:
            for min_name in network.mineral_names:
                init_i.mineral[min_name] = MINERAL_CONCENTRATION

        if zone_config.do_speciation and network.has_exchange:
            tot_exchange_sites: float = 10.0 ** float(init_state_params[init_idx])
            init_idx += 1
            # A small rootfinding step finds the exchange-site
            # partition of the aqueous species at initialization.
            init_i.exchange.update(
                _solve_exchange_initial_state(
                    tot_exchange_sites,
                    network.exchange_species,
                    species_init,
                )
            )

        init_state[zone_name] = init_i

    # ==== Run the reactive transport model ==== #
    # Every measured column is passed to the model.  Each column is
    # scored against a same-named ``{species}_riv`` simulation column:
    # network species against their own river concentrations, and the
    # DIC column against the model-computed derived series (the sum of
    # the aqueous carbonate river series) declared on the model.  The
    # model validates the column names itself and raises for a name it
    # does not recognize.  The measured streamflow is the record's
    # ``q_meas`` column (a catchment-mean depth in millimeters per day,
    # NaN over the spin-up), so the hydrology objective functions are
    # reported against it.  A record without a measured column runs
    # against no measured data: the absent series are passed as
    # ``None`` (the core's sentinel), never as an empty stand-in.
    meas_river_conc: pd.DataFrame | None = (
        input_data.conc_df if not input_data.conc_df.empty else None
    )
    model_res: pt.ModelResults = model.run_both_models(
        forc=input_data.forc,
        precip_conc=input_data.precip_conc,
        init_conc=init_state,
        meas_streamflow=input_data.q_meas,
        meas_river_conc=meas_river_conc,
        verbose=verbose,
        save_results=False,
    )
    gc.collect()

    return GuiRunResults(
        hydro=model_res.hydro, rt=model_res.reactive_transport
    )


def arr_to_hydro_results(
    arr: np.ndarray,
    input_data: GuiInputData,
    verbose: bool = False,
) -> GuiRunResults:
    """
    Run the hydrology model alone for a parameter array.

    The hydrology-only counterpart of `arr_to_results`: builds the
    hydrology model from the array's leading hydro parameters and
    runs it over the full (spin-up-inclusive) record, scoring it
    against the measured streamflow.  The reactive-transport model
    is not run; the returned `GuiRunResults` carries None for its
    reactive-transport results.

    Args:
        arr: The flat parameter array; only the leading hydro
            parameters are used.
        input_data: The assembled GUI inputs.
        verbose: Whether to print verbose output while running the
            model.

    Returns:
        The run's `GuiRunResults`, with no reactive-transport
        results.
    """
    num_hydro: int = input_data.spec.model_class.get_num_hydro_parameters()
    hydro_model: HydrologicalModel = (
        input_data.spec.model_class.hydro_from_array(
            arr[:num_hydro], natural_scales=True
        )
    )
    hydro: pt.HydroModelResults = hydro_model.run_hydro_model(
        input_data.forc,
        meas_streamflow=input_data.q_meas,
        verbose=verbose,
        save_results=False,
    )

    return GuiRunResults(hydro=hydro, rt=None)


def network_measured_species(network: pt.ReactionNetwork) -> list[str]:
    """
    The measured species the figures can show for a network.

    The network's primary and secondary species that are in
    ``RIVER_SPECIES``, in the network's species order, followed by
    ``"DIC"`` when the network carries its carbonate trio (DIC is a
    derived species of the network, not an independent one).  This is
    the figures' species set, and the palette's: the session color
    maps are refilled from these on every load.

    Args:
        network: The run's reaction network.

    Returns:
        The measured species, in plot order.
    """
    aqueous: set[str] = set(network.primary_names) | set(network.secondary_names)
    species: list[str] = [
        name
        for name in network.species_order
        if name in aqueous and name in RIVER_SPECIES
    ]
    if "DIC" not in species and set(CARBONATE_TRIO) <= set(network.species_names):
        species.append("DIC")
    return species


def default_plot_mineral(network: pt.ReactionNetwork | None = None) -> str:
    """
    The default reaction-rate figure's mineral for a network.

    The network's first mineral in its species order; the empty
    string when the network carries no mineral (the reaction-rate
    figure is then simply not shown by the caller).

    Args:
        network: The run's reaction network; the default network
            when None.

    Returns:
        The mineral name, or the empty string.
    """
    net: pt.ReactionNetwork = (
        network if network is not None else default_network()
    )
    return net.mineral_names[0] if net.mineral_names else ""


def default_plot_selection(
    spec: ModelSpec = DEFAULT_MODEL_SPEC,
    network: pt.ReactionNetwork | None = None,
    window: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> PlotSelection:
    """
    The default figure selections for a model spec and network.

    Every measured species of the network (streamflow on) in the
    river-concentration figure, every non-snow zone of the spec in
    the storage and zone-concentration figures, the network's
    dominant mineral in the reaction-rate figure, and the plotting
    window: the record's span when `window` is given, the default
    window otherwise.

    Args:
        spec: The selected model's spec; defaults to the lateral
            three-layer model.
        network: The run's reaction network; the default network
            when None.
        window: The record's forcing span (its first and last day);
            when given, the selection's window is that span's ISO
            dates, otherwise the default window.

    Returns:
        The default `PlotSelection`.
    """
    net: pt.ReactionNetwork = (
        network if network is not None else default_network()
    )
    window_start: str = (
        window[0].strftime("%Y-%m-%d")
        if window is not None
        else DEFAULT_WINDOW_START
    )
    window_end: str = (
        window[1].strftime("%Y-%m-%d")
        if window is not None
        else DEFAULT_WINDOW_END
    )
    return PlotSelection(
        species=network_measured_species(net),
        zones=[zone for zone in spec.zones if zone not in spec.snow_zones],
        mineral=default_plot_mineral(net),
        window_start=window_start,
        window_end=window_end,
    )
# =================== #
