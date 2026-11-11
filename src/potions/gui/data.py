"""
The data loader and run machinery of the East River calibration GUI.

Reads the East River series CSV that ships with the package
(``potions/gui/data/east_river_series.csv``, the sole data source of
the GUI; a different record can be supplied by passing a path to
`load_input_data`) and assembles everything a run needs: the SOC /
DOC / calcite reaction network, the model's parameter bounds, the
reactive-transport zone configurations, the forcing, the
precipitation concentrations, and the measured series.  The model
structure itself is selected by a `potions.gui.model_specs.ModelSpec`
key; every spec runs over the same East River record.

The CSV carries the site's daily record from 2014-05-02 to
2025-09-18: the meteorological forcing (``ppt``, ``temp``, ``pet``),
the measured streamflow (``q_meas``, in millimeters per day), and the
measured river concentrations (``doc_molL``, ``dic_molL``,
``ca_molL``, in moles per liter).  The meteorological forcing is
daily only over 2014-10-01 to 2021-09-30, so the loader restricts the
record to that window.  The record is a single continuous run: the
loader prepends a two-year spin-up (the first two years of the
record, shifted back in time with their measurements set to NaN).

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
from dataclasses import dataclass
from pathlib import Path
from typing import Final

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

# The number of days of the two-year spin-up prepended to the CSV
# record so the model reaches a representative state before the first
# day of the record.
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
            the spin-up.
        conc_df: The measured river concentrations — the network's
            measured columns, in species order with the derived DIC
            last — over the full record, in moles per liter; NaN
            during the spin-up and on days without a sample.
    """

    spec: ModelSpec
    network: pt.ReactionNetwork
    bounds: dict[str, tuple[float, float]]
    zone_configs: dict[str, pt.RtZoneConfiguration]
    forc: pt.ForcingData
    precip_conc: np.ndarray
    q_meas: pd.Series
    conc_df: pd.DataFrame


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


def _load_csv(csv_path: Path) -> pd.DataFrame:
    """
    Read the East River series CSV and prepare it for a run.

    Renames the raw columns to the names the rest of the GUI and the
    model use (``ppt`` to ``precip``, ``doc_molL`` to ``DOC``,
    ``dic_molL`` to ``DIC``, ``ca_molL`` to ``Ca++``) and keeps only
    the rows with complete forcing: the meteorological forcing is
    daily only over 2014-10-01 to 2021-09-30, and the rows outside
    that window carry no forcing to simulate with.

    Args:
        csv_path: Path to the CSV with the daily ``date,ppt,temp,
            pet,q_meas,doc_molL,dic_molL,ca_molL`` record.

    Returns:
        The record indexed by date, sorted ascending, with the
        renamed columns and the forcing window only.
    """
    df: pd.DataFrame = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
    df = df.rename(
        columns={
            "ppt": "precip",
            "doc_molL": "DOC",
            "dic_molL": "DIC",
            "ca_molL": "Ca++",
        }
    )
    forc_cols: list[str] = ["precip", "temp", "pet", "q_meas"]
    return df.loc[df[forc_cols].notna().all(axis=1)].sort_index()


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
        for candidate in _role_names(role):
            match: str | None = lowered.get(candidate.strip().lower())
            if match is not None:
                resolved[role] = match
                break
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


def _with_spinup(
    df: pd.DataFrame, measured_columns: list[str]
) -> pd.DataFrame:
    """
    Prepend the two-year spin-up to the record.

    The first `NUM_SPINUP_DAYS` rows of the record are copied, their
    index shifted back by the same number of days, and their measured
    values (``q_meas`` and the run's measured concentration columns)
    set to NaN, so the model integrates a spin-up period before the
    first day of the record and the measured series are sparse over
    it.

    Args:
        df: The record indexed by daily dates, sorted ascending.
        measured_columns: The run's measured concentration columns
            (from `_measured_columns`).

    Returns:
        The spin-up-inclusive record (the shifted copy of the first
        two years, then the record itself).
    """
    spinup: pd.DataFrame = df.iloc[:NUM_SPINUP_DAYS].copy()
    spinup.index = pd.DatetimeIndex(spinup.index) - pd.Timedelta(
        days=NUM_SPINUP_DAYS
    )
    for column in ("q_meas",) + tuple(measured_columns):
        if column in spinup.columns:
            spinup[column] = np.nan
    return pd.concat([spinup, df])


def load_input_data(
    csv_path: Path = DEFAULT_CSV_PATH,
    model_key: str = DEFAULT_MODEL_KEY,
    network: pt.ReactionNetwork | None = None,
) -> GuiInputData:
    """
    Load and assemble the GUI inputs from the East River series CSV.

    Builds the selected model's bounds, reads the CSV, prepends the
    two-year spin-up, and creates the forcing, the precipitation
    concentrations, and the measured series.  The measured series
    follows the run's network: its primary and secondary species with
    a column in the record, in species order, followed by the derived
    DIC column when the network carries the whole carbonate trio (the
    default network yields ["DOC", "Ca++", "DIC"] — species order,
    DIC last).

    Args:
        csv_path: Path to the East River series CSV; defaults to the
            record shipped with the package.
        model_key: The key of the selected model's spec; defaults to
            the lateral three-layer model.
        network: The run's reaction network; defaults to the SOC / DOC
            / calcite network.

    Returns:
        The assembled `GuiInputData`.

    Raises:
        ValueError: Raised for an unknown model key.
    """
    spec: ModelSpec = get_model_spec(model_key)
    network = network if network is not None else default_network()
    bounds: dict[str, tuple[float, float]] = _bounds(spec, network)

    df: pd.DataFrame = _load_csv(csv_path)
    measured: list[str] = _measured_columns(network, df)
    df = _with_spinup(df, measured)

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
    q_meas: pd.Series = df["q_meas"]
    conc_df: pd.DataFrame = df[measured]
    set_species_colors(measured)

    return GuiInputData(
        spec=spec,
        network=network,
        bounds=bounds,
        zone_configs=spec.zone_configs,
        forc=forc,
        precip_conc=precip_conc,
        q_meas=q_meas,
        conc_df=conc_df,
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
    # does not recognize.  The measured streamflow is the CSV's
    # ``q_meas`` column (a catchment-mean depth in millimeters per day,
    # NaN over the spin-up), so the hydrology objective functions are
    # reported against it.
    model_res: pt.ModelResults = model.run_both_models(
        forc=input_data.forc,
        precip_conc=input_data.precip_conc,
        init_conc=init_state,
        meas_streamflow=input_data.q_meas,
        meas_river_conc=input_data.conc_df,
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

    The network's primary and secondary species that have a measured
    column in the East River record, in the network's species order,
    followed by ``"DIC"`` when the record's DIC column is a derived
    species of the network (its carbonate trio is carried).

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
) -> PlotSelection:
    """
    The default figure selections for a model spec and network.

    Every measured species of the network (streamflow on) in the
    river-concentration figure, every non-snow zone of the spec in
    the storage and zone-concentration figures, the network's
    dominant mineral in the reaction-rate figure, and the whole
    water-year record as the plotting window.

    Args:
        spec: The selected model's spec; defaults to the lateral
            three-layer model.
        network: The run's reaction network; the default network
            when None.

    Returns:
        The default `PlotSelection`.
    """
    net: pt.ReactionNetwork = (
        network if network is not None else default_network()
    )
    return PlotSelection(
        species=network_measured_species(net),
        zones=[zone for zone in spec.zones if zone not in spec.snow_zones],
        mineral=default_plot_mineral(net),
    )
# =================== #
