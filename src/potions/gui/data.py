"""
The data loader and run machinery of the East River lateral-model
calibration GUI.

Reads the East River series CSV that ships with the package
(``potions/gui/data/east_river_series.csv``, the sole data source of
the GUI; a different record can be supplied by passing a path to
`load_input_data`) and assembles everything a run needs: the SOC /
DOC / calcite reaction network, the East River parameter bounds, the
reactive-transport zone configurations, the forcing, the
precipitation concentrations, and the measured series.

The CSV carries the site's daily record from 2014-05-02 to
2025-09-18: the meteorological forcing (``ppt``, ``temp``, ``pet``),
the measured streamflow (``q_meas``, in millimeters per day), and the
measured river concentrations (``doc_molL``, ``dic_molL``,
``ca_molL``, in moles per liter).  The meteorological forcing is
daily only over 2014-10-01 to 2021-09-30, so the loader restricts the
record to that window.  The record is a single continuous run: the
loader prepends a two-year spin-up (the first two years of the
record, shifted back in time with their measurements set to NaN).

The run functions (`arr_to_model` and `arr_to_results`) interpret a
flat parameter array in the bounds' key order: the hydro parameters,
the per-zone reactive-transport parameters, and the trailing
per-speciation-zone log10 total exchange-site concentrations.  They
construct the model and run it, the same way the showcase's
calibration pipeline did, so the GUI can re-run both models on every
sidebar change.
"""

import gc
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import potions as pt
import streamlit as st
from potions.calibration import build_rt_bounds
from potions.reactive_transport.common_networks import get_soc_calcite_network
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

# The reactive-transport zone configurations of the East River model:
# the two snow zones carry no chemistry, and the four non-snow zones
# do both the mineral reactions and the aqueous speciation.
LATERAL_THREE_LAYER_CONFIGS: Final[dict[str, pt.RtZoneConfiguration]] = {
    "snow_hs": pt.RtZoneConfiguration(do_reactions=False, do_speciation=False),
    "snow_rp": pt.RtZoneConfiguration(do_reactions=False, do_speciation=False),
    "surface_hs": pt.RtZoneConfiguration(do_reactions=True, do_speciation=True),
    "surface_rp": pt.RtZoneConfiguration(do_reactions=True, do_speciation=True),
    "ground_hs": pt.RtZoneConfiguration(do_reactions=True, do_speciation=True),
    "ground_rp": pt.RtZoneConfiguration(do_reactions=True, do_speciation=True),
}

# The East River per-zone reactive water storage bounds, in mm, by
# layer: the snow storage is fixed, and the non-snow zones are
# calibrated.
_LAYER_MAX_WATER_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "snow": (5000.0, 5000.0),
    "surface": (150.0, 1500.0),
    "ground": (100.0, 1000.0),
}

# The East River per-layer kinetic bounds, in the order the model
# carries the four parameters of each mineral (sw_threshold, sw_exp,
# q_10, k_a).  The surface layer is fully calibrated; the ground
# layer fixes the soil-moisture and temperature dependencies
# (degenerate bounds) and calibrates only k_a.
_LAYER_KINETIC_BOUNDS: Final[dict[str, dict[str, tuple[float, float]]]] = {
    "surface": {
        "sw_threshold": (0.0, 1.0),
        "sw_exp": (0.5, 2.5),
        "q_10": (1.0, 5.0),
        "k_a": (-15.0, -5.0),
    },
    "ground": {
        "sw_threshold": (0.0, 0.0),
        "sw_exp": (0.0, 0.0),
        "q_10": (1.0, 1.0),
        "k_a": (-15.0, -5.0),
    },
}

# The East River bounds of the calibrated total exchange-site
# concentrations, in log10 mol/L, per layer.
_LAYER_INITIAL_STATE_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "surface": (-5.0, -1.0),
    "ground": (-7.0, -3.0),
}

# The (frozen) initial concentration of every mineral in a reactive
# zone, in mol/L: the mineral state does not vary over the simulation,
# so a single reference value is enough; it is absorbed into the
# per-zone k_a parameter.
MINERAL_CONCENTRATION: Final[float] = 1.0
# =================== #


# ==== Types ==== #
@dataclass(frozen=True)
class GuiInputData:
    """
    All of the inputs a run of the East River GUI needs.

    Assembled once by `load_input_data` and cached for the session.

    Attributes:
        network: The SOC / DOC / calcite reaction network of the run.
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
        conc_df: The measured river concentrations (DOC, DIC, Ca++)
            over the full record, in moles per liter; NaN during the
            spin-up and on days without a sample.
    """

    network: pt.ReactionNetwork
    bounds: dict[str, tuple[float, float]]
    zone_configs: dict[str, pt.RtZoneConfiguration]
    forc: pt.ForcingData
    precip_conc: np.ndarray
    q_meas: pd.Series
    conc_df: pd.DataFrame
# =============== #


# ==== Functions ==== #
def _layer_of(zone_name: str) -> str:
    """
    The layer of a lateral-model zone name.

    The six zone names are the layer name plus the region tag
    (hillslope ``_hs`` or riparian ``_rp``), so the layer is the
    prefix before the first underscore.

    Args:
        zone_name: The name of a zone of the East River model.

    Returns:
        The layer name (snow, surface, or ground).
    """
    return zone_name.split("_", 1)[0]


def _bounds(network: pt.ReactionNetwork) -> dict[str, tuple[float, float]]:
    """
    Assemble the East River flat parameter bounds.

    Starts from the generic bounds of `build_rt_bounds` (the hydro
    parameters in the model's order, then each zone's storage and
    per-mineral kinetic parameters, in the model's per-zone order) and
    applies the East River overrides on the same keys: the per-layer
    storage and kinetic bounds, the pinned per-column drainage alphas,
    the calibrated catchment proportion, and the trailing
    per-speciation-zone initial-state (exchange-site) bounds.

    The key order matches the run's flat parameter array
    (`arr_to_model` / `arr_to_results`).

    Args:
        network: The reaction network of the run.

    Returns:
        The bounds of every flat parameter of the run, in array order.
    """
    bounds: dict[str, tuple[float, float]] = build_rt_bounds(
        pt.LateralThreeLayerModel,
        # build_rt_bounds is annotated with the legacy Python-side
        # ReactionNetwork, but it only reads the network's mineral
        # count, so the core network it actually receives at runtime
        # (and the one from_array consumes) is accepted here.
        network,  # type: ignore[arg-type]
        LATERAL_THREE_LAYER_CONFIGS,
    )

    for zone_name in pt.LateralThreeLayerModel.get_zone_names():
        layer: str = _layer_of(zone_name)
        bounds[f"{zone_name}.max_water"] = _LAYER_MAX_WATER_BOUNDS[layer]

        if not LATERAL_THREE_LAYER_CONFIGS[zone_name].do_reactions:
            continue

        for mineral_id in range(network.num_minerals):
            for field, bnd in _LAYER_KINETIC_BOUNDS[layer].items():
                bounds[f"{zone_name}.{field}[{mineral_id}]"] = bnd

    # Pin the per-column drainage-fraction alphas of the lateral
    # model; the bounds only carry them if the generic hydro bounds do.
    for fixed_alpha in (
        "ground_hs.alpha",
        "ground_rp.alpha",
    ):
        if fixed_alpha in bounds:
            bounds[fixed_alpha] = (1.0, 1.0)

    if "proportion.1" in bounds:
        bounds["proportion.1"] = (0.25, 0.95)

    if network.has_exchange:
        for zone_name in pt.LateralThreeLayerModel.get_zone_names():
            if not LATERAL_THREE_LAYER_CONFIGS[zone_name].do_speciation:
                continue
            init_layer: str = _layer_of(zone_name)
            bounds[f"initial_state.{zone_name}_x"] = (
                _LAYER_INITIAL_STATE_BOUNDS[init_layer]
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


def _with_spinup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepend the two-year spin-up to the record.

    The first `NUM_SPINUP_DAYS` rows of the record are copied, their
    index shifted back by the same number of days, and their measured
    values (streamflow and river concentrations) set to NaN, so the
    model integrates a spin-up period before the first day of the
    record and the measured series are sparse over it.

    Args:
        df: The record indexed by daily dates, sorted ascending.

    Returns:
        The spin-up-inclusive record (the shifted copy of the first
        two years, then the record itself).
    """
    spinup: pd.DataFrame = df.iloc[:NUM_SPINUP_DAYS].copy()
    spinup.index = pd.DatetimeIndex(spinup.index) - pd.Timedelta(
        days=NUM_SPINUP_DAYS
    )
    for species in ("q_meas", "DOC", "DIC", "Ca++"):
        spinup[species] = np.nan
    return pd.concat([spinup, df])


@st.cache_data
def load_input_data(csv_path: Path = DEFAULT_CSV_PATH) -> GuiInputData:
    """
    Load and assemble the GUI inputs from the East River series CSV.

    Builds the reaction network and the East River bounds, reads the
    CSV, prepends the two-year spin-up, and creates the forcing, the
    precipitation concentrations, and the measured series, and caches
    the result for the session since the loader is deterministic.

    Args:
        csv_path: Path to the East River series CSV; defaults to the
            record shipped with the package.

    Returns:
        The assembled `GuiInputData`.
    """
    network: pt.ReactionNetwork = get_soc_calcite_network()
    bounds: dict[str, tuple[float, float]] = _bounds(network)

    df: pd.DataFrame = _with_spinup(_load_csv(csv_path))

    precip_conc: np.ndarray = np.full(
        (
            len(pt.LateralThreeLayerModel.structure[0]),
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
    conc_df: pd.DataFrame = df[["DOC", "DIC", "Ca++"]]

    return GuiInputData(
        network=network,
        bounds=bounds,
        zone_configs=LATERAL_THREE_LAYER_CONFIGS,
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

    model: pt.Model = pt.LateralThreeLayerModel.from_array(
        model_params,
        network=input_data.network,
        config=input_data.zone_configs,
        natural_scales=True,
        derived_species=derived_species,
    )

    return model, init_state_params


def _exchange_residual(
    x_conc: float, tot_exchange_sites: float, mean_doc: float, k_eq: float
) -> float:
    """
    Return the ion-exchange equilibrium residual for a candidate X-
    concentration.

    The residual of the XDOC exchange equilibrium, zero at the
    partition of the total exchange-site concentration into the
    aqueous cation (X-) and the adsorbed (XDOC) fractions.

    Args:
        x_conc: The candidate concentration of the cation X-.
        tot_exchange_sites: The total exchange-site concentration.
        mean_doc: The mean measured DOC concentration.
        k_eq: The exchange equilibrium constant.

    Returns:
        The residual, zero at the solution.
    """
    return k_eq * (tot_exchange_sites - x_conc) - x_conc * mean_doc


def arr_to_results(
    arr: np.ndarray,
    input_data: GuiInputData,
    init_aq_conc: float = 1e-3,
    verbose: bool = False,
) -> pt.ModelResults:
    """
    Run the hydrology and reactive transport models for a parameter
    array.

    Builds the model and the reactive zone initial states (fixed
    mineral concentrations and the calibrated DOC exchange-site
    concentrations) from the parameters and runs the hydrology and
    reactive transport models over the full (spin-up-inclusive)
    record.

    Args:
        arr: The flat parameter array of model parameters followed by
            initial-state parameters.
        input_data: The assembled GUI inputs.
        init_aq_conc: The initial aqueous concentration of all primary
            and secondary species other than DOC in each zone.
        verbose: Whether to print verbose output while running the
            models.

    Returns:
        The full ModelResults including the reactive transport
        simulation.
    """
    model: pt.Model
    init_state_params: np.ndarray
    model, init_state_params = arr_to_model(arr, input_data)

    network: pt.ReactionNetwork = input_data.network
    mean_doc: float = float(input_data.conc_df["DOC"].mean())

    # Construct the initial state
    init_state: dict[str, pt.ChemicalState] = {}
    init_idx: int = 0
    for zone_name, zone_config in input_data.zone_configs.items():
        init_i: pt.ChemicalState = pt.ChemicalState()

        for prim_name in network.primary_names:
            if prim_name == "DOC":
                init_i.primary[prim_name] = mean_doc
            else:
                init_i.primary[prim_name] = init_aq_conc

        for sec_name in network.secondary_names:
            init_i.secondary[sec_name] = init_aq_conc

        if zone_config.do_reactions:
            for min_name in network.mineral_names:
                init_i.mineral[min_name] = MINERAL_CONCENTRATION

        if zone_config.do_speciation and network.has_exchange:
            tot_exchange_sites: float = 10.0 ** float(init_state_params[init_idx])
            init_idx += 1
            k_eq: float = 10.0 ** network.exchange_species[0].log10_k_eq

            # A small rootfinding step finds the exchange-site
            # partition of DOC at initialization.
            residual = partial(
                _exchange_residual,
                tot_exchange_sites,
                mean_doc,
                k_eq,
            )
            x_init: float = float(root_scalar(residual, x0=1e-5).root)
            xdoc_init: float = tot_exchange_sites - x_init

            init_i.exchange["X-"] = x_init
            init_i.exchange["XDOC"] = xdoc_init

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

    return model_res
# =================== #
