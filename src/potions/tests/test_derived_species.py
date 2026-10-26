"""Tests for model-level derived measurements.

Some measured river quantities are not species of the reaction network but
are defined as the sum of several network species; dissolved inorganic
carbon (DIC) is the sum of ``CO2(aq)``, ``HCO3-``, and ``CO3--``. Declaring
such a measurement on the model must:

- add a ``{name}_riv`` column holding the sum of its components' river
  concentrations,
- allow a measurement column named after the derived measurement to be
  scored against that column (including when some measurements are missing),
- reject measurement columns that are neither network species nor declared
  derived measurements,
- reject, at construction time, derived names that collide with a network
  species or another derived measurement, empty component lists, and
  components that are not aqueous species of the network.
"""

import numpy as np
import pandas as pd
import pytest

from ..common_models import ThreeLayerModel
from ..common_types import (
    ChemicalState,
    DerivedSpecies,
    ForcingData,
    ModelResults,
)
from ..core import (
    MineralParameters,
    ReactionNetwork,
    RtParameters,
    RtZone,
    ZoneDimensions,
)
from ..objective_functions import DEFAULT_OBJECTIVE_FUNCTIONS
from ..reactive_transport.common_networks import get_carbon_network
from ..utils import ModelInputError

DIC_COMPONENTS: tuple[str, ...] = ("CO2(aq)", "HCO3-", "CO3--")


@pytest.fixture(scope="module")
def network() -> ReactionNetwork:
    return get_carbon_network()


def _make_rt_zones(network: ReactionNetwork) -> dict[str, RtZone]:
    """One reactive-transport zone per zone of ``ThreeLayerModel``."""
    mp = MineralParameters(
        np.array([0.1]), np.array([2.0]), np.array([1.5]), np.array([10.0])
    )
    dim = ZoneDimensions(100.0, 10.0)
    return {
        "snow": RtZone(
            network,
            RtParameters(dim, mp),
            do_reactions=False,
            do_speciation=False,
            name="snow",
        ),
        "surface": RtZone(
            network,
            RtParameters(dim, mp),
            do_reactions=True,
            do_speciation=True,
            name="surface",
        ),
        "ground": RtZone(
            network,
            RtParameters(dim, mp),
            do_reactions=True,
            do_speciation=True,
            name="ground",
        ),
    }


def _make_model(
    network: ReactionNetwork,
    derived_species: list[DerivedSpecies] | None,
) -> ThreeLayerModel:
    """Build the test model, optionally declaring derived measurements."""
    return ThreeLayerModel(
        scales=[1.0],
        network=network,
        rt_zones=_make_rt_zones(network),
        derived_species=derived_species,
    )


@pytest.fixture(scope="module")
def dates() -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=20, freq="D")


def _make_forcing(dates: pd.DatetimeIndex) -> ForcingData:
    n: int = len(dates)
    return ForcingData(
        precip=pd.Series(np.linspace(0.1, 2.0, n), index=dates),
        temp=pd.Series(np.linspace(0.0, 10.0, n), index=dates),
        pet=pd.Series(np.full(n, 0.5), index=dates),
    )


def _make_init_states(network: ReactionNetwork) -> dict[str, ChemicalState]:
    """A finite, mineral-bearing initial state for every zone."""
    return {
        "snow": ChemicalState(),
        "surface": ChemicalState(
            mineral={sp.name: 0.1 for sp in network.mineral},
            primary={
                "DOC": 1e-4,
                "HCO3-": 1e-3,
                "H+": 1e-7,
            },
            secondary={"CO2(aq)": 1e-6, "CO3--": 1e-6},
            exchange={"X-": 1e-3},
        ),
        "ground": ChemicalState(
            mineral={sp.name: 0.1 for sp in network.mineral},
            primary={
                "DOC": 1e-4,
                "HCO3-": 1e-3,
                "H+": 1e-7,
            },
            secondary={"CO2(aq)": 1e-6, "CO3--": 1e-6},
            exchange={"X-": 1e-3},
        ),
    }


def _make_precip_conc(
    network: ReactionNetwork, num_steps: int
) -> np.ndarray:
    return np.full(
        (1, num_steps, network.num_species), 1e-20, dtype=np.float64
    )


def _make_meas_river_conc(
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    n: int = len(dates)
    return pd.DataFrame(
        {
            "DOC": np.linspace(1e-4, 3e-4, n),
            "DIC": np.linspace(1e-4, 5e-4, n),
        },
        index=dates,
    )


@pytest.fixture(scope="module")
def model(network: ReactionNetwork) -> ThreeLayerModel:
    return _make_model(
        network, [DerivedSpecies(name="DIC", components=DIC_COMPONENTS)]
    )


@pytest.fixture(scope="module")
def results(
    model: ThreeLayerModel,
    network: ReactionNetwork,
    dates: pd.DatetimeIndex,
) -> ModelResults:
    meas = _make_meas_river_conc(dates)
    # Mark a missing measurement: the model excludes it pairwise, exactly as
    # it does for any other measured river concentration.
    meas.loc[dates[3], "DIC"] = np.nan
    return model.run_both_models(
        forc=_make_forcing(dates),
        precip_conc=_make_precip_conc(network, len(dates)),
        init_conc=_make_init_states(network),
        meas_river_conc=meas,
        save_results=False,
    )


# ===================== Derived river concentration ===================== #
def test_derived_riv_is_sum_of_component_riv(
    results: ModelResults,
) -> None:
    rt_sim = results.reactive_transport.simulation
    expected = (
        rt_sim["CO2(aq)_riv"]
        + rt_sim["HCO3-_riv"]
        + rt_sim["CO3--_riv"]
    )
    pd.testing.assert_series_equal(rt_sim["DIC_riv"], expected, check_names=False)


def test_derived_measurement_is_scored(
    results: ModelResults, dates: pd.DatetimeIndex
) -> None:
    rt_sim = results.reactive_transport.simulation
    obj = results.reactive_transport.objective_functions
    assert obj is not None
    assert "DIC" in obj.index

    meas = _make_meas_river_conc(dates)
    meas.loc[dates[3], "DIC"] = np.nan
    expected = np.array(
        [
            func(meas["DIC"], rt_sim["DIC_riv"])
            for _, func in DEFAULT_OBJECTIVE_FUNCTIONS
        ]
    )
    np.testing.assert_allclose(obj.loc["DIC"].to_numpy(), expected)


def test_network_species_still_scored(
    results: ModelResults,
) -> None:
    obj = results.reactive_transport.objective_functions
    assert obj is not None
    assert "DOC" in obj.index


# ===================== Input validation ===================== #
def test_unknown_measurement_column_raises(
    model: ThreeLayerModel,
    network: ReactionNetwork,
    dates: pd.DatetimeIndex,
) -> None:
    meas = pd.DataFrame(
        {"not_a_species": np.full(len(dates), 1e-4)}, index=dates
    )
    with pytest.raises(
        ModelInputError,
        match="neither species of the reaction network",
    ):
        model.run_both_models(
            forc=_make_forcing(dates),
            precip_conc=_make_precip_conc(network, len(dates)),
            init_conc=_make_init_states(network),
            meas_river_conc=meas,
            save_results=False,
        )


# ===================== Construction-time validation ===================== #
def test_network_species_name_collision_raises(
    network: ReactionNetwork,
) -> None:
    with pytest.raises(ValueError, match="already a species"):
        _make_model(
            network, [DerivedSpecies(name="DOC", components=DIC_COMPONENTS)]
        )


def test_duplicate_derived_name_raises(network: ReactionNetwork) -> None:
    derived = [DerivedSpecies(name="DIC", components=DIC_COMPONENTS)]
    with pytest.raises(ValueError, match="declared more than once"):
        _make_model(network, [*derived, *derived])


def test_derived_without_components_raises(
    network: ReactionNetwork,
) -> None:
    with pytest.raises(ValueError, match="has no component species"):
        _make_model(network, [DerivedSpecies(name="DIC", components=())])


def test_non_network_component_raises(network: ReactionNetwork) -> None:
    with pytest.raises(ValueError, match="not a species of the reaction"):
        _make_model(
            network, [DerivedSpecies(name="DIC", components=("CH4",))]
        )


def test_non_aqueous_component_raises(network: ReactionNetwork) -> None:
    with pytest.raises(ValueError, match="must be aqueous"):
        _make_model(
            network, [DerivedSpecies(name="DIC", components=("SOC(s)",))]
        )


def test_derived_species_without_network_raises() -> None:
    with pytest.raises(ValueError, match="requires a reactive transport"):
        ThreeLayerModel(
            scales=[1.0],
            derived_species=[
                DerivedSpecies(name="DIC", components=DIC_COMPONENTS)
            ],
        )
